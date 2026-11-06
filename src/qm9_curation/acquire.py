"""Download the QM9 source files and record their provenance (Phase 1).

Run as ``python -m qm9_curation.acquire``. For each file in
``configs/dataset.yaml`` this:

1. downloads it to ``data/raw/`` unless a copy that matches the published size
   and MD5 is already there (so re-running only re-verifies);
2. fails if the size or MD5 differs from figshare's published value;
3. computes SHA-256, which the rest of the pipeline uses as the file identity;
4. makes the file read-only;
5. writes ``data/raw/MANIFEST.json``.

Retrieval time and HTTP headers are kept from the first download, so
re-verifying an existing copy leaves the manifest unchanged.
"""

import argparse
import hashlib
import json
import os
import ssl
import stat
import tempfile
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import certifi

from qm9_curation import __version__
from qm9_curation.config import RAW_DIR, load_config

CHUNK_BYTES = 1 << 20
USER_AGENT = (
    f"qm9-curation/{__version__} (+https://github.com/akhilnathreddy/qm9-advanced-curation)"
)
MANIFEST_NAME = "MANIFEST.json"


class ChecksumError(RuntimeError):
    """A file's size or digest does not match its published value."""


@dataclass(frozen=True)
class SourceFile:
    name: str
    url: str
    size_bytes: int
    md5: str
    article_id: int
    article_doi: str
    file_id: int
    description: str

    @classmethod
    def from_config(cls, entry: Mapping[str, Any]) -> "SourceFile":
        return cls(
            name=entry["name"],
            url=entry["url"],
            size_bytes=int(entry["size_bytes"]),
            md5=entry["md5"],
            article_id=int(entry["article_id"]),
            article_doi=entry["article_doi"],
            file_id=int(entry["file_id"]),
            description=entry["description"],
        )


@dataclass(frozen=True)
class Digests:
    size_bytes: int
    md5: str
    sha256: str


def digest_file(path: Path) -> Digests:
    md5, sha256 = hashlib.md5(), hashlib.sha256()
    size = 0
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK_BYTES):
            md5.update(chunk)
            sha256.update(chunk)
            size += len(chunk)
    return Digests(size_bytes=size, md5=md5.hexdigest(), sha256=sha256.hexdigest())


def verify(source: SourceFile, digests: Digests) -> None:
    if digests.size_bytes != source.size_bytes:
        raise ChecksumError(
            f"{source.name}: size {digests.size_bytes} != published {source.size_bytes}"
        )
    if digests.md5 != source.md5:
        raise ChecksumError(f"{source.name}: md5 {digests.md5} != published {source.md5}")


def download(url: str, dest: Path) -> dict[str, str | None]:
    """Stream ``url`` to ``dest`` via a ``.part`` file; return selected response headers."""
    part = dest.with_name(dest.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    # certifi's CA bundle, because python.org builds of Python on macOS ship without one.
    context = ssl.create_default_context(cafile=certifi.where())
    with (
        urllib.request.urlopen(request, timeout=60, context=context) as response,
        part.open("wb") as fh,
    ):
        headers = {
            "final_url": response.geturl(),
            "etag": response.headers.get("ETag"),
            "last_modified": response.headers.get("Last-Modified"),
            "content_length": response.headers.get("Content-Length"),
        }
        while chunk := response.read(CHUNK_BYTES):
            fh.write(chunk)
    part.replace(dest)
    return headers


def make_read_only(path: Path) -> None:
    mode = path.stat().st_mode
    path.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def _load_manifest(raw_dir: Path) -> dict[str, Any]:
    path = raw_dir / MANIFEST_NAME
    if not path.exists():
        return {}
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def acquire(
    sources: list[SourceFile], raw_dir: Path, dataset_cfg: Mapping[str, Any]
) -> dict[str, Any]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    previous = {f["name"]: f for f in _load_manifest(raw_dir).get("files", [])}
    entries = []
    for source in sources:
        dest = raw_dir / source.name
        prior = previous.get(source.name)
        if dest.exists() and prior is not None:
            digests = digest_file(dest)
            verify(source, digests)
            retrieval = prior["retrieval"]
        else:
            if dest.exists():
                # Present but without provenance: replace it with a recorded download.
                dest.chmod(dest.stat().st_mode | stat.S_IWUSR)
                dest.unlink()
            headers = download(source.url, dest)
            digests = digest_file(dest)
            try:
                verify(source, digests)
            except ChecksumError:
                dest.unlink()
                raise
            if prior is not None and prior["sha256"] == digests.sha256:
                # Fresh clone: same bytes as the recorded download, so keep its provenance.
                retrieval = prior["retrieval"]
            else:
                retrieval = {
                    "retrieved_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    **headers,
                }
        make_read_only(dest)
        entries.append(
            {
                "name": source.name,
                "description": source.description,
                "url": source.url,
                "figshare_article_id": source.article_id,
                "figshare_article_doi": source.article_doi,
                "figshare_file_id": source.file_id,
                "size_bytes": digests.size_bytes,
                "md5": digests.md5,
                "sha256": digests.sha256,
                "retrieval": retrieval,
            }
        )
    src = dataset_cfg["source"]
    return {
        "dataset": src["name"],
        "paper_doi": src["paper_doi"],
        "collection_doi": src["collection_doi"],
        "license": src["license"],
        "license_url": src["license_url"],
        "files": entries,
    }


def verify_remote(sources: list[SourceFile], raw_dir: Path) -> dict[str, bool]:
    """Download every file again to a temporary directory; compare SHA-256 with the manifest.

    Confirms the source still serves the same bytes (Gate 1). Nothing in
    ``raw_dir`` is touched.
    """
    recorded = {f["name"]: f["sha256"] for f in _load_manifest(raw_dir).get("files", [])}
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        for source in sources:
            dest = Path(tmp) / source.name
            download(source.url, dest)
            results[source.name] = digest_file(dest).sha256 == recorded.get(source.name)
    return results


def write_manifest(manifest: Mapping[str, Any], raw_dir: Path) -> Path:
    path = raw_dir / MANIFEST_NAME
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument(
        "--verify-remote",
        action="store_true",
        help="download everything again to a temp dir and compare with the manifest",
    )
    args = parser.parse_args(argv)
    cfg = load_config("dataset")
    sources = [SourceFile.from_config(entry) for entry in cfg["files"]]
    if args.verify_remote:
        results = verify_remote(sources, args.raw_dir)
        for name, ok in results.items():
            print(f"{'PASS' if ok else 'FAIL'}  second download matches manifest: {name}")
        if not all(results.values()):
            raise SystemExit(1)
        return
    manifest = acquire(sources, args.raw_dir, cfg)
    path = write_manifest(manifest, args.raw_dir)
    for entry in manifest["files"]:
        print(f"ok  {entry['name']:<24} {entry['size_bytes']:>10} B  sha256 {entry['sha256'][:16]}")
    print(f"Manifest written to {path}")


if __name__ == "__main__":
    main()
