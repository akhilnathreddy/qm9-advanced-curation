"""Zenodo upload bundle for the release (the user uploads it from their own account).

Run as ``python -m qm9_curation.export.zenodo`` (``make zenodo``). Writes
``data/zenodo/``:

- ``<name>.zip``: the release directory, zipped deterministically (sorted
  entries, fixed timestamps), so the same release always gives the same zip;
- ``zenodo_metadata.json``: deposition metadata in Zenodo's REST API format
  (``{"metadata": {...}}``), for the web form or the API;
- ``UPLOAD.md``: the upload steps.

Nothing is sent anywhere: publishing on Zenodo mints a DOI that cannot be
deleted, so it stays a manual step (ADR 0014).
"""

import argparse
import json
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from qm9_curation.config import DATA_DIR, load_config
from qm9_curation.export.release import RELEASE_DIR, release_name, sha256_file

ZENODO_DIR = DATA_DIR / "zenodo"
FIXED_TIME = (1980, 1, 1, 0, 0, 0)


def deterministic_zip(src: Path, dest: Path) -> None:
    files = sorted(p for p in src.rglob("*") if p.is_file())
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in files:
            info = zipfile.ZipInfo(f"{src.name}/{path.relative_to(src).as_posix()}", FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            with path.open("rb") as fh:
                zf.writestr(info, fh.read(), compresslevel=9)


def metadata(cfg: Mapping[str, Any], release_readme: str) -> dict[str, Any]:
    description = (
        "<p>"
        + " ".join(str(cfg["description"]).split())
        + "</p><p>Built and verified by the open pipeline at "
        + f'<a href="{cfg["repository"]}">{cfg["repository"]}</a>, which rebuilds '
        "every file of this release from the original QM9 archives with identical "
        "SHA-256 hashes (<code>make reproduce</code>). See <code>README.md</code>, "
        "<code>datasheet.md</code> and <code>croissant.json</code> in the archive.</p>"
    )
    related = [
        {"identifier": s["doi"], "relation": "isDerivedFrom", "scheme": "doi"}
        for s in cfg["sources"]
    ]
    related.append(
        {"identifier": cfg["repository"], "relation": "isSupplementedBy", "scheme": "url"}
    )
    return {
        "metadata": {
            "upload_type": "dataset",
            "title": cfg["title"],
            "creators": [{"name": cfg["author"]}],
            "description": description,
            "version": cfg["version"],
            "publication_date": str(cfg["date_published"]),
            "access_right": "open",
            "license": "cc-zero",
            "keywords": list(cfg["keywords"]),
            "related_identifiers": related,
            "notes": "Release README:\n\n" + release_readme,
        }
    }


UPLOAD_STEPS = """# Uploading {name} to Zenodo

Publishing mints a permanent DOI; a published record cannot be deleted, only
superseded by a new version. Do this from your own account.

1. Optional rehearsal: repeat these steps on https://sandbox.zenodo.org first.
2. Sign in at https://zenodo.org and choose **New upload**.
3. Upload `{zip_name}` ({size_mb:.0f} MB, SHA-256 `{zip_sha}`).
4. Fill the form from `zenodo_metadata.json`: resource type *Dataset*,
   title, creator, description, version {version}, license *Creative Commons
   Zero v1.0 Universal*, keywords and the related identifiers (QM9 and Kim et
   al. DOIs as *Is derived from*, the GitHub repository as *Is supplemented
   by*).
5. Optionally reserve the DOI before publishing and add it to the GitHub
   README and `CITATION.cff` in a PATCH release.
6. **Publish.** Then record the DOI in `configs/release.yaml` (a metadata-only
   change: PATCH version) and in `docs/release.md`.
"""


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, default=RELEASE_DIR)
    parser.add_argument("--out-dir", type=Path, default=ZENODO_DIR)
    args = parser.parse_args(argv)
    cfg = load_config("release")
    name = release_name(cfg)
    src = args.release_root / name
    args.out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = args.out_dir / f"{name}.zip"
    deterministic_zip(src, zip_path)
    meta = metadata(cfg, (src / "README.md").read_text(encoding="utf-8"))
    (args.out_dir / "zenodo_metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    zip_sha = sha256_file(zip_path)
    (args.out_dir / "UPLOAD.md").write_text(
        UPLOAD_STEPS.format(
            name=name,
            zip_name=zip_path.name,
            size_mb=zip_path.stat().st_size / 1e6,
            zip_sha=zip_sha,
            version=cfg["version"],
        ),
        encoding="utf-8",
    )
    print(f"{zip_path} ({zip_path.stat().st_size / 1e6:.0f} MB, sha256 {zip_sha})")


if __name__ == "__main__":
    main()
