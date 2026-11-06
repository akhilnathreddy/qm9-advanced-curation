import hashlib
import json
import os
from pathlib import Path

import pytest

from qm9_curation import acquire
from qm9_curation.acquire import ChecksumError, Digests, SourceFile

CFG = {
    "source": {
        "name": "test",
        "paper_doi": "10.0/test",
        "collection_doi": "10.0/test.c",
        "license": "CC0-1.0",
        "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
    }
}


def make_source(
    tmp_path: Path, content: bytes = b"hello qm9\n", md5: str | None = None
) -> SourceFile:
    origin = tmp_path / "origin" / "file.txt"
    origin.parent.mkdir(exist_ok=True)
    origin.write_bytes(content)
    return SourceFile(
        name="file.txt",
        url=origin.as_uri(),
        size_bytes=len(content),
        md5=md5 or hashlib.md5(content).hexdigest(),
        article_id=1,
        article_doi="10.0/a",
        file_id=2,
        description="test file",
    )


def test_verify_rejects_wrong_size_and_md5(tmp_path: Path) -> None:
    source = make_source(tmp_path)
    with pytest.raises(ChecksumError, match="size"):
        acquire.verify(source, Digests(size_bytes=1, md5=source.md5, sha256=""))
    with pytest.raises(ChecksumError, match="md5"):
        acquire.verify(source, Digests(size_bytes=source.size_bytes, md5="0" * 32, sha256=""))


def test_acquire_downloads_verifies_and_locks(tmp_path: Path) -> None:
    source = make_source(tmp_path)
    raw = tmp_path / "raw"
    manifest = acquire.acquire([source], raw, CFG)
    entry = manifest["files"][0]
    assert entry["sha256"] == hashlib.sha256(b"hello qm9\n").hexdigest()
    assert entry["retrieval"]["retrieved_at_utc"].endswith("Z")
    assert not os.access(raw / "file.txt", os.W_OK)
    assert not (raw / "file.txt.part").exists()


def test_rerun_only_reverifies_and_keeps_manifest(tmp_path: Path) -> None:
    source = make_source(tmp_path)
    raw = tmp_path / "raw"
    first = acquire.acquire([source], raw, CFG)
    acquire.write_manifest(first, raw)
    (tmp_path / "origin" / "file.txt").unlink()  # a re-download would now fail
    second = acquire.acquire([source], raw, CFG)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_corrupted_download_is_removed(tmp_path: Path) -> None:
    source = make_source(tmp_path, md5="0" * 32)
    raw = tmp_path / "raw"
    with pytest.raises(ChecksumError):
        acquire.acquire([source], raw, CFG)
    assert not (raw / "file.txt").exists()


def test_corrupted_local_copy_fails_reverification(tmp_path: Path) -> None:
    source = make_source(tmp_path)
    raw = tmp_path / "raw"
    acquire.write_manifest(acquire.acquire([source], raw, CFG), raw)
    target = raw / "file.txt"
    target.chmod(0o644)
    target.write_bytes(b"tampered!!\n")
    with pytest.raises(ChecksumError):
        acquire.acquire([source], raw, CFG)


def test_verify_remote_compares_against_manifest(tmp_path: Path) -> None:
    source = make_source(tmp_path)
    raw = tmp_path / "raw"
    acquire.write_manifest(acquire.acquire([source], raw, CFG), raw)
    assert acquire.verify_remote([source], raw) == {"file.txt": True}
    (tmp_path / "origin" / "file.txt").write_bytes(b"changed upstream\n")
    assert acquire.verify_remote([source], raw) == {"file.txt": False}


def test_fresh_clone_redownload_keeps_recorded_provenance(tmp_path: Path) -> None:
    source = make_source(tmp_path)
    raw = tmp_path / "raw"
    first = acquire.acquire([source], raw, CFG)
    acquire.write_manifest(first, raw)
    target = raw / "file.txt"
    target.chmod(0o644)
    target.unlink()  # manifest is in git, data is not
    second = acquire.acquire([source], raw, CFG)
    assert second["files"][0]["retrieval"] == first["files"][0]["retrieval"]
