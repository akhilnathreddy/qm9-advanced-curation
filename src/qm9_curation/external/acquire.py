"""Download the QM9-G4(MP2) tables of Kim et al. (2019) and record provenance.

Run as ``python -m qm9_curation.external.acquire``. Same procedure as the
QM9 source files (ADR 0005): published size and MD5 must match, SHA-256 is
recorded, the file is made read-only, and ``data/external/kim2019/MANIFEST.json``
is written.
"""

import argparse
from pathlib import Path

from qm9_curation import acquire
from qm9_curation.config import EXTERNAL_DIR, load_config

KIM2019_DIR = EXTERNAL_DIR / "kim2019"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=KIM2019_DIR)
    args = parser.parse_args(argv)
    cfg = load_config("external")["kim2019"]
    sources = [acquire.SourceFile.from_config(entry) for entry in cfg["files"]]
    manifest = acquire.acquire(sources, args.out_dir, {"source": cfg})
    path = acquire.write_manifest(manifest, args.out_dir)
    for entry in manifest["files"]:
        print(f"ok  {entry['name']}  {entry['size_bytes']} B  sha256 {entry['sha256'][:16]}")
    print(f"Manifest written to {path}")


if __name__ == "__main__":
    main()
