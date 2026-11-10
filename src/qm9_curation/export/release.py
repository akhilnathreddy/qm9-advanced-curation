"""Build the dataset release (Phase 10, ADR 0014).

Run as ``python -m qm9_curation.export.release``. Writes
``data/release/qm9-curated-v<version>/``:

- the release tables as Parquet, copied byte for byte from the pipeline;
- ``qm9_curated.extxyz`` (ASE-readable geometries, one frame per record);
- ``croissant.json`` (Croissant 1.0), ``datasheet.md``, ``README.md``,
  ``CITATION.cff`` and ``LICENSE`` (CC0-1.0);
- ``MANIFEST.sha256``: the SHA-256 of every other file, in ``sha256sum``
  format, so ``sha256sum -c MANIFEST.sha256`` checks a download.

Every output is deterministic: rebuilding gives byte-identical files. The
report ``reports/release/release_report.json`` records file hashes and the
build-side Gate 10 conditions.
"""

import argparse
import hashlib
import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from qm9_curation.config import PROJECT_ROOT, REPORTS_DIR, load_config
from qm9_curation.export import croissant
from qm9_curation.export.xyz import write_extxyz

RELEASE_DIR = PROJECT_ROOT / "data" / "release"
EXTXYZ = "qm9_curated.extxyz"
MANIFEST = "MANIFEST.sha256"
TIERS = ("gold", "silver", "bronze", "quarantine")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def release_name(cfg: Mapping[str, Any]) -> str:
    return f"{cfg['name']}-v{cfg['version']}"


def cite_as(cfg: Mapping[str, Any]) -> str:
    year = str(cfg["date_published"])[:4]
    return (
        f"{cfg['author']} ({year}). {cfg['title']}. Version {cfg['version']}. "
        f"{cfg['repository']}. Derived from Ramakrishnan et al., Sci. Data 1, 140022 "
        "(2014), doi:10.1038/sdata.2014.22."
    )


def citation_cff(cfg: Mapping[str, Any]) -> str:
    doc = {
        "cff-version": "1.2.0",
        "message": "If you use this dataset, please cite it and the original QM9 paper.",
        "type": "dataset",
        "title": cfg["title"],
        "version": cfg["version"],
        "date-released": str(cfg["date_published"]),
        "authors": [{"alias": cfg["author"]}],
        "license": cfg["data_license"],
        "repository-code": cfg["repository"],
        "keywords": list(cfg["keywords"]),
        "references": [
            {
                "type": "article",
                "title": "Quantum chemistry structures and properties of 134 kilo molecules",
                "authors": [
                    {"family-names": "Ramakrishnan", "given-names": "Raghunathan"},
                    {"family-names": "Dral", "given-names": "Pavlo O."},
                    {"family-names": "Rupp", "given-names": "Matthias"},
                    {"family-names": "von Lilienfeld", "given-names": "O. Anatole"},
                ],
                "journal": "Scientific Data",
                "volume": 1,
                "start": 140022,
                "year": 2014,
                "doi": "10.1038/sdata.2014.22",
            },
            {
                "type": "article",
                "title": (
                    "Energy refinement and analysis of structures in the QM9 database "
                    "via a highly accurate quantum chemical method"
                ),
                "authors": [
                    {"family-names": "Kim", "given-names": "Hyungjun"},
                    {"family-names": "Park", "given-names": "Ji Young"},
                    {"family-names": "Choi", "given-names": "Sunghwan"},
                ],
                "journal": "Scientific Data",
                "volume": 6,
                "start": 109,
                "year": 2019,
                "doi": "10.1038/s41597-019-0121-7",
            },
        ],
    }
    return yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=88)


LICENSE_TEXT = """\
QM9 Curated (dataset files in this directory)

To the extent possible under law, the author has waived all copyright and
related or neighboring rights to this dataset under the Creative Commons CC0
1.0 Universal Public Domain Dedication:
https://creativecommons.org/publicdomain/zero/1.0/legalcode

The underlying data are QM9 (Ramakrishnan et al., 2014) and the G4(MP2)
values of Kim et al. (2019), both released under CC0-1.0. Citation of these
works and of this release is requested, not required.

The code that builds this release is MIT-licensed:
https://github.com/akhilnathreddy/qm9-advanced-curation
"""


def readme(
    cfg: Mapping[str, Any], tiers: Mapping[str, int], files: list[dict[str, Any]], n_frames: int
) -> str:
    name = release_name(cfg)
    rows = "\n".join(
        f"| `{f['path']}` | {f['rows']:,} | {f['bytes'] / 1e6:.1f} MB |"
        if f["rows"] is not None
        else f"| `{f['path']}` | — | {f['bytes'] / 1e6:.1f} MB |"
        for f in files
    )
    tier_rows = "\n".join(f"| {t} | {tiers.get(t, 0):,} |" for t in TIERS)
    return f"""# {cfg["title"]}

Version {cfg["version"]} ({cfg["date_published"]}). Data license: {cfg["data_license"]}.
Code and full documentation: {cfg["repository"]}

{" ".join(str(cfg["description"]).split())}

## Quick start

Load the Gold tier and the training rows of a named split (Python, Polars):

<!-- quickstart -->
```python
import polars as pl
mol = pl.read_parquet("{name}/molecules.parquet").filter(pl.col("quality_tier") == "gold")
split = pl.read_parquet("{name}/splits.parquet").filter(pl.col("split_name") == "scaffold_generic", pl.col("seed") == 0)
train = mol.join(split.filter(pl.col("subset") == "train"), on="qm9_id")
test = mol.join(split.filter(pl.col("subset") == "test"), on="qm9_id")
```
<!-- /quickstart -->

Geometries: `ase.io.read("{name}/{EXTXYZ}", index=":")` gives one
`ase.Atoms` per record ({n_frames:,} frames) with properties in `atoms.info`
and Mulliken charges in `atoms.arrays["mulliken_charge"]`. A PyTorch
Geometric loader is in the code repository
(`qm9_curation.export.pyg.QM9Curated`).

## Tiers

| Tier | Records |
|---|---|
{tier_rows}

Gold passes every hard and identity check and is duplicate-free; Silver is
usable with a documented caveat; Bronze geometries are not the labelled
molecule; Quarantine records are not true minima. `tier_reasons` gives the
rule ids. Evaluate on Gold; see `datasheet.md` for intended uses.

## Files

| File | Rows | Size |
|---|---|---|
{rows}

Column definitions are in `croissant.json` (Croissant 1.0) and the
repository's `docs/data_dictionary.md`. Check a download with
`sha256sum -c {MANIFEST}`.

## Citation

{cite_as(cfg)}
See `CITATION.cff`.
"""


def build(
    cfg: Mapping[str, Any],
    tables: Mapping[str, Path],
    out_root: Path,
    datasheet: Path,
    dictionary: Path,
    n_expected: int = 133_885,
) -> dict[str, Any]:
    out = out_root / release_name(cfg)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    files: list[dict[str, Any]] = []
    schemas: dict[str, Any] = {}
    copied_identical = True
    for table, src in tables.items():
        dst = out / f"{table}.parquet"
        shutil.copyfile(src, dst)
        copied_identical &= sha256_file(dst) == sha256_file(src)
        schemas[table] = pl.read_parquet_schema(dst)
        rows = pl.scan_parquet(dst).select(pl.len()).collect().item()
        files.append({"path": dst.name, "rows": rows, "table": table})

    molecules = pl.read_parquet(tables["molecules"])
    atoms = pl.read_parquet(tables["atoms"])
    n_frames = write_extxyz(out / EXTXYZ, molecules, atoms, cfg["extxyz_properties"])
    files.append({"path": EXTXYZ, "rows": n_frames, "table": None})

    sections = croissant.dictionary(dictionary)
    descriptions = {t: croissant.describe(t, list(s), sections) for t, s in schemas.items()}
    undocumented = {t: [c for c, d in desc.items() if not d] for t, desc in descriptions.items()}
    undocumented = {t: cols for t, cols in undocumented.items() if cols}
    sha = {f["path"]: sha256_file(out / f["path"]) for f in files}
    meta = croissant.metadata(
        cfg,
        {f["table"]: (f["path"], sha[f["path"]]) for f in files if f["table"]},
        schemas,
        descriptions,
        cite_as(cfg),
    )
    (out / "croissant.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    shutil.copyfile(datasheet, out / "datasheet.md")
    (out / "CITATION.cff").write_text(citation_cff(cfg), encoding="utf-8")
    (out / "LICENSE").write_text(LICENSE_TEXT, encoding="utf-8")

    tiers = dict(molecules.group_by("quality_tier").len().iter_rows())
    for f in files:
        f["bytes"] = (out / f["path"]).stat().st_size
    (out / "README.md").write_text(readme(cfg, tiers, files, n_frames), encoding="utf-8")

    listed = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    hashes = {rel: sha256_file(out / rel) for rel in listed}
    (out / MANIFEST).write_text(
        "".join(f"{hashes[rel]}  {rel}\n" for rel in listed), encoding="utf-8"
    )

    n = molecules.height
    conditions = {
        "every_source_record_in_release": n == n_expected and molecules["qm9_id"].n_unique() == n,
        "every_record_has_a_tier": bool(molecules["quality_tier"].is_in(TIERS).all()),
        "tables_byte_identical_to_pipeline_outputs": copied_identical,
        "extxyz_has_one_frame_per_record": n_frames == n,
        "every_release_column_documented": not undocumented,
        "manifest_covers_every_file": set(listed)
        == {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()} - {MANIFEST},
    }
    return {
        "name": release_name(cfg),
        "version": cfg["version"],
        "date_published": str(cfg["date_published"]),
        "tiers": {t: tiers.get(t, 0) for t in TIERS},
        "files": [
            {**{k: v for k, v in f.items() if k != "table"}, "sha256": hashes[f["path"]]}
            for f in files
        ],
        "undocumented_columns": undocumented,
        "output_sha256": hashes,
        "manifest_sha256": sha256_file(out / MANIFEST),
        "gate_10_conditions": conditions,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=RELEASE_DIR)
    parser.add_argument("--datasheet", type=Path, default=PROJECT_ROOT / "docs" / "datasheet.md")
    parser.add_argument(
        "--dictionary", type=Path, default=PROJECT_ROOT / "docs" / "data_dictionary.md"
    )
    parser.add_argument(
        "--report", type=Path, default=REPORTS_DIR / "release" / "release_report.json"
    )
    args = parser.parse_args(argv)
    cfg = load_config("release")
    tables = {t: PROJECT_ROOT / p for t, p in cfg["tables"].items()}
    report = build(cfg, tables, args.out_root, args.datasheet, args.dictionary)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for condition, ok in report["gate_10_conditions"].items():
        print(f"{'PASS' if ok else 'FAIL'}  {condition}")
    if report["undocumented_columns"]:
        print(f"Undocumented columns: {report['undocumented_columns']}")


if __name__ == "__main__":
    main()
