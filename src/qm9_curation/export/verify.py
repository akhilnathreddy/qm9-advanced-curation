"""Verify a built release against the published manifest (Gate 10, ``make reproduce``).

Run as ``python -m qm9_curation.export.verify``. Checks, for the release built
under ``data/release/``:

1. every file's SHA-256 equals the published manifest
   (``release/<name>.sha256``, committed to git). With ``--compare-with`` (another
   build, e.g. the CI artifact), each differing Parquet file also gets the
   largest numeric difference per column, to locate the cause;
2. the release's own ``MANIFEST.sha256`` matches its files;
3. ASE reads every extended-XYZ frame, and the coordinates equal ``atoms``;
4. ``croissant.json`` validates with mlcroissant (no errors);
5. the README quick start (at most 5 lines) runs and loads the Gold tier and
   a named split.

Writes ``reports/release/verify_report.json`` and exits non-zero on failure.
Needs the dev dependency group (ASE, mlcroissant), which ``uv sync`` installs.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from qm9_curation.config import PROJECT_ROOT, REPORTS_DIR, load_config
from qm9_curation.export.release import EXTXYZ, MANIFEST, RELEASE_DIR, release_name, sha256_file

PUBLISHED_DIR = PROJECT_ROOT / "release"


def read_manifest(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        sha, rel = line.split("  ", 1)
        out[rel] = sha
    return out


def parquet_diff(a: Path, b: Path) -> dict[str, Any]:
    """Largest absolute difference per numeric column; equality for the rest."""
    fa, fb = pl.read_parquet(a), pl.read_parquet(b)
    if fa.schema != fb.schema or fa.height != fb.height:
        return {"schema_or_rows_differ": True}
    out: dict[str, Any] = {}
    for col, dtype in fa.schema.items():
        if dtype.is_float():
            x, y = fa[col].to_numpy(), fb[col].to_numpy()
            both = np.isfinite(x) & np.isfinite(y)
            if not np.array_equal(np.isfinite(x), np.isfinite(y)):
                out[col] = "missing values differ"
            elif both.any() and not np.array_equal(x[both], y[both]):
                out[col] = float(np.max(np.abs(x[both] - y[both])))
        elif not fa[col].equals(fb[col]):
            out[col] = "values differ"
    return out


def check_manifest(release: Path, published: Path) -> dict[str, Any]:
    built = {
        p.relative_to(release).as_posix(): sha256_file(p)
        for p in sorted(release.rglob("*"))
        if p.is_file() and p.name != MANIFEST
    }
    expected = read_manifest(published)
    own = read_manifest(release / MANIFEST)
    differing = sorted(k for k in expected if built.get(k) != expected[k])
    return {
        "published_manifest": (
            str(published.relative_to(PROJECT_ROOT))
            if published.is_relative_to(PROJECT_ROOT)
            else str(published)
        ),
        "missing_files": sorted(set(expected) - set(built)),
        "extra_files": sorted(set(built) - set(expected)),
        "differing_files": differing,
        "own_manifest_matches": own == built,
        "identical": built == expected,
    }


def check_extxyz(release: Path) -> dict[str, Any]:
    from ase.io import iread

    atoms = pl.read_parquet(release / "atoms.parquet").sort("qm9_id", "atom_idx")
    xyz = atoms.select("x_ang", "y_ang", "z_ang").to_numpy()
    numbers = atoms["atomic_number"].to_numpy()
    n_frames, k, exact = 0, 0, True
    for frame in iread(release / EXTXYZ, index=":"):
        n = len(frame)
        exact &= np.array_equal(frame.positions, xyz[k : k + n]) and np.array_equal(
            frame.numbers, numbers[k : k + n]
        )
        k += n
        n_frames += 1
    molecules = pl.scan_parquet(release / "molecules.parquet").select(pl.len()).collect().item()
    return {
        "frames": n_frames,
        "atoms": k,
        "one_frame_per_record": n_frames == molecules,
        "coordinates_exact": bool(exact and k == len(xyz)),
    }


def check_croissant(release: Path) -> dict[str, Any]:
    import warnings

    with warnings.catch_warnings():
        # rdflib (used by mlcroissant) emits its own deprecation warnings.
        warnings.simplefilter("ignore", DeprecationWarning)
        import mlcroissant as mlc

        dataset = mlc.Dataset(jsonld=release / "croissant.json")
    issues = dataset.metadata.issues
    return {
        "errors": sorted(str(e) for e in issues.errors),
        "warnings": sorted(str(w) for w in issues.warnings),
        "record_sets": [rs.name for rs in dataset.metadata.record_sets],
        "valid": not issues.errors,
    }


def check_quickstart(release: Path) -> dict[str, Any]:
    text = (release / "README.md").read_text(encoding="utf-8")
    block = re.search(r"<!-- quickstart -->\n```python\n(.*?)```", text, re.S)
    if block is None:
        return {"found": False, "ok": False}
    code = block.group(1)
    lines = [ln for ln in code.splitlines() if ln.strip()]
    probe = code + (
        "assert mol.height > 0 and (mol['quality_tier'] == 'gold').all()\n"
        "assert train.height > 0 and test.height > 0\n"
        "print(mol.height, train.height, test.height)\n"
    )
    run = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=release.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "found": True,
        "lines": len(lines),
        "output": run.stdout.strip(),
        "stderr_tail": run.stderr.strip()[-500:],
        "ok": run.returncode == 0 and len(lines) <= 5,
    }


def verify(release: Path, published: Path, compare_with: Path | None = None) -> dict[str, Any]:
    manifest = check_manifest(release, published)
    if compare_with is not None:
        manifest["parquet_differences"] = {
            rel: parquet_diff(release / rel, compare_with / rel)
            for rel in manifest["differing_files"]
            if rel.endswith(".parquet") and (compare_with / rel).exists()
        }
    report: dict[str, Any] = {
        "release": release.name,
        "manifest": manifest,
        "extxyz": check_extxyz(release),
        "croissant": check_croissant(release),
        "quickstart": check_quickstart(release),
    }
    report["gate_10_verify_conditions"] = {
        "release_matches_published_manifest": manifest["identical"],
        "release_manifest_matches_files": manifest["own_manifest_matches"],
        "ase_reads_every_frame_with_exact_coordinates": report["extxyz"]["one_frame_per_record"]
        and report["extxyz"]["coordinates_exact"],
        "croissant_metadata_valid": report["croissant"]["valid"],
        "gold_tier_and_named_split_load_in_5_lines": report["quickstart"]["ok"],
    }
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, default=RELEASE_DIR)
    parser.add_argument("--published-dir", type=Path, default=PUBLISHED_DIR)
    parser.add_argument(
        "--compare-with",
        type=Path,
        default=None,
        help="Another build of the same release (e.g. the CI artifact): per-column "
        "differences of every Parquet file that differs from the published manifest.",
    )
    parser.add_argument(
        "--report", type=Path, default=REPORTS_DIR / "release" / "verify_report.json"
    )
    args = parser.parse_args(argv)
    name = release_name(load_config("release"))
    report = verify(
        args.release_root / name, args.published_dir / f"{name}.sha256", args.compare_with
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for condition, ok in report["gate_10_verify_conditions"].items():
        print(f"{'PASS' if ok else 'FAIL'}  {condition}")
    if report["manifest"]["differing_files"]:
        print(f"Differs from the published manifest: {report['manifest']['differing_files']}")
    if not all(report["gate_10_verify_conditions"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
