"""Record a phase exit-gate result in ``reports/gates/phase_<N>.json``.

Each gate lists its conditions (plan v2, section 4). The Makefile target for a
gate runs the commands that prove those conditions first, so reaching this
step means they passed. Where a condition is recorded in a report file, it is
read back here and the gate fails unless every one of them holds. The record
ties the result to the code and environment that produced it.
"""

import argparse
import json
from pathlib import Path
from typing import Any

from qm9_curation.config import REPORTS_DIR
from qm9_curation.env_report import collect

GATE_CONDITIONS: dict[int, tuple[str, ...]] = {
    0: (
        "uv sync --locked succeeds",
        "ruff check and ruff format --check pass",
        "mypy --strict passes on src/",
        "pytest passes",
        "dvc repro succeeds",
    ),
    1: (
        "every raw file matches its published size and MD5 (acquire)",
        "a second download of every file matches the manifest SHA-256 (acquire --verify-remote)",
        "inventory conditions all hold (reports/data_quality/inventory.json)",
        "pytest passes, including the 1k fixture tests",
    ),
    2: (
        "every record parses under the strict layout (no failures)",
        "every record serializes back to its exact source bytes",
        "every numeric token's value is held exactly by the float64 tables",
        "all Pandera schemas validate",
        "atom and frequency row counts match the per-molecule counts",
        "a second pipeline run produces byte-identical Parquet files",
    ),
    3: (
        "profile.json has every section and one profile row per molecule",
        "every issue has a triage class and an owning phase",
        "geometric duplicate pairs are energy-consistent (|dU0| < 0.1 kcal/mol)",
        "clean checks pass (no clashes, no missing values, GDB-17 SMILES parse)",
        "a second profile run produces byte-identical outputs, including the HTML",
    ),
    4: (
        "every registered check has a passing and a failing test case",
        "one flag row per molecule with a result for every check",
        "every tolerance is set and justified in ADR 0008",
        "a second validation run produces a byte-identical flag table",
    ),
    5: (
        "every record has a tier, and every non-Gold record a machine-readable reason",
        "source values are preserved exactly; unit conversions are invertible",
        "one representative per duplicate group; Gold is duplicate-free",
        "the curated table matches its schema",
        "a second curation run produces a byte-identical table",
    ),
    6: (
        "every feature column is in the feature dictionary",
        "features read only atoms and coordinates (leakage guard)",
        "invariance tests pass (pose and atom order)",
        "no non-finite values in any feature table",
        "a second feature run produces byte-identical tables",
    ),
    7: (
        "external G4(MP2) data verified (published MD5, CC0) and its counts match the paper",
        "atom references checked; values withheld wherever a reference is invalid",
        "distribution tests, correlations and residual tables cover every target and tier",
        "every Layer 4 flag is soft: one review row per record, tiers unchanged",
        "unresolved uncertainties are listed in docs/limitations.md",
        "a second review run produces byte-identical tables",
    ),
    8: (
        "every split assigns every record exactly once, for every seed",
        "no leakage key (exact structure, tautomer, connectivity, geometry InChIKey block) "
        "or split unit occurs in two subsets",
        "holdout rules respected; grouped splits within the fraction tolerance",
        "distribution-shift report for every split (targets, composition, NN Tanimoto)",
        "analysis plan frozen (splits fixed)",
        "a second split run produces byte-identical tables",
    ),
}

# Report files whose boolean conditions must all be true for the gate to pass.
GATE_REPORTS: dict[int, tuple[tuple[str, str], ...]] = {
    1: (("data_quality/inventory.json", "gate_1_conditions"),),
    2: (("data_quality/parse_report.json", "gate_2_conditions"),),
    3: (("data_quality/profile.json", "gate_3_conditions"),),
    4: (("validation/validation_report.json", "gate_4_conditions"),),
    5: (("curation/curation_report.json", "gate_5_conditions"),),
    6: (("features/feature_report.json", "gate_6_conditions"),),
    7: (("statistical/statistical_report.json", "gate_7_conditions"),),
    8: (("splits/split_report.json", "gate_8_conditions"),),
}


def gate_record(phase: int, reports_dir: Path = REPORTS_DIR) -> dict[str, Any]:
    if phase not in GATE_CONDITIONS:
        raise ValueError(f"No gate defined for phase {phase}")
    checks: dict[str, bool] = {}
    for rel_path, key in GATE_REPORTS.get(phase, ()):
        report = json.loads((reports_dir / rel_path).read_text(encoding="utf-8"))
        checks.update({f"{rel_path}:{name}": bool(ok) for name, ok in report[key].items()})
    return {
        "phase": phase,
        "passed": all(checks.values()),
        "conditions": list(GATE_CONDITIONS[phase]),
        "checks": checks,
        "environment": collect(),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, default=REPORTS_DIR / "gates")
    args = parser.parse_args(argv)
    record = gate_record(args.phase)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"phase_{args.phase}.json"
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Gate {args.phase} {'PASSED' if record['passed'] else 'FAILED'}; recorded in {out}")
    if not record["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
