"""Record a phase exit-gate result in ``reports/gates/phase_<N>.json``.

Each gate lists its conditions (plan v2, section 4). The Makefile target for a
gate runs the commands that prove those conditions first, so reaching this
step means they passed; the record ties that result to the code and
environment that produced it.
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
}


def gate_record(phase: int) -> dict[str, Any]:
    if phase not in GATE_CONDITIONS:
        raise ValueError(f"No gate defined for phase {phase}")
    return {
        "phase": phase,
        "passed": True,
        "conditions": list(GATE_CONDITIONS[phase]),
        "environment": collect(),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, default=REPORTS_DIR / "gates")
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"phase_{args.phase}.json"
    out.write_text(
        json.dumps(gate_record(args.phase), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Gate {args.phase} recorded in {out}")


if __name__ == "__main__":
    main()
