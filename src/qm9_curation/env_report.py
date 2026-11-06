"""Record the software environment a pipeline run used.

Run as ``python -m qm9_curation.env_report --out reports/gates/env.json``.
The output deliberately has no timestamp so that two runs in the same
environment produce byte-identical files.
"""

import argparse
import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

from qm9_curation import __version__
from qm9_curation.config import PROJECT_ROOT
from qm9_curation.units import CODATA_RELEASE

TRACKED_PACKAGES = (
    "numpy",
    "scipy",
    "pandas",
    "polars",
    "pyarrow",
    "rdkit",
    "pandera",
    "pyyaml",
    "matplotlib",
)


def _git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        commit = run("rev-parse", "HEAD")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": bool(run("status", "--porcelain"))}


def collect() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "qm9_curation_version": __version__,
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": f"{platform.system()}-{platform.machine()}",
        "constants": CODATA_RELEASE,
        "packages": packages,
        "git": _git_state(),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(collect(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
