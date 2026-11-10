"""Provenance and MLflow tracking for benchmark runs.

Every run is logged to a local MLflow file store (``mlruns/``, git-ignored)
with the tags plan v2 §7.2 asks for: ``git_commit``, ``dataset_version``,
``split_name``, ``tier_filter``, ``seed``, plus the model, target, experiment
and the run key. The file store is in MLflow's maintenance mode but still
supported; ``MLFLOW_ALLOW_FILE_STORE`` opts in (the SQLite store needs the
full ``mlflow`` package and its server dependencies).

``code_sha256`` hashes the benchmark source and config, so a run produced
from an uncommitted tree can still be matched to the commit that contains
exactly that code.
"""

import hashlib
import json
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from qm9_curation.config import CONFIG_DIR, PROJECT_ROOT

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

BENCHMARK_SRC = Path(__file__).resolve().parent


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_sha256(config_dir: Path = CONFIG_DIR) -> str:
    h = hashlib.sha256()
    for path in [*sorted(BENCHMARK_SRC.glob("*.py")), config_dir / "benchmark.yaml"]:
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def git_state(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return {
            "git_commit": git("rev-parse", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": "unknown", "git_dirty": True}


def run_key(spec: Mapping[str, Any], params: Mapping[str, Any]) -> str:
    """Stable key of a run: its spec and the model parameters (incl. version)."""
    blob = json.dumps({"spec": spec, "params": params}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:20]


class Tracker:
    def __init__(self, store: Path, experiment: str, provenance: Mapping[str, Any]):
        import mlflow

        self.mlflow = mlflow
        store.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(store.resolve().as_uri())
        mlflow.set_experiment(experiment)
        self.provenance = dict(provenance)

    def log(
        self,
        key: str,
        spec: Mapping[str, Any],
        params: Mapping[str, Any],
        metrics: Mapping[str, float],
        info: Mapping[str, Any],
    ) -> str:
        tags = {
            "run_key": key,
            "git_commit": self.provenance["git_commit"],
            "git_dirty": str(self.provenance["git_dirty"]),
            "code_sha256": self.provenance["code_sha256"],
            "dataset_version": self.provenance["dataset_version"],
            "split_name": spec["split"],
            "tier_filter": spec["tiers"],
            "seed": str(spec["seed"]),
            "model": spec["model"],
            "target": spec["target"],
            "experiment": spec["experiment"],
        }
        flat = {f"param.{k}": json.dumps(v, default=str) for k, v in params.items()}
        with self.mlflow.start_run(run_name=key, tags=tags) as run:
            self.mlflow.log_params({**flat, **{k: str(v) for k, v in spec.items()}})
            self.mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
            self.mlflow.log_dict(dict(info), "info.json")
            return str(run.info.run_id)
