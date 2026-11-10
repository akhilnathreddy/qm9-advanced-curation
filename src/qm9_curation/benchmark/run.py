"""Run the benchmark grid (Phase 9): tuning, RQ3 + stress, RQ2 OOF, RQ5 ablation.

Run as ``python -m qm9_curation.benchmark.run``. Each run is one
(experiment, model, target, split, seed, training tiers, N, fold) cell. It
writes ``data/benchmark/runs/<key>.parquet`` (``qm9_id, role, y, pred``) and
``<key>.json`` (spec, parameters, info, metrics, MLflow run id, provenance)
and is logged to MLflow. The key hashes the spec and the model parameters
(including the model version), so a finished run is never repeated and a
changed model is never served from cache: the grid is resumable, and several
processes may work on disjoint ``--models`` at once.

Order: tuning (validation only) -> RQ3 and stress splits -> RQ2 out-of-fold
residuals of the level with the best ``random`` validation MAE -> RQ5.
Tuning runs predict validation rows only; every other run predicts val and
test once.
"""

import argparse
import json
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from qm9_curation.benchmark import data as bdata
from qm9_curation.benchmark import models as bmodels
from qm9_curation.benchmark.metrics import metrics
from qm9_curation.benchmark.tracking import Tracker, code_sha256, git_state, run_key, sha256_file
from qm9_curation.config import (
    DATA_DIR,
    INTERIM_DIR,
    PROCESSED_DIR,
    PROJECT_ROOT,
    SPLITS_DIR,
    load_config,
)

RUNS_DIR = DATA_DIR / "benchmark" / "runs"
GNN_MODELS = ("dmpnn", "schnet")


class Context:
    """Inputs shared by every run, loaded once; graph packs on first use."""

    def __init__(self, cfg: dict[str, Any], processed: Path, splits_dir: Path, interim: Path):
        self.cfg = cfg
        self.processed, self.interim = processed, interim
        self.inputs = bdata.load_inputs(processed, splits_dir, list(cfg["targets"]))
        names = [
            *cfg["experiments"]["rq3"]["splits"],
            *cfg["experiments"]["stress"]["splits"],
        ]
        self.splits = bdata.load_splits(splits_dir, names)
        self._packed: dict[str, Any] = {}
        self._rows: dict[tuple[str, int], dict[str, np.ndarray]] = {}

    def rows(self, split: str, seed: int) -> dict[str, np.ndarray]:
        if (split, seed) not in self._rows:
            self._rows[split, seed] = bdata.subset_rows(self.inputs, self.splits, split, seed)
        return self._rows[split, seed]

    def packed(self, model: str) -> Any:
        if model not in GNN_MODELS:
            return None
        if model not in self._packed:
            from qm9_curation.benchmark import gnn

            if model == "dmpnn":
                feat = self.processed / "features"
                self._packed[model] = gnn.pack_graphs(
                    pl.read_parquet(feat / "graph_nodes.parquet"),
                    pl.read_parquet(feat / "graph_edges.parquet"),
                )
            else:
                self._packed[model] = gnn.pack_atoms(
                    pl.read_parquet(self.interim / "atoms.parquet")
                )
        return self._packed[model]


def spec(
    experiment: str,
    model: str,
    target: str,
    split: str,
    seed: int,
    tiers: str = "all",
    n_train: int | None = None,
    fold: int | None = None,
    grid_point: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "experiment": experiment,
        "model": model,
        "target": target,
        "split": split,
        "seed": seed,
        "tiers": tiers,
        "n_train": n_train,
        "fold": fold,
        "grid_point": dict(grid_point or {}),
    }


def model_params(
    cfg: Mapping[str, Any], model: str, target: str, tuned: Mapping[str, Any]
) -> dict[str, Any]:
    params = dict(cfg["models"][model])
    params.update(tuned.get(model, {}).get(target, {}))
    return params


class Runner:
    def __init__(
        self,
        ctx: Context,
        runs_dir: Path,
        tracker: Tracker | None,
        provenance: Mapping[str, Any],
        log: bool = True,
    ):
        self.ctx, self.runs_dir, self.tracker = ctx, runs_dir, tracker
        self.provenance = dict(provenance)
        self.verbose = log
        runs_dir.mkdir(parents=True, exist_ok=True)

    def path(self, key: str) -> Path:
        return self.runs_dir / f"{key}.json"

    def done(self, s: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any] | None:
        p = self.path(run_key(s, params))
        return json.loads(p.read_text()) if p.exists() else None

    def run(self, s: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        cached = self.done(s, params)
        if cached is not None:
            return cached
        key = run_key(s, params)
        ctx, inputs = self.ctx, self.ctx.inputs
        rows = ctx.rows(s["split"], s["seed"])
        tiers = ctx.cfg["experiments"]["rq5"]["tier_sets"].get(s["tiers"], ctx.cfg["train_tiers"])
        train = rows["train"]
        predict: dict[str, np.ndarray]
        if s["experiment"] == "rq2":
            folds = inputs.frame["leakage_group_id"].to_numpy()[train] % int(
                ctx.cfg["experiments"]["rq2"]["n_folds"]
            )
            predict = {"oof": train[folds == s["fold"]]}
            train = train[folds != s["fold"]]
        train = bdata.tier_subsample(inputs, train, tiers, s["n_train"], s["seed"])
        if s["experiment"] == "tuning":
            predict = {"val": rows["val"]}
        elif s["experiment"] != "rq2":
            predict = {"val": rows["val"], "test": rows["test"]}
        readout = ctx.cfg["targets"][s["target"]]["readout"]
        start = time.time()
        fit = bmodels.fit_predict(
            s["model"],
            inputs,
            s["target"],
            train,
            rows["val"],
            predict,
            params,
            s["seed"],
            packed=ctx.packed(s["model"]),
            mean_readout=readout == "mean",
        )
        seconds = time.time() - start
        y = inputs.frame[s["target"]].to_numpy()
        ids = inputs.ids
        table = pl.concat(
            [
                pl.DataFrame(
                    {
                        "qm9_id": ids[r].astype(np.uint32),
                        "role": [role] * len(r),
                        "y": y[r],
                        "pred": fit.predictions[role],
                    }
                )
                for role, r in predict.items()
            ]
        )
        scores = {
            f"{role}_{k}": v
            for role, r in predict.items()
            for k, v in metrics(y[r], fit.predictions[role]).items()
        }
        record: dict[str, Any] = {
            "key": key,
            "spec": s,
            "params": params,
            "n_train": len(train),
            "train_tier_counts": {
                t: int(c)
                for t, c in zip(
                    *np.unique(inputs.frame["quality_tier"].to_numpy()[train], return_counts=True),
                    strict=True,
                )
            },
            "metrics": scores,
            "info": fit.info,
            "seconds": seconds,
            "provenance": self.provenance,
        }
        record["mlflow_run_id"] = (
            self.tracker.log(key, s, params, scores, record) if self.tracker else None
        )
        table.write_parquet(self.runs_dir / f"{key}.parquet", compression="zstd")
        self.path(key).write_text(json.dumps(record, indent=1, sort_keys=True, default=float))
        if self.verbose:
            main = scores.get("test_mae", scores.get("val_mae", scores.get("oof_mae")))
            print(
                f"{s['experiment']:7s} {s['model']:12s} {s['target']:18s} {s['split']:28s} "
                f"seed={s['seed']} tiers={s['tiers']} n={len(train)} fold={s['fold']} "
                f"mae={main:.4f} {seconds:.0f}s",
                flush=True,
            )
        return record


def covers(cfg: Mapping[str, Any], model: str, experiment: str, split: str | None = None) -> bool:
    cov = cfg["coverage"][model]
    if experiment == "rq3" and split == "random" and "rq3_random" in cov:
        return True
    return experiment in cov


def tune(runner: Runner, models: Sequence[str]) -> dict[str, Any]:
    """Grid search on random seed 0 validation rows; returns {model: {target: params}}.

    Models outside ``models`` are not trained here; their winners are read
    from finished tuning runs when all exist (another process ran them).
    """
    cfg = runner.ctx.cfg
    t = cfg["tuning"]
    out: dict[str, Any] = {}
    for model in cfg["models"]:
        if model not in t:
            continue
        grid = t[model]["grid"]
        names = sorted(grid)
        points = [
            dict(zip(names, vals, strict=True)) for vals in _product([grid[n] for n in names])
        ]
        out[model] = {}
        for target in cfg["targets"]:
            scores = []
            for point in points:
                params = {**cfg["models"][model], **point}
                s = spec(
                    "tuning",
                    model,
                    target,
                    t["split"],
                    t["seed"],
                    n_train=t[model].get("n_train"),
                    grid_point=point,
                )
                rec = runner.run(s, params) if model in models else runner.done(s, params)
                if rec is not None:
                    scores.append((rec["metrics"]["val_mae"], point))
            if len(scores) == len(points):
                out[model][target] = min(scores, key=lambda x: x[0])[1]
        if len(out[model]) < len(cfg["targets"]):
            del out[model]
    return out


def _product(lists: Sequence[Sequence[Any]]) -> Iterator[tuple[Any, ...]]:
    if not lists:
        yield ()
        return
    for head in lists[0]:
        for tail in _product(lists[1:]):
            yield (head, *tail)


def best_random_model(runner: Runner, tuned: Mapping[str, Any], target: str) -> str | None:
    """Level with the lowest mean ``random`` validation MAE; None if any is missing."""
    cfg = runner.ctx.cfg
    scores = {}
    for model in cfg["coverage"]:
        if not covers(cfg, model, "rq3", "random"):
            continue
        params = model_params(cfg, model, target, tuned)
        recs = [
            runner.done(spec("rq3", model, target, "random", seed), params) for seed in cfg["seeds"]
        ]
        if any(r is None for r in recs):
            return None
        scores[model] = float(np.mean([r["metrics"]["val_mae"] for r in recs if r]))
    return str(min(scores, key=scores.__getitem__))


def grid(runner: Runner, models: Sequence[str], experiments: Sequence[str]) -> None:
    ctx, cfg = runner.ctx, runner.ctx.cfg
    tuned = tune(runner, models)
    targets = list(cfg["targets"])
    if "rq3" in experiments or "stress" in experiments:
        splits = [
            *(("rq3", s) for s in cfg["experiments"]["rq3"]["splits"]),
            *(("stress", s) for s in cfg["experiments"]["stress"]["splits"]),
        ]
        for model in models:
            for exp, split in splits:
                if exp not in experiments or not covers(cfg, model, exp, split):
                    continue
                for seed in cfg["seeds"]:
                    for target in targets:
                        runner.run(
                            spec(exp, model, target, split, seed),
                            model_params(cfg, model, target, tuned),
                        )
    if "rq2" in experiments:
        r2 = cfg["experiments"]["rq2"]
        for target in targets:
            best = best_random_model(runner, tuned, target)
            if best is None:
                print(f"rq2 {target}: waiting for every random-split run", flush=True)
                continue
            if best not in models:
                continue
            for fold in range(int(r2["n_folds"])):
                runner.run(
                    spec("rq2", best, target, r2["split"], r2["seed"], fold=fold),
                    model_params(cfg, best, target, tuned),
                )
    if "rq5" in experiments:
        r5 = cfg["experiments"]["rq5"]
        for model in models:
            if model not in r5["models"] or not covers(cfg, model, "rq5"):
                continue
            for seed in cfg["seeds"]:
                train = ctx.rows(r5["split"], seed)["train"]
                n_max = bdata.max_common_n(ctx.inputs, train, list(r5["tier_sets"].values()))
                for n in [n for n in r5["n_grid"] if n <= n_max]:
                    for tiers in r5["tier_sets"]:
                        for target in targets:
                            runner.run(
                                spec("rq5", model, target, r5["split"], seed, tiers, n),
                                model_params(cfg, model, target, tuned),
                            )


def provenance(processed: Path, splits_dir: Path) -> dict[str, Any]:
    version = load_config("params", PROJECT_ROOT) if (PROJECT_ROOT / "params.yaml").exists() else {}
    base = version.get("pipeline", {}).get("dataset_version", "unknown")
    mol = sha256_file(processed / "molecules.parquet")[:12]
    spl = sha256_file(splits_dir / "splits.parquet")[:12]
    return {
        **git_state(),
        "code_sha256": code_sha256(),
        "dataset_version": f"{base}+molecules.{mol}.splits.{spl}",
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--splits-dir", type=Path, default=SPLITS_DIR)
    parser.add_argument("--interim-dir", type=Path, default=INTERIM_DIR)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--mlruns", type=Path, default=PROJECT_ROOT / "mlruns")
    parser.add_argument("--models", default=",".join(bmodels.LEVELS))
    parser.add_argument("--experiments", default="rq3,stress,rq2,rq5")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config("benchmark")
    ctx = Context(cfg, args.processed_dir, args.splits_dir, args.interim_dir)
    prov = provenance(args.processed_dir, args.splits_dir)
    tracker = None if args.no_mlflow else Tracker(args.mlruns, "qm9-benchmark", prov)
    runner = Runner(ctx, args.runs_dir, tracker, prov)
    models = [m for m in args.models.split(",") if m]
    grid(runner, models, args.experiments.split(","))


if __name__ == "__main__":
    main()
