"""Benchmark inputs: labels, tiers, splits and per-level feature matrices.

Everything is aligned on one sorted ``qm9_id`` axis (:class:`Inputs`), so a
model receives row positions and never re-joins tables. Only records with
every feature set (RDKit perception succeeded; all Gold records do) enter the
benchmark, so every level sees the same molecules; the 22 Bronze records
without features and the 5 quarantine records are excluded and counted.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from scipy import sparse

from qm9_curation.features.molecular import ECFP_SIZE
from qm9_curation.review.stats import ELEMENT_COUNTS

TIERS = ("gold", "silver", "bronze")


@dataclass
class Inputs:
    frame: pl.DataFrame  # qm9_id, quality_tier, leakage_group_id, targets (sorted by qm9_id)
    composition: np.ndarray  # (n, 5) element counts
    descriptors: np.ndarray  # (n, 184) RDKit descriptors
    ecfp: sparse.csr_matrix  # (n, 2048) ECFP4 counts
    coulomb: np.ndarray  # (n, 29) sorted Coulomb-matrix eigenvalues
    excluded: dict[str, int]

    @property
    def ids(self) -> np.ndarray:
        return self.frame["qm9_id"].to_numpy()


def load_inputs(
    processed_dir: Path, splits_dir: Path, targets: Sequence[str], tiers: Sequence[str] = TIERS
) -> Inputs:
    feat = processed_dir / "features"
    mol = pl.read_parquet(processed_dir / "molecules.parquet").select(
        "qm9_id", "quality_tier", *targets
    )
    index = pl.read_parquet(feat / "index.parquet")
    groups = pl.read_parquet(splits_dir / "split_groups.parquet").select(
        "qm9_id", "leakage_group_id"
    )
    joined = mol.join(index, on="qm9_id").join(groups, on="qm9_id")
    keep = pl.col("quality_tier").is_in(list(tiers)) & pl.col("rdkit_features_available")
    excluded = {
        "quarantine": int((joined["quality_tier"] == "quarantine").sum()),
        "no_rdkit_features": int(
            joined.filter(pl.col("quality_tier").is_in(list(tiers)))["rdkit_features_available"]
            .not_()
            .sum()
        ),
    }
    frame = (
        joined.filter(keep).drop("rdkit_features_available", "unavailable_reason").sort("qm9_id")
    )
    ids = frame.select("qm9_id")

    def aligned(name: str) -> pl.DataFrame:
        table = ids.join(pl.read_parquet(feat / f"{name}.parquet"), on="qm9_id", how="left")
        if table.height != frame.height or table.null_count().sum_horizontal().item():
            raise ValueError(f"{name} does not cover every benchmark record")
        return table.drop("qm9_id")

    fp = ids.join(pl.read_parquet(feat / "ecfp4.parquet"), on="qm9_id", how="left")
    rows = np.repeat(np.arange(fp.height), fp["bits"].list.len().to_numpy())
    ecfp = sparse.csr_matrix(
        (
            fp["counts"].explode().to_numpy().astype(np.float32),
            (rows, fp["bits"].explode().to_numpy().astype(np.int64)),
        ),
        shape=(fp.height, ECFP_SIZE),
    )
    return Inputs(
        frame=frame,
        composition=aligned("composition").select(ELEMENT_COUNTS).to_numpy().astype(np.float64),
        descriptors=aligned("rdkit_descriptors").to_numpy().astype(np.float32),
        ecfp=ecfp,
        coulomb=aligned("coulomb_eigenvalues").to_numpy().astype(np.float64),
        excluded=excluded,
    )


def load_splits(splits_dir: Path, names: Sequence[str]) -> pl.DataFrame:
    return (
        pl.read_parquet(splits_dir / "splits.parquet")
        .filter(pl.col("split_name").cast(pl.String).is_in(list(names)))
        .with_columns(pl.col("split_name").cast(pl.String), pl.col("subset").cast(pl.String))
    )


def subset_rows(
    inputs: Inputs, splits: pl.DataFrame, split: str, seed: int
) -> dict[str, np.ndarray]:
    """Row positions (into ``inputs``) of train (all tiers), val ∩ Gold and test ∩ Gold."""
    assigned = (
        inputs.frame.select("qm9_id", "quality_tier")
        .with_row_index("row")
        .join(
            splits.filter((pl.col("split_name") == split) & (pl.col("seed") == seed)).select(
                "qm9_id", "subset"
            ),
            on="qm9_id",
            how="left",
        )
    )
    if assigned["subset"].null_count():
        raise ValueError(f"{split}/{seed} does not assign every benchmark record")
    gold = pl.col("quality_tier") == "gold"

    def rows(cond: pl.Expr) -> np.ndarray:
        return assigned.filter(cond)["row"].to_numpy().astype(np.int64)

    return {
        "train": rows(pl.col("subset") == "train"),
        "val": rows((pl.col("subset") == "val") & gold),
        "test": rows((pl.col("subset") == "test") & gold),
    }


def tier_subsample(
    inputs: Inputs, rows: np.ndarray, tiers: Sequence[str], n: int | None, seed: int
) -> np.ndarray:
    """Training rows restricted to ``tiers``, then a seeded subsample of size ``n``."""
    tier = inputs.frame["quality_tier"].to_numpy()[rows]
    pool = rows[np.isin(tier, list(tiers))]
    if n is None or n >= len(pool):
        if n is not None and n > len(pool):
            raise ValueError(f"cannot draw {n} from {len(pool)} {list(tiers)} rows")
        return np.sort(pool)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(pool, size=n, replace=False))


def max_common_n(inputs: Inputs, rows: np.ndarray, tier_sets: Sequence[Sequence[str]]) -> int:
    tier = inputs.frame["quality_tier"].to_numpy()[rows]
    return min(int(np.isin(tier, list(t)).sum()) for t in tier_sets)
