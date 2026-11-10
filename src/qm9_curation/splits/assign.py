"""Subset assignment for the two kinds of split.

* **grouped:** whole split units (a grouping key merged with leakage groups)
  go to one subset. Units larger than half the test target always go to
  train (as in balanced scaffold splitting); the rest are shuffled with the
  seed and filled into test, then val, until each reaches its target size;
  everything else is train.
* **holdout:** a rule fixes the test set (same for every seed). Train and
  val come from the remaining units in the ratio train : val of the
  configured fractions. A unit is in test if any member meets the rule, so a
  unit is never split.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
import polars as pl

SUBSETS = ("train", "val", "test")


def grouped(units: pl.DataFrame, fractions: Mapping[str, float], seed: int) -> pl.DataFrame:
    """``units``: qm9_id, unit_id. Returns qm9_id, subset."""
    n = units.height
    sizes = units.group_by("unit_id").len().sort("unit_id")
    target = {s: fractions[s] * n for s in ("test", "val")}
    big = sizes.filter(pl.col("len") > target["test"] / 2)
    rest = sizes.filter(pl.col("len") <= target["test"] / 2)
    order = np.random.default_rng(seed).permutation(rest.height)
    filled = {"test": 0, "val": 0}
    subset_of: dict[int, str] = {u: "train" for u in big["unit_id"].to_list()}
    unit_ids, lens = rest["unit_id"].to_list(), rest["len"].to_list()
    for i in order:
        u, size = unit_ids[i], lens[i]
        for s in ("test", "val"):
            if filled[s] + size <= target[s]:
                subset_of[u] = s
                filled[s] += size
                break
        else:
            subset_of[u] = "train"
    return units.select(
        "qm9_id",
        pl.col("unit_id").replace_strict(subset_of, return_dtype=pl.String).alias("subset"),
    )


def holdout(
    units: pl.DataFrame, in_test: pl.Series, fractions: Mapping[str, float], seed: int
) -> pl.DataFrame:
    """``in_test``: per-record rule result aligned with ``units``."""
    frame = units.with_columns(in_test.alias("_rule"))
    test_units = frame.filter(pl.col("_rule"))["unit_id"].unique()
    pool = frame.filter(~pl.col("unit_id").is_in(test_units.implode()))
    val_share = fractions["val"] / (fractions["train"] + fractions["val"])
    # Reuse the grouped filler with a single held-out share, then call it val.
    pooled = grouped(
        pool.select("qm9_id", "unit_id"), {"test": val_share, "val": 0.0}, seed
    ).with_columns(pl.col("subset").replace({"test": "val"}))
    tested = frame.filter(pl.col("unit_id").is_in(test_units.implode())).select(
        "qm9_id", pl.lit("test").alias("subset")
    )
    return pl.concat([pooled, tested]).sort("qm9_id")


def holdout_rule(frame: pl.DataFrame, rule: Mapping[str, Any]) -> pl.Series:
    """Per-record test membership for a ``test_when`` rule."""
    if "n_heavy_min" in rule:
        expr = pl.col("n_heavy") >= rule["n_heavy_min"]
    elif "contains_element" in rule:
        expr = pl.col(f"n_{rule['contains_element']}") > 0
    elif "tail" in rule:
        t = rule["tail"]
        # The round(fraction * n) most extreme records; ties at the boundary all included.
        rank = pl.col(t["target"]).rank("min", descending=t["side"] == "upper")
        expr = rank <= (pl.len() * t["fraction"]).round()
    else:
        raise ValueError(f"Unknown holdout rule: {dict(rule)}")
    return frame.select(expr.alias("in_test")).to_series()
