"""Leakage assertions and distribution-shift report for every split.

Assertions (every split, every seed) — Gate 8 fails on any violation:

* every record is assigned exactly once;
* no leakage key (exact-structure, tautomer, GDB connectivity, geometry
  InChIKey connectivity block) and no split unit occurs in two subsets;
* holdout splits: every record meeting the rule is in test, and train/val
  contain none;
* grouped splits: achieved fractions within the configured tolerance.

Shift report (seed 0, Gold records, as Phase 9 evaluates on Gold): test vs.
train target distributions, composition shift, and the test -> train
nearest-neighbour Tanimoto distribution on a fixed sample.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import polars as pl
from scipy import sparse

from qm9_curation.review.stats import compare_distributions, summary
from qm9_curation.splits.assign import SUBSETS
from qm9_curation.splits.fingerprints import max_similarity
from qm9_curation.splits.groups import LEAKAGE_KEYS


def overlaps(assigned: pl.DataFrame, keys: Sequence[str]) -> dict[str, int]:
    """Number of key values present in more than one subset."""
    return {
        k: int(
            assigned.filter(pl.col(k).is_not_null())
            .group_by(k)
            .agg(pl.col("subset").n_unique().alias("n"))
            .filter(pl.col("n") > 1)
            .height
        )
        for k in keys
    }


def check_assignment(
    assigned: pl.DataFrame,
    n_records: int,
    spec: Mapping[str, Any],
    fractions: Mapping[str, float],
    tolerance: float,
    rule: pl.Series | None,
) -> dict[str, Any]:
    """``assigned``: one split x seed joined with the group columns (sorted by qm9_id)."""
    counts = dict(assigned.group_by("subset").len().rows())
    achieved = {s: counts.get(s, 0) / assigned.height for s in SUBSETS}
    leaks = overlaps(assigned, [*LEAKAGE_KEYS, "leakage_group_id", "unit_id"])
    out: dict[str, Any] = {
        "counts": {s: counts.get(s, 0) for s in SUBSETS},
        "fractions": achieved,
        "overlaps": leaks,
        "covers_every_record_once": assigned.height == n_records
        and assigned["qm9_id"].is_unique().all()
        and set(counts) <= set(SUBSETS),
        "no_key_in_two_subsets": not any(leaks.values()),
    }
    if spec["kind"] == "grouped":
        out["fractions_within_tolerance"] = all(
            abs(achieved[s] - fractions[s]) <= tolerance for s in SUBSETS
        )
    else:
        assert rule is not None
        test = assigned["subset"] == "test"
        out["rule_records_all_in_test"] = bool((~rule | test).all())
        out["test_records_count_meeting_rule"] = int((rule & test).sum())
        out["test_records_pulled_in_by_group"] = int((~rule & test).sum())
        out["holdout_rule_respected"] = out["rule_records_all_in_test"]
    return out


def shift_report(
    assigned: pl.DataFrame,
    curated: pl.DataFrame,
    composition: pl.DataFrame,
    fp_ids: np.ndarray,
    fp: sparse.csr_matrix,
    targets: Sequence[str],
    nn_sample: int,
) -> dict[str, Any]:
    """Seed-0 test vs. train shift, on Gold records."""
    frame = (
        assigned.select("qm9_id", "subset")
        .join(curated.select("qm9_id", "quality_tier", *targets), on="qm9_id")
        .join(composition, on="qm9_id")
    )
    gold = frame.filter(pl.col("quality_tier") == "gold")
    train, test = (gold.filter(pl.col("subset") == s) for s in ("train", "test"))
    out: dict[str, Any] = {
        "gold_counts": {s: int((gold["subset"] == s).sum()) for s in SUBSETS},
        "targets": {
            t: compare_distributions(train[t].to_numpy(), test[t].to_numpy()) for t in targets
        },
        "composition": {
            "n_heavy": compare_distributions(
                train["n_heavy"].cast(pl.Float64).to_numpy(),
                test["n_heavy"].cast(pl.Float64).to_numpy(),
            ),
            "fraction_containing": {
                e: {
                    s: float(np.mean(sub[f"n_{e}"].to_numpy() > 0)) if sub.height else 0.0
                    for s, sub in (("train", train), ("test", test))
                }
                for e in ("N", "O", "F")
            },
        },
    }
    # Nearest neighbour: sampled Gold test molecules vs. every train record with a fingerprint.
    row_of = {q: i for i, q in enumerate(fp_ids.tolist())}
    train_rows = [
        row_of[q]
        for q in frame.filter(pl.col("subset") == "train")["qm9_id"].to_list()
        if q in row_of
    ]
    test_ids = [q for q in test["qm9_id"].to_list() if q in row_of]
    rng = np.random.default_rng(0)
    pick = sorted(rng.choice(len(test_ids), size=min(nn_sample, len(test_ids)), replace=False))
    query = fp[[row_of[test_ids[i]] for i in pick]]
    out["nn_tanimoto_test_to_train"] = summary(max_similarity(query, fp[train_rows]))
    return out


def naive_random_leakage(groups: pl.DataFrame, test_fraction: float, seed: int) -> dict[str, Any]:
    """What a record-level random split (the common literature practice) would leak."""
    rng = np.random.default_rng(seed)
    n = groups.height
    test = np.zeros(n, dtype=bool)
    test[rng.permutation(n)[: round(test_fraction * n)]] = True
    frame = groups.select("leakage_group_id", "structure_group_id").with_columns(
        pl.Series("test", test)
    )
    out: dict[str, Any] = {"n_test": int(test.sum())}
    for key in ("structure_group_id", "leakage_group_id"):
        mixed = (
            frame.group_by(key)
            .agg(pl.col("test").any().alias("t"), (~pl.col("test")).any().alias("o"))
            .filter(pl.col("t") & pl.col("o"))
        )
        leaked = frame.join(mixed.select(key), on=key).filter(pl.col("test")).height
        out[f"test_records_sharing_{key}_with_train_or_val"] = leaked
    return out
