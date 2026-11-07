"""Duplicate representatives and config-driven quality tiers (Phase 5).

Every record gets the worst tier among the rules it matches (``gold`` when
none match) and the ids of all matching rules, in config order, as its
reasons. Nothing is deleted. Rule predicates are named here so the config
can only choose between reviewed definitions, never inject expressions.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import polars as pl

TIER_ORDER = ("gold", "silver", "bronze", "quarantine")
TIER_RANK = {t: i for i, t in enumerate(TIER_ORDER)}

PREDICATES: dict[str, Callable[[], pl.Expr]] = {
    "hard_check_failed": lambda: pl.col("n_failed_hard") > 0,
    "fragmented": lambda: ~pl.col("single_fragment__pass"),
    "structure_changed": lambda: (
        pl.col("single_fragment__pass") & ~pl.col("gdb_matches_geometry__pass")
    ),
    "listed_uncharacterized_but_consistent": lambda: (
        ~pl.col("not_uncharacterized__pass")
        & pl.col("gdb_matches_geometry__pass")
        & pl.col("single_fragment__pass")
    ),
    "duplicate_not_representative": lambda: ~pl.col("is_structure_representative"),
    "convergence_concern": lambda: ~pl.col("not_hard_to_converge__pass"),
}


def choose_representatives(groups: pl.DataFrame, validation: pl.DataFrame) -> pl.DataFrame:
    """One representative per exact-structure group: cleanest record, then lowest id.

    "Cleanest" orders by failed hard checks, failed identity checks, failed
    soft checks; ties go to the smallest ``qm9_id``, so the choice is
    deterministic and never depends on property values.
    """
    ranked = groups.select("qm9_id", "structure_group_id").join(
        validation.select("qm9_id", "n_failed_hard", "n_failed_identity", "n_failed_soft"),
        on="qm9_id",
    )
    rep = (
        ranked.sort(
            ["structure_group_id", "n_failed_hard", "n_failed_identity", "n_failed_soft", "qm9_id"]
        )
        .group_by("structure_group_id", maintain_order=True)
        .first()
        .select("structure_group_id", pl.col("qm9_id").alias("structure_representative_id"))
    )
    return (
        ranked.select("qm9_id", "structure_group_id")
        .join(rep, on="structure_group_id")
        .with_columns(
            (pl.col("qm9_id") == pl.col("structure_representative_id")).alias(
                "is_structure_representative"
            )
        )
        .sort("qm9_id")
    )


def rule_columns(rules: Sequence[Mapping[str, Any]]) -> list[pl.Expr]:
    unknown = [r["predicate"] for r in rules if r["predicate"] not in PREDICATES]
    if unknown:
        raise ValueError(f"Unknown predicate(s) in curation rules: {unknown}")
    bad_tier = [r["tier"] for r in rules if r["tier"] not in TIER_RANK]
    if bad_tier:
        raise ValueError(f"Unknown tier(s) in curation rules: {bad_tier}")
    return [PREDICATES[r["predicate"]]().fill_null(False).alias(f"rule__{r['id']}") for r in rules]


def assign_tiers(frame: pl.DataFrame, rules: Sequence[Mapping[str, Any]]) -> pl.DataFrame:
    with_rules = frame.with_columns(rule_columns(rules))
    rank = pl.max_horizontal(
        pl.lit(0, dtype=pl.UInt8),
        *(
            pl.when(pl.col(f"rule__{r['id']}"))
            .then(pl.lit(TIER_RANK[r["tier"]], dtype=pl.UInt8))
            .otherwise(0)
            for r in rules
        ),
    )
    reasons = (
        pl.concat_list(
            *(
                pl.when(pl.col(f"rule__{r['id']}"))
                .then(pl.lit([r["id"]]))
                .otherwise(pl.lit([], dtype=pl.List(pl.String)))
                for r in rules
            )
        )
        if rules
        else pl.lit([], dtype=pl.List(pl.String))
    )
    tier_names = pl.Series(TIER_ORDER)
    return with_rules.with_columns(
        rank.replace_strict(
            list(range(len(TIER_ORDER))), list(tier_names), return_dtype=pl.String
        ).alias("quality_tier"),
        reasons.alias("tier_reasons"),
    ).drop(f"rule__{r['id']}" for r in rules)


def tier_counts(frame: pl.DataFrame) -> dict[str, int]:
    counts = dict(frame.group_by("quality_tier").len().rows())
    return {t: int(counts.get(t, 0)) for t in TIER_ORDER}


def waterfall(frame: pl.DataFrame, rules: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Apply rules cumulatively in config order; report matches and tier counts after each."""
    steps: list[dict[str, Any]] = [
        {
            "after_rule": None,
            "matched": None,
            "newly_out_of_gold": None,
            "tiers": tier_counts(assign_tiers(frame, [])),
        }
    ]
    previous_gold = frame.height
    for i, rule in enumerate(rules):
        tiered = assign_tiers(frame, rules[: i + 1])
        counts = tier_counts(tiered)
        matched = int(frame.select(PREDICATES[rule["predicate"]]().fill_null(False).sum()).item())
        steps.append(
            {
                "after_rule": rule["id"],
                "matched": matched,
                "newly_out_of_gold": previous_gold - counts["gold"],
                "tiers": counts,
            }
        )
        previous_gold = counts["gold"]
    return steps


def rule_impact(frame: pl.DataFrame, rules: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """For each rule: how many records change tier if that rule alone is switched off."""
    full = assign_tiers(frame, rules).select("qm9_id", "quality_tier")
    impact = {}
    for rule in rules:
        without = assign_tiers(frame, [r for r in rules if r["id"] != rule["id"]]).select(
            "qm9_id", pl.col("quality_tier").alias("t2")
        )
        impact[rule["id"]] = int(
            full.join(without, on="qm9_id").filter(pl.col("quality_tier") != pl.col("t2")).height
        )
    return impact
