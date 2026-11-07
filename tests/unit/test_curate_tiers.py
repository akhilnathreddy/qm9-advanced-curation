import polars as pl
import pytest

from qm9_curation.curate.tiers import (
    assign_tiers,
    choose_representatives,
    rule_impact,
    tier_counts,
    waterfall,
)

RULES = [
    {"id": "R1", "predicate": "hard_check_failed", "tier": "quarantine", "reason": ""},
    {"id": "R2", "predicate": "fragmented", "tier": "bronze", "reason": ""},
    {"id": "R5", "predicate": "duplicate_not_representative", "tier": "silver", "reason": ""},
    {"id": "R6", "predicate": "convergence_concern", "tier": "silver", "reason": ""},
]


def frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "qm9_id": [1, 2, 3, 4, 5],
            "n_failed_hard": [0, 1, 0, 0, 1],
            "single_fragment__pass": [True, True, False, True, False],
            "gdb_matches_geometry__pass": [True] * 5,
            "not_uncharacterized__pass": [True] * 5,
            "is_structure_representative": [True, True, True, False, True],
            "not_hard_to_converge__pass": [True, False, True, True, True],
        }
    )


def test_worst_tier_wins_and_all_reasons_are_kept_in_rule_order() -> None:
    t = assign_tiers(frame(), RULES).sort("qm9_id")
    assert t["quality_tier"].to_list() == ["gold", "quarantine", "bronze", "silver", "quarantine"]
    assert t["tier_reasons"].to_list() == [[], ["R1", "R6"], ["R2"], ["R5"], ["R1", "R2"]]


def test_unknown_predicate_or_tier_is_rejected() -> None:
    with pytest.raises(ValueError, match="predicate"):
        assign_tiers(frame(), [{"id": "X", "predicate": "nope", "tier": "gold"}])
    with pytest.raises(ValueError, match="tier"):
        assign_tiers(frame(), [{"id": "X", "predicate": "fragmented", "tier": "platinum"}])


def test_waterfall_is_cumulative_and_impact_counts_unique_effects() -> None:
    steps = waterfall(frame(), RULES)
    assert steps[0]["tiers"]["gold"] == 5
    assert [s["newly_out_of_gold"] for s in steps[1:]] == [2, 1, 1, 0]
    assert steps[-1]["tiers"] == tier_counts(assign_tiers(frame(), RULES))
    impact = rule_impact(frame(), RULES)
    assert impact == {"R1": 2, "R2": 1, "R5": 1, "R6": 0}  # R6 only hits a quarantined record


def test_representative_is_cleanest_then_lowest_id() -> None:
    groups = pl.DataFrame(
        {"qm9_id": [10, 11, 12, 20], "structure_group_id": [10, 10, 10, 20]},
        schema={"qm9_id": pl.UInt32, "structure_group_id": pl.UInt32},
    )
    validation = pl.DataFrame(
        {
            "qm9_id": [10, 11, 12, 20],
            "n_failed_hard": [0, 0, 0, 0],
            "n_failed_identity": [1, 0, 0, 0],
            "n_failed_soft": [0, 1, 0, 0],
        },
        schema={
            "qm9_id": pl.UInt32,
            "n_failed_hard": pl.UInt8,
            "n_failed_identity": pl.UInt8,
            "n_failed_soft": pl.UInt8,
        },
    )
    reps = choose_representatives(groups, validation)
    assert reps.filter(pl.col("is_structure_representative"))["qm9_id"].to_list() == [12, 20]
    assert set(reps["structure_representative_id"].to_list()) == {12, 20}
