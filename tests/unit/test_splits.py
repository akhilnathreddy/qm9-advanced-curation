import numpy as np
import polars as pl
import pytest
from rdkit import DataStructs

from qm9_curation.splits import assign
from qm9_curation.splits.audit import check_assignment, overlaps
from qm9_curation.splits.fingerprints import (
    bit_matrix,
    leader_clusters,
    max_similarity,
    nearest,
    tanimoto_block,
)
from qm9_curation.splits.groups import components, leakage_groups, scaffold_key

FRACTIONS = {"train": 0.8, "val": 0.1, "test": 0.1}


def test_components_merge_chains_and_label_by_smallest_id() -> None:
    ids = [10, 11, 12, 13, 14]
    a = ["x", "x", None, "y", None]
    b = [None, "p", "p", "y", None]
    assert components(ids, [a, b]) == [10, 10, 10, 13, 14]


def test_leakage_groups_join_on_the_geometry_inchikey_block() -> None:
    curated = pl.DataFrame(
        {
            "qm9_id": [1, 2, 3],
            "structure_group_id": [1, 2, 3],
            "tautomer_group_id": [1, 2, 3],
            "connectivity_group_id": [1, 2, 3],
            # 1 and 2: different GDB labels, same constitution from the geometry
            "inchikey_geometry": ["AAAAAAAAAAAAAA-XXXX", "AAAAAAAAAAAAAA-YYYY", None],
        }
    )
    assert leakage_groups(curated).to_list() == [1, 1, 3]


def test_scaffold_keys() -> None:
    assert scaffold_key("CCCO") == "acyclic:C3O1"
    assert scaffold_key("CC(C)N") == scaffold_key("CCCN") == "acyclic:C3N1"
    assert scaffold_key("c1ccccc1O") == scaffold_key("C1CCCCC1") == "C1CCCCC1"
    assert scaffold_key(None) is None


def units_frame(sizes: list[int]) -> pl.DataFrame:
    unit = np.repeat(np.arange(len(sizes)), sizes)
    return pl.DataFrame(
        {"qm9_id": np.arange(1, len(unit) + 1), "unit_id": unit},
        schema={"qm9_id": pl.UInt32, "unit_id": pl.UInt32},
    )


def test_grouped_keeps_units_whole_and_hits_fractions() -> None:
    units = units_frame([3, 1, 2] * 400 + [150])  # one unit larger than half the test target
    out = assign.grouped(units, FRACTIONS, seed=0).join(units, on="qm9_id")
    assert (
        out.group_by("unit_id")
        .agg(pl.col("subset").n_unique())
        .filter(pl.col("subset") > 1)
        .is_empty()
    )
    frac = dict(out.group_by("subset").len().with_columns(pl.col("len") / out.height).rows())
    assert frac["test"] == pytest.approx(0.1, abs=0.005)
    assert frac["val"] == pytest.approx(0.1, abs=0.005)
    assert out.filter(pl.col("unit_id") == 1200)["subset"].unique().to_list() == ["train"]
    again = assign.grouped(units, FRACTIONS, seed=0)
    other = assign.grouped(units, FRACTIONS, seed=1)
    assert again.equals(assign.grouped(units, FRACTIONS, seed=0))
    assert not again.equals(other)


def test_holdout_pulls_whole_units_into_test() -> None:
    units = units_frame([2] * 500)
    rule = pl.Series([False] * 1000).scatter([0, 10, 11], True)  # unit 0 has one rule member
    out = assign.holdout(units, rule, FRACTIONS, seed=3)
    test = set(out.filter(pl.col("subset") == "test")["qm9_id"].to_list())
    assert test == {1, 2, 11, 12}
    pool = out.filter(pl.col("subset") != "test")
    assert (pool["subset"] == "val").mean() == pytest.approx(1 / 9, abs=0.01)


def test_holdout_rules() -> None:
    frame = pl.DataFrame(
        {
            "n_heavy": [9, 8, 9, 3],
            "n_F": [0, 1, 0, 0],
            "gap_ev": [1.0, 2.0, 3.0, 4.0],
            "u0_atomization_ev": [-1.0, -2.0, -3.0, -4.0],
        }
    )
    assert assign.holdout_rule(frame, {"n_heavy_min": 9}).to_list() == [True, False, True, False]
    assert assign.holdout_rule(frame, {"contains_element": "F"}).to_list() == [
        False,
        True,
        False,
        False,
    ]
    upper = {"tail": {"target": "gap_ev", "side": "upper", "fraction": 0.25}}
    lower = {"tail": {"target": "u0_atomization_ev", "side": "lower", "fraction": 0.25}}
    assert assign.holdout_rule(frame, upper).to_list() == [False, False, False, True]
    assert assign.holdout_rule(frame, lower).to_list() == [False, False, False, True]
    with pytest.raises(ValueError):
        assign.holdout_rule(frame, {"nonsense": 1})


def ecfp_frame(bits: list[list[int]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "qm9_id": list(range(1, len(bits) + 1)),
            "bits": bits,
            "counts": [[1] * len(b) for b in bits],
        },
        schema={"qm9_id": pl.UInt32, "bits": pl.List(pl.UInt16), "counts": pl.List(pl.UInt16)},
    )


def test_tanimoto_matches_rdkit() -> None:
    rng = np.random.default_rng(0)
    bits = [
        sorted(rng.choice(2048, size=int(rng.integers(5, 30)), replace=False).tolist())
        for _ in range(40)
    ]
    m = bit_matrix(ecfp_frame(bits))
    fps = []
    for b in bits:
        v = DataStructs.ExplicitBitVect(2048)
        v.SetBitsFromList(b)
        fps.append(v)
    ref = np.array([DataStructs.BulkTanimotoSimilarity(f, fps[20:]) for f in fps[:20]])
    assert np.allclose(tanimoto_block(m[:20], m[20:]), ref)
    assert np.allclose(max_similarity(m[:20], m[20:]), ref.max(axis=1))
    assert (nearest(m[:20], m[20:]) == ref.argmax(axis=1)).all()


def test_leader_clusters_group_identical_fingerprints() -> None:
    bits = [[1, 2, 3], [1, 2, 3], [100, 200, 300], [1, 2, 3, 4], [100, 200, 300, 400]]
    clusters = leader_clusters(ecfp_frame(bits), distance_threshold=0.5)
    assert clusters["cluster_id"].to_list() == [1, 1, 3, 1, 3]


def test_check_assignment_detects_leaks_and_gaps() -> None:
    frame = pl.DataFrame(
        {
            "qm9_id": [1, 2, 3, 4],
            "subset": ["train", "test", "train", "val"],
            "structure_group_id": [1, 1, 3, 4],
            "tautomer_group_id": [1, 2, 3, 4],
            "connectivity_group_id": [1, 2, 3, 4],
            "ik_block": ["a", "b", None, "d"],
            "leakage_group_id": [1, 2, 3, 4],
            "unit_id": [1, 2, 3, 4],
        }
    )
    assert overlaps(frame, ["structure_group_id", "ik_block"]) == {
        "structure_group_id": 1,
        "ik_block": 0,
    }
    spec = {"kind": "grouped"}
    res = check_assignment(frame, 4, spec, FRACTIONS, 0.5, None)
    assert not res["no_key_in_two_subsets"]
    assert (
        check_assignment(frame, 5, spec, FRACTIONS, 0.5, None)["covers_every_record_once"] is False
    )
    rule = pl.Series([False, True, True, False])
    hold = check_assignment(frame, 4, {"kind": "holdout"}, FRACTIONS, 0.5, rule)
    assert hold["holdout_rule_respected"] is False  # record 3 meets the rule but is in train
