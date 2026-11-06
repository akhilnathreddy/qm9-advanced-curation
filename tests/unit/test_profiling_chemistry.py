import polars as pl

from qm9_curation.profiling.chemistry import profile_molecule
from qm9_curation.profiling.duplicates import _union_find, duplicate_groups
from qm9_curation.profiling.issues import TRIAGE_CLASSES, build_issues
from qm9_curation.profiling.properties import robust_z


def test_benzene_like_ring_and_scaffold() -> None:
    r = profile_molecule(
        1,
        "C1=CC=CC=C1",
        "c1ccccc1",
        "InChI=1S/C6H6/c1-2-4-6-5-3-1/h1-6H",
        "InChI=1S/C6H6/c1-2-4-6-5-3-1/h1-6H",
    )
    assert r["n_rings"] == 1 and r["ring_sizes"] == "6" and r["n_aromatic_atoms"] == 6
    assert r["murcko_scaffold"] == "c1ccccc1" and r["generic_scaffold"] == "C1CCCCC1"
    assert r["inchikey_gdb"] == "UHOVQNZJYSORNB-UHFFFAOYSA-N"


def test_kekule_forms_share_canonical_smiles() -> None:
    a = profile_molecule(1, "C#CCC1=CN=CC=N1", "x", "InChI=1S/x", "InChI=1S/x")
    b = profile_molecule(2, "C#CCC1=NC=CN=C1", "x", "InChI=1S/x", "InChI=1S/x")
    assert a["canonical_smiles_gdb"] == b["canonical_smiles_gdb"]


def test_open_babel_style_zwitterion_does_not_parse_and_falls_back() -> None:
    r = profile_molecule(80, "[NH3+]CC([O-])=O", "[NH3]CC(=O)[O]", "InChI=1S/x", "InChI=1S/x")
    assert r["gdb_rdkit_ok"] and not r["relaxed_rdkit_ok"]
    assert r["n_charged_atoms"] == 2  # ring/charge stats fall back to the GDB-17 molecule
    assert r["murcko_scaffold"] == ""


def test_union_find_merges_transitively() -> None:
    root = _union_find([1, 2, 3, 4, 5], [(3, 2), (2, 5)])
    assert root == {1: 1, 2: 2, 3: 2, 4: 4, 5: 2}


def test_duplicate_groups_combine_smiles_and_geometry() -> None:
    chem = pl.DataFrame(
        {
            "qm9_id": [1, 2, 3, 4],
            "inchikey_gdb": [
                "AAAAAAAAAAAAAA-X-N",
                "AAAAAAAAAAAAAA-X-N",
                "BBBBBBBBBBBBBB-Y-N",
                "CCCCCCCCCCCCCC-Z-N",
            ],
            "canonical_smiles_gdb": ["c1ccccc1", "c1ccccc1", "CO", "OC=O"],
        },
        schema={"qm9_id": pl.UInt32, "inchikey_gdb": pl.String, "canonical_smiles_gdb": pl.String},
    )
    pairs = pl.DataFrame(
        {"qm9_id_a": [3], "qm9_id_b": [4], "max_distance_diff_ang": [0.001]},
        schema={"qm9_id_a": pl.UInt32, "qm9_id_b": pl.UInt32, "max_distance_diff_ang": pl.Float64},
    )
    g = duplicate_groups(chem, pairs)
    assert g["structure_group_id"].to_list() == [1, 1, 3, 3]
    assert g["tautomer_group_size"].to_list() == [2, 2, 1, 1]
    assert g["in_smiles_duplicate"].to_list() == [True, True, False, False]
    assert g["in_geometric_duplicate"].to_list() == [False, False, True, True]


def test_robust_z_is_computed_within_strata() -> None:
    from qm9_curation.schema import PROPERTY_COLUMNS

    n = 40
    data = {c: [1.0] * n for c in PROPERTY_COLUMNS}
    data["mu_debye"] = [1.0 + 0.01 * (i % 5) for i in range(20)] + [
        5.0 + 0.01 * (i % 5) for i in range(20)
    ]
    frame = pl.DataFrame({"qm9_id": list(range(1, n + 1)), "n_heavy": [5] * 20 + [9] * 20, **data})
    z = robust_z(frame)["z_mu_debye"]
    assert z.abs().max() < 2  # each stratum is internally homogeneous


def test_every_issue_rule_has_a_valid_triage() -> None:
    profile = {
        "composition": {"n_heavy_histogram": {"9": 10}},
        "chemistry": {
            "gdb_vs_relaxed_flat_smiles": [
                {"comparison": "disagree", "uncharacterized": False, "n": 3}
            ],
            "rdkit_parse_failures": {"relaxed": 1, "gdb": 0},
            "relaxed_smiles_with_radicals": 1,
            "relaxed_smiles_with_fragments": 1,
            "gdb_smiles_with_charges": 1,
            "scaffolds": {"empty_scaffold_molecules": 1},
        },
        "duplicates": {
            "structure": {"n_molecules_in_such_groups": 2, "max_u0_spread_kcal_per_mol": 0.0},
            "tautomer": {"n_molecules_in_such_groups": 2},
        },
        "properties": {"mu_debye": {"n_robust_z_above_review": 1}},
        "geometry": {"molecules_with_negative_frequency": 1},
        "parse": {"doubled_frequencies": 1, "linear_molecules": 1},
        "rotational": {
            "a_zero_and_3n_minus_5_frequencies": 1,
            "a_zero_but_3n_minus_6_frequencies": 1,
            "a_above_1e4_ghz": 1,
        },
    }
    issues = build_issues(profile, {})
    assert len({i["id"] for i in issues}) == len(issues) == 14
    assert all(i["triage"] in TRIAGE_CLASSES for i in issues)
