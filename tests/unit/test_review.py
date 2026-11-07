from pathlib import Path

import numpy as np
import polars as pl
import pytest
from scipy import sparse

from qm9_curation.external import kim2019
from qm9_curation.review import multifidelity, scientific
from qm9_curation.review.build import review_table
from qm9_curation.review.stats import (
    compare_distributions,
    fit_per_element,
    mann_whitney_greater,
    robust_z,
)
from qm9_curation.review.suspicion import ridge_oof, suspicion_scores
from qm9_curation.units import HARTREE_TO_EV
from tests.unit.test_external_kim2019 import ATOMS, MEMBERS, write_archive

GROUND = {"H": 2, "C": 3, "N": 4, "O": 3, "F": 2}
# (formula, counts H C N O F, n_heavy)
FORMULAS = [
    ("C1H4", (4, 1, 0, 0, 0)), ("H3N1", (3, 0, 1, 0, 0)), ("H2O1", (2, 0, 0, 1, 0)),
    ("C2H6", (6, 2, 0, 0, 0)), ("C1H3F1", (3, 1, 0, 0, 1)), ("C1H5N1", (5, 1, 1, 0, 0)),
    ("C1H4O1", (4, 1, 0, 1, 0)), ("C2H3N1", (3, 2, 1, 0, 0)), ("C1H1F3", (1, 1, 0, 0, 3)),
    ("C2H4O1", (4, 2, 0, 1, 0)),
]  # fmt: skip
# Per-element B3LYP - G4(MP2) total-energy offsets used to build the synthetic data (Ha).
OFFSET = {"H": 0.001, "C": -0.05, "N": -0.056, "O": -0.064, "F": -0.068}


def frames(noise: float = 0.0) -> tuple[pl.DataFrame, pl.DataFrame, list[tuple]]:
    rng = np.random.default_rng(0)
    cur, comp, kim_rows = [], [], []
    for i, (formula, counts) in enumerate(FORMULAS, start=1):
        n = dict(zip("HCNOF", counts, strict=True))
        u0 = -40.0 * (counts[1] + counts[2] + counts[3] + counts[4]) - 0.5 * counts[0]
        zpve = 0.01 * sum(counts)
        cur.append(
            {
                "qm9_id": i,
                "quality_tier": "gold",
                "n_heavy": sum(counts[1:]),
                "source_u0_ha": u0,
                "source_zpve_ha": zpve,
                "u0_atomization_ev": -10.0 * i,
                "formula": formula,
                "smiles_geometry": "C",
                "u0_ev": u0 * HARTREE_TO_EV,
            }
        )
        comp.append({"qm9_id": i, **{f"n_{e}": n[e] for e in "HCNOF"}})
        g4 = u0 - sum(n[e] * OFFSET[e] for e in "HCNOF") + noise * rng.standard_normal()
        kim_rows.append((i, formula, g4, u0 - zpve, zpve))
    curated = pl.DataFrame(cur).with_columns(pl.col("qm9_id").cast(pl.UInt32))
    composition = pl.DataFrame(comp).with_columns(pl.col("qm9_id").cast(pl.UInt32))
    return curated, composition, kim_rows


def load(tmp_path: Path, rows: list[tuple], **kw: object) -> kim2019.Kim2019:
    return kim2019.load(write_archive(tmp_path / "k.tar.gz", rows, **kw), MEMBERS)  # type: ignore[arg-type]


def test_statuses_and_withheld_atomization(tmp_path: Path) -> None:
    curated, composition, rows = frames()
    rows = [r for r in rows if r[0] != 10]  # id 10 did not converge
    i, f, g, b, z = rows[3]
    rows[3] = (i, f, g, b - 2e-3, z)  # re-optimization lowered B3LYP by > 1 kcal/mol
    kim = load(tmp_path, rows, not_converged=[10])
    xref, info = multifidelity.xref_table(curated, composition, kim, GROUND, 1.59e-3)
    status = dict(xref.select("qm9_id", "g4mp2_status").iter_rows())
    assert status[4] == "geometry_changed" and status[10] == "not_converged"
    assert sum(s == "ok" for s in status.values()) == 8
    assert info["invalid_reference_elements"] == ["N"]
    with_n = xref.join(composition, on="qm9_id").filter(pl.col("n_N") > 0)
    assert with_n["g4mp2_u0_atomization_ev"].is_null().all()
    assert (
        with_n.filter(pl.col("g4mp2_status") != "not_converged")["g4mp2_atomization_missing_reason"]
        .str.contains("N")
        .all()
    )
    no_n = xref.join(composition, on="qm9_id").filter(
        (pl.col("n_N") == 0) & (pl.col("qm9_id") != 10)
    )
    assert no_n["g4mp2_u0_atomization_ev"].is_not_null().all()
    assert info["formulas_agree"]


def test_ground_state_references_give_every_atomization(tmp_path: Path) -> None:
    curated, composition, rows = frames()
    atoms = {**ATOMS, "N": (ATOMS["N"][0], ATOMS["N"][1], 4)}
    xref, info = multifidelity.xref_table(
        curated, composition, load(tmp_path, rows, atoms=atoms), GROUND, 1.59e-3
    )
    assert info["invalid_reference_elements"] == []
    assert xref["g4mp2_u0_atomization_ev"].is_not_null().all()


def test_per_element_offsets_leave_no_residual(tmp_path: Path) -> None:
    curated, composition, rows = frames()
    xref, info = multifidelity.xref_table(curated, composition, load(tmp_path, rows), GROUND, 1e-3)
    assert np.abs(xref["delta_residual_ev"].to_numpy()).max() < 1e-9
    fitted = info["total_energy_fit"]
    for e in "HCNOF":
        assert fitted[f"n_{e}"] == pytest.approx(OFFSET[e] * HARTREE_TO_EV, abs=1e-9)


def test_noise_survives_the_per_element_fit(tmp_path: Path) -> None:
    curated, composition, rows = frames(noise=1e-3)
    xref, _ = multifidelity.xref_table(curated, composition, load(tmp_path, rows), GROUND, 1.0)
    assert np.abs(xref["delta_residual_ev"].to_numpy()).max() > 1e-3


def test_nitrogen_offset_recovers_a_shifted_reference(tmp_path: Path) -> None:
    curated, composition, rows = frames()
    shift = 0.0876  # Hartree, raises the N reference as a doublet would
    atoms = {**ATOMS, "N": (ATOMS["N"][0] + shift, ATOMS["N"][1], 2)}
    # G4(MP2) energies whose AE equals the B3LYP AE when the true N reference is used.
    refs = {e: v[0] for e, v in ATOMS.items()}
    rows = [
        (i, f, sum(n * refs[e] for e, n in kim2019.formula_counts(f).items())
         + ae / HARTREE_TO_EV, b, z)
        for (i, f, _, b, z), ae in zip(rows, curated["u0_atomization_ev"], strict=True)
    ]  # fmt: skip
    kim = load(tmp_path, rows, atoms=atoms)
    xref, _ = multifidelity.xref_table(curated, composition, kim, GROUND, 1.0)
    coefs = multifidelity.nitrogen_offset(curated, composition, xref, kim)
    assert coefs["n_N"] == pytest.approx(shift * HARTREE_TO_EV, abs=1e-8)
    assert coefs["n_C"] == pytest.approx(0.0, abs=1e-8)
    assert multifidelity.authors_atomization_reproduced(kim) < 1e-9


def test_fit_per_element_recovers_coefficients() -> None:
    _, composition, _ = frames()
    y = sum(pl.col(f"n_{e}") * c for e, c in zip("HCNOF", (1.0, 2.0, 3.0, 4.0, 5.0), strict=True))
    frame = composition.with_columns((y + 0.5).alias("y"))
    coefs, pred = fit_per_element(frame, "y", intercept=True)
    assert coefs["n_O"] == pytest.approx(4.0) and coefs["intercept"] == pytest.approx(0.5)
    assert np.allclose(frame.select(pred).to_series().to_numpy(), frame["y"].to_numpy())


def test_robust_z_and_two_sample_tests() -> None:
    frame = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 100.0]})
    z = frame.select(robust_z(pl.col("x"))).to_series().to_list()
    assert z[2] == 0.0 and z[4] > 40
    assert frame.select(robust_z(pl.lit(1.0) * pl.col("x") * 0)).to_series().is_null().all()
    a = np.linspace(0, 1, 200)
    same = compare_distributions(a, a)
    assert same["ks_statistic"] == 0.0 and same["wasserstein"] == 0.0
    assert compare_distributions(a, a + 0.5)["wasserstein"] == pytest.approx(0.5)
    mw = mann_whitney_greater(a + 10, a)
    assert mw["prob_a_greater"] == 1.0 and mw["pvalue"] < 1e-10


def test_ridge_oof_fits_linear_data() -> None:
    rng = np.random.default_rng(1)
    x = sparse.random(400, 30, density=0.2, random_state=2, format="csr") * 5
    w = rng.normal(size=30)
    y = x @ w + 3.0
    pred = ridge_oof(x, y, np.arange(400) % 5, alpha=1e-8)
    assert np.abs(pred - y).max() < 1e-5


def test_suspicion_ranks_an_injected_label_error_first() -> None:
    rng = np.random.default_rng(3)
    n = 300
    bits = [sorted(rng.choice(2048, size=8, replace=False).tolist()) for _ in range(n)]
    counts = [[1] * 8 for _ in range(n)]
    ecfp = pl.DataFrame(
        {"qm9_id": np.arange(1, n + 1), "bits": bits, "counts": counts},
        schema={"qm9_id": pl.UInt32, "bits": pl.List(pl.UInt16), "counts": pl.List(pl.UInt16)},
    )
    comp = pl.DataFrame(
        {"qm9_id": np.arange(1, n + 1), **{f"n_{e}": rng.integers(0, 5, n) for e in "HCNOF"}}
    ).with_columns(pl.col("qm9_id").cast(pl.UInt32))
    y = comp.select(sum(pl.col(f"n_{e}") * k for k, e in enumerate("HCNOF", 1))).to_series()
    y = y.cast(pl.Float64).to_numpy() + 0.01 * rng.standard_normal(n)
    y[41] += 5.0
    curated = comp.select("qm9_id").with_columns(
        pl.col("qm9_id").alias("structure_group_id"),
        pl.Series("u0_atomization_ev", y),
        pl.Series("gap_ev", np.full(n, 7.0) + 0.01 * rng.standard_normal(n)),
    )
    scores = suspicion_scores(curated, comp, ecfp, ["u0_atomization_ev", "gap_ev"], 5, 1e-3)
    assert scores.sort("suspicion_score", descending=True)["qm9_id"][0] == 42


def test_review_reasons_follow_thresholds_and_keep_tiers() -> None:
    ids = pl.Series("qm9_id", [1, 2, 3], dtype=pl.UInt32)
    curated = pl.DataFrame({"qm9_id": ids, "quality_tier": ["gold", "silver", "bronze"]})
    comp = pl.DataFrame(
        {
            "qm9_id": ids,
            "composition_residual_ev": [0.0, 0.0, 0.0],
            "composition_robust_z": [7.0, -6.0, -6.5],
        }
    )
    thermal = pl.DataFrame(
        {
            "qm9_id": ids,
            "thermal_energy_robust_z": [0.0, 0.0, 9.0],
            "minus_ts_robust_z": [None, 0.0, 0.0],
        }
    )
    xref = pl.DataFrame(
        {
            "qm9_id": ids,
            "g4mp2_status": ["ok", "not_converged", "geometry_changed"],
            "delta_residual_ev": [0.1, None, None],
            "delta_robust_z": [1.0, None, None],
        }
    )
    sus = pl.DataFrame({"qm9_id": ids, "suspicion_score": [6.1, None, 1.0]})
    review = review_table(curated, comp, thermal, xref, sus, threshold=6.0)
    reasons = dict(review.select("qm9_id", "review_reasons").iter_rows())
    assert reasons[1] == "composition_outlier,model_suspicion"
    assert reasons[2] == "g4mp2_not_converged"  # |z| = 6 is not above the threshold
    assert reasons[3] == "composition_outlier,g4mp2_geometry_changed,thermal_energy_outlier"
    assert review["quality_tier"].to_list() == ["gold", "silver", "bronze"]


def test_structure_notes_and_constitution() -> None:
    assert "strained small ring" in scientific.structure_notes("C1CC1")["notes"]
    norbornane = scientific.structure_notes("C1CC2CCC1C2")
    assert "bridged polycycle" in norbornane["notes"] and norbornane["smallest_ring"] == 5
    assert "conjugated triple bonds" in scientific.structure_notes("C#CC#C")["notes"]
    assert scientific.structure_notes(None)["notes"] == "no geometry SMILES"
    assert scientific.constitution("C[C@H](N)O") == scientific.constitution("C[C@@H](N)O")
    assert scientific.constitution("C[C@H](N)O") != scientific.constitution("CC(N)C")
