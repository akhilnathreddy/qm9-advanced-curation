"""Layer 3 — physics invariants recomputed from the record itself.

QM9 stores several quantities from one calculation, so many fields can be
recomputed from others. Every check returns, per molecule, a **residual**
(how far off) and a **pass** flag at the configured tolerance, so tolerances
can be audited against the residual distribution (ADR 0008).

All functions are vectorized over the interim tables.
"""

from collections.abc import Mapping

import numpy as np
import polars as pl

from qm9_curation.profiling.geometry import batches
from qm9_curation.units import (
    INV_CM_TO_HARTREE,
    ISOTOPE_MASS_AMU,
    ROT_CONST_GHZ_AMU_ANG2,
    thermal_energy_hartree,
)

MASS_BY_Z = {
    1: ISOTOPE_MASS_AMU["H"],
    6: ISOTOPE_MASS_AMU["C"],
    7: ISOTOPE_MASS_AMU["N"],
    8: ISOTOPE_MASS_AMU["O"],
    9: ISOTOPE_MASS_AMU["F"],
}


def _result(frame: pl.DataFrame, name: str, residual: pl.Expr, passed: pl.Expr) -> pl.DataFrame:
    return frame.select(
        "qm9_id", residual.alias(f"{name}__residual"), passed.alias(f"{name}__pass")
    )


def gap_consistency(mol: pl.DataFrame, tol: float) -> pl.DataFrame:
    res = (pl.col("gap_ha") - (pl.col("lumo_ha") - pl.col("homo_ha"))).abs()
    return _result(mol, "gap_consistency", res, res <= tol)


def enthalpy_minus_energy(mol: pl.DataFrame, tol: float) -> pl.DataFrame:
    res = (pl.col("h298_ha") - pl.col("u298_ha") - thermal_energy_hartree()).abs()
    return _result(mol, "enthalpy_minus_energy", res, res <= tol)


def free_energy_order(mol: pl.DataFrame) -> pl.DataFrame:
    res = pl.col("h298_ha") - pl.col("g298_ha")  # T*S, must be positive
    return _result(mol, "free_energy_order", res, res > 0)


def thermal_order(mol: pl.DataFrame) -> pl.DataFrame:
    res = pl.col("u298_ha") - pl.col("u0_ha")  # thermal energy above 0 K, must be positive
    return _result(mol, "thermal_order", res, res > 0)


def mulliken_neutral(mol: pl.DataFrame, atoms: pl.DataFrame, per_atom_tol: float) -> pl.DataFrame:
    sums = atoms.group_by("qm9_id").agg(pl.col("mulliken_charge_e").sum().abs().alias("s"))
    j = mol.select("qm9_id", "n_atoms").join(sums, on="qm9_id")
    # Each charge is printed to 6 decimals, so rounding alone allows n_atoms * 0.5e-6.
    return _result(
        j, "mulliken_neutral", pl.col("s"), pl.col("s") <= pl.col("n_atoms") * per_atom_tol
    )


def zpve_from_frequencies(mol: pl.DataFrame, freq: pl.DataFrame, rel_tol: float) -> pl.DataFrame:
    """Relative ZPVE residual per frequency set; the record passes if any set is consistent.

    Also reports which set is consistent, resolving the 433 doubled records,
    and whether that set contains an imaginary mode. Gaussian's ZPVE sums
    real modes only, so imaginary (negative) modes are excluded.
    """
    per_set = (
        freq.group_by("qm9_id", "set_idx")
        .agg(
            (
                pl.col("wavenumber_cm1").filter(pl.col("wavenumber_cm1") > 0).sum()
                * 0.5
                * INV_CM_TO_HARTREE
            ).alias("zc"),
            (pl.col("wavenumber_cm1") < 0).sum().alias("n_imag"),
        )
        .join(mol.select("qm9_id", "zpve_ha"), on="qm9_id")
        .with_columns(((pl.col("zc") - pl.col("zpve_ha")) / pl.col("zpve_ha")).abs().alias("rel"))
    )
    best = (
        per_set.sort(["qm9_id", "rel", "set_idx"]).group_by("qm9_id", maintain_order=True).first()
    )
    other = per_set.join(
        best.select("qm9_id", pl.col("set_idx").alias("best_set")), on="qm9_id"
    ).filter(pl.col("set_idx") != pl.col("best_set"))
    return (
        best.select(
            "qm9_id",
            pl.col("rel").alias("zpve_from_frequencies__residual"),
            (pl.col("rel") <= rel_tol).alias("zpve_from_frequencies__pass"),
            pl.col("set_idx").cast(pl.UInt8).alias("zpve_consistent_set"),
            (pl.col("n_imag") == 0).alias("real_minimum__pass"),
            pl.col("n_imag").cast(pl.Float64).alias("real_minimum__residual"),
        )
        .join(
            other.select("qm9_id", pl.col("rel").alias("zpve_other_set_residual")),
            on="qm9_id",
            how="left",
        )
        .sort("qm9_id")
    )


def principal_moments(z: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """Principal moments of inertia (amu * Å^2), ascending, for a batch of molecules."""
    masses = np.vectorize(MASS_BY_Z.__getitem__)(z.astype(int))
    com = (masses[..., None] * xyz).sum(1, keepdims=True) / masses.sum(1)[:, None, None]
    r = xyz - com
    r2 = (r**2).sum(-1)
    inertia = np.einsum("mi,mi,jk->mjk", masses, r2, np.eye(3)) - np.einsum(
        "mi,mij,mik->mjk", masses, r, r
    )
    return np.linalg.eigvalsh(inertia)


def _rel(source: str, calc: str) -> pl.Expr:
    return ((pl.col(source) - pl.col(calc)) / pl.col(calc)).abs()


def rotational_constants(
    mol: pl.DataFrame, atoms: pl.DataFrame, rel_tol: float, linear_moment: float
) -> pl.DataFrame:
    """Recompute A >= B >= C (GHz) from geometry; compare B and C, and A when finite.

    A molecule is linear by geometry when its smallest principal moment is
    below ``linear_moment`` (amu * Å^2); A is then undefined and not compared.
    """
    rows = []
    for batch in batches(atoms):
        moments = principal_moments(batch.z, batch.xyz)
        rows.append(
            pl.DataFrame(
                {
                    "qm9_id": batch.qm9_ids,
                    "i_min": moments[:, 0],
                    "b_calc": ROT_CONST_GHZ_AMU_ANG2 / moments[:, 1],
                    "c_calc": ROT_CONST_GHZ_AMU_ANG2 / moments[:, 2],
                    "a_calc": np.where(
                        moments[:, 0] > linear_moment,
                        ROT_CONST_GHZ_AMU_ANG2 / np.maximum(moments[:, 0], 1e-12),
                        np.nan,
                    ),
                }
            )
        )
    calc = pl.concat(rows).with_columns(pl.col("qm9_id").cast(pl.UInt32))
    j = mol.select("qm9_id", "rot_a_ghz", "rot_b_ghz", "rot_c_ghz").join(calc, on="qm9_id")
    linear = pl.col("i_min") <= linear_moment
    res = pl.max_horizontal(
        _rel("rot_b_ghz", "b_calc"),
        _rel("rot_c_ghz", "c_calc"),
        pl.when(linear).then(None).otherwise(_rel("rot_a_ghz", "a_calc")),
    )
    return j.select(
        "qm9_id",
        res.alias("rotational_constants__residual"),
        (res <= rel_tol).alias("rotational_constants__pass"),
        linear.alias("linear_by_geometry"),
        pl.col("i_min").alias("smallest_moment_amu_ang2"),
    ).sort("qm9_id")


def linearity_representation(
    mol: pl.DataFrame, rot: pl.DataFrame, a_infinite_ghz: float
) -> pl.DataFrame:
    """Linear by geometry must coincide with A = 0 / huge and a 3N-5 frequency count (soft)."""
    j = mol.select("qm9_id", "rot_a_ghz", "is_linear").join(
        rot.select("qm9_id", "linear_by_geometry"), on="qm9_id"
    )
    a_marks_linear = (pl.col("rot_a_ghz") == 0) | (pl.col("rot_a_ghz") > a_infinite_ghz)
    consistent = (pl.col("linear_by_geometry") == a_marks_linear) & (
        pl.col("linear_by_geometry") == pl.col("is_linear")
    )
    return j.select(
        "qm9_id",
        pl.lit(None, dtype=pl.Float64).alias("linearity_representation__residual"),
        consistent.alias("linearity_representation__pass"),
    )


def run_physics(
    mol: pl.DataFrame, atoms: pl.DataFrame, freq: pl.DataFrame, tol: Mapping[str, float]
) -> pl.DataFrame:
    rot = rotational_constants(
        mol, atoms, tol["rotational_constant_rel_tol"], tol["linear_moment_amu_ang2"]
    )
    parts = [
        gap_consistency(mol, tol["gap_abs_tol_hartree"]),
        enthalpy_minus_energy(mol, tol["h_minus_u_abs_tol_hartree"]),
        free_energy_order(mol),
        thermal_order(mol),
        mulliken_neutral(mol, atoms, tol["mulliken_per_atom_tol"]),
        zpve_from_frequencies(mol, freq, tol["zpve_rel_tol"]),
        rot,
        linearity_representation(mol, rot, tol["rot_a_infinite_ghz"]),
    ]
    out = parts[0]
    for part in parts[1:]:
        out = out.join(part, on="qm9_id", how="left")
    return out.sort("qm9_id")
