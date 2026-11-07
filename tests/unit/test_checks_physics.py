"""Every registered check has a passing and a failing case (Gate 4)."""

import math

import numpy as np
import polars as pl
import pytest

from qm9_curation.checks import physics
from qm9_curation.checks.build import assemble
from qm9_curation.checks.format import run_format
from qm9_curation.checks.registry import CHECK_NAMES, CHECKS
from qm9_curation.checks.statistical import run_statistical
from qm9_curation.units import ROT_CONST_GHZ_AMU_ANG2, thermal_energy_hartree

TOL = dict(
    gap_abs_tol_hartree=1.5e-4,
    h_minus_u_abs_tol_hartree=1.5e-6,
    mulliken_per_atom_tol=5e-7,
    zpve_rel_tol=1e-4,
    rotational_constant_rel_tol=1e-4,
    linear_moment_amu_ang2=0.1,
    rot_a_infinite_ghz=1e4,
)
# Water (qm9_id 3) as stored in the source: geometry, charges and frequencies.
WATER_XYZ = [
    (-0.0343604951, 0.9775395708, 0.0076015923),
    (0.0647664923, 0.0205721989, 0.0015346341),
    (0.8717903737, 1.3007924048, 0.0006931336),
]
WATER_Q = [-0.589706, 0.294853, 0.294853]
WATER_FREQ = [1671.4222, 3803.6305, 3907.698]


def molecules(**overrides: float) -> pl.DataFrame:
    row = dict(
        qm9_id=3,
        n_atoms=3,
        n_heavy=1,
        rot_a_ghz=799.58812,
        rot_b_ghz=437.90386,
        rot_c_ghz=282.94545,
        homo_ha=-0.2928,
        lumo_ha=0.0687,
        gap_ha=0.3615,
        zpve_ha=0.021375,
        u0_ha=-76.404702,
        u298_ha=-76.401867,
        h298_ha=-76.400922,
        g298_ha=-76.422349,
        is_linear=False,
    )
    row.update(overrides)
    return pl.DataFrame([row]).with_columns(pl.col("qm9_id", "n_atoms", "n_heavy").cast(pl.UInt32))


def atoms(
    xyz: list[tuple[float, float, float]] = WATER_XYZ,
    charges: list[float] = WATER_Q,
    elements: tuple[str, ...] = ("O", "H", "H"),
) -> pl.DataFrame:
    z = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "Cl": 17}
    return pl.DataFrame(
        {
            "qm9_id": [3] * len(xyz),
            "atom_idx": list(range(len(xyz))),
            "element": list(elements),
            "atomic_number": [z[e] for e in elements],
            "x_ang": [p[0] for p in xyz],
            "y_ang": [p[1] for p in xyz],
            "z_ang": [p[2] for p in xyz],
            "mulliken_charge_e": charges,
        },
        schema={
            "qm9_id": pl.UInt32,
            "atom_idx": pl.UInt8,
            "element": pl.String,
            "atomic_number": pl.UInt8,
            "x_ang": pl.Float64,
            "y_ang": pl.Float64,
            "z_ang": pl.Float64,
            "mulliken_charge_e": pl.Float64,
        },
    )


def frequencies(sets: list[list[float]]) -> pl.DataFrame:
    rows = [(3, s, m, f) for s, values in enumerate(sets) for m, f in enumerate(values)]
    return pl.DataFrame(
        rows,
        schema={
            "qm9_id": pl.UInt32,
            "set_idx": pl.UInt8,
            "mode_idx": pl.UInt16,
            "wavenumber_cm1": pl.Float64,
        },
        orient="row",
    )


def run(
    mol: pl.DataFrame | None = None, at: pl.DataFrame | None = None, fr: pl.DataFrame | None = None
) -> dict[str, object]:
    out = physics.run_physics(
        mol if mol is not None else molecules(),
        at if at is not None else atoms(),
        fr if fr is not None else frequencies([WATER_FREQ]),
        TOL,
    )
    return out.row(0, named=True)


def test_water_passes_every_physics_check() -> None:
    row = run()
    for name in (
        "gap_consistency",
        "enthalpy_minus_energy",
        "free_energy_order",
        "thermal_order",
        "mulliken_neutral",
        "zpve_from_frequencies",
        "rotational_constants",
        "real_minimum",
        "linearity_representation",
    ):
        assert row[f"{name}__pass"] is True, name
    assert row["linear_by_geometry"] is False


@pytest.mark.parametrize(
    ("check", "kwargs"),
    [
        ("gap_consistency", {"mol": molecules(gap_ha=0.3620)}),
        ("enthalpy_minus_energy", {"mol": molecules(h298_ha=-76.400900)}),
        ("free_energy_order", {"mol": molecules(g298_ha=-76.400000)}),
        ("thermal_order", {"mol": molecules(u298_ha=-76.404800)}),
        ("mulliken_neutral", {"at": atoms(charges=[-0.589706, 0.294853, 0.294900])}),
        ("zpve_from_frequencies", {"fr": frequencies([[1671.4222, 3803.6305, 3990.0]])}),
        ("rotational_constants", {"mol": molecules(rot_b_ghz=438.5)}),
        ("real_minimum", {"fr": frequencies([[-12.0, 1671.4222, 3803.6305, 3907.698]])}),
        ("linearity_representation", {"mol": molecules(rot_a_ghz=0.0)}),
    ],
)
def test_each_physics_check_fails_when_it_should(
    check: str, kwargs: dict[str, pl.DataFrame]
) -> None:
    assert run(**kwargs)[f"{check}__pass"] is False


def test_zpve_picks_the_consistent_set_of_a_doubled_record() -> None:
    row = run(fr=frequencies([[1700.0, 3850.0, 3950.0], WATER_FREQ]))
    assert row["zpve_consistent_set"] == 1 and row["zpve_from_frequencies__pass"] is True
    assert row["zpve_other_set_residual"] > 1e-3


def test_h_minus_u_reference_is_kT() -> None:
    assert math.isclose(-76.400922 - -76.401867, thermal_energy_hartree(), abs_tol=1.5e-6)


def test_linear_molecule_by_geometry_skips_a() -> None:
    # Acetylene (qm9_id 4): A undefined; B = C from the source.
    xyz = [
        (0.5995394918, 0.0, 1.0),
        (-0.5995394918, 0.0, 1.0),
        (-1.6616385861, 0.0, 1.0),
        (1.6616385861, 0.0, 1.0),
    ]
    mol = molecules(
        n_atoms=4,
        n_heavy=2,
        rot_a_ghz=0.0,
        rot_b_ghz=35.6100361,
        rot_c_ghz=35.6100361,
        is_linear=True,
    )
    row = physics.rotational_constants(
        mol, atoms(xyz, [0.0] * 4, ("C", "C", "H", "H")), 1e-4, 0.1
    ).row(0, named=True)
    assert row["linear_by_geometry"] is True and row["rotational_constants__pass"] is True


def test_rotational_constant_conversion() -> None:
    moments = physics.principal_moments(np.array([[8.0, 1.0, 1.0]]), np.array([WATER_XYZ]))
    assert ROT_CONST_GHZ_AMU_ANG2 / moments[0, 0] == pytest.approx(799.58812, rel=1e-4)


def test_format_checks_pass_and_fail() -> None:
    mol = molecules()
    ok = run_format(mol, atoms(), frequencies([WATER_FREQ]), 9).row(0, named=True)
    assert all(
        ok[f"{n}__pass"]
        for n in ("elements_allowed", "heavy_atom_limit", "frequency_count", "coordinates_finite")
    )
    bad_el = run_format(mol, atoms(elements=("O", "H", "Cl")), frequencies([WATER_FREQ]), 9).row(
        0, named=True
    )
    assert bad_el["elements_allowed__pass"] is False
    assert (
        run_format(molecules(n_heavy=10), atoms(), frequencies([WATER_FREQ]), 9).row(0, named=True)[
            "heavy_atom_limit__pass"
        ]
        is False
    )
    assert (
        run_format(mol, atoms(), frequencies([WATER_FREQ[:2]]), 9).row(0, named=True)[
            "frequency_count__pass"
        ]
        is False
    )
    nan_xyz = [(math.nan, 0.0, 0.0), *WATER_XYZ[1:]]
    assert (
        run_format(mol, atoms(nan_xyz), frequencies([WATER_FREQ]), 9).row(0, named=True)[
            "coordinates_finite__pass"
        ]
        is False
    )


def test_statistical_checks_pass_and_fail() -> None:
    profile = pl.DataFrame(
        {"qm9_id": [1, 2], "max_abs_robust_z": [2.0, 9.0]},
        schema={"qm9_id": pl.UInt32, "max_abs_robust_z": pl.Float64},
    )
    out = run_statistical(profile, 6.0, [2])
    assert out["not_statistical_outlier__pass"].to_list() == [True, False]
    assert out["not_hard_to_converge__pass"].to_list() == [True, False]


def test_every_registered_check_is_exercised_by_this_module_or_identity_tests() -> None:
    tested = {
        "elements_allowed",
        "heavy_atom_limit",
        "frequency_count",
        "coordinates_finite",
        "gap_consistency",
        "enthalpy_minus_energy",
        "free_energy_order",
        "thermal_order",
        "mulliken_neutral",
        "zpve_from_frequencies",
        "rotational_constants",
        "real_minimum",
        "linearity_representation",
        "not_statistical_outlier",
        "not_hard_to_converge",
        # Layer 2, exercised in test_checks_identity.py:
        "single_fragment",
        "gdb_matches_geometry",
        "not_uncharacterized",
        "relaxed_smiles_matches_geometry",
    }
    assert tested == set(CHECK_NAMES)


def test_bitmask_and_severity_counts() -> None:
    base = pl.DataFrame({"qm9_id": [1, 2]}, schema={"qm9_id": pl.UInt32})
    cols = {f"{c.name}__pass": [True, True] for c in CHECKS}
    cols["gap_consistency__pass"] = [True, False]
    cols["not_hard_to_converge__pass"] = [True, False]
    flags = assemble([base.with_columns(**{k: pl.Series(v) for k, v in cols.items()})])
    second = flags.row(1, named=True)
    gap_bit = 1 << CHECK_NAMES.index("gap_consistency")
    soft_bit = 1 << CHECK_NAMES.index("not_hard_to_converge")
    assert second["flag_bitmask"] == gap_bit | soft_bit
    assert (second["n_failed_hard"], second["n_failed_soft"], second["n_failed_identity"]) == (
        1,
        1,
        0,
    )
    assert flags.row(0, named=True)["flag_bitmask"] == 0
