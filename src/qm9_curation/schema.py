"""Pandera schemas for the parsed (interim) tables.

These check structure, not chemistry: column names and dtypes, nullability,
keys, allowed categorical values, and that every number is finite. Numeric
bounds are deliberately wide; they exist to reject NaN and infinity (which
compare outside any range in Polars), not to judge physical plausibility,
which is Phase 4's job.

Column names carry their unit. Values are in the **source** units; unit
conversion happens in Phase 5 and produces separately named columns.
"""

from typing import Any, ClassVar

import pandera.polars as pa
import polars as pl

ELEMENTS = ["H", "C", "N", "O", "F"]
ATOMIC_NUMBERS = [1, 6, 7, 8, 9]
FINITE = 1e9  # any finite source value is far inside this bound

PROPERTY_COLUMNS = (
    "rot_a_ghz",
    "rot_b_ghz",
    "rot_c_ghz",
    "mu_debye",
    "alpha_bohr3",
    "homo_ha",
    "lumo_ha",
    "gap_ha",
    "r2_bohr2",
    "zpve_ha",
    "u0_ha",
    "u298_ha",
    "h298_ha",
    "g298_ha",
    "cv_cal_per_mol_k",
)


def _finite() -> Any:  # pa.Field is a factory function, not a type
    return pa.Field(ge=-FINITE, le=FINITE)


class Molecules(pa.DataFrameModel):
    qm9_id: pl.UInt32 = pa.Field(unique=True, ge=1)
    n_atoms: pl.UInt8 = pa.Field(ge=1)
    n_heavy: pl.UInt8 = pa.Field(ge=1, le=9)
    formula: pl.String
    rot_a_ghz: pl.Float64 = _finite()
    rot_b_ghz: pl.Float64 = _finite()
    rot_c_ghz: pl.Float64 = _finite()
    mu_debye: pl.Float64 = _finite()
    alpha_bohr3: pl.Float64 = _finite()
    homo_ha: pl.Float64 = _finite()
    lumo_ha: pl.Float64 = _finite()
    gap_ha: pl.Float64 = _finite()
    r2_bohr2: pl.Float64 = _finite()
    zpve_ha: pl.Float64 = _finite()
    u0_ha: pl.Float64 = _finite()
    u298_ha: pl.Float64 = _finite()
    h298_ha: pl.Float64 = _finite()
    g298_ha: pl.Float64 = _finite()
    cv_cal_per_mol_k: pl.Float64 = _finite()
    smiles_gdb: pl.String = pa.Field(str_length={"min_value": 1})
    smiles_relaxed: pl.String = pa.Field(str_length={"min_value": 1})
    inchi_gdb: pl.String = pa.Field(str_startswith="InChI=")
    inchi_relaxed: pl.String = pa.Field(str_startswith="InChI=")
    n_modes: pl.UInt16
    n_frequency_sets: pl.UInt8 = pa.Field(isin=[1, 2])
    is_linear: pl.Boolean
    has_fortran_exponent: pl.Boolean
    source_member: pl.String = pa.Field(str_matches=r"^dsgdb9nsd_\d{6}\.xyz$")
    source_sha256: pl.String = pa.Field(str_matches=r"^[0-9a-f]{64}$")
    source_archive_sha256: pl.String = pa.Field(str_matches=r"^[0-9a-f]{64}$")

    class Config:
        strict = True
        coerce = False


class Atoms(pa.DataFrameModel):
    qm9_id: pl.UInt32 = pa.Field(ge=1)
    atom_idx: pl.UInt8
    element: pl.String = pa.Field(isin=ELEMENTS)
    atomic_number: pl.UInt8 = pa.Field(isin=ATOMIC_NUMBERS)
    x_ang: pl.Float64 = _finite()
    y_ang: pl.Float64 = _finite()
    z_ang: pl.Float64 = _finite()
    mulliken_charge_e: pl.Float64 = _finite()

    class Config:
        strict = True
        coerce = False
        unique: ClassVar[list[str]] = ["qm9_id", "atom_idx"]


class Frequencies(pa.DataFrameModel):
    qm9_id: pl.UInt32 = pa.Field(ge=1)
    set_idx: pl.UInt8 = pa.Field(isin=[0, 1])
    mode_idx: pl.UInt16
    wavenumber_cm1: pl.Float64 = _finite()

    class Config:
        strict = True
        coerce = False
        unique: ClassVar[list[str]] = ["qm9_id", "set_idx", "mode_idx"]


class ParseLog(pa.DataFrameModel):
    qm9_id: pl.UInt32 = pa.Field(ge=1)
    event: pl.String = pa.Field(isin=["fortran_exponent", "doubled_frequencies"])
    location: pl.String

    class Config:
        strict = True
        coerce = False


SCHEMAS: dict[str, type[pa.DataFrameModel]] = {
    "molecules": Molecules,
    "atoms": Atoms,
    "frequencies": Frequencies,
    "parse_log": ParseLog,
}
