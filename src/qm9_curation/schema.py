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


TIERS = ["gold", "silver", "bronze", "quarantine"]
IDENTITY_CLASSES = ["consistent", "proton_moved", "rearranged", "fragmented", "unperceived"]


class CuratedMolecules(pa.DataFrameModel):
    """``data/processed/molecules.parquet`` (Phase 5). One row per source record."""

    qm9_id: pl.UInt32 = pa.Field(unique=True, ge=1)
    quality_tier: pl.String = pa.Field(isin=TIERS)
    tier_reasons: pl.String
    formula: pl.String
    n_atoms: pl.UInt8
    n_heavy: pl.UInt8 = pa.Field(ge=1, le=9)
    smiles_gdb: pl.String
    canonical_smiles_gdb: pl.String
    smiles_geometry: pl.String = pa.Field(nullable=True)
    inchikey_geometry: pl.String = pa.Field(
        nullable=True, str_matches=r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$"
    )
    geometry_identifier_error: pl.String = pa.Field(nullable=True)
    identity_class: pl.String = pa.Field(isin=IDENTITY_CLASSES)
    structure_group_id: pl.UInt32
    structure_representative_id: pl.UInt32
    is_structure_representative: pl.Boolean
    tautomer_group_id: pl.UInt32
    connectivity_group_id: pl.UInt32
    is_linear: pl.Boolean
    gdb_charge_separated: pl.Boolean
    uncharacterized_listed: pl.Boolean
    mu_debye: pl.Float64 = pa.Field(ge=0)
    alpha_ang3: pl.Float64 = pa.Field(gt=0)
    homo_ev: pl.Float64 = _finite()
    lumo_ev: pl.Float64 = _finite()
    gap_ev: pl.Float64 = pa.Field(gt=0)
    r2_ang2: pl.Float64 = pa.Field(gt=0)
    zpve_ev: pl.Float64 = pa.Field(gt=0)
    u0_ev: pl.Float64 = _finite()
    u298_ev: pl.Float64 = _finite()
    h298_ev: pl.Float64 = _finite()
    g298_ev: pl.Float64 = _finite()
    cv_cal_per_mol_k: pl.Float64 = pa.Field(gt=0)
    u0_atomization_ev: pl.Float64 = pa.Field(lt=0)
    u298_atomization_ev: pl.Float64 = pa.Field(lt=0)
    h298_atomization_ev: pl.Float64 = pa.Field(lt=0)
    g298_atomization_ev: pl.Float64 = pa.Field(lt=0)
    rot_a_ghz: pl.Float64 = pa.Field(nullable=True, gt=0)
    rot_a_missing_reason: pl.String = pa.Field(nullable=True)
    rot_b_ghz: pl.Float64 = pa.Field(gt=0)
    rot_c_ghz: pl.Float64 = pa.Field(gt=0)
    source_rot_a_ghz: pl.Float64
    source_rot_b_ghz: pl.Float64
    source_rot_c_ghz: pl.Float64
    source_mu_debye: pl.Float64
    source_alpha_bohr3: pl.Float64
    source_homo_ha: pl.Float64
    source_lumo_ha: pl.Float64
    source_gap_ha: pl.Float64
    source_r2_bohr2: pl.Float64
    source_zpve_ha: pl.Float64
    source_u0_ha: pl.Float64
    source_u298_ha: pl.Float64
    source_h298_ha: pl.Float64
    source_g298_ha: pl.Float64
    source_cv_cal_per_mol_k: pl.Float64
    zpve_consistent_set: pl.UInt8 = pa.Field(isin=[0, 1])
    flag_bitmask: pl.UInt32
    n_failed_hard: pl.UInt8
    n_failed_identity: pl.UInt8
    n_failed_soft: pl.UInt8
    source_member: pl.String
    source_sha256: pl.String = pa.Field(str_matches=r"^[0-9a-f]{64}$")
    source_archive_sha256: pl.String = pa.Field(str_matches=r"^[0-9a-f]{64}$")

    class Config:
        strict = True
        coerce = False


SCHEMAS: dict[str, type[pa.DataFrameModel]] = {
    "molecules": Molecules,
    "atoms": Atoms,
    "frequencies": Frequencies,
    "parse_log": ParseLog,
}
