"""Read the QM9-G4(MP2) tables of Kim et al. (2019) straight from the archive.

Only the members named in ``configs/external.yaml`` are read, in memory;
nothing is extracted to disk and none of the authors' scripts is run.

What the tables hold (paper, Table 1, checked against the atom logs):

* ``G4MP2``: G4(MP2) internal energy at 0 K (includes the scaled ZPE), Hartree.
  Gaussian 16 A.03 prints the G4MP2 summary block with labels shifted by two
  places; the archive entry ``G4MP2=`` that the authors parsed is the 0 K
  energy (for H: -0.5020936 = E(CCSD(T)) + HLC + corrections).
* ``B3LYP/6-31g(2df,p)``: B3LYP total electronic energy at the re-optimized
  geometry (no ZPE), so it compares with QM9's U0 - ZPVE.
* ``Zero-point correction``: unscaled B3LYP ZPE at the re-optimized geometry.
"""

import io
import re
import tarfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import polars as pl

ELEMENTS = ("H", "C", "N", "O", "F")
MULTIPLICITY = re.compile(rb"Charge\s*=\s*-?\d+\s+Multiplicity\s*=\s*(\d+)")
FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d+)")


@dataclass(frozen=True)
class Kim2019:
    molecules: pl.DataFrame  # one row per converged molecule
    atoms: pl.DataFrame  # one row per element
    authors_atomization: pl.DataFrame
    not_converged: list[int]
    multi_molecule: list[int]
    duplicate_pairs: list[tuple[int, int]]


def read_members(archive: Path, members: Mapping[str, str]) -> dict[str, bytes]:
    """Return the bytes of each configured member (``{element}`` expands to every element)."""
    wanted: dict[str, str] = {}
    for key, name in members.items():
        if "{element}" in name:
            wanted |= {f"{key}:{e}": name.format(element=e) for e in ELEMENTS}
        else:
            wanted[key] = name
    out = {}
    with tarfile.open(archive, "r:gz") as tar:
        for key, name in wanted.items():
            info = tar.getmember(name)
            if not info.isfile():
                raise ValueError(f"{archive.name}: {name} is not a regular file")
            fh = tar.extractfile(info)
            assert fh is not None
            out[key] = fh.read()
    return out


def _indices(raw: bytes) -> list[int]:
    # Some lists are written by numpy.savetxt ("2.12e+02"), others as integers.
    return sorted(int(float(tok)) for tok in raw.split())


def _pairs(raw: bytes) -> list[tuple[int, int]]:
    pairs = []
    for line in raw.decode("ascii").splitlines():
        if line.strip():
            a, b = (int(t) for t in line.split())
            pairs.append((a, b))
    return pairs


def multiplicity(log: bytes) -> int:
    match = MULTIPLICITY.search(log)
    if match is None:
        raise ValueError("No 'Multiplicity =' line in Gaussian log")
    return int(match.group(1))


def formula_counts(chem: str) -> dict[str, int]:
    """``C1H4`` -> {"C": 1, "H": 4}; every element of the formula must carry a count."""
    counts = {e: int(n) for e, n in FORMULA_TOKEN.findall(chem)}
    if "".join(f"{e}{n}" for e, n in counts.items()) != chem:
        raise ValueError(f"Unexpected formula format: {chem!r}")
    return counts


def parse(raw: Mapping[str, bytes]) -> Kim2019:
    mol = pl.read_csv(io.BytesIO(raw["molecules"]))
    molecules = mol.select(
        pl.col("index").cast(pl.UInt32).alias("qm9_id"),
        pl.col("Chem").alias("kim_formula"),
        pl.col("G4MP2").alias("g4mp2_u0_ha"),
        pl.col("B3LYP/6-31g(2df,p)").alias("g4mp2_b3lyp_ha"),
        pl.col("Zero-point correction").alias("g4mp2_zpve_ha"),
    ).sort("qm9_id")
    atom = pl.read_csv(io.BytesIO(raw["atoms"]))
    atoms = (
        atom.select(
            pl.col("Chem").alias("element"),
            pl.col("G4MP2").alias("g4mp2_u0_ha"),
            pl.col("B3LYP/6-31g(2df,p)").alias("b3lyp_ha"),
        )
        .with_columns(
            pl.col("element")
            .replace_strict({e: multiplicity(raw[f"atom_logs:{e}"]) for e in ELEMENTS})
            .alias("multiplicity")
        )
        .sort("element")
    )
    authors = pl.read_csv(io.BytesIO(raw["authors_atomization"])).select(
        pl.col("index").cast(pl.UInt32).alias("qm9_id"),
        pl.col("G4MP2").alias("authors_g4mp2_atomization_ha"),
    )
    return Kim2019(
        molecules=molecules,
        atoms=atoms,
        authors_atomization=authors.sort("qm9_id"),
        not_converged=_indices(raw["not_converged"]),
        multi_molecule=_indices(raw["multi_molecule"]),
        duplicate_pairs=_pairs(raw["duplicate_pairs"]),
    )


def load(archive: Path, members: Mapping[str, str]) -> Kim2019:
    return parse(read_members(archive, members))
