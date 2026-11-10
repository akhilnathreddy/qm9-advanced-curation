"""Extended XYZ export (ASE-readable), one frame per record.

Each frame's comment line carries ``qm9_id``, ``quality_tier``, ``formula``
and the release properties in converted units (eV, Debye, Å, ...); each atom
line carries the element, Cartesian coordinates (Å) and the Mulliken charge
(e). Floats are written with Python's shortest round-trip ``repr``, so the
file is identical on every platform and reads back to the exact float64
values. Missing values are written as ``nan``. The file is not compressed:
gzip output depends on the zlib build, which would break the cross-platform
hash check (ADR 0014).
"""

from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import polars as pl

ATOM_PROPERTIES = "species:S:1:pos:R:3:mulliken_charge:R:1"


def _num(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "nan"
    return repr(float(value))


def frames(
    molecules: pl.DataFrame, atoms: pl.DataFrame, properties: Sequence[str]
) -> Iterator[str]:
    """Yield one extended-XYZ frame (with trailing newline) per molecule, by qm9_id."""
    mol = molecules.select("qm9_id", "quality_tier", "formula", *properties).sort("qm9_id")
    at = atoms.sort("qm9_id", "atom_idx")
    ids = at["qm9_id"].to_numpy()
    elements = at["element"].to_list()
    xyzq = at.select("x_ang", "y_ang", "z_ang", "mulliken_charge_e").to_numpy()
    starts = np.searchsorted(ids, mol["qm9_id"].to_numpy(), side="left")
    ends = np.searchsorted(ids, mol["qm9_id"].to_numpy(), side="right")
    for row, start, end in zip(mol.iter_rows(named=True), starts, ends, strict=True):
        if end <= start:
            raise ValueError(f"qm9_id {row['qm9_id']} has no atoms")
        info = " ".join(
            [
                f"Properties={ATOM_PROPERTIES}",
                f"qm9_id={row['qm9_id']}",
                f"quality_tier={row['quality_tier']}",
                f"formula={row['formula']}",
                *(f"{p}={_num(row[p])}" for p in properties),
                'pbc="F F F"',
            ]
        )
        lines = [str(end - start), info]
        for k in range(start, end):
            x, y, z, q = xyzq[k]
            lines.append(f"{elements[k]} {_num(x)} {_num(y)} {_num(z)} {_num(q)}")
        yield "\n".join(lines) + "\n"


def write_extxyz(
    path: Path, molecules: pl.DataFrame, atoms: pl.DataFrame, properties: Sequence[str]
) -> int:
    """Write every frame to ``path``; returns the number of frames."""
    n = 0
    with path.open("w", encoding="ascii", newline="\n") as fh:
        for frame in frames(molecules, atoms, properties):
            fh.write(frame)
            n += 1
    return n
