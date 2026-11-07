"""The only data feature code is allowed to read (leakage guard).

Features describe the molecule a model is given: its atoms and 3D
coordinates. Everything else in the source is either a prediction target or
a by-product of the same quantum calculation (Mulliken charges, frequencies,
ZPVE, rotational constants) and would leak information about the targets.
Every feature function calls :func:`require_allowed` on its input, so passing
a frame that carries any other column fails loudly.
"""

import polars as pl

ALLOWED_ATOM_COLUMNS = ("qm9_id", "atom_idx", "element", "atomic_number", "x_ang", "y_ang", "z_ang")


class LeakageError(ValueError):
    """A feature function received a column outside the allowed inputs."""


def restrict(atoms: pl.DataFrame) -> pl.DataFrame:
    return atoms.select(ALLOWED_ATOM_COLUMNS).sort(["qm9_id", "atom_idx"])


def require_allowed(frame: pl.DataFrame) -> None:
    extra = sorted(set(frame.columns) - set(ALLOWED_ATOM_COLUMNS))
    if extra:
        raise LeakageError(f"Feature input has disallowed columns: {extra}")
