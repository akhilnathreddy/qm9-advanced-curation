"""Scientific consistency: composition residuals and inter-target structure.

* **Composition residuals.** U0 atomization energy is fitted on atom counts
  (Gold only); the residual is what bonding contributes beyond a per-atom
  average. The largest residuals are annotated with ring strain, cage and
  polyyne descriptors from the geometry SMILES. Chemically real outliers are
  documented, not removed.
* **Inter-target structure.** U0, U, H and G atomization energies should be
  almost collinear; their differences (thermal energy, -TS) are scored with
  robust z within size strata, so a record whose thermochemistry is off
  relative to molecules of the same size is routed to review.
"""

from typing import Any

import numpy as np
import polars as pl
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from qm9_curation.review.stats import (
    ELEMENT_COUNTS,
    fit_per_element,
    robust_z_by_size,
    summary,
)

ATOMIZATION_TARGETS = (
    "u0_atomization_ev",
    "u298_atomization_ev",
    "h298_atomization_ev",
    "g298_atomization_ev",
)


def composition_residuals(
    curated: pl.DataFrame, composition: pl.DataFrame
) -> tuple[pl.DataFrame, dict[str, float]]:
    frame = curated.select("qm9_id", "quality_tier", "n_heavy", "u0_atomization_ev").join(
        composition.select("qm9_id", *ELEMENT_COUNTS), on="qm9_id"
    )
    coefs, pred = fit_per_element(
        frame.filter(pl.col("quality_tier") == "gold"), "u0_atomization_ev", intercept=True
    )
    frame = frame.with_columns(
        (pl.col("u0_atomization_ev") - pred).alias("composition_residual_ev")
    )
    frame = frame.with_columns(
        robust_z_by_size(frame, "composition_residual_ev").alias("composition_robust_z")
    )
    return frame.select("qm9_id", "composition_residual_ev", "composition_robust_z"), coefs


def constitution(smiles: str | None) -> str | None:
    """Canonical SMILES without stereo: same atoms and bonds, any configuration."""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    return Chem.MolToSmiles(mol, isomericSmiles=False) if mol is not None else None


def structure_notes(smiles: str | None) -> dict[str, Any]:
    """Descriptors that explain large bonding residuals; heuristics, not verdicts."""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return {
            "smallest_ring": None,
            "n_rings": None,
            "n_bridgeheads": None,
            "n_triple_bonds": None,
            "notes": "no geometry SMILES",
        }
    rings = [len(r) for r in mol.GetRingInfo().AtomRings()]
    bridge = int(rdMolDescriptors.CalcNumBridgeheadAtoms(mol))
    triple = sum(b.GetBondType() == Chem.BondType.TRIPLE for b in mol.GetBonds())
    notes = []
    if rings and min(rings) <= 4:
        notes.append("strained small ring")
    if bridge >= 2:
        notes.append("bridged polycycle")
    if triple >= 2:
        notes.append("conjugated triple bonds")
    if len(rings) >= 3:
        notes.append("three or more rings")
    return {
        "smallest_ring": min(rings) if rings else None,
        "n_rings": len(rings),
        "n_bridgeheads": bridge,
        "n_triple_bonds": triple,
        "notes": ", ".join(notes) or "no strain or conjugation descriptor",
    }


def annotate_top(
    scores: pl.DataFrame, curated: pl.DataFrame, value: str, z: str, top_n: int
) -> list[dict[str, Any]]:
    top = (
        scores.join(
            curated.select("qm9_id", "quality_tier", "formula", "smiles_geometry"), on="qm9_id"
        )
        .filter(pl.col(z).is_not_null())
        .sort(pl.col(z).abs(), "qm9_id", descending=[True, False])
        .head(top_n)
    )
    return [
        {
            **{
                k: row[k]
                for k in ("qm9_id", "quality_tier", "formula", "smiles_geometry", value, z)
            },
            **structure_notes(row["smiles_geometry"]),
        }
        for row in top.iter_rows(named=True)
    ]


def note_counts(rows: list[dict[str, Any]], baseline: pl.DataFrame) -> dict[str, Any]:
    """How often each descriptor appears among the top rows vs. in all Gold records."""
    labels = (
        "strained small ring",
        "bridged polycycle",
        "conjugated triple bonds",
        "three or more rings",
    )
    gold = [
        structure_notes(s)["notes"]
        for s in baseline.filter(pl.col("quality_tier") == "gold")["smiles_geometry"].to_list()
    ]
    return {
        label: {
            "top": sum(label in r["notes"] for r in rows) / max(len(rows), 1),
            "gold": sum(label in n for n in gold) / max(len(gold), 1),
        }
        for label in labels
    }


def thermal_scores(curated: pl.DataFrame) -> pl.DataFrame:
    frame = curated.select(
        "qm9_id",
        "n_heavy",
        (pl.col("u298_ev") - pl.col("u0_ev")).alias("thermal_energy_ev"),
        (pl.col("g298_ev") - pl.col("h298_ev")).alias("minus_ts_ev"),
    )
    return frame.select(
        "qm9_id",
        "thermal_energy_ev",
        "minus_ts_ev",
        robust_z_by_size(frame, "thermal_energy_ev").alias("thermal_energy_robust_z"),
        robust_z_by_size(frame, "minus_ts_ev").alias("minus_ts_robust_z"),
    )


def collinearity(curated: pl.DataFrame) -> dict[str, Any]:
    """Pearson r between the four atomization energies, Gold and all records."""
    out: dict[str, Any] = {}
    for name, frame in (
        ("gold", curated.filter(pl.col("quality_tier") == "gold")),
        ("all", curated),
    ):
        m = np.corrcoef(frame.select(ATOMIZATION_TARGETS).to_numpy().T)
        out[name] = {
            a: {b: float(m[i, j]) for j, b in enumerate(ATOMIZATION_TARGETS)}
            for i, a in enumerate(ATOMIZATION_TARGETS)
        }
        out[f"{name}_min_r"] = float(m.min())
    return out


def thermal_summary(thermal: pl.DataFrame, curated: pl.DataFrame) -> dict[str, Any]:
    gold = thermal.join(curated.select("qm9_id", "quality_tier"), on="qm9_id").filter(
        pl.col("quality_tier") == "gold"
    )
    return {
        "thermal_energy_ev_gold": summary(gold["thermal_energy_ev"].to_numpy()),
        "minus_ts_ev_gold": summary(gold["minus_ts_ev"].to_numpy()),
    }
