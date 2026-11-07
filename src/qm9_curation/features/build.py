"""Build all feature tables and the feature dictionary (Phase 6).

Run as ``python -m qm9_curation.features.build``. Reads only the allowed
inputs (``features.inputs``). Writes to ``data/processed/features/``:

=========================  ===================================================
``index``                  qm9_id, whether RDKit features exist, and why not
``composition``            atom counts per element
``coulomb_eigenvalues``    29 sorted Coulomb-matrix eigenvalues (a.u.)
``rdkit_descriptors``      RDKit descriptors (constant / non-finite dropped)
``ecfp4``                  ECFP4 count fingerprint, sparse (bits, counts)
``maccs``                  MACCS keys, sparse (on bits)
``graph_nodes``            per-atom features, ``atom_idx`` = source atom order
``graph_edges``            per-bond features incl. bond length (Å)
=========================  ===================================================

and ``reports/features/feature_report.json`` (dictionary, quality checks,
Gate 6 conditions).
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import polars as pl
from rdkit import __version__ as rdkit_version

from qm9_curation.config import INTERIM_DIR, PROCESSED_DIR, REPORTS_DIR
from qm9_curation.features.composition import ELEMENTS, composition
from qm9_curation.features.coulomb import MAX_ATOMS, coulomb_eigen_features
from qm9_curation.features.inputs import ALLOWED_ATOM_COLUMNS, restrict
from qm9_curation.features.molecular import (
    DESCRIPTOR_NAMES,
    ECFP_RADIUS,
    ECFP_SIZE,
    molecule_features,
)
from qm9_curation.parallel import default_workers, map_chunks

CORRELATION_REPORT_THRESHOLD = 0.99
UNAVAILABLE = "bond orders could not be assigned"


def _features_chunk(rows: list[tuple[Any, ...]]) -> list[dict[str, Any]]:
    return [molecule_features(*row) for row in rows]


def molecular_tables(atoms: pl.DataFrame, workers: int) -> dict[str, pl.DataFrame]:
    per_mol = (
        atoms.group_by("qm9_id", maintain_order=True)
        .agg(pl.col("element"), pl.concat_list("x_ang", "y_ang", "z_ang").alias("xyz"))
        .sort("qm9_id")
    )
    rows = [(q, e, [tuple(c) for c in xyz]) for q, e, xyz in per_mol.iter_rows()]
    results = map_chunks(_features_chunk, rows, workers)
    ok = [r for r in results if r["ok"]]
    return {
        "index": pl.DataFrame(
            {
                "qm9_id": [r["qm9_id"] for r in results],
                "rdkit_features_available": [r["ok"] for r in results],
                "unavailable_reason": [None if r["ok"] else UNAVAILABLE for r in results],
            },
            schema={
                "qm9_id": pl.UInt32,
                "rdkit_features_available": pl.Boolean,
                "unavailable_reason": pl.String,
            },
        ),
        "rdkit_descriptors": pl.DataFrame(
            [{"qm9_id": r["qm9_id"], **r["descriptors"]} for r in ok],
            schema={"qm9_id": pl.UInt32, **{n: pl.Float64 for n in DESCRIPTOR_NAMES}},
            orient="row",
        ),
        "ecfp4": pl.DataFrame(
            {
                "qm9_id": [r["qm9_id"] for r in ok],
                "bits": [r["ecfp4_bits"] for r in ok],
                "counts": [r["ecfp4_counts"] for r in ok],
            },
            schema={"qm9_id": pl.UInt32, "bits": pl.List(pl.UInt16), "counts": pl.List(pl.UInt16)},
        ),
        "maccs": pl.DataFrame(
            {"qm9_id": [r["qm9_id"] for r in ok], "bits": [r["maccs_bits"] for r in ok]},
            schema={"qm9_id": pl.UInt32, "bits": pl.List(pl.UInt8)},
        ),
        "graph_nodes": pl.DataFrame(
            [n for r in ok for n in r["nodes"]],
            schema={
                "qm9_id": pl.UInt32,
                "atom_idx": pl.UInt8,
                "atomic_number": pl.UInt8,
                "degree": pl.UInt8,
                "total_h": pl.UInt8,
                "formal_charge": pl.Int8,
                "hybridization": pl.UInt8,
                "is_aromatic": pl.Boolean,
                "in_ring": pl.Boolean,
            },
            orient="row",
        ),
        "graph_edges": pl.DataFrame(
            [e for r in ok for e in r["edges"]],
            schema={
                "qm9_id": pl.UInt32,
                "atom_i": pl.UInt8,
                "atom_j": pl.UInt8,
                "bond_order": pl.Float32,
                "is_aromatic": pl.Boolean,
                "is_conjugated": pl.Boolean,
                "in_ring": pl.Boolean,
                "length_ang": pl.Float64,
            },
            orient="row",
        ),
    }


def clean_descriptors(desc: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, list[str]]]:
    """Drop descriptors that are constant or have any non-finite value; report which."""
    names = [c for c in desc.columns if c != "qm9_id"]
    non_finite = [c for c in names if not desc[c].is_finite().all()]
    constant = [c for c in names if c not in non_finite and desc[c].n_unique() == 1]
    return desc.drop(non_finite + constant), {"non_finite": non_finite, "constant": constant}


def highly_correlated_pairs(desc: pl.DataFrame, threshold: float) -> list[dict[str, Any]]:
    names = [c for c in desc.columns if c != "qm9_id"]
    corr = desc.select(names).corr()
    pairs = []
    for i, a in enumerate(names):
        col = corr[a].to_list()
        for j in range(i + 1, len(names)):
            r = col[j]
            if r is not None and math.isfinite(r) and abs(r) >= threshold:
                pairs.append({"a": a, "b": names[j], "r": round(r, 4)})
    return pairs


def dictionary(tables: dict[str, pl.DataFrame]) -> list[dict[str, str]]:
    def entry(table: str, column: str, family: str, unit: str, description: str) -> dict[str, str]:
        dtype = str(tables[table].schema[column])
        return {
            "table": table,
            "column": column,
            "family": family,
            "dtype": dtype,
            "unit": unit,
            "description": description,
        }

    out = []
    for t in tables:
        out.append(entry(t, "qm9_id", "key", "-", "Molecule id (joins every table)"))
    out.append(
        entry(
            "index",
            "rdkit_features_available",
            "index",
            "-",
            "RDKit-based features exist for this molecule",
        )
    )
    out.append(entry("index", "unavailable_reason", "index", "-", "Why not, when they don't"))
    for e in ELEMENTS:
        out.append(entry("composition", f"n_{e}", "composition", "count", f"Number of {e} atoms"))
    out.append(entry("composition", "n_atoms", "composition", "count", "Number of atoms"))
    out.append(
        entry("composition", "n_heavy", "composition", "count", "Number of non-hydrogen atoms")
    )
    for k in range(MAX_ATOMS):
        out.append(
            entry(
                "coulomb_eigenvalues",
                f"cm_eig_{k:02d}",
                "coulomb",
                "Hartree (a.u.)",
                f"Coulomb-matrix eigenvalue {k}, descending, zero-padded",
            )
        )
    for c in tables["rdkit_descriptors"].columns:
        if c != "qm9_id":
            out.append(
                entry(
                    "rdkit_descriptors",
                    c,
                    "rdkit_descriptor",
                    "descriptor-specific",
                    f"RDKit {rdkit_version} descriptor {c}, heavy-atom molecule",
                )
            )
    out.append(
        entry(
            "ecfp4",
            "bits",
            "fingerprint",
            "index",
            f"Morgan radius {ECFP_RADIUS}, {ECFP_SIZE}-bit folded; non-zero positions",
        )
    )
    out.append(entry("ecfp4", "counts", "fingerprint", "count", "Counts at the positions in bits"))
    out.append(entry("maccs", "bits", "fingerprint", "index", "MACCS keys set (0-166)"))
    node_desc = {
        "atom_idx": "Atom position in the source record",
        "atomic_number": "Z",
        "degree": "Explicit neighbours (incl. H)",
        "total_h": "Attached hydrogens",
        "formal_charge": "Formal charge from bond-order perception",
        "hybridization": "0 s, 1 sp, 2 sp2, 3 sp3, 4 other",
        "is_aromatic": "Aromatic atom",
        "in_ring": "Ring atom",
    }
    for c, d in node_desc.items():
        out.append(entry("graph_nodes", c, "graph", "-", d))
    edge_desc = {
        "atom_i": "Lower atom index",
        "atom_j": "Higher atom index",
        "bond_order": "1, 1.5 (aromatic), 2, 3",
        "is_aromatic": "Aromatic bond",
        "is_conjugated": "Conjugated bond",
        "in_ring": "Ring bond",
        "length_ang": "Bond length from the B3LYP geometry",
    }
    for c, d in edge_desc.items():
        out.append(entry("graph_edges", c, "graph", "Å" if c == "length_ang" else "-", d))
    return out


def build(atoms_all: pl.DataFrame, workers: int) -> tuple[dict[str, pl.DataFrame], dict[str, Any]]:
    atoms = restrict(atoms_all)
    tables = {
        "composition": composition(atoms),
        "coulomb_eigenvalues": coulomb_eigen_features(atoms),
    }
    tables.update(molecular_tables(atoms, workers))
    tables["rdkit_descriptors"], dropped = clean_descriptors(tables["rdkit_descriptors"])
    tables = {
        k: v.sort([c for c in ("qm9_id", "atom_idx", "atom_i", "atom_j") if c in v.columns])
        for k, v in tables.items()
    }

    entries = dictionary(tables)
    documented = {(e["table"], e["column"]) for e in entries}
    all_columns = {(t, c) for t, f in tables.items() for c in f.columns}
    n_mol = atoms["qm9_id"].n_unique()
    n_ok = int(tables["index"]["rdkit_features_available"].sum())
    available = tables["index"].filter("rdkit_features_available")
    n_atoms_available = int(tables["composition"].join(available, on="qm9_id")["n_atoms"].sum())
    numeric_finite = all(
        frame[c].is_finite().all()
        for frame in tables.values()
        for c, dt in frame.schema.items()
        if dt.is_float()
    )
    report: dict[str, Any] = {
        "inputs": list(ALLOWED_ATOM_COLUMNS),
        "rdkit_version": rdkit_version,
        "row_counts": {k: v.height for k, v in tables.items()},
        "rdkit_features_available": n_ok,
        "descriptors_dropped": dropped,
        "descriptors_kept": tables["rdkit_descriptors"].width - 1,
        "descriptor_pairs_abs_r_at_least_0.99": highly_correlated_pairs(
            tables["rdkit_descriptors"], CORRELATION_REPORT_THRESHOLD
        ),
        "dictionary": entries,
        "gate_6_conditions": {
            "every_feature_column_documented": all_columns == documented,
            "only_allowed_inputs_used": list(atoms.columns) == list(ALLOWED_ATOM_COLUMNS),
            "per_molecule_tables_cover_every_molecule": all(
                tables[t].height == n_mol for t in ("index", "composition", "coulomb_eigenvalues")
            ),
            "rdkit_tables_match_available_molecules": all(
                tables[t].height == n_ok for t in ("rdkit_descriptors", "ecfp4", "maccs")
            ),
            "graph_nodes_match_atom_counts": tables["graph_nodes"].height == n_atoms_available,
            "no_non_finite_feature_values": numeric_finite,
        },
    }
    return tables, report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interim-dir", type=Path, default=INTERIM_DIR)
    parser.add_argument("--out-dir", type=Path, default=PROCESSED_DIR / "features")
    parser.add_argument(
        "--report", type=Path, default=REPORTS_DIR / "features" / "feature_report.json"
    )
    parser.add_argument("--workers", type=int, default=default_workers())
    args = parser.parse_args(argv)
    tables, report = build(pl.read_parquet(args.interim_dir / "atoms.parquet"), args.workers)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    digests = {}
    for name, frame in tables.items():
        path = args.out_dir / f"{name}.parquet"
        frame.write_parquet(path, compression="zstd", compression_level=9, statistics=True)
        digests[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    report["output_sha256"] = digests
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for condition, ok in report["gate_6_conditions"].items():
        print(f"{'PASS' if ok else 'FAIL'}  {condition}")
    print(
        "rows:",
        report["row_counts"],
        "| descriptors kept:",
        report["descriptors_kept"],
        "| dropped:",
        report["descriptors_dropped"],
    )


if __name__ == "__main__":
    main()
