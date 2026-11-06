"""Parse the raw archive into typed Parquet tables (Phase 2).

Run as ``python -m qm9_curation.parse.build``. Writes ``molecules``,
``atoms``, ``frequencies`` and ``parse_log`` to ``data/interim/`` and a
report with the Gate 2 conditions to
``reports/data_quality/parse_report.json``.

"Lossless" is shown at two levels (ADR 0006):

1. **Text:** every record parses under the strict layout and serializes
   back to its exact source bytes (checked inside ``parse_record``).
2. **Value:** every numeric token's decimal value equals the decimal value
   of the float64 stored in the tables, so the tables hold every source
   value exactly at source precision. Only formatting (``0.`` vs ``0.0``,
   ``*^`` notation, sign padding) is not carried into the tables; the raw
   archive keeps it.
"""

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Iterator
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandera.errors
import polars as pl

from qm9_curation.config import INTERIM_DIR, RAW_DIR, REPORTS_DIR, load_config
from qm9_curation.inventory import ARCHIVE_NAME, MEMBER_RE, iter_members
from qm9_curation.parse.xyz import FORTRAN_EXPONENT, ParseError, Record, parse_record, to_float
from qm9_curation.schema import PROPERTY_COLUMNS, SCHEMAS

ATOMIC_NUMBER = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}
TABLE_SORT_KEYS = {
    "molecules": ["qm9_id"],
    "atoms": ["qm9_id", "atom_idx"],
    "frequencies": ["qm9_id", "set_idx", "mode_idx"],
    "parse_log": ["qm9_id", "event", "location"],
}


def hill_formula(elements: Iterable[str]) -> str:
    """Hill-order formula: C, then H, then the rest alphabetically."""
    counts = Counter(elements)
    order = ["C", "H"] if "C" in counts else []
    order += sorted(e for e in counts if e not in order)
    return "".join(e + (str(counts[e]) if counts[e] > 1 else "") for e in order)


def value_is_exact(token: str) -> bool:
    """True if float64(token) represents the same decimal value as the token."""
    value = to_float(token)
    shortest = repr(value)
    if shortest == token:
        return True
    return Decimal(token.replace(FORTRAN_EXPONENT, "E")) == Decimal(shortest)


class TableBuilder:
    def __init__(self, archive_sha256: str) -> None:
        self.archive_sha256 = archive_sha256
        self.rows: dict[str, list[dict[str, Any]]] = {name: [] for name in TABLE_SORT_KEYS}
        self.inexact: list[dict[str, Any]] = []
        self.n_tokens = 0

    def add(self, record: Record, member: str, raw: bytes) -> None:
        qid = record.index
        elements = [a.element for a in record.atoms]
        sets = record.frequency_sets
        exponent_locations = [loc for loc, tok in record.number_tokens() if FORTRAN_EXPONENT in tok]
        self.rows["molecules"].append(
            {
                "qm9_id": qid,
                "n_atoms": record.n_atoms,
                "n_heavy": sum(e != "H" for e in elements),
                "formula": hill_formula(elements),
                **{
                    col: to_float(tok)
                    for col, tok in zip(PROPERTY_COLUMNS, record.properties, strict=True)
                },
                "smiles_gdb": record.smiles_gdb,
                "smiles_relaxed": record.smiles_relaxed,
                "inchi_gdb": record.inchi_gdb,
                "inchi_relaxed": record.inchi_relaxed,
                "n_modes": record.n_modes,
                "n_frequency_sets": len(sets),
                "is_linear": record.n_modes == 3 * record.n_atoms - 5,
                "has_fortran_exponent": bool(exponent_locations),
                "source_member": member,
                "source_sha256": hashlib.sha256(raw).hexdigest(),
                "source_archive_sha256": self.archive_sha256,
            }
        )
        for i, atom in enumerate(record.atoms):
            self.rows["atoms"].append(
                {
                    "qm9_id": qid,
                    "atom_idx": i,
                    "element": atom.element,
                    "atomic_number": ATOMIC_NUMBER[atom.element],
                    "x_ang": to_float(atom.x),
                    "y_ang": to_float(atom.y),
                    "z_ang": to_float(atom.z),
                    "mulliken_charge_e": to_float(atom.charge),
                }
            )
        for s, values in enumerate(sets):
            for m, tok in enumerate(values):
                self.rows["frequencies"].append(
                    {"qm9_id": qid, "set_idx": s, "mode_idx": m, "wavenumber_cm1": to_float(tok)}
                )
        for loc in exponent_locations:
            self.rows["parse_log"].append(
                {"qm9_id": qid, "event": "fortran_exponent", "location": loc}
            )
        if len(sets) == 2:
            self.rows["parse_log"].append(
                {"qm9_id": qid, "event": "doubled_frequencies", "location": "frequencies"}
            )
        for loc, tok in record.number_tokens():
            self.n_tokens += 1
            if not value_is_exact(tok):
                self.inexact.append({"qm9_id": qid, "location": loc, "token": tok})

    def frames(self) -> dict[str, pl.DataFrame]:
        frames = {}
        for name, rows in self.rows.items():
            # Column order and Polars dtypes come from the Pandera schema (``.type`` unwraps
            # Pandera's engine dtype), so the tables are built to match it by construction.
            schema = {col: dt.type for col, dt in SCHEMAS[name].to_schema().dtypes.items()}
            frame = pl.DataFrame(rows, schema=schema, orient="row")
            frames[name] = frame.sort(TABLE_SORT_KEYS[name])
        return frames


def validate_frames(frames: dict[str, pl.DataFrame]) -> dict[str, str | None]:
    """Validate each table; return None per table on success, else the error summary."""
    results: dict[str, str | None] = {}
    for name, frame in frames.items():
        try:
            SCHEMAS[name].validate(frame, lazy=True)
            results[name] = None
        except pandera.errors.SchemaErrors as exc:
            results[name] = str(exc)[:2000]
    return results


def iter_archive(archive: Path) -> Iterator[tuple[str, bytes]]:
    for member, data in iter_members(archive):
        if data is not None and MEMBER_RE.match(member.name):
            yield member.name, data


def iter_directory(directory: Path) -> Iterator[tuple[str, bytes]]:
    """Source records stored as loose ``.xyz`` files (e.g. the 1k test fixture)."""
    for path in sorted(directory.glob("dsgdb9nsd_*.xyz")):
        yield path.name, path.read_bytes()


def build(
    members: Iterable[tuple[str, bytes]], archive_sha256: str, expected_n: int | None = None
) -> tuple[dict[str, pl.DataFrame], dict[str, Any]]:
    builder = TableBuilder(archive_sha256)
    failures: list[dict[str, Any]] = []
    n_members = 0
    for name, data in members:
        n_members += 1
        try:
            record = parse_record(data.decode("ascii"), name)
        except ParseError as exc:
            failures.append({"member": name, "line": exc.line_no, "reason": exc.reason})
            continue
        builder.add(record, name, data)
    frames = builder.frames()
    schema_errors = validate_frames(frames)

    mol = frames["molecules"]
    expected_freq_rows = int((mol["n_modes"].cast(pl.Int64) * mol["n_frequency_sets"]).sum())
    gate = {
        "all_records_parsed": not failures
        and mol.height == (n_members if expected_n is None else expected_n),
        "text_round_trip_exact_for_all": not failures,
        "values_exact_for_all_tokens": not builder.inexact,
        "schemas_valid": all(err is None for err in schema_errors.values()),
        "atom_rows_match_atom_counts": frames["atoms"].height == int(mol["n_atoms"].sum()),
        "frequency_rows_match_mode_counts": frames["frequencies"].height == expected_freq_rows,
    }
    log_counts = frames["parse_log"].group_by("event").len().sort("event")
    report = {
        "gate_2_conditions": gate,
        "n_members": n_members,
        "n_parsed": mol.height,
        "parse_failures": failures,
        "n_numeric_tokens_checked": builder.n_tokens,
        "inexact_values": builder.inexact[:50],
        "schema_errors": schema_errors,
        "row_counts": {name: frame.height for name, frame in frames.items()},
        "parse_log_events": dict(zip(log_counts["event"], log_counts["len"], strict=True)),
        "molecules_with_fortran_exponent": int(mol["has_fortran_exponent"].sum()),
        "linear_molecules": int(mol["is_linear"].sum()),
    }
    return frames, report


def write_tables(frames: dict[str, pl.DataFrame], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    digests = {}
    for name, frame in frames.items():
        path = out_dir / f"{name}.parquet"
        frame.write_parquet(path, compression="zstd", compression_level=9, statistics=True)
        digests[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digests


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=INTERIM_DIR)
    parser.add_argument(
        "--report", type=Path, default=REPORTS_DIR / "data_quality" / "parse_report.json"
    )
    args = parser.parse_args(argv)
    manifest = json.loads((args.raw_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    archive_sha = next(f["sha256"] for f in manifest["files"] if f["name"] == ARCHIVE_NAME)
    expected_n = load_config("dataset")["expected"]["n_molecules"]
    frames, report = build(iter_archive(args.raw_dir / ARCHIVE_NAME), archive_sha, expected_n)
    report["output_sha256"] = write_tables(frames, args.out_dir)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for condition, ok in report["gate_2_conditions"].items():
        print(f"{'PASS' if ok else 'FAIL'}  {condition}")
    print(f"Tables written to {args.out_dir}; report at {args.report}")


if __name__ == "__main__":
    main()
