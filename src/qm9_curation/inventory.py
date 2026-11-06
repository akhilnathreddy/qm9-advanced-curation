"""Inventory of the raw QM9 files (Phase 1).

Run as ``python -m qm9_curation.inventory``. Streams the archive without
extracting it and reports structure-level facts: member names and indices,
line layout, header tags, ``*^`` exponents, element and size distributions,
frequency counts, and the contents of ``uncharacterized.txt`` and
``atomref.txt``.

This is deliberately *not* the Phase 2 parser. It splits lines on whitespace
to count things; it does not interpret values beyond what a count needs.
Anything it finds that looks wrong is listed as an observation for Phase 2
and Phase 4 to handle properly.
"""

import argparse
import json
import re
import tarfile
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qm9_curation.config import RAW_DIR, REPORTS_DIR, load_config

ARCHIVE_NAME = "dsgdb9nsd.xyz.tar.bz2"
UNCHARACTERIZED_NAME = "uncharacterized.txt"
ATOMREF_NAME = "atomref.txt"
MEMBER_RE = re.compile(r"^dsgdb9nsd_(\d{6})\.xyz$")
FORTRAN_EXPONENT = "*^"
N_PROPERTY_FIELDS = 15
EXAMPLE_LIMIT = 25


@dataclass
class ArchiveStats:
    n_members: int = 0
    non_file_members: list[str] = field(default_factory=list)
    bad_names: list[str] = field(default_factory=list)
    indices: Counter[int] = field(default_factory=Counter)
    index_mismatch: list[str] = field(default_factory=list)
    bad_line_count: list[str] = field(default_factory=list)
    bad_property_count: list[str] = field(default_factory=list)
    header_tags: Counter[str] = field(default_factory=Counter)
    exponent_records: Counter[str] = field(default_factory=Counter)
    exponent_examples: list[str] = field(default_factory=list)
    elements: Counter[str] = field(default_factory=Counter)
    n_atoms: Counter[int] = field(default_factory=Counter)
    n_heavy: Counter[int] = field(default_factory=Counter)
    freq_class: Counter[str] = field(default_factory=Counter)
    freq_other: list[dict[str, int]] = field(default_factory=list)
    negative_freq: list[int] = field(default_factory=list)
    uncompressed_bytes: int = 0


def iter_members(archive: Path) -> Iterator[tuple[tarfile.TarInfo, bytes | None]]:
    with tarfile.open(archive, "r:bz2") as tf:
        for member in tf:
            fh = tf.extractfile(member) if member.isfile() else None
            yield member, (fh.read() if fh is not None else None)


def inspect_record(name: str, text: str, stats: ArchiveStats) -> None:
    lines = text.rstrip("\n").split("\n")
    n_atoms = int(lines[0].strip())
    stats.n_atoms[n_atoms] += 1
    if len(lines) != n_atoms + 5:
        stats.bad_line_count.append(name)
        return

    header = lines[1].split()
    stats.header_tags[header[0]] += 1
    match = MEMBER_RE.match(name)
    if match and int(header[1]) != int(match.group(1)):
        stats.index_mismatch.append(name)
    if len(header) - 2 != N_PROPERTY_FIELDS:
        stats.bad_property_count.append(name)

    sections = {
        "properties": lines[1],
        "atoms": "\n".join(lines[2 : 2 + n_atoms]),
        "frequencies": lines[2 + n_atoms],
    }
    for section, content in sections.items():
        if FORTRAN_EXPONENT in content:
            stats.exponent_records[section] += 1
            if len(stats.exponent_examples) < EXAMPLE_LIMIT:
                stats.exponent_examples.append(f"{name}:{section}")

    atoms = [line.split()[0] for line in lines[2 : 2 + n_atoms]]
    stats.elements.update(atoms)
    stats.n_heavy[sum(1 for a in atoms if a != "H")] += 1

    freqs = lines[2 + n_atoms].split()
    nonlinear, linear = 3 * n_atoms - 6, 3 * n_atoms - 5
    if len(freqs) == nonlinear:
        stats.freq_class["3N-6"] += 1
    elif len(freqs) == linear:
        stats.freq_class["3N-5"] += 1
    elif len(freqs) in (2 * nonlinear, 2 * linear):
        # Two frequency sets on one line. Record whether the halves are textually identical.
        half = len(freqs) // 2
        kind = "2x(3N-6)" if len(freqs) == 2 * nonlinear else "2x(3N-5)"
        same = freqs[:half] == freqs[half:]
        stats.freq_class[f"{kind} {'identical' if same else 'differing'} halves"] += 1
        stats.freq_other.append(
            {"index": int(header[1]), "n_atoms": n_atoms, "n_frequencies": len(freqs)}
        )
    else:
        stats.freq_class["other"] += 1
        stats.freq_other.append(
            {"index": int(header[1]), "n_atoms": n_atoms, "n_frequencies": len(freqs)}
        )
    if any(f.startswith("-") for f in freqs):
        stats.negative_freq.append(int(header[1]))


def scan_archive(archive: Path) -> ArchiveStats:
    stats = ArchiveStats()
    for member, data in iter_members(archive):
        stats.n_members += 1
        if data is None:
            stats.non_file_members.append(member.name)
            continue
        stats.uncompressed_bytes += len(data)
        match = MEMBER_RE.match(member.name)
        if match is None:
            stats.bad_names.append(member.name)
        else:
            stats.indices[int(match.group(1))] += 1
        inspect_record(member.name, data.decode("ascii"), stats)
    return stats


def parse_uncharacterized(path: Path) -> list[int]:
    """Indices listed in ``uncharacterized.txt``: every line whose first token is an integer."""
    indices = []
    for line in path.read_text(encoding="ascii").splitlines():
        tokens = line.split()
        if tokens and tokens[0].isdigit():
            indices.append(int(tokens[0]))
    return indices


def parse_atomref(path: Path) -> dict[str, dict[str, float]]:
    columns = ("zpve", "u0", "u298", "h298", "g298", "cv")
    table = {}
    for line in path.read_text(encoding="ascii").splitlines():
        tokens = line.split()
        if len(tokens) == len(columns) + 1 and tokens[0] in {"H", "C", "N", "O", "F"}:
            table[tokens[0]] = dict(zip(columns, map(float, tokens[1:]), strict=True))
    return table


def _sorted_counts(counter: Counter[Any]) -> dict[str, int]:
    return {str(k): counter[k] for k in sorted(counter)}


def build_report(raw_dir: Path) -> dict[str, Any]:
    expected = load_config("dataset")["expected"]
    stats = scan_archive(raw_dir / ARCHIVE_NAME)
    n_expected = expected["n_molecules"]
    full_range = set(range(1, n_expected + 1))
    present = set(stats.indices)
    duplicated = sorted(i for i, c in stats.indices.items() if c > 1)

    unchar = parse_uncharacterized(raw_dir / UNCHARACTERIZED_NAME)
    unchar_set = set(unchar)
    atomref = parse_atomref(raw_dir / ATOMREF_NAME)

    gate = {
        "n_members_equals_expected": stats.n_members == n_expected,
        "indices_are_1_to_n_exactly_once": present == full_range and not duplicated,
        "uncharacterized_count_equals_expected": len(unchar) == expected["n_uncharacterized"],
        "uncharacterized_unique": len(unchar_set) == len(unchar),
        "uncharacterized_subset_of_indices": unchar_set <= present,
        "elements_within_expected": set(stats.elements) <= set(expected["elements"]),
        "atomref_has_all_elements": set(atomref) == set(expected["elements"]),
    }
    return {
        "gate_1_conditions": gate,
        "archive": {
            "n_members": stats.n_members,
            "uncompressed_bytes": stats.uncompressed_bytes,
            "non_file_members": stats.non_file_members,
            "bad_member_names": stats.bad_names,
            "missing_indices": sorted(full_range - present)[:EXAMPLE_LIMIT],
            "unexpected_indices": sorted(present - full_range)[:EXAMPLE_LIMIT],
            "duplicated_indices": duplicated[:EXAMPLE_LIMIT],
            "header_index_mismatch": stats.index_mismatch[:EXAMPLE_LIMIT],
            "bad_line_count": stats.bad_line_count[:EXAMPLE_LIMIT],
            "bad_property_count": stats.bad_property_count[:EXAMPLE_LIMIT],
            "header_tags": dict(stats.header_tags),
        },
        "composition": {
            "element_atom_counts": _sorted_counts(stats.elements),
            "n_atoms_histogram": _sorted_counts(stats.n_atoms),
            "n_heavy_histogram": _sorted_counts(stats.n_heavy),
            "max_heavy_atoms": max(stats.n_heavy),
        },
        "observations_for_later_phases": {
            "fortran_exponent_records_by_section": dict(stats.exponent_records),
            "fortran_exponent_examples": stats.exponent_examples,
            "frequency_count_classes": dict(stats.freq_class),
            "frequency_count_other": stats.freq_other,
            "records_with_negative_frequency": stats.negative_freq,
        },
        "uncharacterized": {
            "n_entries": len(unchar),
            "n_unique": len(unchar_set),
            "min_index": min(unchar),
            "max_index": max(unchar),
        },
        "atomref_hartree": atomref,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out", type=Path, default=REPORTS_DIR / "data_quality" / "inventory.json")
    args = parser.parse_args(argv)
    report = build_report(args.raw_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for condition, ok in report["gate_1_conditions"].items():
        print(f"{'PASS' if ok else 'FAIL'}  {condition}")
    print(f"Inventory written to {args.out}")


if __name__ == "__main__":
    main()
