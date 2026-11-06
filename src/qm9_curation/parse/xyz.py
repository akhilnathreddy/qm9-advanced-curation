"""Strict, lossless parser and exact serializer for QM9 ``.xyz`` records.

A record is parsed into tokens, the original text of every field, and the
parser then checks that serializing those tokens reproduces the input
byte-for-byte. Any line that does not match the expected layout raises
:class:`ParseError` with the record name, line number and reason, so the
parser cannot silently skip or misread content.

Layout (``docs/data_dictionary.md``; all separators are tabs):

====================  =========================================================
line 1                atom count
line 2                ``gdb <index>`` then 15 property tokens, trailing tab
lines 3 .. n+2        element, x, y, z, Mulliken charge; non-negative numbers
                      carry one leading space
line n+3              frequency tokens (one or two sets)
line n+4              GDB-17 SMILES, relaxed SMILES, trailing tab
line n+5              GDB-17 InChI, relaxed InChI
====================  =========================================================
"""

import re
from dataclasses import dataclass

HEADER_TAG = "gdb"
N_PROPERTIES = 15
ELEMENTS = ("H", "C", "N", "O", "F")
# Fortran-style exponent as written in the source, e.g. "2.1997*^-6".
FORTRAN_EXPONENT = "*^"
NUMBER_RE = re.compile(r"^-?\d+(?:\.\d*)?(?:(?:\*\^|[eE])[-+]?\d+)?$")


class ParseError(ValueError):
    """A record does not match the QM9 layout."""

    def __init__(self, name: str, line_no: int, reason: str) -> None:
        super().__init__(f"{name} line {line_no}: {reason}")
        self.name = name
        self.line_no = line_no
        self.reason = reason


@dataclass(frozen=True)
class Atom:
    element: str
    x: str
    y: str
    z: str
    charge: str


@dataclass(frozen=True)
class Record:
    """One QM9 molecule as token strings, exactly as written in the source."""

    index: int
    properties: tuple[str, ...]
    atoms: tuple[Atom, ...]
    frequencies: tuple[str, ...]
    smiles_gdb: str
    smiles_relaxed: str
    inchi_gdb: str
    inchi_relaxed: str

    @property
    def n_atoms(self) -> int:
        return len(self.atoms)

    @property
    def n_modes(self) -> int:
        """Vibrational modes per set: 3N-5 for linear molecules, else 3N-6."""
        n = 3 * self.n_atoms
        return n - 5 if len(self.frequencies) in (n - 5, 2 * (n - 5)) else n - 6

    @property
    def frequency_sets(self) -> tuple[tuple[str, ...], ...]:
        k = self.n_modes
        return tuple(self.frequencies[i : i + k] for i in range(0, len(self.frequencies), k))

    def number_tokens(self) -> list[tuple[str, str]]:
        """Every numeric token with a label saying where it sits in the record."""
        tokens = [(f"property[{i}]", t) for i, t in enumerate(self.properties)]
        for i, atom in enumerate(self.atoms):
            for field in ("x", "y", "z", "charge"):
                tokens.append((f"atom[{i}].{field}", getattr(atom, field)))
        tokens.extend((f"frequency[{i}]", t) for i, t in enumerate(self.frequencies))
        return tokens


def to_float(token: str) -> float:
    """Convert a source number token, including Fortran ``*^`` exponents."""
    return float(token.replace(FORTRAN_EXPONENT, "e"))


def _signed(token: str) -> str:
    return token if token.startswith("-") else f" {token}"


def serialize_record(record: Record) -> str:
    lines = [
        str(record.n_atoms),
        f"{HEADER_TAG} {record.index}\t" + "\t".join(record.properties) + "\t",
        *(
            a.element + "\t" + "\t".join(_signed(t) for t in (a.x, a.y, a.z, a.charge))
            for a in record.atoms
        ),
        "\t".join(record.frequencies),
        f"{record.smiles_gdb}\t{record.smiles_relaxed}\t",
        f"{record.inchi_gdb}\t{record.inchi_relaxed}",
    ]
    return "\n".join(lines) + "\n"


def _number(name: str, line_no: int, token: str, what: str) -> str:
    if not NUMBER_RE.match(token):
        raise ParseError(name, line_no, f"{what}: {token!r} is not a number")
    return token


def parse_record(text: str, name: str) -> Record:
    if not text.endswith("\n"):
        raise ParseError(name, 0, "record does not end with a newline")
    lines = text[:-1].split("\n")

    if not lines[0].isdigit():
        raise ParseError(name, 1, f"atom count {lines[0]!r} is not an integer")
    n_atoms = int(lines[0])
    if len(lines) != n_atoms + 5:
        raise ParseError(name, 0, f"{len(lines)} lines, expected {n_atoms + 5}")

    header = lines[1].split("\t")
    tag, _, index_text = header[0].partition(" ")
    if tag != HEADER_TAG or not index_text.isdigit():
        raise ParseError(name, 2, f"header starts with {header[0]!r}, expected 'gdb <index>'")
    properties = tuple(header[1:-1])
    if len(properties) != N_PROPERTIES or header[-1] != "":
        raise ParseError(name, 2, f"{len(header) - 1} fields after tag, expected 15 + trailing tab")
    for i, t in enumerate(properties):
        _number(name, 2, t, f"property {i + 1}")

    atoms = []
    for k in range(n_atoms):
        line_no = k + 3
        fields = lines[k + 2].split("\t")
        if len(fields) != 5 or fields[0] not in ELEMENTS:
            raise ParseError(name, line_no, "atom line must be element + 4 numbers")
        values = [_number(name, line_no, f.removeprefix(" "), "atom field") for f in fields[1:]]
        atoms.append(Atom(fields[0], *values))

    freq_line = n_atoms + 3
    frequencies = tuple(lines[n_atoms + 2].split("\t"))
    for t in frequencies:
        _number(name, freq_line, t, "frequency")

    smiles = lines[n_atoms + 3].split("\t")
    if len(smiles) != 3 or smiles[2] != "":
        raise ParseError(name, n_atoms + 4, "SMILES line must be 2 fields + trailing tab")
    inchi = lines[n_atoms + 4].split("\t")
    if len(inchi) != 2 or not all(s.startswith("InChI=") for s in inchi):
        raise ParseError(name, n_atoms + 5, "InChI line must be 2 InChI strings")

    record = Record(
        index=int(index_text),
        properties=properties,
        atoms=tuple(atoms),
        frequencies=frequencies,
        smiles_gdb=smiles[0],
        smiles_relaxed=smiles[1],
        inchi_gdb=inchi[0],
        inchi_relaxed=inchi[1],
    )
    if len(frequencies) not in {k * record.n_modes for k in (1, 2)}:
        raise ParseError(
            name,
            freq_line,
            f"{len(frequencies)} frequencies; expected {record.n_modes} or {2 * record.n_modes}",
        )

    # Strictness check: the tokens must reproduce the input exactly.
    rebuilt = serialize_record(record).split("\n")
    for line_no, (original, again) in enumerate(zip(lines, rebuilt, strict=False), start=1):
        if original != again:
            raise ParseError(name, line_no, f"layout differs from spec: {original!r}")
    return record
