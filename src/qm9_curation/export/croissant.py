"""Croissant 1.0 JSON-LD metadata for the release (MLCommons).

Every release table becomes a ``cr:FileObject`` (with its SHA-256) and a
``cr:RecordSet`` with one ``cr:Field`` per column. Field descriptions come
from ``docs/data_dictionary.md``, so metadata and documentation cannot drift
apart: a column the dictionary does not describe is reported, and the
release gate fails on it. Generated without the ``mlcroissant`` package; the
gate validates the output with it.
"""

import fnmatch
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import polars as pl

# The JSON-LD context of Croissant 1.0, as produced by mlcroissant.
CONTEXT: dict[str, Any] = {
    "@language": "en",
    "@vocab": "https://schema.org/",
    "citeAs": "cr:citeAs",
    "column": "cr:column",
    "conformsTo": "dct:conformsTo",
    "cr": "http://mlcommons.org/croissant/",
    "rai": "http://mlcommons.org/croissant/RAI/",
    "data": {"@id": "cr:data", "@type": "@json"},
    "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
    "dct": "http://purl.org/dc/terms/",
    "equivalentProperty": "cr:equivalentProperty",
    "examples": {"@id": "cr:examples", "@type": "@json"},
    "extract": "cr:extract",
    "field": "cr:field",
    "fileProperty": "cr:fileProperty",
    "fileObject": "cr:fileObject",
    "fileSet": "cr:fileSet",
    "format": "cr:format",
    "includes": "cr:includes",
    "isLiveDataset": "cr:isLiveDataset",
    "jsonPath": "cr:jsonPath",
    "key": "cr:key",
    "md5": "cr:md5",
    "parentField": "cr:parentField",
    "path": "cr:path",
    "recordSet": "cr:recordSet",
    "references": "cr:references",
    "regex": "cr:regex",
    "repeated": "cr:repeated",
    "replace": "cr:replace",
    "samplingRate": "cr:samplingRate",
    "sc": "https://schema.org/",
    "separator": "cr:separator",
    "source": "cr:source",
    "subField": "cr:subField",
    "transform": "cr:transform",
}

# Release table -> heading of its section in docs/data_dictionary.md.
DICTIONARY_SECTIONS = {
    "molecules": "## Curated table",
    "atoms": "### `atoms`",
    "frequencies": "### `frequencies`",
    "review": "## Review table",
    "g4mp2_xref": "## External cross-reference",
    "splits": "### `splits.parquet`",
    "split_groups": "### `split_groups.parquet`",
}
SHARED = {"qm9_id": "QM9 record index (1-based), as in the source file name; the join key"}

KEYS = {
    "molecules": ["qm9_id"],
    "atoms": ["qm9_id", "atom_idx"],
    "frequencies": ["qm9_id", "set_idx", "mode_idx"],
    "g4mp2_xref": ["qm9_id"],
    "review": ["qm9_id"],
    "splits": ["qm9_id", "split_name", "seed"],
    "split_groups": ["qm9_id"],
}


def _expand(cell: str) -> list[str]:
    """Column patterns in one dictionary cell (``a, b``, ``u0/u298_x``, ``{a,b}``, ``x_*``)."""
    cell = cell.replace("`", "").replace("\\", "")
    braces = re.fullmatch(r"(.*)\{([^}]*)\}(.*)", cell)
    if braces:
        pre, alts, post = braces.groups()
        return [p for alt in alts.split(",") for p in _expand(pre + alt.strip() + post)]
    out = []
    for token in (t.strip() for t in cell.split(", ")):
        token = re.sub(r"<[^>]+>", "*", token)
        if "/" in token:
            parts = token.split("/")
            suffix = parts[-1][parts[-1].index("_") :] if "_" in parts[-1] else ""
            out += [p if p.endswith(suffix) else p + suffix for p in parts]
        elif token:
            out.append(token)
    return out


def dictionary(path: Path) -> dict[str, list[tuple[str, str]]]:
    """Section heading -> [(column pattern, description)] from the data dictionary."""
    sections: dict[str, list[tuple[str, str]]] = {}
    current = None
    header: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            current, header = line, []
            sections[current] = []
            continue
        if current is None or not line.startswith("|") or line.startswith("|---"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if not header:
            header = [c.lower() for c in cells]
            continue
        row = dict(zip(header, cells, strict=False))
        meaning = row.get("meaning", "")
        unit = row.get("unit", "")
        text = f"{meaning} ({unit})" if unit and unit not in ("—", "-") else meaning
        for pattern in _expand(cells[0]):
            sections[current].append((pattern, text))
    return sections


def describe(
    table: str, columns: list[str], sections: Mapping[str, list[tuple[str, str]]]
) -> dict[str, str | None]:
    """Column -> description (None when the dictionary does not cover it)."""
    heading = DICTIONARY_SECTIONS[table]
    entries = next((v for k, v in sections.items() if k.startswith(heading)), [])
    out: dict[str, str | None] = {}
    for col in columns:
        match = next((d for p, d in entries if fnmatch.fnmatchcase(col, p)), None)
        out[col] = match or SHARED.get(col)
    return out


def data_type(dtype: pl.DataType) -> str:
    if dtype.is_integer():
        return "sc:Integer"
    if dtype.is_float():
        return "sc:Float"
    if dtype == pl.Boolean:
        return "sc:Boolean"
    return "sc:Text"


def metadata(
    cfg: Mapping[str, Any],
    files: Mapping[str, tuple[str, str]],
    schemas: Mapping[str, Mapping[str, pl.DataType]],
    descriptions: Mapping[str, Mapping[str, str | None]],
    cite_as: str,
) -> dict[str, Any]:
    """Croissant document. ``files``: table -> (relative path, sha256)."""
    distribution = [
        {
            "@type": "cr:FileObject",
            "@id": f"{table}-file",
            "name": f"{table}-file",
            "contentUrl": rel,
            "encodingFormat": "application/x-parquet",
            "sha256": sha,
        }
        for table, (rel, sha) in files.items()
    ]
    record_sets = []
    for table, schema in schemas.items():
        fields = []
        for col, dtype in schema.items():
            field: dict[str, Any] = {
                "@type": "cr:Field",
                "@id": f"{table}/{col}",
                "name": col,
                "dataType": data_type(dtype),
                "source": {"fileObject": {"@id": f"{table}-file"}, "extract": {"column": col}},
            }
            if descriptions[table].get(col):
                field["description"] = descriptions[table][col]
            fields.append(field)
        record_sets.append(
            {
                "@type": "cr:RecordSet",
                "@id": table,
                "name": table,
                "key": [{"@id": f"{table}/{k}"} for k in KEYS[table]],
                "field": fields,
            }
        )
    return {
        "@context": CONTEXT,
        "@type": "sc:Dataset",
        "name": cfg["name"],
        "description": " ".join(str(cfg["description"]).split()),
        "conformsTo": "http://mlcommons.org/croissant/1.0",
        "citeAs": cite_as,
        "creator": {"@type": "sc:Person", "name": cfg["author"]},
        "datePublished": str(cfg["date_published"]),
        "keywords": list(cfg["keywords"]),
        "license": cfg["data_license_url"],
        "url": cfg["repository"],
        "version": cfg["version"],
        "isBasedOn": [s["url"] for s in cfg["sources"]],
        "distribution": distribution,
        "recordSet": record_sets,
    }
