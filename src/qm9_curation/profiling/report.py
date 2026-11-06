"""Self-contained HTML data-quality report (Phase 3).

Figures are rendered with Matplotlib's Agg backend and embedded as base64
PNGs with no timestamp metadata, so the same profile yields the same file.
"""

import base64
import html
import io
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from qm9_curation.schema import PROPERTY_COLUMNS

INK, ACCENT, MUTED = "#1f2a37", "#2f6db5", "#9aa5b1"
TRIAGE_LABEL = {
    "expected_chemistry": "Expected chemistry",
    "known_source_issue": "Known source issue",
    "needs_investigation": "Needs investigation",
}


def _png(fig: Any) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _style(ax: Any) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK, labelsize=8)
    ax.grid(axis="y", color="#e5e9ef", linewidth=0.6)
    ax.set_axisbelow(True)


def _bar(counts: dict[str, int], title: str, xlabel: str) -> str:
    fig, ax = plt.subplots(figsize=(5, 2.6))
    keys = sorted(counts, key=lambda k: float(k))
    ax.bar(keys, [counts[k] for k in keys], color=ACCENT)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set_xlabel(xlabel, fontsize=8, color=INK)
    _style(ax)
    return _png(fig)


def _property_grid(molecules: pl.DataFrame) -> str:
    fig, axes = plt.subplots(3, 5, figsize=(13, 6.6))
    for ax, col in zip(axes.flat, PROPERTY_COLUMNS, strict=True):
        values = molecules[col].to_numpy()
        lo, hi = np.quantile(values, [0.001, 0.999])
        inside = values[(values >= lo) & (values <= hi)]
        ax.hist(inside, bins=60, color=ACCENT)
        ax.set_yscale("log")
        ax.set_title(col, fontsize=9, color=INK, loc="left")
        _style(ax)
    fig.suptitle(
        "Property distributions (central 99.8%, log count)",
        fontsize=10,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    return _png(fig)


def _hist(values: np.ndarray, title: str, xlabel: str, log: bool = False) -> str:
    fig, ax = plt.subplots(figsize=(5, 2.6))
    ax.hist(values, bins=80, color=ACCENT)
    if log:
        ax.set_yscale("log")
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set_xlabel(xlabel, fontsize=8, color=INK)
    _style(ax)
    return _png(fig)


def img(b64: str) -> str:
    return f'<img alt="" src="data:image/png;base64,{b64}">'


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in row) + "</tr>" for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _fmt(x: float) -> str:
    return f"{x:.6g}"


CSS = """
:root{--bg:#ffffff;--fg:#1f2a37;--muted:#5b6673;--line:#e3e8ee;--card:#f6f8fa;
--exp:#2e7d4f;--known:#8a6d1f;--inv:#b3412e}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#11161c;--fg:#e6ebf0;
--muted:#9aa5b1;--line:#2a323c;--card:#1a2129}}
:root[data-theme="dark"]{--bg:#11161c;--fg:#e6ebf0;--muted:#9aa5b1;--line:#2a323c;--card:#1a2129}
body{background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,-apple-system,sans-serif;
margin:0 auto;max-width:1100px;padding:24px 16px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:32px 0 8px;border-bottom:1px solid var(--line);
padding-bottom:4px}p.sub{color:var(--muted);margin:0 0 20px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.card b{display:block;font-size:20px}.card span{color:var(--muted);font-size:12px}
table{border-collapse:collapse;width:100%;font-size:12.5px;margin:8px 0;display:block;overflow-x:auto}
th,td{border-bottom:1px solid var(--line);padding:5px 8px;text-align:left;vertical-align:top}
th{color:var(--muted);font-weight:600}
.tag{border-radius:10px;padding:1px 8px;font-size:11.5px;color:#fff;white-space:nowrap}
.expected_chemistry{background:var(--exp)}.known_source_issue{background:var(--known)}
.needs_investigation{background:var(--inv)}
img{max-width:100%;background:#fff;border-radius:6px;border:1px solid var(--line)}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px}
code{font-size:12px}
"""


def render(profile: dict[str, Any], molecules: pl.DataFrame, per_molecule: pl.DataFrame) -> str:
    comp, chem, dup = profile["composition"], profile["chemistry"], profile["duplicates"]
    cards = [
        (f"{comp['n_molecules']:,}", "molecules"),
        (f"{comp['n_unique_formulas']:,}", "unique formulas"),
        (f"{chem['rings']['acyclic_fraction']:.1%}", "acyclic"),
        (
            f"{dup['structure']['n_molecules_in_such_groups']:,}",
            "in exact-structure duplicate groups",
        ),
        (
            f"{dup['tautomer']['n_molecules_in_such_groups']:,}",
            "share a standard InChI (tautomers)",
        ),
        (f"{chem['rdkit_parse_failures']['relaxed']:,}", "relaxed SMILES RDKit cannot parse"),
    ]
    card_html = "".join(
        f'<div class="card"><b>{v}</b><span>{html.escape(k)}</span></div>' for v, k in cards
    )
    issue_rows = "".join(
        f"<tr><td>{i['id']}</td><td>{html.escape(i['title'])}</td><td>{i['count']:,}</td>"
        f'<td><span class="tag {i["triage"]}">{TRIAGE_LABEL[i["triage"]]}</span></td>'
        f"<td>{html.escape(i['rationale'])}</td><td>{html.escape(i['handled_in'])}</td>"
        f"<td><code>{', '.join(map(str, i['examples'][:5]))}</code></td></tr>"
        for i in profile["issues"]
    )
    issues_table = (
        "<table><thead><tr><th>ID</th><th>Issue</th><th>Count</th><th>Triage</th><th>Why</th>"
        f"<th>Handled in</th><th>Examples (qm9_id)</th></tr></thead><tbody>{issue_rows}</tbody></table>"
    )
    clean = _table(
        ["Check", "Passed", "Value"],
        [[c["check"], "yes" if c["passed"] else "NO", c["value"]] for c in profile["clean_checks"]],
    )
    prop_rows = []
    for col, s in profile["properties"].items():
        q = s["quantiles"]
        prop_rows.append(
            [
                col,
                _fmt(q["0.0"]),
                _fmt(q["0.01"]),
                _fmt(q["0.5"]),
                _fmt(q["0.99"]),
                _fmt(q["1.0"]),
                s["zero"],
                s["negative"],
                s["n_robust_z_above_review"],
            ]
        )
    props = _table(
        ["Property", "min", "p1", "median", "p99", "max", "zeros", "negative", "|z|>6"], prop_rows
    )
    dup_rows = [
        [level, d["n_groups_with_2_or_more"], d["n_molecules_in_such_groups"], d["largest_group"]]
        for level, d in dup.items()
    ]
    dup_table = _table(["Level", "Groups (≥2)", "Molecules in groups", "Largest group"], dup_rows)
    smiles_cmp = _table(
        ["GDB-17 vs relaxed (flat SMILES)", "In uncharacterized list", "Molecules"],
        [
            [r["comparison"], r["uncharacterized"], r["n"]]
            for r in chem["gdb_vs_relaxed_flat_smiles"]
        ],
    )
    top_formulas = _table(
        ["Formula", "Isomers"], [[r["formula"], r["n"]] for r in comp["largest_formula_groups"]]
    )
    top_scaffolds = _table(
        ["Generic scaffold", "Molecules"],
        [[r["scaffold"] or "(acyclic)", r["n"]] for r in chem["scaffolds"]["top_generic"]],
    )
    isomer_sizes = molecules.group_by("formula").len()["len"].to_numpy()
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>QM9 Data Profile</title>
<style>{CSS}</style></head><body>
<h1>QM9 data profile — Phase 3</h1>
<p class="sub">Source: interim tables built from the verified raw archive. Descriptive only:
nothing has been removed or changed. Generated by <code>python -m qm9_curation.profiling.build</code>.</p>
<div class="cards">{card_html}</div>
<h2>Issues and triage</h2>{issues_table}
<h2>Checks that came back clean</h2>{clean}
<h2>Composition</h2><div class="grid2">
{img(_bar(comp["n_heavy_histogram"], "Molecules by heavy-atom count", "heavy atoms"))}
{img(_hist(isomer_sizes, "Constitutional isomers per formula", "molecules sharing a formula", log=True))}
</div>{top_formulas}
<h2>Rings and scaffolds</h2><div class="grid2">
{img(_bar(chem["rings"]["n_rings_histogram"], "Molecules by ring count", "rings (SSSR)"))}
{img(_bar(chem["rings"]["ring_size_counts"], "Rings by size", "ring size"))}
</div>{top_scaffolds}
<h2>Duplicates</h2>{dup_table}
<p>Exact-structure groups join identical canonical GDB-17 SMILES with pairs whose relaxed geometries
match (sorted Coulomb-matrix eigenvalues within {profile["thresholds"]["eigen_radius"]}, every sorted
interatomic distance within {profile["thresholds"]["distance_tol_ang"]} Å). Mirror images are
indistinguishable to this test. Largest U0 spread inside any group:
{dup["structure"]["max_u0_spread_kcal_per_mol"]:.3f} kcal/mol.</p>
<h2>SMILES consistency</h2>{smiles_cmp}
<h2>Properties</h2>{img(_property_grid(molecules))}{props}
<h2>Geometry</h2><div class="grid2">
{img(_hist(per_molecule["min_interatomic_ang"].to_numpy(), "Shortest interatomic distance per molecule", "Å"))}
{img(_hist(per_molecule["radius_of_gyration_ang"].to_numpy(), "Radius of gyration", "Å"))}
</div></body></html>
"""
