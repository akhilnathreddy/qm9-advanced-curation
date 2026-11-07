"""Rule-based issue list with explicit triage (Phase 3).

Each rule turns a profile finding into an issue with a triage class:

* ``expected_chemistry`` — real and correct; affects how data is used
  (e.g. splits), not whether it is right;
* ``known_source_issue`` — a defect the source documents or Phase 1 found;
* ``needs_investigation`` — unexplained or potentially wrong; owned by a
  named later phase.

The triage of every rule is written here, with its rationale, so the
classification is reviewable in code rather than buried in a notebook.
"""

from typing import Any

TRIAGE_CLASSES = ("expected_chemistry", "known_source_issue", "needs_investigation")


def _issue(
    issue_id: str,
    title: str,
    count: int,
    triage: str,
    rationale: str,
    handled_in: str,
    examples: list[int] | None = None,
) -> dict[str, Any]:
    assert triage in TRIAGE_CLASSES
    return {
        "id": issue_id,
        "title": title,
        "count": count,
        "triage": triage,
        "rationale": rationale,
        "handled_in": handled_in,
        "examples": (examples or [])[:10],
    }


def build_issues(profile: dict[str, Any], examples: dict[str, list[int]]) -> list[dict[str, Any]]:
    comp, chem, dup = profile["composition"], profile["chemistry"], profile["duplicates"]
    props, geo, rot = profile["properties"], profile["geometry"], profile["rotational"]
    spread = dup["structure"]["max_u0_spread_kcal_per_mol"]
    disagree_char = sum(
        r["n"]
        for r in chem["gdb_vs_relaxed_flat_smiles"]
        if r["comparison"] == "disagree" and not r["uncharacterized"]
    )
    issues = [
        _issue(
            "P01",
            "Exact structure duplicates (Kekulé forms, resonance forms, conformers)",
            dup["structure"]["n_molecules_in_such_groups"],
            "needs_investigation",
            "The same molecule appears under more than one qm9_id with near-identical geometry "
            f"and energy (max spread {spread:.3f} kcal/mol). "
            "Leakage risk if members land in different splits.",
            "Phase 5 (choose a representative per group) and Phase 8 (keep groups together)",
            examples.get("structure"),
        ),
        _issue(
            "P02",
            "Tautomers share one standard InChI / InChIKey",
            dup["tautomer"]["n_molecules_in_such_groups"],
            "expected_chemistry",
            "Standard InChI has a mobile-H layer; GDB-17 lists tautomers as distinct molecules "
            "with distinct energies. Not duplicates: never deduplicate by InChI.",
            "Phase 8 (tautomer-group-aware splits)",
            examples.get("tautomer"),
        ),
        _issue(
            "P03",
            "Relaxed SMILES not parseable by RDKit",
            chem["rdkit_parse_failures"]["relaxed"],
            "needs_investigation",
            "Authors' relaxed SMILES come from Open Babel bond perception; many encode "
            "zwitterions as neutral radicals (e.g. [NH3]...[O]). GDB-17 SMILES all parse.",
            "Phase 4 Layer 2 (independent bond perception from 3D)",
            examples.get("relaxed_unparseable"),
        ),
        _issue(
            "P04",
            "Relaxed SMILES with radical centres",
            chem["relaxed_smiles_with_radicals"],
            "needs_investigation",
            "QM9 molecules are closed-shell singlets; radicals in the relaxed SMILES most likely "
            "reflect failed bond-order perception, not the electronic structure.",
            "Phase 4 Layer 2",
            examples.get("radicals"),
        ),
        _issue(
            "P05",
            "Relaxed structure split into fragments",
            chem["relaxed_smiles_with_fragments"],
            "needs_investigation",
            "Relaxed SMILES contains '.', i.e. the optimized geometry is two or more molecules "
            "(e.g. NH3 leaving). Strong candidate for a hard flag.",
            "Phase 4 Layer 2 (fragment check from 3D bonds)",
            examples.get("fragments"),
        ),
        _issue(
            "P06",
            "GDB-17 vs relaxed SMILES disagree outside the uncharacterized list",
            disagree_char,
            "needs_investigation",
            "String-level disagreement can be real connectivity change or only aromaticity / "
            "charge / bond-order notation. Needs a graph-level comparison.",
            "Phase 4 Layer 2 (three-way graph agreement; RQ1)",
            examples.get("disagree"),
        ),
        _issue(
            "P07",
            "Charge-separated (zwitterionic) GDB-17 SMILES",
            chem["gdb_smiles_with_charges"],
            "expected_chemistry",
            "GDB-17 enumerates charge-separated forms; molecules are net neutral. Several "
            "relax to a neutral resonance form (see P01).",
            "Phase 5 (keep as written; flag)",
            examples.get("gdb_charged"),
        ),
        _issue(
            "P08",
            "Doubled frequency sets",
            profile["parse"]["doubled_frequencies"],
            "known_source_issue",
            "Found in Phase 1 (O1); both sets are kept in the frequencies table.",
            "Phase 4 (ZPVE invariant chooses the consistent set)",
            examples.get("doubled"),
        ),
        _issue(
            "P09",
            "Imaginary frequencies (not a true minimum)",
            geo["molecules_with_negative_frequency"],
            "known_source_issue",
            "All are on the authors' difficult-to-converge list (Phase 1, O2).",
            "Phase 4 Layer 3 (hard flag)",
            examples.get("negative_freq"),
        ),
        _issue(
            "P10",
            "Linear molecules store rotational constant A = 0",
            profile["parse"]["linear_molecules"],
            "expected_chemistry",
            "A is undefined (infinite) for linear molecules; 0 is a placeholder.",
            "Phase 5 (represent as missing with reason)",
            examples.get("linear"),
        ),
        _issue(
            "P14",
            "Linearity represented inconsistently",
            rot["a_zero_but_3n_minus_6_frequencies"] + rot["a_above_1e4_ghz"],
            "needs_investigation",
            "Physically linear molecules appear three ways: A = 0 with 3N-5 frequencies "
            f"({rot['a_zero_and_3n_minus_5_frequencies']}), A = 0 with 3N-6 "
            f"({rot['a_zero_but_3n_minus_6_frequencies']}), and finite but huge A "
            f"({rot['a_above_1e4_ghz']}, polyynes). is_linear (frequency count) "
            "misses the last two groups.",
            "Phase 4 Layer 3 (moments of inertia from geometry); Phase 5 (representation)",
            examples.get("linearity"),
        ),
        _issue(
            "P11",
            "Property values with |robust z| > 6 within heavy-atom strata",
            sum(p["n_robust_z_above_review"] for p in props.values()),
            "needs_investigation",
            "Statistically unusual is not wrong; these go to chemical review. Counts are summed "
            "over properties (a molecule can count more than once).",
            "Phase 7 (composition-residual analysis)",
            examples.get("robust_z"),
        ),
        _issue(
            "P12",
            "Size distribution dominated by 9-heavy-atom molecules",
            int(comp["n_heavy_histogram"].get("9", 0)),
            "expected_chemistry",
            "GDB-17 enumeration grows combinatorially with size.",
            "Phase 8 (size-extrapolation split); Phase 7 (stratified statistics)",
        ),
        _issue(
            "P13",
            "Acyclic molecules have an empty Murcko scaffold",
            chem["scaffolds"]["empty_scaffold_molecules"],
            "expected_chemistry",
            "A plain scaffold split would put all of them in one bucket.",
            "Phase 8 (generic scaffold split with acyclic buckets)",
        ),
    ]
    return issues


def clean_checks(profile: dict[str, Any]) -> list[dict[str, Any]]:
    """Checks that came back clean, recorded so their absence of issues is visible."""
    geo, props = profile["geometry"], profile["properties"]
    return [
        {
            "check": "no interatomic distance below 0.9 Å (no atom clashes)",
            "passed": geo["min_interatomic_ang"]["min"] >= 0.9,
            "value": geo["min_interatomic_ang"]["min"],
        },
        {
            "check": "no missing or NaN property values",
            "passed": all(p["null"] == 0 and p["nan"] == 0 for p in props.values()),
            "value": 0,
        },
        {
            "check": "GDB-17 SMILES all parse with RDKit",
            "passed": profile["chemistry"]["rdkit_parse_failures"]["gdb"] == 0,
            "value": profile["chemistry"]["rdkit_parse_failures"]["gdb"],
        },
    ]
