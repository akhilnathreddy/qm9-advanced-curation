# QM9 Advanced Dataset Curation — Plan v2

**Status:** Planning (v2 supersedes v1)
**Domain:** Computational chemistry × machine learning
**Primary dataset:** QM9 (Ramakrishnan et al., *Sci. Data* 2014)
**Date:** 2026-11-06

> v1 described *what* a careful curation project contains. v2 makes it
> **testable and publishable**: research questions with hypotheses,
> physics-based consistency checks that use QM9's own redundancy,
> graded quality tiers instead of keep/drop, a suite of splits, a
> curation-impact study, and quantitative exit gates for every phase.

---

## 0. What changed from v1

| Area | v1 | v2 upgrade |
|---|---|---|
| Framing | Curation project | Curation project **plus a research study**: "How much do curation and split choices change QM9 benchmark conclusions?" (§1) |
| Validation | RDKit parsing + generic checks | **Four-layer validation stack**: format → chemical identity → physics invariants → statistical/ML-assisted anomalies (§5) |
| Chemical identity | Parse SMILES | **Three-way graph agreement**: GDB-17 SMILES vs. relaxed-geometry SMILES vs. bonds re-derived independently from 3D coordinates |
| Numeric checks | Ranges and distributions | **Recompute properties from the record itself**: ZPVE from frequencies, rotational constants from geometry, H − U = k_BT, gap = LUMO − HOMO, Σ Mulliken charges ≈ 0 |
| Record outcome | Keep / quarantine | **Quality tiers** (Gold / Silver / Bronze / Quarantine) and a **per-record flag bitmask**; users pick the subset; nothing is deleted |
| Label quality | Not addressed | **Multi-fidelity cross-check** against published higher-level (G4MP2) energies for QM9 to estimate the label-noise floor |
| Splits | Random + scaffold | **Split suite**: random, isomer-group, scaffold with a fix for acyclic molecules, fingerprint clusters, size extrapolation, element holdout, property-tail extrapolation |
| Benchmarks | Simple baselines | **Baseline ladder** from composition-only to 3D GNN, **learning curves**, multiple seeds with CIs, and a **curation ablation** on a fixed Gold test set |
| Engineering | Tool list | Locked env (uv), typed schemas (Pandera), property-based tests (Hypothesis), DVC stage graph, **bit-for-bit rebuild test**, CI on a fixture subset |
| Release | Data card | Datasheet + **Croissant** metadata, dataset semantic versioning, multi-format export (Parquet, extxyz, PyG) |
| Governance | Rules list | Phase exit gates, a risk register, and a decision log in ADR format |
| Open decisions | Five open | **All five get a recommended answer** (§2) |

---

## 1. Research framing

The dataset and pipeline are the main deliverables. Framing the work
around explicit questions gives it a result, not just an artifact.

| ID | Research question | Hypothesis (falsifiable) | Evidence produced |
|---|---|---|---|
| RQ1 | How many QM9 records fail independent identity and physics consistency checks beyond the 3,054 marked "uncharacterized"? | A non-trivial number of additional records fail at least one check. | Flag counts per check, overlap (UpSet) plot |
| RQ2 | Do flagged records hurt models? | Flagged records have larger out-of-fold residuals than Gold records for the same model. | Residual distributions by tier, Mann–Whitney test |
| RQ3 | How much do split choices inflate reported performance? | MAE on random splits is clearly lower than on isomer-group, cluster and size-extrapolation splits. | MAE table across splits × models with CIs |
| RQ4 | What is the label-noise floor? | B3LYP vs. G4MP2 differences exceed the MAE of the best models on random splits, meaning those models fit B3LYP rather than physical reality. | Δ(B3LYP − G4MP2) distribution vs. model MAE |
| RQ5 | Does training on curated tiers change test error on a fixed Gold test set? | Gold-only training gives equal or better Gold-test MAE at matched N. | Curation ablation with learning curves |

**Pre-registration:** write RQs, metrics, splits and analysis plan into
`docs/analysis_plan.md` and commit it **before** Phase 9 runs, so the
benchmark can't be tuned after the fact.

---

## 2. Recommended answers to v1's open decisions

| Decision | Recommendation | Reason |
|---|---|---|
| Execution environment | **MacBook (local) for Phases 0–8 and classical ML; Colab or a cloud GPU only for 3D GNNs.** | The raw data is a few hundred MB; RDKit and descriptors run fine on Apple silicon. Apple MPS can run small GNNs, but a CUDA box is more reliable for PyG. |
| Source package | **The figshare collection linked from the 2014 *Sci. Data* paper** (`dsgdb9nsd.xyz.tar.bz2` plus `uncharacterized.txt` and the atom-reference file). Use processed redistributions (PyG, DeepChem, TorchDrug) **only as cross-checks**, never as the source. | Redistributions apply their own undocumented filtering and unit changes. v1 listed `qm9.org`; that is not confirmed as an official source, so don't use it as the source of record. |
| Env manager / versions | **uv** with `pyproject.toml` + `uv.lock`; Python 3.12; RDKit from PyPI (`rdkit`). Fall back to conda-forge only if a package forces it. | Fast, reproducible lockfile; one tool. |
| Validation framework | **Pandera** (schemas that live in code, work with Polars and pandas) plus custom check functions. Skip Great Expectations. | Lighter, testable with pytest, no extra config layer. |
| First benchmark targets | **(a) HOMO–LUMO gap**, which needs no reference energies, and **(b) U0 atomization energy**, which requires atom-reference subtraction and so also tests the curation pipeline. | One "easy" target and one that exposes unit and reference bugs. |
| Release license | Inherit and respect the source license. **Check the figshare record's license during Phase 1**, then decide (likely CC BY 4.0 or CC0, whichever the source allows). | Can't be decided before acquisition. |

---

## 3. Data model and lineage

### 3.1 Multi-table layout (Parquet)

QM9 is naturally relational. One flat CSV loses information or bloats it.

| Table | Grain | Key columns |
|---|---|---|
| `molecules` | 1 row / molecule | `qm9_id`, `n_atoms`, `n_heavy`, formula, 15 source properties (raw units), SMILES_gdb, SMILES_relaxed, InChI_gdb, InChI_relaxed, `source_file_sha256` |
| `atoms` | 1 row / atom | `qm9_id`, `atom_idx`, element, Z, x, y, z (Å), mulliken_charge |
| `frequencies` | 1 row / mode | `qm9_id`, `mode_idx`, freq_cm-1 |
| `bonds_derived` | 1 row / bond | `qm9_id`, i, j, bond_order, method (`smiles_relaxed` / `geom_rdkit`) |
| `targets` | 1 row / molecule | normalized units (eV, Debye, Å³, …), atomization energies, per-atom variants |
| `flags` | 1 row / molecule | one boolean per check, `flag_bitmask`, `quality_tier`, `tier_reason` |
| `features_*` | 1 row / molecule | one table per feature family, versioned separately |
| `splits` | 1 row / (molecule, split_name) | `split_name`, `fold`, `subset ∈ {train,val,test}`, `seed` |
| `external_xref` | 1 row / molecule | G4MP2 energies (if license allows), PubChem CID, InChIKey matches |

All tables join on `qm9_id`. Raw property values are never overwritten.
Normalized values live in `targets` under unit-suffixed column names
(`u0_atomization_ev`, not `u0`).

### 3.2 Lineage per record

Each `molecules` row carries: archive SHA-256, member filename,
pipeline git commit, pipeline config hash, and `dataset_version`. That
traces any curated value back to its exact source bytes.

### 3.3 Quality tiers

| Tier | Definition (draft; finalize in Phase 5) |
|---|---|
| **Gold** | Not in `uncharacterized.txt`; all three graph sources agree; all physics invariants pass within tolerance; no unresolved anomaly |
| **Silver** | Graphs agree and data is physically consistent, but there is a soft flag (e.g., stereo ambiguity, charge-separated SMILES, borderline invariant residual) |
| **Bronze** | Parses and is numerically complete, but there is identity disagreement or the record is in `uncharacterized.txt` |
| **Quarantine** | Parse failure, non-finite values, impossible geometry, or a hard invariant violation |

Every tier is shipped. The data card states which tier each benchmark used.

---

## 4. Roadmap with exit gates

Every phase ends with a **gate**: measurable conditions that must hold
before the next phase starts. Gate results go in
`reports/gates/phase_N.json`.

### Phase 0 — Setup and protocol (≈ 3 days)

- Git repo, `uv` env and lockfile, `ruff`, `mypy`, `pre-commit`.
- DVC initialized; local remote (later an S3/GDrive remote if needed).
- GitHub Actions CI: lint, type-check, unit tests, and a **mini-pipeline
  on a 1,000-molecule fixture** checked into `tests/fixtures/`.
- Decision log in ADR format (`docs/adr/0001-*.md`): context → decision
  → consequences.
- `docs/analysis_plan.md` skeleton (RQs from §1).

**Gate 0:** `uv sync && pytest && dvc repro` succeeds on a clean clone;
CI is green.

### Phase 1 — Acquisition (≈ 2 days)

- Script downloads from the figshare collection; records URL, retrieval
  timestamp, HTTP headers, size and SHA-256 in `data/raw/MANIFEST.json`.
- Raw archives are read-only (`chmod a-w`) and tracked by DVC.
- Record the license string from the figshare record.
- Inventory: number of `.xyz` members, `uncharacterized.txt` line count,
  atom-reference file contents.

**Gate 1:** member count = **133,885**; uncharacterized list = **3,054**
IDs; checksums are recorded and re-verify on a second download.

### Phase 2 — Lossless parser and schema (≈ 4 days)

Each QM9 `.xyz` record, per the source paper (confirm against the files):
line 1 is the atom count; line 2 holds the tag, index and 15 properties;
then one line per atom with element, x, y, z and Mulliken charge; then
the harmonic frequencies; then the two SMILES (GDB-17 and relaxed); then
the two InChIs.

- **Strict parser**: fail loudly on any deviation. Handle the known
  Fortran-style exponent token (`*^`) explicitly, and **log every
  record where it occurs**.
- **Round-trip test**: parse → serialize → byte-compare against the
  source for 100% of records (after documented whitespace
  normalization). This proves the parser is lossless.
- Data dictionary with source unit, normalized unit, physical meaning,
  and conversion factor for all 15 properties (rotational constants in
  GHz, dipole in Debye, α in Bohr³, energies in Hartree, ⟨R²⟩ in Bohr²,
  Cv in cal/mol·K, etc.).

**Gate 2:** 133,885/133,885 records parsed **or** each failure listed with
its reason; round-trip passes; Pandera schema validates all tables.

### Phase 3 — Profiling (≈ 3 days)

Everything from v1, plus:

- Composition profile: element counts, heavy-atom counts, stoichiometry
  frequency (how many constitutional isomers per formula).
- Ring statistics, including the **fraction of acyclic molecules**
  (this decides whether scaffold splitting is meaningful; §6).
- Duplicate analysis at four levels: identical InChI, identical
  InChIKey connectivity block (stereo-insensitive), identical canonical
  SMILES, and near-identical geometry (RMSD after alignment between
  same-formula molecules).
- Linear/near-linear molecules and their rotational constants
  (degenerate or extreme values are expected; document, don't delete).

**Gate 3:** machine-readable `reports/data_quality/profile.json` plus
an HTML report; issues list triaged into "expected chemistry" vs
"needs investigation".

### Phase 4 — Four-layer validation (≈ 1.5 weeks) — *core of v2*

See §5 for the full check catalogue. Each check is a pure function
`record → (passed: bool, residual: float | None, detail: str)` with its
own unit tests and a tolerance set in `configs/validation.yaml`.

**Gate 4:** every check has tests covering a passing case, a failing
case and an edge case; the flag table is complete for all records;
tolerances are justified in an ADR.

### Phase 5 — Standardization and tiering (≈ 4 days)

- Canonical SMILES and InChIKey from the **relaxed geometry**
  (stereo assigned from 3D), keeping the source strings.
- Normalized units in `targets`; atomization energies computed with the
  atom-reference energies shipped with the release (never with
  energies from another level of theory).
- Explicit policy on charge-separated (zwitterionic) SMILES: keep as
  written and flag; do not "neutralize" because the geometry is the
  ground truth.
- Tier assignment (§3.3) driven entirely by `configs/curation_rules.yaml`.
- **Waterfall report**: records per tier and per rule, both
  individually and cumulatively.

**Gate 5:** tier counts are reproducible, every non-Gold record has a
machine-readable reason, and a rule-toggle test shows each rule's
effect.

### Phase 6 — Feature engineering (≈ 1 week)

| Family | Examples | Notes |
|---|---|---|
| Composition | atom counts per element, formula vector | Basis of the "dressed atom" baseline |
| 2D descriptors | RDKit descriptors, topological indices | Drop constant features; record why |
| Fingerprints | ECFP4/6 (counts), MACCS, atom-pair | Fixed bit width; collisions noted |
| 3D global | Coulomb matrix (sorted eigenvalues), Bag-of-Bonds | Classic QML representations for QM9 |
| 3D local | SOAP / ACSF (via `dscribe`) | Optional; heavier |
| Graphs | PyG `Data` objects with atom/bond features and positions | For GNNs |

Requirements:

- **Invariance tests**: rotating, translating or permuting atoms leaves
  each feature unchanged (within tolerance) wherever that is the
  intended behaviour.
- **Leakage guard**: a test fails the build if any feature column is
  derived from a `targets` column or from frequencies/thermochemistry
  when predicting thermochemical targets. Frequencies and ZPVE are
  products of the same calculation, so for energy targets they count
  as "post-outcome" information.
- Deterministic generation, checked by hashing the output twice.

**Gate 6:** feature dictionary complete; invariance and leakage tests
pass; feature tables hash identically across two runs.

### Phase 7 — Statistical, scientific and multi-fidelity validation (≈ 1 week)

- Raw vs. curated distribution comparison (KS / Wasserstein per
  target, per tier).
- Inter-target correlation structure (e.g., U0, U, H, G should be
  nearly collinear; deviations point to record issues).
- **Composition-residual analysis**: fit a linear model of energy on
  atom counts, then inspect the largest residuals by hand.
  Chemically real outliers (strained rings, unusual bonding) are
  documented as such, not removed.
- **Multi-fidelity cross-check** (RQ4): join published G4MP2 energies
  for QM9 (Kim et al., 2019; see §11) on ID/InChI after confirming that
  their license allows it. Quantify Δ(B3LYP − G4MP2) and flag molecules
  with extreme Δ as candidates for multi-reference character or
  geometry problems.
- **ML-assisted anomaly detection**: out-of-fold residuals from a
  strong model (Phase 9 early run) give each record a "suspicion
  score". These records go to **review**, never to automatic removal.

**Gate 7:** statistical validation report; list of unresolved
uncertainties in `docs/limitations.md`.

### Phase 8 — Split suite (≈ 4 days)

See §6. Every split is stored as `(qm9_id, split_name, subset, seed)`,
and a leakage audit runs on each one.

**Gate 8:** each split passes its leakage assertions (e.g., no InChIKey
connectivity block appears in two subsets for isomer-safe splits);
distribution-shift report produced.

### Phase 9 — Benchmarking and curation ablation (≈ 1.5 weeks)

See §7. Run only after `docs/analysis_plan.md` is frozen.

**Gate 9:** all RQ1–RQ5 have a result (supported, refuted or
inconclusive), with CIs; every run is tracked in MLflow with git commit
and data version.

### Phase 10 — Release (≈ 4 days)

- Dataset semantic versioning: **MAJOR** when the record set or a
  tier definition changes, **MINOR** when columns or features are added,
  **PATCH** for docs or metadata fixes.
- Datasheet (Gebru et al. format) **and** Croissant JSON-LD metadata
  so the dataset is machine-discoverable.
- Exports: Parquet (canonical), extended XYZ (ASE-readable), and a
  PyG `InMemoryDataset` loader in the package.
- `make reproduce` rebuilds the full release from raw archives, and a
  CI job checks the output hashes against the published manifest.
- If the source license allows, deposit on Zenodo for a DOI.

**Gate 10:** a clean machine reproduces the release with identical
hashes; an outside reader can load the Gold tier and a named split in
≤ 5 lines of code.

### Phase 11 (optional) — Uncharacterized-record recovery

Senthil, Chakraborty & Ramakrishnan (*Chem. Sci.* 2021) studied the
unstable/uncharacterized QM9 molecules with connectivity-preserving
geometry optimization. This is most likely the follow-up v1 mentions;
**confirm the 2,830 / 3,054 figures against the paper**. If their
recovered geometries are released under a compatible license, add them
as a **separate, clearly labelled table** with their own provenance
(`source = "Senthil2021"`), never merged into the original record set.

---

## 5. Validation check catalogue

### Layer 1 — Format and completeness

| Check | Rule |
|---|---|
| Atom-count consistency | header `n_atoms` = number of atom lines |
| Element whitelist | ⊂ {H, C, N, O, F} |
| Heavy-atom limit | ≤ 9 heavy atoms |
| Finite numerics | no NaN/inf in coordinates, properties, frequencies |
| Frequency count | 3N − 6 (non-linear) or 3N − 5 (linear) |
| Exponent token | records that used `*^` are logged and parsed correctly |

### Layer 2 — Chemical identity (three-way graph agreement)

1. Graph A: from **GDB-17 SMILES** (the intended molecule).
2. Graph B: from **relaxed-geometry SMILES** (what the source says the
   optimized structure is).
3. Graph C: bonds **re-derived independently from 3D coordinates**
   using RDKit `rdDetermineBonds` (connectivity, then bond orders with
   total charge 0).

| Outcome | Interpretation | Default tier impact |
|---|---|---|
| A = B = C | Consistent | none |
| A ≠ B, B = C | Structure changed during optimization (should match uncharacterized list) | Bronze |
| A = B ≠ C | Bond perception disagrees; check for ambiguous bond orders or charge separation | Silver, review |
| All differ | Investigate | Bronze / review |

Compare graphs by InChI (stereo-insensitive and stereo-sensitive
variants separately) **and** by graph isomorphism to avoid
canonicalization artifacts. Report agreement against
`uncharacterized.txt` as a confusion matrix; that matrix is itself a
result for RQ1.

Geometry sanity: minimum interatomic distance above a covalent-radius
threshold (no clashes), every atom bonded (no fragments in a single
molecule), bond lengths within element-pair ranges.

### Layer 3 — Physics invariants (recompute from the record)

QM9 stores redundant quantities from one calculation, so many fields
can be recomputed and checked against each other.

| Invariant | Recompute | Expected |
|---|---|---|
| HOMO–LUMO gap | `lumo − homo` | equals `gap` to print precision |
| Enthalpy vs. internal energy | `H − U` | k_B·T at 298.15 K ≈ 9.44 × 10⁻⁴ Ha (ideal gas) |
| Free energy ordering | `G < H` | always (T·S > 0) |
| Thermal ordering | `U > U0` | always at 298.15 K |
| ZPVE from frequencies | ½ Σ ν̃ᵢ × 4.5563 × 10⁻⁶ Ha/cm⁻¹ | ≈ `zpve` (tolerance set after checking whether source frequencies are scaled) |
| Rotational constants | principal moments of inertia from coordinates and isotopic masses → B = h / (8π²I) in GHz | ≈ A, B, C (note ordering and linear-molecule degeneracy) |
| Charge neutrality | Σ Mulliken charges | ≈ 0 |
| Real minimum | all frequencies > 0 | an imaginary/negative mode means the structure is not a minimum: flag hard |
| Spatial extent | ⟨R²⟩ vs. geometric radius of gyration | monotone relation; large residuals flagged |

Each invariant stores the **residual**, not just pass/fail, so
tolerances can be studied and revised openly.

### Layer 4 — Statistical and ML-assisted

- Robust z-scores (median/MAD) **within formula or size strata**, never
  globally (a 9-heavy-atom molecule is not an outlier relative to a
  2-atom one).
- Composition-model residuals (Phase 7).
- Out-of-fold model residuals (suspicion score).
- All Layer 4 flags are **soft**: they route to review, never directly
  to Quarantine.

---

## 6. Split suite

| Split | Construction | What it tests | Caveat |
|---|---|---|---|
| `random` | Seeded shuffle, 80/10/10, 5 seeds | Interpolation; comparability with literature | Inflated by close analogues |
| `isomer_group` | Group by molecular formula; whole groups go to one subset | Leakage between constitutional isomers | Composition-driven targets become harder |
| `scaffold_generic` | Bemis–Murcko generic scaffolds; **acyclic molecules get their own buckets by chain-length/heteroatom pattern** | Ring-system novelty | Plain scaffold split puts all acyclic molecules in one empty-scaffold bucket and degenerates; measure that fraction in Phase 3 |
| `cluster_ecfp` | Butina or k-means clustering on ECFP4 Tanimoto; clusters assigned to subsets | Structural novelty | Depends on threshold; report it |
| `size_extrapolation` | Train ≤ 8 heavy atoms, test = 9 | Extrapolation to larger molecules | Test set is large; subsample if needed |
| `element_holdout` | Test = all F-containing molecules | Chemical-space extrapolation | Report as a stress test, not a headline |
| `property_tail` | Test = top 5% of target value | Extrapolation in target space | Target-specific |

**Leakage audit per split:** InChIKey-connectivity overlap across
subsets, nearest-neighbour Tanimoto distribution (test → train),
target distribution shift (Wasserstein), and composition shift.

**External-overlap audit:** if any model is pretrained on another
dataset (PubChemQC, QM7-X, etc.), compute InChIKey overlap with the test
set and report it.

---

## 7. Benchmark protocol

### 7.1 Baseline ladder (simple → complex)

| Level | Model | Input | Why it's there |
|---|---|---|---|
| 0 | Train-mean | — | Floor |
| 1 | Linear "dressed atom" | composition counts | Shows how much energy is explained by stoichiometry alone; essential for atomization energies |
| 2 | Ridge / LightGBM | RDKit descriptors, ECFP counts | 2D cheminformatics baseline |
| 3 | Kernel ridge regression | Coulomb matrix, Bag-of-Bonds | Classic QML baseline for QM9 |
| 4 | 2D GNN | Chemprop D-MPNN | Strong graph baseline without 3D |
| 5 | 3D GNN | SchNet or PaiNN (PyG) | 3D-aware; reference point for literature |

### 7.2 Protocol

- **Metrics:** MAE (primary), RMSE, R², and the fraction of predictions
  within chemical accuracy (1 kcal/mol ≈ 0.0434 eV) for energies.
- **Learning curves:** N ∈ {1k, 2k, 5k, 10k, 25k, 50k, 100k}, plotted on
  log–log axes. The slope is more informative than one MAE number.
- **Seeds and CIs:** ≥ 5 seeds; bootstrap 95% CIs on test MAE; paired
  comparisons on the same test molecules.
- **Hyperparameters** are tuned on validation only; the test set is
  touched once per (model, split, target).
- **Curation ablation (RQ5):** train on {all records, Gold+Silver, Gold}
  at matched N, always evaluate on the **same Gold test set**.
- **Sanity alarms:** results much better than published numbers for
  the same model and split trigger a leakage investigation before they
  are reported.
- **Tracking:** MLflow run tags include `git_commit`, `dataset_version`,
  `split_name`, `tier_filter`, `seed`.

---

## 8. Repository structure (v2)

```text
qm9-advanced-curation/
├── pyproject.toml / uv.lock
├── dvc.yaml                      # stage graph: acquire → parse → validate → tier → features → splits → benchmark
├── params.yaml                   # DVC-tracked parameters
├── configs/
│   ├── dataset.yaml
│   ├── validation.yaml           # tolerances per check
│   ├── curation_rules.yaml       # tier rules
│   └── splits.yaml
├── data/{raw,interim,processed,splits,external}/
├── src/qm9_curation/
│   ├── acquire.py
│   ├── parse/                    # strict lossless parser + round-trip
│   ├── schema.py                 # Pandera schemas
│   ├── checks/
│   │   ├── format.py             # Layer 1
│   │   ├── identity.py           # Layer 2 (three-way graph)
│   │   ├── physics.py            # Layer 3 invariants
│   │   └── statistical.py        # Layer 4
│   ├── tiering.py
│   ├── units.py                  # all conversion constants in one place, with sources
│   ├── features/
│   ├── splits/
│   ├── benchmark/
│   ├── export/                   # parquet, extxyz, PyG
│   └── reporting/
├── tests/
│   ├── fixtures/qm9_1k/          # small committed subset
│   ├── unit/                     # one file per check module
│   ├── property/                 # Hypothesis: invariance, round-trip
│   └── pipeline/                 # end-to-end on fixture + hash determinism
├── reports/{gates,data_quality,validation,statistical,splits,benchmark}/
├── docs/
│   ├── analysis_plan.md          # frozen before Phase 9
│   ├── adr/                      # decision records
│   ├── data_dictionary.md
│   ├── datasheet.md
│   ├── croissant.json
│   ├── curation_methodology.md
│   └── limitations.md
└── notebooks/                    # exploration only; never the source of a released artifact
```

Rule: **notebooks explore, `src/` produces.** Anything that ends up in
the release is generated by a DVC stage calling `src/` code.

---

## 9. Testing strategy

| Kind | Examples |
|---|---|
| Unit | each check on hand-built molecules (e.g., water, methane, HCN for the linear case, a planted bad record) |
| Golden records | ~20 real QM9 IDs with values verified by hand, committed as expected outputs |
| Property-based (Hypothesis) | random rotations/translations/permutations leave invariant features unchanged; parse ∘ serialize = id |
| Mutation-style | inject known corruptions (swap two coordinates, flip a sign on an energy, drop a frequency) and assert the right check fires |
| Pipeline | full DVC run on the 1k fixture in CI |
| Determinism | two clean runs produce byte-identical Parquet (fixed row order, fixed compression, no timestamps in files) |

---

## 10. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Bond perception from 3D is wrong for charge-separated or unusual bonding | Medium | Wrong identity flags | Use graph-agreement as a *flag*, not a delete; hand-review a sample of disagreements |
| Invariant tolerances set too tight or too loose | Medium | Spurious or missed flags | Store residuals; set tolerances from residual distributions; record in an ADR |
| Unit / atom-reference mistakes in atomization energies | Medium | Silent wrong targets | Golden-record tests; cross-check against one processed redistribution |
| External data (G4MP2) license blocks redistribution | Medium | RQ4 can't ship joined data | Ship join *code* and derived statistics only |
| Scope creep into GNN tuning | High | Delays release | Baseline ladder is fixed in the analysis plan; extra models go to a v2.x release |
| Benchmark tuned after seeing test results | Low–Medium | Invalid conclusions | Frozen analysis plan; one test evaluation per config |
| Mac-specific build differences (RDKit, LightGBM) | Low | Non-reproducible hashes | CI on Linux is the reference build |

---

## 11. Timeline (≈ 10–11 weeks part-time)

| Week | Milestone |
|---|---|
| 1 | Gates 0–1: repo, env, CI, verified raw data |
| 2 | Gate 2: lossless parser, schema, data dictionary |
| 3 | Gate 3: profiling report |
| 4–5 | Gate 4: four-layer validation |
| 5–6 | Gate 5: standardization, tiers, waterfall |
| 6–7 | Gate 6: features |
| 7–8 | Gates 7–8: statistical and multi-fidelity validation, split suite; **freeze analysis plan** |
| 9–10 | Gate 9: benchmarks and ablation |
| 11 | Gate 10: release v1.0.0 |

---

## 12. Success criteria (quantitative where possible)

- 100% of the 133,885 source records are accounted for: parsed and
  tiered, or quarantined with a reason.
- The parser round-trips 100% of records.
- Every validation check has unit tests and a documented tolerance.
- The release rebuilds from raw archives with identical hashes on a
  clean Linux machine.
- RQ1–RQ5 each have a reported outcome with uncertainty.
- The data card states for each tier: counts, intended use and known
  limitations.
- Success is still **not** maximum rows retained or minimum MAE; it is
  a trustworthy dataset whose every decision can be traced.

---

## 13. First working session

1. Create the repo with the §8 skeleton, `uv` env, pre-commit and CI.
2. Write ADR-0001 (environment: local Mac + CI on Linux) and ADR-0002
   (source of record: figshare collection).
3. Read the 2014 *Sci. Data* paper's file-format section and note the
   exact property order and units in `docs/data_dictionary.md`.
4. Download the archives, write `MANIFEST.json` with SHA-256, and make
   the raw files read-only.
5. Confirm 133,885 members and 3,054 uncharacterized IDs (Gate 1).
6. Parse 10 records by hand and turn them into the first golden-record
   tests.

**First milestone:** Gate 1 passed. No cleaning before Gate 2.

---

## 14. Weekly progress review (carried over from v1)

v1 notes that a recurring review task was created and the user then
said "no wait." Its status is still unknown. **Check whether it exists
and is active** before relying on it; if kept, have it report against
the phase gates in §4 (which gate is open, which conditions are unmet).

---

## 15. References

Checked against OpenAlex on 2026-11-06 unless marked otherwise.

- Ramakrishnan, Dral, Rupp, von Lilienfeld. *Quantum chemistry
  structures and properties of 134 kilo molecules.* Sci. Data 1, 140022
  (2014). https://doi.org/10.1038/sdata.2014.22 ✔
- Kim, Park, Choi. *Energy refinement and analysis of structures in the
  QM9 database via a highly accurate quantum chemical method.* Sci. Data
  6, 109 (2019). https://doi.org/10.1038/s41597-019-0121-7 ✔
  (G4MP2 energies for QM9; source for the multi-fidelity check)
- Senthil, Chakraborty, Ramakrishnan. *Troubleshooting unstable
  molecules in chemical space.* Chem. Sci. (2021).
  https://doi.org/10.1039/d0sc05591c ✔ (probable source of the
  2,830/3,054 recovery figure; confirm in the paper)
- Ward et al. *Machine learning prediction of accurate atomization
  energies of organic molecules from low-fidelity quantum chemical
  calculations.* MRS Commun. (2019).
  https://doi.org/10.1557/mrc.2019.107 ✔ (multi-fidelity / Δ-learning)
- Nandi, Vegge, Bhowmik. *MultiXC-QM9.* Sci. Data (2023).
  https://doi.org/10.1038/s41597-023-02690-2 ✔ (multi-functional
  energies; alternative cross-check)
- Wu et al. *MoleculeNet: a benchmark for molecular machine learning.*
  Chem. Sci. (2018; OpenAlex lists 2017).
  https://doi.org/10.1039/c7sc02664a ✔
- Gilmer et al. *Neural Message Passing for Quantum Chemistry.* ICML
  2017. https://doi.org/10.48550/arxiv.1704.01212 ✔
- Heid et al. *Chemprop.* J. Chem. Inf. Model. (2023).
  https://doi.org/10.1021/acs.jcim.3c01250 ✔
- Narayanan et al. G4MP2 energies for the QM9 set (Chem. Sci., 2019) —
  ✘ *not found by lookup; verify before citing.*
- Gebru et al. *Datasheets for Datasets* (CACM 2021) — standard; verify
  citation details.
- MLCommons Croissant metadata format — https://mlcommons.org/croissant/
  (verify URL).
- Tool docs: RDKit (`rdDetermineBonds`), Pandera, DVC, MLflow, PyG,
  dscribe.

---

## Working principle (v2)

**Understand → audit → validate against physics → tier, don't delete →
split honestly → benchmark against a frozen plan → release reproducibly.**
Each phase closes with a gate, and each decision becomes an ADR.
