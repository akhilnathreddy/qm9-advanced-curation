# QM9 Advanced Curation — project report

**Release:** QM9 Curated v1.0.0 (2026-11-10). **Code:** MIT. **Data:** CC0-1.0.
**Repository:** https://github.com/akhilnathreddy/qm9-advanced-curation
**Author:** akhilnathreddy

This report consolidates the whole project into one document:
- the question and design;
- every phase, with its gate;
- every decision, with the alternatives considered;
- the results of the five research questions;
- the limitations that remain;
- exact reproduction steps.

The detail lives in the documents it cites. Decisions are in
`docs/adr/0001`–`0014`, results in `docs/<topic>.md`, and open issues in
`docs/limitations.md`.

---

## 1. Summary

**What this is.** QM9 has 133,885 small organic molecules with
B3LYP/6-31G(2df,p) properties (Ramakrishnan et al., *Sci. Data* 1, 140022,
2014). It is the most-used benchmark for molecular property prediction. It is
usually consumed through redistributions that drop records silently, convert
units without saying so, and split at random. This project rebuilds QM9 from
the original archives with these guarantees:
- every value is parsed losslessly;
- every record is checked in four layers: format, chemical identity, physics
  invariants and statistics;
- every record gets a quality tier and a machine-readable reason, instead of
  being deleted;
- duplicates and near-duplicates are grouped, so no split can leak them;
- the labels are compared with a higher level of theory, G4(MP2);
- the benchmark asks how much curation and split choice change the
  conclusions people draw from QM9.

**The release** keeps all 133,885 records:

| Tier | Records | Meaning |
|---|---|---|
| Gold | 130,659 | Passes every check |
| Silver | 1,630 | Usable, with a documented caveat |
| Bronze | 1,591 | The geometry is not the labelled molecule |
| Quarantine | 5 | Not a true minimum |

It also ships geometries, frequencies, a G4(MP2) cross-reference, review
scores and eight leakage-audited splits × 5 seeds. Formats:
- Parquet;
- extended XYZ;
- Croissant 1.0 metadata;
- a PyTorch Geometric loader.

It rebuilds from the raw archives on a clean machine (`make reproduce`).
12 of its 14 files are byte-identical. The two tables holding least-squares
results match within a relative 10⁻⁹, because their last bits depend on the
BLAS library and the CPU.

**Answers to the research questions** (frozen analysis plan, §2):

| RQ | Question | Verdict | Key number |
|---|---|---|---|
| RQ1 | Do records fail checks beyond the 3,054 "uncharacterized"? | supported | 8 unlisted records fail a hard or identity check; 1,466 listed records are in fact consistent |
| RQ2 | Do flagged records hurt models? | supported | Bronze out-of-fold error ≈ 2× Gold; Silver slightly higher (4/4 tests, Holm p ≤ 10⁻³) |
| RQ3 | Does split choice inflate performance? | mixed* | Learned models: 23 of 24 split cells inflated; scaffold and size splits raise U0 error 3–4× |
| RQ4 | Is model error below the label-noise floor? | supported | Best U0 MAE 0.050 eV vs. 0.123 eV B3LYP–G4(MP2) residual |
| RQ5 | Does Gold-only training help on a Gold test set? | supported | Yes, but by ≈ 1 meV (< 1% of the MAE) at N = 100,000 |

\* `mixed` only because the frozen rule counts the mean and dressed-atom
baselines, which have nothing to overfit. For every learned model the answer
is a clear yes (§5.3).

**Take-home messages:**
1. Most of QM9 is sound. The problems that do exist are structural (wrong
   molecule, duplicates), not numerical: every physics invariant holds to
   print precision.
2. The conventional fix, dropping the 3,054 "uncharacterized" records,
   discards 1,466 good molecules and misses 8 bad ones.
3. Random-split MAEs on QM9 are optimistic by 3–4× for new scaffolds or
   larger molecules.
4. The best models already fit B3LYP more closely than B3LYP agrees with
   G4(MP2). Further QM9 gains measure fidelity to DFT, not to physics.

---

## 2. Research design

Plan v2 (`docs/plan_v2.md`) framed the curation as a study: *how much do
curation and split choices change QM9 benchmark conclusions?* It asked five
falsifiable questions, RQ1–RQ5.

**Pre-registration.**
- `docs/analysis_plan.md` fixed, before any Phase 9 model was trained
  (commit 4b08cc5, 2026-11-07):
  - the questions, metrics and populations;
  - the splits;
  - the decision rules (α = 0.05, Holm correction within each RQ, bootstrap
    95% CIs);
  - the models.
- Gate 9 checks that the text above `## Amendments` is unchanged
  (SHA-256 recorded).
- Every later change is a dated amendment with its reason. That includes a
  disclosure of test numbers seen in exploratory runs before the grid.

**Populations.**
- Evaluation is always on `test ∩ Gold`.
- Validation (`val ∩ Gold`) is used only for early stopping and
  hyperparameters.
- Training uses Gold, Silver and Bronze by default. RQ5 compares tier sets
  at matched N.

**Governance.**
- Each phase ends with an exit gate, `make gateN`. Each gate runs the full
  checks, forces a rebuild of the phase's stage, checks the rebuild is
  byte-identical, and records the result with the environment in
  `reports/gates/phase_N.json`.
- Decisions are ADRs, with context, decision, alternatives and consequences.

---

## 3. Pipeline

One DVC stage graph (`dvc.yaml`), run by `make repro`:

| Stage | Phase | Output | What it does |
|---|---|---|---|
| `acquire` | 1 | `data/raw/` | Download from figshare; verify the published MD5 and size; record SHA-256 |
| `inventory` | 1 | `reports/data_quality/inventory.json` | Count and characterize the source files |
| `parse` | 2 | `data/interim/{molecules,atoms,frequencies,parse_log}` | Lossless parse, proven by round-trip |
| `profile` | 3 | `molecule_profile`, `geometric_duplicate_pairs`, `profile.html` | Profile and triage issues |
| `validate` | 4 | `data/interim/validation.parquet` | 19 checks in 4 layers, flag bitmask |
| `curate` | 5 | `data/processed/molecules.parquet` | Standardize, choose duplicate representatives, assign tiers |
| `features` | 6 | `data/processed/features/` | Composition, Coulomb, RDKit, ECFP4, MACCS, graph tables |
| `acquire_external`, `review` | 7 | `g4mp2_xref.parquet`, `review.parquet` | G4(MP2) cross-check and soft statistical review |
| `splits` | 8 | `data/splits/` | 8 leakage-audited splits × 5 seeds |
| `benchmark`, `benchmark_analysis` | 9 | `data/benchmark/`, `reports/benchmark/` | 682 tracked runs; decision rules |
| `release` | 10 | `data/release/qm9-curated-v1.0.0/` | Release files, metadata, manifest |

**Engineering.**
- Python 3.12, with locked dependencies (`uv.lock`).
- Ruff, `mypy --strict`, pytest with Hypothesis (202 tests), and Pandera
  schemas.
- pre-commit hooks.
- GitHub Actions CI on Linux, warning-free (`pytest -W error`).
- A 1,000-molecule fixture lets CI run the pipeline without the raw data.

---

## 4. Phases

### Phase 0 — Setup (Gate 0)
- **Built:** the repository, the locked environment, linting, typing,
  pre-commit hooks, DVC, CI on a fixture, ADRs and the analysis-plan
  skeleton.
- **Decisions:**
  - develop on macOS arm64, with Linux CI as the reference build (ADR 0001);
  - the figshare collection of the 2014 paper is the source of record, with
    redistributions as cross-checks only (ADR 0002);
  - uv, DVC, ruff, mypy and pytest (ADR 0003);
  - Pandera schemas plus plain-Python checks (ADR 0004).

### Phase 1 — Acquisition (Gate 1)
- **Files:** the four source files from figshare collection
  doi:10.6084/m9.figshare.c.978904.v5, all CC0-1.0. Each is checked against
  figshare's published MD5, identified by SHA-256 in `data/raw/MANIFEST.json`,
  and re-downloaded once to confirm (ADR 0005).
- **Inventory:** 133,885 records and 3,054 uncharacterized IDs, both as
  published.
- **Findings** (`docs/raw_data_observations.md`):
  - 433 records have two frequency sets on one line;
  - 5 have imaginary frequencies;
  - the header tag is `gdb`, not `gdb9`;
  - the authors' list compares B3LYP with Corina starting geometries, not
    with GDB-17;
  - 83% of molecules have 9 heavy atoms.

### Phase 2 — Lossless parsing (Gate 2)
- **Text level:** every record re-serializes to its exact source bytes.
- **Value level:** every one of 18,078,000 numeric tokens equals the stored
  float64 exactly.
- **Tables:** four, in source units, with the unit in each column name. Both
  frequency sets are kept (ADR 0006).

### Phase 3 — Profiling (Gate 3)
- **Statistics:** 621 formulas, 22,804 Murcko scaffolds, and 14,117 acyclic
  molecules.
- **Duplicates:** 453 molecules in 222 groups, by SMILES and geometric match.
  The geometric thresholds were chosen at a plateau where every matched pair
  agrees in U0 within 0.023 kcal/mol.
- **Tautomers:** 3,652 groups share a standard InChI. They are distinct
  molecules, so they are grouped but never deduplicated (ADR 0007).
- **Authors' relaxed SMILES:** unreliable: 746 unparseable and 12,075 with
  radicals.
- **Triage:** every issue has a class and an owning phase
  (`docs/profiling.md`).

### Phase 4 — Validation (Gate 4)
- **19 checks in four layers** (`docs/validation.md`, ADR 0008):
  - format;
  - chemical identity: the GDB-17 graph against the graph perceived from 3D
    with extended-Hückel connectivity. The default covalent-radius method
    produced 6,812 false positives on strained cages; Hückel left 5;
  - physics invariants: the gap, H − U = k_BT, G < H, the Mulliken sum, ZPVE
    from frequencies, rotational constants from geometry, and linearity;
  - robust statistics.
- **Tolerances** come from print precision, not percentiles.
- **Results:**
  - every physics invariant holds for every record except the 5
    imaginary-frequency structures;
  - 96.3% of records pass every check;
  - the identity check finds 1,010 proton transfers, 229 fragmentations
    and 354 framework rearrangements. Every proton transfer and
    fragmentation is on the authors' list. 5 rearrangements are not;
  - 1,466 listed records are in fact the intended molecule. The authors
    flagged them because the *starting* geometry was odd.
  - Only one doubled-frequency record (129152) has an inconsistent set.

### Phase 5 — Standardization and tiers (Gate 5)
- **Units and atomization:** 15 source properties kept unchanged
  (`source_*`). Energies in eV (CODATA 2018); atomization energies from the
  release's own atom references.
- **Identifiers:** geometry-derived SMILES and InChIKey.
- **Duplicates:** one representative per group, chosen without looking at
  property values.
- **Tiers:** six ordered rules (ADR 0009): Gold 130,659, Silver 1,630,
  Bronze 1,591, Quarantine 5.
- **Not used as tier rules**, so Gold isn't made artificially "typical":
  statistical outliers, mismatches with the authors' SMILES, and linearity
  notation.
- **Conventional subset:** `uncharacterized_listed` reproduces it exactly
  (130,831 records).

### Phase 6 — Features (Gate 6)
- **Tables** (`docs/features.md`, ADR 0010):
  - composition;
  - sorted Coulomb eigenvalues;
  - 184 RDKit descriptors;
  - ECFP4 counts and MACCS;
  - atom and bond graph tables.
- **Leakage guard:** features may read only elements and coordinates, which
  a test enforces. Charges, frequencies and targets cannot leak in.
- **Missing features:** 22 Bronze records have no bond orders, so no RDKit
  features.

### Phase 7 — Statistical and multi-fidelity validation (Gate 7)
- **External data:** the G4(MP2) data of Kim, Park and Choi (*Sci. Data* 6,
  109, 2019, CC0) are joined on `qm9_id`. Every count matches the paper.
- **A published reference error:** the G4(MP2) nitrogen atom reference is a
  doublet, not the ⁴S ground state.
  - A fit gives 2.380 eV per N atom; NIST's ²D–⁴S splitting is 2.3835 eV.
  - Every N-containing published G4(MP2) atomization energy is therefore off
    by about 2.4 eV per N.
  - Those values are withheld (ADR 0011).
- **Label-noise floor:** B3LYP − G4(MP2) on Gold, after a per-element fit,
  has an SD of 0.153 eV and an MAE of 0.123 eV (2.8 kcal/mol). This residual
  does not depend on atom references.
- **Soft review:** robust z within heavy-atom strata, distribution tests by
  tier, and an out-of-fold suspicion score. 384 records carry a review
  reason, and no tier changed (`docs/statistical_validation.md`).

### Phase 8 — Splits (Gate 8)
- **Leakage groups:** union-find over the exact structure, the tautomer, GDB
  connectivity and the geometry InChIKey block. That gives 129,343 groups,
  the largest with 9 records.
- **Why it matters:** a record-level random split would leave 5.8% of test
  records sharing a group with train, and 43 exact duplicates.
- **Eight splits × 5 seeds:**
  - `random`, `isomer_group`, `scaffold_generic`, `cluster_ecfp` and
    `size_extrapolation`;
  - stress tests: `element_holdout` (F) and the upper 5% tails of the gap
    and of U0.
- **Audit:** every split is checked for leakage and coverage, and the shift
  is reported (ADR 0012, `docs/splits.md`).
- **The analysis plan was frozen here.**

### Phase 9 — Benchmark and curation ablation (Gate 9)
- **Ladder** (ADR 0013, `docs/benchmark.md`); levels 2–5 learn the residual
  of level 1:
  - 0: mean;
  - 1: dressed atom;
  - 2: histogram GBM on descriptors and ECFP4;
  - 3: KRR on Coulomb eigenvalues;
  - 4: D-MPNN;
  - 5: SchNet.
- **Grid:** 682 runs, 40.6 h on one M2 Pro laptop, every run tracked in
  MLflow with the commit, code hash and data version.
- **Compute budget (chosen by the user):** D-MPNN does not run RQ5; SchNet
  runs only the `random` split and RQ2. Both are amendments.
- **Two analysis bugs found and fixed before the write-up:**
  - RQ5 ignored `coverage`;
  - the RQ1 count of listed records passing the identity check.

### Phase 10 — Release (Gate 10)
- **Release v1.0.0** (ADR 0014, `docs/release.md`, `docs/datasheet.md`):
  - seven Parquet tables in a canonical layout;
  - extended XYZ with exact floats;
  - Croissant 1.0, with descriptions parsed from the data dictionary;
  - a datasheet, a README with a 5-line quick start, `CITATION.cff`, a CC0
    license and `MANIFEST.sha256`;
  - a PyG loader behind an optional extra.
- **Reproducibility:**
  - `make reproduce` from a clean checkout took 11 min 20 s on macOS and
    about 20 min on Linux CI.
  - The first Linux run showed that three tables differed only in Parquet
    chunking. The canonical layout now removes that.
  - Two tables hold least-squares floats (`review`, `g4mp2_xref`). They
    differ in the last bits: by up to 4.5 × 10⁻¹¹ between macOS and Linux,
    and by up to 5 × 10⁻¹² between Linux machines, because NumPy's OpenBLAS
    picks kernels by CPU.
  - Verification requires the other 12 files to be byte-identical to the
    published manifest, and those two tables to be within a relative 10⁻⁹
    of the published copy, which is downloaded and hash-checked. Every run
    on macOS and Linux passed this.
- **Zenodo:** published 2026-11-10 from the Linux reference build,
  [doi:10.5281/zenodo.24105666](https://doi.org/10.5281/zenodo.24105666) (all versions: doi:10.5281/zenodo.24105665).

### Phase 11 — Uncharacterized-record recovery (optional)
Not started. It would add re-optimized geometries for the uncharacterized
records (Senthil et al., *Chem. Sci.* 2021) as a separate, labelled table,
never merged into the record set.

---

## 5. Results

### 5.1 RQ1 — records failing checks beyond the authors' list
- **Not on the list but failing:** 8 records fail a hard or identity check
  that the list misses:
  - 5 framework rearrangements, mostly strained cages;
  - 3 imaginary-frequency structures.
- **On the list but sound:** 1,466 of the 3,054 listed records are the
  intended GDB-17 molecule by independent perception. They are Silver, not
  dropped.
- **Corroboration:** Kim et al. found the same 229 multi-molecule records,
  and all 453 of our duplicates are among their duplicate pairs.

### 5.2 RQ2 — do flagged records hurt models?
Out-of-fold |residuals|, 5 folds by leakage group, using the best `random`
model per target:

| Target | Model | Silver vs Gold P(>) | Bronze vs Gold P(>) | Bronze MAE / Gold MAE |
|---|---|---|---|---|
| gap | SchNet | 0.549 (p = 5 × 10⁻¹⁰) | 0.634 (p = 10⁻⁶¹) | 0.264 / 0.125 eV |
| U0 | D-MPNN | 0.525 (p = 9 × 10⁻⁴) | 0.645 (p = 10⁻⁷¹) | 0.126 / 0.053 eV |

Supported on all four tests. Phase 7's ridge model had seen no Silver
effect; the stronger models resolve a small one.

### 5.3 RQ3 — split inflation
Gold-test MAE on `random` (eV, 5 seeds):

| Level | Model | Gap | U0 atomization |
|---|---|---|---|
| 0 | mean | 1.070 | 8.119 |
| 1 | dressed atom | 0.818 | 0.862 |
| 2 | GBM | 0.144 | 0.088 |
| 3 | KRR (20k molecules) | 0.622 | 0.552 |
| 4 | D-MPNN | 0.110 | **0.050** |
| 5 | SchNet | **0.107** | 0.051 |

Ratio of each split's MAE to the `random` MAE:

| Model, target | isomer | scaffold | cluster | size |
|---|---|---|---|---|
| GBM, U0 | 1.44 | 2.92 | 1.11 | 3.36 |
| D-MPNN, U0 | 1.08 | 3.76 | 1.15 | 3.09 |
| D-MPNN, gap | 1.08 | 1.72 | 1.11 | 1.79 |

- 23 of 24 cells for the learned levels inflate. The exception is KRR U0 on
  `cluster_ecfp`.
- `cluster_ecfp` matters little (1.04–1.15×), because QM9 is a dense
  enumeration and the test set stays close to train.
- **Stress test:** with no fluorine in train, every level fails on U0
  (≈ 6 eV). The dressed-atom baseline has no F coefficient, and every level
  learns its residual.

### 5.4 RQ4 — the label-noise floor
- **Comparison:** the best `random` U0 MAE is 0.050 eV [0.049, 0.050], from
  D-MPNN; SchNet reaches 0.051 and GBM 0.088. All are below the 0.123 eV
  B3LYP − G4(MP2) residual MAE.
- **Conclusion:** the best models already reproduce B3LYP labels more
  closely than B3LYP agrees with G4(MP2).

### 5.5 RQ5 — curated training on a fixed Gold test set
Level 2, paired bootstrap on the same Gold test molecules:

| N | Gap: gold − all [95% CI] | U0: gold − all [95% CI] |
|---|---|---|
| 1,000 | +0.0035 [+0.0013, +0.0057] | −0.0046 [−0.0069, −0.0023] |
| 10,000 | +0.0008 [−0.0006, +0.0022] | −0.0019 [−0.0032, −0.0006] |
| 100,000 | −0.0006 [−0.0012, −0.0000] | −0.0011 [−0.0017, −0.0006] |

- Gold-only is not worse at the largest N, so the verdict is supported.
- **The effect is about 1 meV, under 1% of the error.** It changes sign at
  smaller N, and the learning-curve slopes are identical across tier sets.
- The ~2.4% of non-Gold training records barely affect Gold-test accuracy.
  Their effect shows in their own residuals (RQ2).

---

## 6. Decisions

| ADR | Decision | Main alternative rejected, and why |
|---|---|---|
| 0001 | macOS for development, Linux CI as the reference build | Colab or cloud for everything: session limits, no persistent cache |
| 0002 | The figshare collection of the 2014 paper is the source of record | PyG and DeepChem copies: already filtered and converted |
| 0003 | uv, DVC, ruff, mypy --strict, pytest + Hypothesis | conda, Poetry: slower; RDKit now ships wheels |
| 0004 | Pandera schemas plus plain-Python checks | Great Expectations: heavy, poor fit for per-record residuals |
| 0005 | figshare as the DVC "remote", MD5 + SHA-256 manifest, second-download check | `dvc import-url`: signed redirect URLs make ETags unreliable |
| 0006 | Lossless parsing proven by text round-trip and value exactness | A lenient regex parse: silent loss |
| 0007 | Group duplicates by SMILES and geometry; never deduplicate by InChI | InChIKey dedup: would delete about 4,400 distinct tautomers |
| 0008 | Tolerances from print precision; Hückel bond perception | Percentile tolerances: fail 0.1% by construction; covalent radii: 6,812 false cages |
| 0009 | Tiers with reasons; keep source values; geometry identifiers | Drop all 3,054 listed records: loses 1,466 sound ones, misses 8 |
| 0010 | Features read only elements and coordinates; describe the geometry molecule | Features from the GDB-17 SMILES: wrong molecule for Bronze |
| 0011 | Withhold N-containing G4(MP2) atomization energies; use a reference-free residual | Invent a corrected N reference |
| 0012 | Leakage groups by four keys; grouped and holdout splits; full audit | Record-level random split: 5.8% leakage |
| 0013 | Ladder 0–5 in plain PyTorch and scikit-learn; Δ-learning; Option B budget | Full grid (30–35 h); LightGBM, Chemprop and PyG packages |
| 0014 | Canonical Parquet; uncompressed extended XYZ; Croissant from the dictionary; Linux manifest; user-run Zenodo upload | Byte copies (chunking-dependent); gzip (zlib-dependent); CC BY |

---

## 7. Limitations

`docs/limitations.md` has the full list. The most important:
- **Labels.**
  - Every label is a B3LYP value, with one geometry per molecule that is not
    guaranteed to be the global minimum.
  - The noise floor is measured against G4(MP2), not against experiment.
- **Tier judgement.** The Silver tier for the 1,466 listed-but-consistent
  records is a judgement call. `uncharacterized_listed` lets users reverse
  it.
- **Silver and Bronze are not random samples.** Their chemistry differs from
  Gold's, so tier ablations change coverage as well as label quality.
- **Splits.** QM9 is dense: cluster and scaffold splits reduce test → train
  similarity only modestly (median Tanimoto 0.55–0.57 vs. 0.62).
- **Benchmark budget.**
  - The neural levels are under-trained (30 epochs); their numbers are upper
    bounds on their error.
  - KRR uses only Coulomb eigenvalues and 20,000 molecules.
  - Coverage is limited: no D-MPNN in RQ5, SchNet on `random` only.
- **Release.** Two of 14 files (the least-squares tables) are reproducible
  to 10⁻⁹, not bit for bit, on any platform. The archived `datasheet.md`
  says the release rebuilds "with identical file hashes". That overstates it
  for these two files; this is an erratum, to be corrected in the next
  version.
- **Open items:**
  - recompute the suspicion score with the best model;
  - a ZPE-separated RQ4 comparison;
  - Phase 11.

---

## 8. Reproducing everything

**Requirements:** git, [uv](https://docs.astral.sh/uv/) and internet access to
figshare. Disk: about 1.5 GB for data and outputs, plus about 1.5 GB for the
environment with the `ml` and `pyg` extras (PyTorch). uv installs Python 3.12 and every
dependency from `uv.lock`.

```bash
git clone https://github.com/akhilnathreddy/qm9-advanced-curation.git
cd qm9-advanced-curation
uv sync --locked --extra ml --extra pyg
```

**Rebuild the release from the raw archives and verify every file** (about
11 min on an M2 Pro, about 20 min on a 4-core Linux runner):

```bash
make reproduce
```

Every file must match `release/qm9-curated-v1.0.0.sha256`, except
`review.parquet` and `g4mp2_xref.parquet`. Those two must be within a
relative 10⁻⁹ of the published archive, which `make reproduce` downloads and
hash-checks. To use a copy you already have:

```bash
make reproduce REFERENCE=/path/to/qm9-curated-v1.0.0
```

The data release is the tag `v1.0.0`. Use `main` for the tooling: it carries
the verification rule above, which was added after the tag.

**Run every pipeline stage, including the benchmark.** This is about 40 h of
compute. The runs are cached and resumable.

```bash
make repro
```

**Phase gates:**

```bash
make gate9
```

```bash
make gate10
```

**Checks:**

```bash
make check
```

**Load the release in Python:**

```python
import polars as pl
mol = pl.read_parquet("qm9-curated-v1.0.0/molecules.parquet").filter(pl.col("quality_tier") == "gold")
split = pl.read_parquet("qm9-curated-v1.0.0/splits.parquet").filter(pl.col("split_name") == "scaffold_generic", pl.col("seed") == 0)
train = mol.join(split.filter(pl.col("subset") == "train"), on="qm9_id")
```

---

## 9. Citation

- **This release:** QM9 Curated v1.0.0, Zenodo, [doi:10.5281/zenodo.24105666](https://doi.org/10.5281/zenodo.24105666). See `CITATION.cff`.
- **QM9:** Ramakrishnan, Dral, Rupp & von Lilienfeld, *Sci. Data* 1, 140022
  (2014), doi:10.1038/sdata.2014.22.
- **G4(MP2) cross-reference:** Kim, Park & Choi, *Sci. Data* 6, 109 (2019),
  doi:10.1038/s41597-019-0121-7.
