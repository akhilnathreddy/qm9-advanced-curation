# 0012 — Split suite and leakage audit

- **Status:** Accepted
- **Date:** 2026-11-07
- **Phase:** 8

## Context
RQ3 asks how much split choice inflates reported performance, and RQ2, RQ4
and RQ5 need a fixed evaluation set. Two traps need care:
- QM9 has the same molecule under more than one index. Phase 7 found
  stereoisomer, conformer and relaxed-to-same pairs that ADR 0007's
  exact-structure groups do not join.
- Plain scaffold splits put all 14,117 acyclic molecules into one
  empty-scaffold bucket.

## Decision — scope
- **Every split assigns all 133,885 records, for every seed.** Tier
  filtering (Gold-only test, tiered training) happens at use time.
  - This lets one split serve every tier, as RQ5 requires.
  - Quarantine records are assigned too but never used (analysis plan).
- **Storage:** `data/splits/splits.parquet`, in the long format
  `(qm9_id, split_name, seed, subset)` that plan v2 asks for. Seeds 0–4.

## Decision — leakage groups (every split)
- **Components:** the union-find components of records that share any of
  - the exact-structure group;
  - the standard-InChI (tautomer) group;
  - the GDB-17 connectivity group;
  - the connectivity block (first 14 characters) of the InChIKey perceived
    from the geometry.
- **Why the fourth key:** it closes the Phase 7 gap, merging stereoisomers,
  other conformers and records that relaxed to the same molecule.
- **Result:** 129,343 groups, the largest with 9 records; 8,267 records are in
  groups of two or more.
- **Split units:** each split's grouping key (formula, scaffold, cluster) is
  merged with the leakage groups, so no split can separate a leakage group.

## Decision — the splits

| Split | Unit | Assignment |
|---|---|---|
| `random` | leakage group | grouped 80/10/10 |
| `isomer_group` | formula (621) | grouped |
| `scaffold_generic` | generic Bemis–Murcko scaffold of the geometry structure; acyclic molecules bucketed by heavy-atom composition (`acyclic:C4N1O1`, 138 buckets) | grouped |
| `cluster_ecfp` | RDKit LeaderPicker (sphere exclusion) on ECFP4 bits at Tanimoto distance 0.65 (7,940 leaders); each molecule to its nearest leader | grouped |
| `size_extrapolation` | leakage group | test = 9 heavy atoms; train/val from ≤ 8 |
| `element_holdout` | leakage group | test = contains F |
| `property_tail_gap` | leakage group | test = largest 5% of gap |
| `property_tail_u0_atomization` | leakage group | test = most negative 5% of U0 atomization energy |

**Grouped assignment:**
- Units larger than half the test target always go to train (balanced
  scaffold splitting).
- The rest are shuffled with the seed and fill test, then val, without
  overshooting.
- Everything else goes to train.

**Holdout assignment:**
- A rule fixes the test set, so it is the same for every seed.
- A unit goes to test if any member meets the rule.
- Train and val come from the rest in a ratio of 8 : 1.

## Decision — the leakage audit (Gate 8)

**Assertions, for every split and seed:**
- every record is assigned exactly once;
- no exact-structure, tautomer, connectivity, geometry InChIKey block, leakage
  group or split unit occurs in two subsets;
- holdout rules are respected;
- grouped fractions are within ±0.01 of target. Achieved: within 10⁻⁵.

**Shift report (seed 0, Gold, test vs. train):**
- KS and Wasserstein for the gap and the U0 atomization energy;
- the heavy-atom-count shift;
- element-presence fractions;
- the max-Tanimoto (test → train) distribution for 2,000 sampled test
  molecules, computed exactly with sparse products.

## Alternatives considered
- **Record-level random split, the literature default:** 774 of 13,388 test
  records (5.8%) would share a leakage group with train/val, and 43 would be
  exact duplicates. `random` here differs from the literature only by
  removing that leakage.
- **Butina clustering:** quadratic in memory at 134k molecules. The leader
  algorithm is Butina's linear-time sphere-exclusion form.
- **k-means on fingerprints:** not a Tanimoto metric, and it would need
  scikit-learn in the default environment.

## Consequences
- **The cluster and scaffold splits only modestly reduce test → train
  similarity.**
  - Median NN Tanimoto: 0.57 (cluster) and 0.55 (scaffold), against 0.62
    (random).
  - QM9 is a dense enumeration, and molecules near a cluster boundary keep
    close neighbours across it. Reported as-is, not tuned away.
- **Merging with leakage groups creates large units.**
  - Cluster: 15,018 records; scaffold: 23,418.
  - They go to train, which is the documented balanced-split behaviour.
- **The isomer split has the highest NN similarity (0.63).**
  - The same scaffolds recur across formulas.
  - It tests composition novelty, not structural novelty; `docs/splits.md`
    says so.
