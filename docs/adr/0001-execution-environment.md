# 0001 — Execution environment

- **Status:** Accepted
- **Date:** 2026-11-06
- **Phase:** 0

## Context
The pipeline must run on the author's MacBook (Apple silicon) and be
reproducible elsewhere. The raw data is a few hundred MB; Phases 0–8 and the
classical baselines are CPU-bound. 3D GNNs (Phase 9) benefit from CUDA.

## Decision
- Develop and run Phases 0–8 and classical ML locally on macOS arm64.
- Treat **CI on Linux (ubuntu-latest)** as the reference build for
  reproducibility and output hashes.
- Run 3D GNN experiments on Colab or a cloud GPU, using the same lockfile.

## Alternatives considered
- *Colab for everything*: session limits, no persistent DVC cache, harder to
  pin the environment.
- *Cloud VM from day one*: cost and setup time not justified before Phase 9.

## Consequences
- Byte-identity checks are defined against the Linux build; macOS results
  must match within tolerance, and any difference is investigated.
- GPU work needs a documented remote setup before Phase 9.
