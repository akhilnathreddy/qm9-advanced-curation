# Working notes for Claude

Start every session by reading `docs/handoff.md`: it gives the current phase,
what is done, what is next, and open decisions.

## Project shape
- **Phases and gates:** the project follows `docs/plan_v2.md`, phase by phase.
  Each phase ends with an exit gate (`make gateN`) that runs:
  1. `sync check repro`;
  2. a forced re-run of the phase's stage, with a byte-identity check;
  3. `python -m qm9_curation.reporting.gates --phase N`.
- **Docs per phase:** decisions go in ADRs (`docs/adr/NNNN-*.md`), results in
  a `docs/<topic>.md`, open issues in `docs/limitations.md`, and column
  definitions in `docs/data_dictionary.md`. The README "Current phase" line is
  updated at each phase end.
- **Frozen analysis plan:** `docs/analysis_plan.md` has been frozen since
  commit 4b08cc5. Never edit the text above `## Amendments`. Every change is a
  dated amendment, and anything seen before it (for example test numbers) is
  disclosed.
- **Commands:**
  - environment: `uv sync --locked --extra ml` (Python 3.12);
  - checks: `make check` (ruff, ruff format, mypy --strict, pytest);
  - pipeline: `make repro` (DVC).
  - CI also needs `uv run pytest -W error` to stay warning-free.

## Git and GitHub
- Repo: `akhilnathreddy/qm9-advanced-curation` (public). Each phase has its
  own stacked branch: `phase-N/<name>`, created from the previous phase's
  branch.
- **Commit or push only when the user asks.**
- Push with
  `git -c credential.helper= -c credential.helper='!gh auth git-credential' push ...`.
  Never change the global git config, and never handle passwords or tokens.
- Commit messages end with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. PR descriptions
  end with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
- The default branch stays `phase-0/setup` (the user's choice, 2026-11-09).
  There is no `main` yet; create one before the release.

## Data
- Data is versioned by DVC, never git. Its outputs are read-only hard links
  into `.dvc/cache`.
- Sources are QM9 (figshare, CC0-1.0) and Kim et al. 2019 G4(MP2) (CC0-1.0).
  Both are verified by MD5/SHA-256 manifests.
- The user plans to publish the curated dataset in Phase 10. A Zenodo deposit
  goes through the user's own account.
