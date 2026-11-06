# 0005 — Acquisition and raw-data versioning

- **Status:** Accepted
- **Date:** 2026-11-06
- **Phase:** 1

## Context
ADR 0002 chose the figshare collection as the source of record, pending
confirmation of URLs and license. The raw files must be reproducible from a
clean clone, verifiably identical to what the authors published, and
protected against accidental edits. There is no DVC remote yet.

## Decision
- **Source confirmed:** collection doi:10.6084/m9.figshare.c.978904.v5.
  Four files are fetched: `dsgdb9nsd.xyz.tar.bz2`, `uncharacterized.txt`,
  `atomref.txt`, `readme.txt`. Every article is licensed **CC0-1.0**
  (reported by the figshare API on 2026-11-06).
- **Integrity:** each file is checked against figshare's published size and
  MD5, then identified by SHA-256 in `data/raw/MANIFEST.json`, which is
  versioned in git. Gate 1 also downloads everything a second time and
  compares SHA-256.
- **figshare is the remote:** `dvc repro` on a clean clone downloads and
  verifies the files (stage `acquire`, `persist: true`), so no separate DVC
  remote is needed for now.
- **Immutability:** DVC uses `cache.type: hardlink,copy`. Raw files are
  hardlinked to the read-only DVC cache, so the workspace copies stay
  read-only.
- **CI** has no raw data. It runs `dvc repro env_report` and the test suite,
  which includes a committed 1,000-molecule fixture (allowed by CC0).
- `validation.txt` and the C7H10O2 isomer set are not fetched now; they are
  added when Phase 7 needs them.

## Alternatives considered
- *`dvc import-url`*: tracks upstream changes, but figshare download URLs
  redirect to signed storage URLs, which makes ETag-based checks unreliable.
- *Separate DVC remote (S3 / Google Drive)*: not needed while figshare
  serves byte-identical files; revisit if it stops doing so.

## Consequences
- If figshare ever serves different bytes, `acquire` fails loudly instead of
  silently using new data.
- The fixture must be rebuilt (`make fixture`) if the selection logic changes;
  `SAMPLE.json` records the seed and the archive SHA-256.
