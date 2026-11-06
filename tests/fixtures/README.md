# Test fixtures

`qm9_1k/` holds 1,000 QM9 `.xyz` records copied byte-for-byte from the
source archive (QM9 is CC0). `qm9_1k/SAMPLE.json` lists each molecule, why
it was chosen, the seed, and the archive SHA-256. Rebuild with
`make fixture`; the selection is deterministic.
