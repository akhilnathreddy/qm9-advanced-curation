"""Build the 1,000-molecule CI fixture from the raw archive (Phase 1).

Run as ``python -m qm9_curation.fixture``. Copies source ``.xyz`` files
byte-for-byte into ``tests/fixtures/qm9_1k/`` and writes ``SAMPLE.json``
(IDs, seed, selection reason, archive SHA-256). QM9 is CC0, so redistributing
the records is allowed.

Selection:
1. every edge case the inventory found (linear molecules, imaginary
   frequencies, molecules the authors list as hard to converge), plus seeded
   samples of doubled-frequency records, ``*^`` records and uncharacterized
   molecules;
2. the rest filled by a seeded sample stratified by heavy-atom count, with
   each stratum guaranteed at least a few members so rare sizes are covered.
"""

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from qm9_curation.acquire import MANIFEST_NAME
from qm9_curation.config import PROJECT_ROOT, RAW_DIR, load_config
from qm9_curation.inventory import (
    ARCHIVE_NAME,
    FORTRAN_EXPONENT,
    MEMBER_RE,
    UNCHARACTERIZED_NAME,
    iter_members,
    parse_uncharacterized,
)

FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures" / "qm9_1k"
SAMPLE_SIZE = 1000
SEED = 20261106
MIN_PER_STRATUM = 3
SAMPLED_EDGE_CASES = {"doubled_frequencies": 10, "fortran_exponent": 10, "uncharacterized": 20}


def record_traits(text: str) -> dict[str, Any]:
    lines = text.rstrip("\n").split("\n")
    n_atoms = int(lines[0])
    atoms = [line.split()[0] for line in lines[2 : 2 + n_atoms]]
    freqs = lines[2 + n_atoms].split()
    return {
        "n_heavy": sum(1 for a in atoms if a != "H"),
        "linear": len(freqs) == 3 * n_atoms - 5,
        "doubled_frequencies": len(freqs) == 2 * (3 * n_atoms - 6),
        "negative_frequency": any(f.startswith("-") for f in freqs),
        "fortran_exponent": FORTRAN_EXPONENT in text,
    }


def select(
    traits: dict[int, dict[str, Any]],
    uncharacterized: set[int],
    hard_to_converge: set[int],
    seed: int,
    size: int,
) -> dict[int, str]:
    rng = random.Random(seed)
    chosen: dict[int, str] = {}

    def take(ids: list[int], reason: str) -> None:
        for i in ids:
            chosen.setdefault(i, reason)

    take(sorted(i for i, t in traits.items() if t["linear"]), "linear")
    take(sorted(i for i, t in traits.items() if t["negative_frequency"]), "negative_frequency")
    take(sorted(hard_to_converge & traits.keys()), "hard_to_converge")
    pools = {
        "doubled_frequencies": [i for i, t in traits.items() if t["doubled_frequencies"]],
        "fortran_exponent": [i for i, t in traits.items() if t["fortran_exponent"]],
        "uncharacterized": sorted(uncharacterized & traits.keys()),
    }
    for reason, k in SAMPLED_EDGE_CASES.items():
        pool = sorted(set(pools[reason]) - chosen.keys())
        take(sorted(rng.sample(pool, min(k, len(pool)))), reason)

    strata: dict[int, list[int]] = defaultdict(list)
    for i, t in traits.items():
        if i not in chosen:
            strata[t["n_heavy"]].append(i)
    for n_heavy in sorted(strata):
        pool = sorted(strata[n_heavy])
        take(sorted(rng.sample(pool, min(MIN_PER_STRATUM, len(pool)))), "stratum_minimum")

    remaining = size - len(chosen)
    if remaining < 0:
        raise ValueError(f"Edge cases alone exceed the sample size ({len(chosen)} > {size})")
    total = sum(len(v) for v in strata.values())
    quotas = {n: round(remaining * len(v) / total) for n, v in strata.items()}
    # Fix rounding drift on the largest stratum so the total is exact.
    largest = max(strata, key=lambda n: len(strata[n]))
    quotas[largest] += remaining - sum(quotas.values())
    for n_heavy in sorted(strata):
        pool = sorted(set(strata[n_heavy]) - chosen.keys())
        take(sorted(rng.sample(pool, min(quotas[n_heavy], len(pool)))), "stratified")
    return chosen


def build(
    raw_dir: Path, out_dir: Path, seed: int = SEED, size: int = SAMPLE_SIZE
) -> dict[str, Any]:
    archive = raw_dir / ARCHIVE_NAME
    contents: dict[int, bytes] = {}
    traits: dict[int, dict[str, Any]] = {}
    for member, data in iter_members(archive):
        match = MEMBER_RE.match(member.name)
        if data is None or match is None:
            continue
        index = int(match.group(1))
        contents[index] = data
        traits[index] = record_traits(data.decode("ascii"))

    uncharacterized = set(parse_uncharacterized(raw_dir / UNCHARACTERIZED_NAME))
    hard_to_converge = set(load_config("dataset")["known_issues"]["hard_to_converge"])
    chosen = select(traits, uncharacterized, hard_to_converge, seed, size)

    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.xyz"):
        old.unlink()
    for index in sorted(chosen):
        (out_dir / f"dsgdb9nsd_{index:06d}.xyz").write_bytes(contents[index])

    manifest = json.loads((raw_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    archive_sha = next(f["sha256"] for f in manifest["files"] if f["name"] == ARCHIVE_NAME)
    reasons: dict[str, int] = defaultdict(int)
    for reason in chosen.values():
        reasons[reason] += 1
    sample = {
        "source_archive": ARCHIVE_NAME,
        "source_archive_sha256": archive_sha,
        "license": manifest["license"],
        "seed": seed,
        "size": len(chosen),
        "counts_by_reason": dict(sorted(reasons.items())),
        "molecules": [
            {
                "index": i,
                "reason": chosen[i],
                "sha256": hashlib.sha256(contents[i]).hexdigest(),
                **traits[i],
            }
            for i in sorted(chosen)
        ],
    }
    (out_dir / "SAMPLE.json").write_text(
        json.dumps(sample, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return sample


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=FIXTURE_DIR)
    args = parser.parse_args(argv)
    sample = build(args.raw_dir, args.out_dir)
    print(f"{sample['size']} molecules written to {args.out_dir}")
    for reason, count in sample["counts_by_reason"].items():
        print(f"  {reason:<22} {count}")


if __name__ == "__main__":
    main()
