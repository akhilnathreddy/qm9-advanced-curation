import hashlib
import json
from pathlib import Path

import pytest

from qm9_curation.config import load_config
from qm9_curation.fixture import record_traits
from qm9_curation.inventory import ArchiveStats, inspect_record

pytestmark = pytest.mark.pipeline


def test_fixture_matches_its_sample_record(qm9_1k_dir: Path) -> None:
    sample = json.loads((qm9_1k_dir / "SAMPLE.json").read_text())
    files = sorted(qm9_1k_dir.glob("*.xyz"))
    assert len(files) == sample["size"] == 1000
    assert [int(f.stem.split("_")[1]) for f in files] == [m["index"] for m in sample["molecules"]]
    assert sample["license"] == "CC0-1.0"


def test_fixture_files_are_byte_identical_to_source(qm9_1k_dir: Path) -> None:
    sample = json.loads((qm9_1k_dir / "SAMPLE.json").read_text())
    for molecule in sample["molecules"]:
        data = (qm9_1k_dir / f"dsgdb9nsd_{molecule['index']:06d}.xyz").read_bytes()
        assert hashlib.sha256(data).hexdigest() == molecule["sha256"], molecule["index"]


def test_fixture_records_have_valid_structure(qm9_1k_dir: Path) -> None:
    stats = ArchiveStats()
    for path in qm9_1k_dir.glob("*.xyz"):
        inspect_record(path.name, path.read_text(encoding="ascii"), stats)
    assert not (stats.bad_line_count or stats.index_mismatch or stats.bad_property_count)
    assert set(stats.elements) <= set(load_config("dataset")["expected"]["elements"])
    assert stats.header_tags == {"gdb": 1000}


def test_fixture_covers_known_edge_cases(qm9_1k_dir: Path) -> None:
    traits = [record_traits(p.read_text(encoding="ascii")) for p in qm9_1k_dir.glob("*.xyz")]
    assert sum(t["linear"] for t in traits) == 5
    assert sum(t["negative_frequency"] for t in traits) == 5
    assert any(t["doubled_frequencies"] for t in traits)
    assert any(t["fortran_exponent"] for t in traits)
    assert {t["n_heavy"] for t in traits} == set(range(1, 10))
    saddles = load_config("dataset")["known_issues"]["saddle_points"]
    for index in saddles:
        assert (qm9_1k_dir / f"dsgdb9nsd_{index:06d}.xyz").exists()
