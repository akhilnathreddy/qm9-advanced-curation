import json
from pathlib import Path

from qm9_curation import env_report


def test_report_has_core_fields() -> None:
    report = env_report.collect()
    assert report["python"].startswith("3.12")
    assert report["packages"]["rdkit"] is not None
    assert report["constants"] == "CODATA 2018"


def test_report_is_deterministic(tmp_path: Path) -> None:
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    env_report.main(["--out", str(a)])
    env_report.main(["--out", str(b)])
    assert a.read_bytes() == b.read_bytes()
    assert "timestamp" not in json.loads(a.read_text())
