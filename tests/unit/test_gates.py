import json
from pathlib import Path

import pytest

from qm9_curation.reporting import gates


def test_gate0_record(tmp_path: Path) -> None:
    gates.main(["--phase", "0", "--out-dir", str(tmp_path)])
    assert json.loads((tmp_path / "phase_0.json").read_text())["passed"] is True


def test_unknown_phase_is_rejected() -> None:
    with pytest.raises(ValueError):
        gates.gate_record(99)


def write_inventory(reports: Path, conditions: dict[str, bool]) -> None:
    (reports / "data_quality").mkdir(parents=True)
    (reports / "data_quality" / "inventory.json").write_text(
        json.dumps({"gate_1_conditions": conditions})
    )


def test_gate1_reads_inventory_conditions(tmp_path: Path) -> None:
    write_inventory(tmp_path, {"a": True, "b": True})
    record = gates.gate_record(1, reports_dir=tmp_path)
    assert record["passed"] is True
    assert set(record["checks"]) == {
        "data_quality/inventory.json:a",
        "data_quality/inventory.json:b",
    }


def test_gate1_fails_if_any_condition_fails(tmp_path: Path) -> None:
    write_inventory(tmp_path, {"a": True, "b": False})
    assert gates.gate_record(1, reports_dir=tmp_path)["passed"] is False
