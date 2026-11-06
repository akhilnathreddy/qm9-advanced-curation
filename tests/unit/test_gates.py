from pathlib import Path

import pytest

from qm9_curation.reporting import gates


def test_gate0_record(tmp_path: Path) -> None:
    gates.main(["--phase", "0", "--out-dir", str(tmp_path)])
    assert (tmp_path / "phase_0.json").exists()


def test_unknown_phase_is_rejected() -> None:
    with pytest.raises(ValueError):
        gates.gate_record(99)
