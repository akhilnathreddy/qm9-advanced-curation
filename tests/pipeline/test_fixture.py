from pathlib import Path

import pytest


@pytest.mark.pipeline
def test_fixture_has_1000_molecules(qm9_1k_dir: Path) -> None:
    assert len(list(qm9_1k_dir.glob("*.xyz"))) == 1000
