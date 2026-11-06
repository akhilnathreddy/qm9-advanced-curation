from pathlib import Path

import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "qm9_1k"


@pytest.fixture
def qm9_1k_dir() -> Path:
    if not any(FIXTURE_DIR.glob("*.xyz")):
        pytest.skip("1k fixture not populated yet (Phase 1)")
    return FIXTURE_DIR


GOLDEN_DIR = Path(__file__).parent / "fixtures" / "golden"


@pytest.fixture
def golden() -> dict[int, str]:
    """Real QM9 source records, byte-for-byte, keyed by index."""
    return {
        int(p.stem.split("_")[1]): p.read_text(encoding="ascii")
        for p in sorted(GOLDEN_DIR.glob("*.xyz"))
    }
