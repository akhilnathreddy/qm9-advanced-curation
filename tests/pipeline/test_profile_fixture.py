from pathlib import Path

import pytest

from qm9_curation.parse import build as parse_build
from qm9_curation.profiling.build import build_profile
from qm9_curation.profiling.report import render

pytestmark = pytest.mark.pipeline


def test_fixture_profile_and_report(qm9_1k_dir: Path) -> None:
    frames, _ = parse_build.build(parse_build.iter_directory(qm9_1k_dir), "0" * 64)
    mol, atoms, freq = frames["molecules"], frames["atoms"], frames["frequencies"]
    profile, per_molecule, _ = build_profile(mol, atoms, freq, set())
    assert all(profile["gate_3_conditions"].values()), profile["gate_3_conditions"]
    assert per_molecule.height == 1000
    assert profile["geometry"]["molecules_with_negative_frequency"] == 5
    page = render(profile, mol, per_molecule)
    assert page.startswith("<!doctype html>") and "QM9 data profile" in page
    assert render(profile, mol, per_molecule) == page  # deterministic
