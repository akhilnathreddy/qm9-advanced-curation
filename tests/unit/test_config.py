from pathlib import Path

import pytest

from qm9_curation.config import CONFIG_NAMES, load_config


@pytest.mark.parametrize("name", CONFIG_NAMES)
def test_every_config_loads(name: str) -> None:
    assert isinstance(load_config(name), dict)


def test_expected_counts_match_source_paper() -> None:
    expected = load_config("dataset")["expected"]
    assert expected["n_molecules"] == 133885
    assert expected["n_uncharacterized"] == 3054
    assert set(expected["elements"]) == {"H", "C", "N", "O", "F"}


def test_non_mapping_config_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "bad.yaml").write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_config("bad", config_dir=tmp_path)
