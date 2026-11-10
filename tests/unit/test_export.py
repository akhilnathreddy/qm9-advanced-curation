"""Release exports (Phase 10): extended XYZ, Croissant, release build, verify, PyG."""

import math
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from qm9_curation.config import PROJECT_ROOT, load_config
from qm9_curation.export import croissant, release, verify, xyz

DICTIONARY = PROJECT_ROOT / "docs" / "data_dictionary.md"
DATASHEET = PROJECT_ROOT / "docs" / "datasheet.md"
PROPS = load_config("release")["extxyz_properties"]


def toy_tables(tmp: Path, n: int = 6) -> dict[str, Path]:
    """Seven small release tables with real column names (all documented)."""
    ids = np.arange(1, n + 1, dtype=np.uint32)
    molecules = pl.DataFrame(
        {
            "qm9_id": ids,
            "quality_tier": ["gold"] * (n - 2) + ["silver", "bronze"],
            "formula": ["CH4"] * n,
            **{p: np.linspace(-1.0, 1.0, n) / 3.0 for p in PROPS},
        }
    ).with_columns(
        pl.when(pl.col("qm9_id") == 2).then(None).otherwise(pl.col("mu_debye")).alias("mu_debye")
    )
    rng = np.random.default_rng(0)
    atoms = pl.DataFrame(
        {
            "qm9_id": np.repeat(ids, 5),
            "atom_idx": np.tile(np.arange(5, dtype=np.uint8), n),
            "element": ["C", "H", "H", "H", "H"] * n,
            "atomic_number": np.tile(np.array([6, 1, 1, 1, 1], dtype=np.uint8), n),
            "x_ang": rng.normal(size=5 * n),
            "y_ang": rng.normal(size=5 * n),
            "z_ang": rng.normal(size=5 * n) / 7.0,
            "mulliken_charge_e": rng.normal(size=5 * n) / 3.0,
        }
    )
    frequencies = pl.DataFrame(
        {
            "qm9_id": np.repeat(ids, 9),
            "set_idx": np.zeros(9 * n, dtype=np.uint8),
            "mode_idx": np.tile(np.arange(9, dtype=np.uint16), n),
            "wavenumber_cm1": rng.uniform(1000, 3000, 9 * n),
        }
    )
    subsets = ["train", "train", "test", "val", "test", "test"][:n]
    splits = pl.DataFrame(
        {
            "qm9_id": ids,
            "split_name": ["scaffold_generic"] * n,
            "seed": np.zeros(n, dtype=np.uint8),
            "subset": subsets,
        }
    )
    frames = {
        "molecules": molecules,
        "atoms": atoms,
        "frequencies": frequencies,
        "g4mp2_xref": pl.DataFrame({"qm9_id": ids, "g4mp2_status": ["ok"] * n}),
        "review": pl.DataFrame({"qm9_id": ids, "suspicion_score": rng.uniform(size=n)}),
        "splits": splits,
        "split_groups": pl.DataFrame({"qm9_id": ids, "leakage_group_id": ids}),
    }
    out = {}
    for name, frame in frames.items():
        out[name] = tmp / f"src_{name}.parquet"
        frame.write_parquet(out[name])
    return out


def test_expand_dictionary_cell_patterns() -> None:
    assert croissant._expand("a, b") == ["a", "b"]
    assert croissant._expand("u0/u298/h298_atomization_ev") == [
        "u0_atomization_ev",
        "u298_atomization_ev",
        "h298_atomization_ev",
    ]
    assert croissant._expand("oof_residual_{u0_atomization_ev,gap_ev}") == [
        "oof_residual_u0_atomization_ev",
        "oof_residual_gap_ev",
    ]
    assert croissant._expand("unit_<split>") == ["unit_*"]


def test_every_real_release_column_is_documented() -> None:
    sections = croissant.dictionary(DICTIONARY)
    for table, path in load_config("release")["tables"].items():
        if not (PROJECT_ROOT / path).exists():
            pytest.skip("pipeline outputs not built")
        cols = list(pl.read_parquet_schema(PROJECT_ROOT / path))
        missing = [c for c, d in croissant.describe(table, cols, sections).items() if not d]
        assert not missing, (table, missing)


def test_extxyz_round_trips_through_ase(tmp_path: Path) -> None:
    from ase.io import read

    tables = toy_tables(tmp_path)
    mol, atoms = pl.read_parquet(tables["molecules"]), pl.read_parquet(tables["atoms"])
    path = tmp_path / "t.extxyz"
    assert xyz.write_extxyz(path, mol, atoms, PROPS) == mol.height
    frames = read(path, index=":")
    pos = np.concatenate([f.get_positions() for f in frames])
    assert np.array_equal(pos, atoms.select("x_ang", "y_ang", "z_ang").to_numpy())
    charges = np.concatenate([f.arrays["mulliken_charge"] for f in frames])
    assert np.array_equal(charges, atoms["mulliken_charge_e"].to_numpy())
    assert [f.info["qm9_id"] for f in frames] == mol["qm9_id"].to_list()
    assert frames[0].info["gap_ev"] == mol["gap_ev"][0]
    assert math.isnan(frames[1].info["mu_debye"])  # null written as nan


def build_toy(tmp_path: Path) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    cfg = load_config("release")
    tables = toy_tables(tmp_path)
    report = release.build(cfg, tables, tmp_path / "out", DATASHEET, DICTIONARY, n_expected=6)
    return cfg, tmp_path / "out" / release.release_name(cfg), report


def test_release_build_is_complete_documented_and_deterministic(tmp_path: Path) -> None:
    cfg, out, report = build_toy(tmp_path)
    assert all(report["gate_10_conditions"].values()), report["gate_10_conditions"]
    manifest = verify.read_manifest(out / release.MANIFEST)
    assert set(manifest) == {p.name for p in out.iterdir() if p.name != release.MANIFEST}
    assert all(release.sha256_file(out / k) == v for k, v in manifest.items())
    first = dict(report["output_sha256"])
    again = release.build(cfg, toy_tables(tmp_path), tmp_path / "out", DATASHEET, DICTIONARY, 6)
    assert again["output_sha256"] == first


def test_verify_accepts_own_build_and_flags_a_changed_file(tmp_path: Path) -> None:
    _, out, _ = build_toy(tmp_path)
    published = tmp_path / "published.sha256"
    published.write_text((out / release.MANIFEST).read_text())
    report = verify.verify(out, published)
    assert all(report["gate_10_verify_conditions"].values()), report
    assert report["quickstart"]["lines"] <= 5
    (out / "LICENSE").write_text("changed\n")
    report = verify.verify(out, published)
    assert report["manifest"]["differing_files"] == ["LICENSE"]
    assert not report["gate_10_verify_conditions"]["release_matches_published_manifest"]


def test_parquet_diff_locates_the_differing_column(tmp_path: Path) -> None:
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    pl.DataFrame({"x": [1.0, 2.0], "s": ["p", "q"]}).write_parquet(a)
    pl.DataFrame({"x": [1.0, 2.0 + 1e-15], "s": ["p", "q"]}).write_parquet(b)
    diff = verify.parquet_diff(a, b)
    assert set(diff) == {"x"} and 0 < diff["x"] < 1e-14


def test_pyg_dataset_and_split_indices(tmp_path: Path) -> None:
    pytest.importorskip("torch_geometric")
    from qm9_curation.export.pyg import QM9Curated

    _, out, _ = build_toy(tmp_path)
    ds = QM9Curated(tmp_path / "pyg", out, tiers=("gold",))
    assert len(ds) == 4
    assert ds[0].z.tolist() == [6, 1, 1, 1, 1] and tuple(ds[0].y.shape) == (1, 2)
    idx = ds.split_indices("scaffold_generic", seed=0)
    assert [len(idx[s]) for s in ("train", "val", "test")] == [2, 1, 1]
    assert len(QM9Curated(tmp_path / "pyg", out, tiers=("gold", "silver"))) == 5
    with pytest.raises(KeyError):
        ds.split_indices("random", seed=0)
