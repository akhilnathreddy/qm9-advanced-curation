from pathlib import Path

import numpy as np
import polars as pl
import pytest
from tests.unit.test_curate_standardize import ATOMREF
from tests.unit.test_external_kim2019 import MEMBERS, write_archive

from qm9_curation.checks import build as checks_build
from qm9_curation.config import load_config
from qm9_curation.curate.build import curated_frame
from qm9_curation.curate.standardize import identifiers_from_geometry
from qm9_curation.external import kim2019
from qm9_curation.features import build as feature_build
from qm9_curation.parse import build as parse_build
from qm9_curation.profiling.build import build_profile
from qm9_curation.review import build as review_build

pytestmark = pytest.mark.pipeline


def curated_fixture(
    qm9_1k_dir: Path, tmp_path: Path
) -> tuple[pl.DataFrame, dict[str, pl.DataFrame]]:
    frames, _ = parse_build.build(parse_build.iter_directory(qm9_1k_dir), "0" * 64)
    interim, raw = tmp_path / "interim", tmp_path / "raw"
    parse_build.write_tables(frames, interim)
    unchar = set(frames["molecules"]["qm9_id"].head(20).to_list())
    _, profile, _ = build_profile(
        frames["molecules"], frames["atoms"], frames["frequencies"], unchar
    )
    profile.write_parquet(interim / "molecule_profile.parquet")
    raw.mkdir()
    (raw / "uncharacterized.txt").write_text("".join(f"  {i}  X X X 1.0\n" for i in sorted(unchar)))
    validation, _ = checks_build.build(interim, raw, workers=2)
    ids = identifiers_from_geometry(frames["atoms"], workers=2)
    rules = load_config("curation_rules")["rules"]
    curated = curated_frame(
        frames["molecules"], frames["atoms"], validation, profile, ATOMREF, rules, ids
    )
    features, _ = feature_build.build(frames["atoms"], workers=2)
    return curated, features


def synthetic_kim(curated: pl.DataFrame, composition: pl.DataFrame, path: Path) -> Path:
    """G4(MP2)-like energies: per-element offsets plus noise; one changed, one unconverged."""
    rng = np.random.default_rng(7)
    ids = curated["qm9_id"].sort().to_list()
    changed, unconverged = ids[30], ids[40]
    offset = {"H": 0.0011, "C": -0.0502, "N": -0.0565, "O": -0.0636, "F": -0.0678}
    frame = curated.select("qm9_id", "source_u0_ha", "source_zpve_ha").join(
        composition, on="qm9_id"
    )
    rows = []
    for r in frame.iter_rows(named=True):
        if r["qm9_id"] == unconverged:
            continue
        chem = "".join(f"{e}{r[f'n_{e}']}" for e in ("C", "H", "N", "O", "F") if r[f"n_{e}"])
        shift = sum(r[f"n_{e}"] * v for e, v in offset.items())
        g4 = r["source_u0_ha"] - shift + 2e-3 * rng.standard_normal()
        b3 = r["source_u0_ha"] - r["source_zpve_ha"] - (5e-3 if r["qm9_id"] == changed else 0.0)
        rows.append((r["qm9_id"], chem, g4, b3, r["source_zpve_ha"]))
    fragmented = curated.filter(pl.col("identity_class") == "fragmented")["qm9_id"].to_list()
    return write_archive(
        path,
        rows,
        not_converged=[unconverged],
        multi_molecule=fragmented,
        duplicate_pairs=[(ids[1], ids[2])],
    )


def test_fixture_review_end_to_end(qm9_1k_dir: Path, tmp_path: Path) -> None:
    curated, features = curated_fixture(qm9_1k_dir, tmp_path)
    archive = synthetic_kim(curated, features["composition"], tmp_path / "kim.tar.gz")
    kim = kim2019.load(archive, MEMBERS)
    ext_cfg = load_config("external")["kim2019"]
    ext_cfg["expected"] = {
        "n_molecules": 999,
        "n_not_converged": 1,
        "n_multi_molecule": len(kim.multi_molecule),
        "n_duplicate_pairs": 1,
        "n_energy_change_above_1_kcal_mol": 1,
    }
    manifest = {"license": ext_cfg["license"], "files": ext_cfg["files"]}
    stat_cfg = load_config("statistical")
    args = (curated, features["composition"], features["ecfp4"], kim, stat_cfg, ext_cfg, [40])
    tables, report, info = review_build.build(*args)
    conditions = review_build.gate_conditions(
        tables["review"],
        curated,
        tables["g4mp2_xref"],
        features["composition"],
        kim,
        info,
        report,
        ext_cfg,
        manifest,
    )
    assert all(conditions.values()), conditions

    xref = tables["g4mp2_xref"]
    status = dict(xref.select("qm9_id", "g4mp2_status").iter_rows())
    ids = curated["qm9_id"].sort().to_list()
    assert status[ids[30]] == "geometry_changed" and status[ids[40]] == "not_converged"
    mf = report["multi_fidelity"]
    assert mf["invalid_reference_elements"] == ["N"]
    # The residual keeps only the injected noise (2 mHa ~ 0.054 eV), not the offsets.
    assert mf["residual_difference_gold"]["std"] == pytest.approx(0.054, rel=0.25)
    assert report["cross_checks"]["multi_molecule_vs_fragmented"]["identical_sets"]

    again, report_again, _ = review_build.build(*args)
    for name, frame in tables.items():
        assert frame.equals(again[name]), name  # deterministic
    assert report_again["suspicion"]["oof_mae"] == report["suspicion"]["oof_mae"]
