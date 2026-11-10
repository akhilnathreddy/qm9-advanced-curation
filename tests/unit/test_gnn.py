import numpy as np
import polars as pl
import pytest

torch = pytest.importorskip("torch")

from qm9_curation.benchmark import gnn  # noqa: E402


def toy_graphs() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Water (O-H, O-H) and methanol heavy+H graph (C-O, C-H x3, O-H)."""
    nodes = []
    for qid, zs in ((1, [8, 1, 1]), (2, [6, 8, 1, 1, 1, 1])):
        for k, z in enumerate(zs):
            nodes.append(
                {
                    "qm9_id": qid,
                    "atom_idx": k,
                    "atomic_number": z,
                    "degree": 1,
                    "total_h": 0,
                    "formal_charge": 0,
                    "hybridization": 3,
                    "is_aromatic": False,
                    "in_ring": False,
                }
            )
    bonds = [(1, 0, 1), (1, 0, 2), (2, 0, 1), (2, 0, 2), (2, 0, 3), (2, 0, 4), (2, 1, 5)]
    edges = [
        {
            "qm9_id": q,
            "atom_i": i,
            "atom_j": j,
            "bond_order": 1.0,
            "is_aromatic": False,
            "is_conjugated": False,
            "in_ring": False,
            "length_ang": 1.0,
        }
        for q, i, j in bonds
    ]
    return pl.DataFrame(nodes), pl.DataFrame(edges)


def toy_atoms(n_mol: int = 6, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for q in range(1, n_mol + 1):
        n = int(rng.integers(3, 7))
        for k in range(n):
            x, y, z = rng.normal(scale=1.2, size=3)
            rows.append(
                {
                    "qm9_id": q,
                    "atom_idx": k,
                    "atomic_number": int(rng.choice([1, 6, 7, 8])),
                    "x_ang": x,
                    "y_ang": y,
                    "z_ang": z,
                }
            )
    return pl.DataFrame(rows)


def test_pack_graphs_reverse_edges() -> None:
    p = gnn.pack_graphs(*toy_graphs())
    assert p.ids.tolist() == [1, 2]
    assert p.atom_x.shape == (9, gnn.N_ATOM_FEATURES)
    assert p.edge_ptr is not None and p.edge_ptr.tolist() == [0, 4, 14]
    b = gnn.batch(p, np.array([0, 1]), torch.device("cpu"))
    src, dst, rev = b["src"], b["dst"], b["rev"]
    assert torch.equal(src[rev], dst) and torch.equal(dst[rev], src)
    assert torch.equal(rev[rev], torch.arange(len(rev)))
    assert int(b["mol"].max()) == 1


def test_batch_order_matches_packed_order() -> None:
    p = gnn.pack_graphs(*toy_graphs())
    b = gnn.batch(p, np.array([1, 0]), torch.device("cpu"))
    assert b["mol"].tolist() == [0] * 6 + [1] * 3  # methanol first, then water
    assert b["src"].max() < 9


def test_dmpnn_is_invariant_to_batch_composition() -> None:
    torch.manual_seed(0)
    p = gnn.pack_graphs(*toy_graphs())
    model = gnn.DMPNN(hidden=16, depth=3, ffn_hidden=16).eval()
    with torch.no_grad():
        both = model(gnn.batch(p, np.array([0, 1]), torch.device("cpu")))
        alone = model(gnn.batch(p, np.array([1]), torch.device("cpu")))
    assert torch.allclose(both[1], alone[0], atol=1e-6)


def _rotate(atoms: pl.DataFrame, seed: int) -> pl.DataFrame:
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    xyz = atoms.select("x_ang", "y_ang", "z_ang").to_numpy() @ q.T + np.array([1.0, -2.0, 0.5])
    return atoms.with_columns(
        pl.Series("x_ang", xyz[:, 0]), pl.Series("y_ang", xyz[:, 1]), pl.Series("z_ang", xyz[:, 2])
    )


def test_schnet_is_rotation_translation_and_permutation_invariant() -> None:
    torch.manual_seed(0)
    atoms = toy_atoms()
    model = gnn.SchNet(hidden=16, filters=16, interactions=2, n_rbf=10, cutoff=5.0).eval()
    rows = np.arange(6)
    with torch.no_grad():
        base = model(gnn.batch(gnn.pack_atoms(atoms), rows, torch.device("cpu")))
        moved = model(gnn.batch(gnn.pack_atoms(_rotate(atoms, 1)), rows, torch.device("cpu")))
        shuffled = atoms.sample(fraction=1.0, shuffle=True, seed=3).with_columns(
            pl.int_range(pl.len()).over("qm9_id").alias("atom_idx")
        )
        perm = model(gnn.batch(gnn.pack_atoms(shuffled), rows, torch.device("cpu")))
    assert torch.allclose(base, moved, atol=1e-4)
    assert torch.allclose(base, perm, atol=1e-4)


@pytest.mark.parametrize("which", ["dmpnn", "schnet"])
def test_training_is_seeded_and_fits(which: str) -> None:
    if which == "dmpnn":
        p = gnn.pack_graphs(*toy_graphs())
        rows, y = np.array([0, 1]), np.array([1.0, -1.0])
        arch = {"hidden": 16, "depth": 2, "ffn_hidden": 16}
    else:
        p = gnn.pack_atoms(toy_atoms())
        rows, y = np.arange(6), np.linspace(-1, 1, 6)
        arch = {"hidden": 16, "filters": 16, "interactions": 2, "n_rbf": 10, "cutoff": 5.0}
    params = {
        "model": which,
        "device": "cpu",
        "arch": arch,
        "epochs": 40,
        "batch_size": 2,
        "warmup_epochs": 1,
        "lr_max": 1e-2,
        "lr_min": 1e-3,
    }
    a = gnn.train(p, rows, y, rows, y, params, mean_readout=False, seed=0)
    b = gnn.train(p, rows, y, rows, y, params, mean_readout=False, seed=0)
    assert np.allclose(a.predict(rows), b.predict(rows))
    assert a.history[-1]["train_mse_std"] < a.history[0]["train_mse_std"]
    assert np.abs(a.predict(rows) - y).mean() < np.abs(y - y.mean()).mean()
