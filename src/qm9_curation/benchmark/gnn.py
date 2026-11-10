"""Graph neural networks of baseline levels 4 and 5, in plain PyTorch.

* Level 4 — **D-MPNN**, the directed message-passing network of Chemprop
  (Yang et al., *J. Chem. Inf. Model.* 2019), on the Phase 6 graph tables
  (all-atom graphs perceived from the geometry).
* Level 5 — **SchNet** (Schütt et al., *J. Chem. Phys.* 2018) on the
  coordinates, all atom pairs within a cosine cutoff.

Both are re-implementations (ADR 0013): one featurization path (Phase 6), no
Lightning / PyG dependency, and seeded, deterministic CPU execution for tests.
Molecules are packed into flat arrays once (:class:`Packed`); a batch is a
vectorized slice, so the data path costs little next to the model.

The readout is a **sum** over atoms for extensive targets (energies) and a
**mean** for intensive ones (gap), as configured per target.
"""

import math
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
import torch
from torch import nn

ELEMENTS = (1, 6, 7, 8, 9)
N_ATOM_FEATURES = len(ELEMENTS) + 6 + 5 + 3 + 5 + 2  # Z, degree, H, charge, hybrid, flags
N_BOND_FEATURES = 4 + 3  # order (1, 1.5, 2, 3), aromatic, conjugated, ring


def _one_hot(values: np.ndarray, choices: tuple[float, ...]) -> np.ndarray:
    """One-hot; values outside ``choices`` map to the last column."""
    out = np.zeros((len(values), len(choices)), dtype=np.float32)
    idx = np.full(len(values), len(choices) - 1)
    for k, c in enumerate(choices):
        idx[values == c] = k
    out[np.arange(len(values)), idx] = 1.0
    return out


def atom_features(nodes: pl.DataFrame) -> np.ndarray:
    return np.hstack(
        [
            _one_hot(nodes["atomic_number"].to_numpy(), ELEMENTS),
            _one_hot(nodes["degree"].to_numpy(), (0, 1, 2, 3, 4, 5)),
            _one_hot(nodes["total_h"].to_numpy(), (0, 1, 2, 3, 4)),
            _one_hot(nodes["formal_charge"].to_numpy(), (-1, 1, 0)),
            _one_hot(nodes["hybridization"].to_numpy(), (0, 1, 2, 3, 4)),
            nodes.select("is_aromatic", "in_ring").to_numpy().astype(np.float32),
        ]
    )


def bond_features(edges: pl.DataFrame) -> np.ndarray:
    return np.hstack(
        [
            _one_hot(edges["bond_order"].to_numpy(), (1.0, 1.5, 2.0, 3.0)),
            edges.select("is_aromatic", "is_conjugated", "in_ring").to_numpy().astype(np.float32),
        ]
    )


@dataclass
class Packed:
    """Molecules packed into flat arrays, in ``ids`` order.

    Atoms of molecule k are rows ``atom_ptr[k]:atom_ptr[k+1]``; directed
    edges (graphs) are rows ``edge_ptr[k]:edge_ptr[k+1]`` with atom indices
    local to the molecule. ``rev`` gives each directed edge's reverse, also
    local to the molecule's edge block.
    """

    ids: np.ndarray
    atom_ptr: np.ndarray
    atom_x: np.ndarray  # (n_atoms, F) float32 — graph features, or Z for SchNet
    pos: np.ndarray | None  # (n_atoms, 3) float32 for SchNet
    edge_ptr: np.ndarray | None
    edge_src: np.ndarray | None
    edge_dst: np.ndarray | None
    edge_x: np.ndarray | None
    edge_rev: np.ndarray | None

    def index(self, ids: np.ndarray) -> np.ndarray:
        """Row positions of ``ids`` (all must be present)."""
        pos = np.searchsorted(self.ids, ids)
        if not np.array_equal(self.ids[np.minimum(pos, len(self.ids) - 1)], ids):
            raise KeyError("ids missing from the packed set")
        return pos


def _ptr(counts: np.ndarray) -> np.ndarray:
    return np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)


def pack_graphs(nodes: pl.DataFrame, edges: pl.DataFrame) -> Packed:
    nodes = nodes.sort("qm9_id", "atom_idx")
    edges = edges.sort("qm9_id", "atom_i", "atom_j")
    ids, n_atoms = np.unique(nodes["qm9_id"].to_numpy(), return_counts=True)
    e_counts = (
        pl.DataFrame({"qm9_id": ids})
        .join(edges.group_by("qm9_id").len(), on="qm9_id", how="left")
        .sort("qm9_id")["len"]
        .fill_null(0)
        .to_numpy()
    )
    bx = bond_features(edges)
    i = edges["atom_i"].to_numpy().astype(np.int64)
    j = edges["atom_j"].to_numpy().astype(np.int64)
    m = len(i)
    # Directed edges, per molecule: its forward block, then its reverse block.
    mol_of_edge = np.repeat(np.arange(len(ids)), e_counts)
    order = np.argsort(np.concatenate([mol_of_edge, mol_of_edge]), kind="stable")
    src = np.concatenate([i, j])[order]
    dst = np.concatenate([j, i])[order]
    ex = np.concatenate([bx, bx])[order]
    edge_ptr = _ptr(2 * e_counts)
    # Reverse partner: row r (undirected bond b, direction d) <-> bond b, 1 - d.
    pair = np.concatenate([np.arange(m), np.arange(m)])[order]
    direction = np.concatenate([np.zeros(m, int), np.ones(m, int)])[order]
    lookup = np.empty((m, 2), dtype=np.int64)
    lookup[pair, direction] = np.arange(2 * m)
    rev_global = lookup[pair, 1 - direction]
    rev = rev_global - np.repeat(edge_ptr[:-1], 2 * e_counts)
    return Packed(
        ids=ids,
        atom_ptr=_ptr(n_atoms),
        atom_x=atom_features(nodes),
        pos=None,
        edge_ptr=edge_ptr,
        edge_src=src,
        edge_dst=dst,
        edge_x=ex,
        edge_rev=rev,
    )


def pack_atoms(atoms: pl.DataFrame) -> Packed:
    atoms = atoms.sort("qm9_id", "atom_idx")
    ids, n_atoms = np.unique(atoms["qm9_id"].to_numpy(), return_counts=True)
    return Packed(
        ids=ids,
        atom_ptr=_ptr(n_atoms),
        atom_x=atoms["atomic_number"].to_numpy().astype(np.int64)[:, None],
        pos=atoms.select("x_ang", "y_ang", "z_ang").to_numpy().astype(np.float32),
        edge_ptr=None,
        edge_src=None,
        edge_dst=None,
        edge_x=None,
        edge_rev=None,
    )


def _gather(ptr: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Concatenated ranges ptr[r]:ptr[r+1] for r in rows; and the offset of each item."""
    starts, ends = ptr[rows], ptr[rows + 1]
    counts = ends - starts
    offsets = _ptr(counts)
    flat = np.repeat(starts - offsets[:-1], counts) + np.arange(offsets[-1])
    return flat, offsets


_PAIRS: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _pairs(n: int) -> tuple[np.ndarray, np.ndarray]:
    if n not in _PAIRS:
        i, j = np.where(~np.eye(n, dtype=bool))
        _PAIRS[n] = (i, j)
    return _PAIRS[n]


def batch(packed: Packed, rows: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    """Collate molecules ``rows`` (positions in ``packed``) into one disjoint graph."""
    atoms, a_off = _gather(packed.atom_ptr, rows)
    n_atoms = np.diff(a_off)
    out: dict[str, Any] = {
        "x": packed.atom_x[atoms],
        "mol": np.repeat(np.arange(len(rows)), n_atoms),
        "n_mol": len(rows),
    }
    if packed.edge_ptr is not None:
        assert packed.edge_src is not None and packed.edge_dst is not None
        assert packed.edge_x is not None and packed.edge_rev is not None
        edges, e_off = _gather(packed.edge_ptr, rows)
        shift_a = np.repeat(a_off[:-1], np.diff(e_off))
        shift_e = np.repeat(e_off[:-1], np.diff(e_off))
        out["src"] = packed.edge_src[edges] + shift_a
        out["dst"] = packed.edge_dst[edges] + shift_a
        out["edge_x"] = packed.edge_x[edges]
        out["rev"] = packed.edge_rev[edges] + shift_e
    if packed.pos is not None:
        out["pos"] = packed.pos[atoms]
        ii, jj = [], []
        for m, n in enumerate(n_atoms.tolist()):
            i, j = _pairs(n)
            ii.append(i + a_off[m])
            jj.append(j + a_off[m])
        out["pi"] = np.concatenate(ii)
        out["pj"] = np.concatenate(jj)
    tensors: dict[str, Any] = {}
    for k, v in out.items():
        if isinstance(v, int):
            tensors[k] = v
        elif v.dtype.kind in "iu":
            tensors[k] = torch.from_numpy(v.astype(np.int64)).to(device)
        else:
            tensors[k] = torch.from_numpy(v.astype(np.float32)).to(device)
    return tensors


def _readout(h: torch.Tensor, mol: torch.Tensor, n_mol: int, mean: bool) -> torch.Tensor:
    out = torch.zeros(n_mol, h.shape[1], device=h.device, dtype=h.dtype).index_add_(0, mol, h)
    if mean:
        count = torch.zeros(n_mol, device=h.device, dtype=h.dtype).index_add_(
            0, mol, torch.ones_like(mol, dtype=h.dtype)
        )
        out = out / count.clamp(min=1.0)[:, None]
    return out


class DMPNN(nn.Module):
    """Chemprop's bond-centred message passing (defaults: hidden 300, depth 3)."""

    def __init__(self, hidden: int = 300, depth: int = 3, ffn_hidden: int = 300, mean=False):  # type: ignore[no-untyped-def]
        super().__init__()
        self.depth, self.mean = depth, mean
        self.w_i = nn.Linear(N_ATOM_FEATURES + N_BOND_FEATURES, hidden, bias=False)
        self.w_h = nn.Linear(hidden, hidden, bias=False)
        self.w_o = nn.Linear(N_ATOM_FEATURES + hidden, hidden)
        self.ffn = nn.Sequential(nn.Linear(hidden, ffn_hidden), nn.ReLU(), nn.Linear(ffn_hidden, 1))

    def forward(self, b: dict[str, Any]) -> torch.Tensor:
        x, src, dst, rev = b["x"], b["src"], b["dst"], b["rev"]
        h0 = torch.relu(self.w_i(torch.cat([x[src], b["edge_x"]], dim=1)))
        h = h0
        for _ in range(self.depth - 1):
            incoming = torch.zeros(x.shape[0], h.shape[1], device=h.device).index_add_(0, dst, h)
            message = incoming[src] - h[rev]
            h = torch.relu(h0 + self.w_h(message))
        incoming = torch.zeros(x.shape[0], h.shape[1], device=h.device).index_add_(0, dst, h)
        atom = torch.relu(self.w_o(torch.cat([x, incoming], dim=1)))
        out: torch.Tensor = self.ffn(_readout(atom, b["mol"], b["n_mol"], self.mean)).squeeze(1)
        return out


class _ShiftedSoftplus(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.softplus(x) - math.log(2.0)


class _Interaction(nn.Module):
    def __init__(self, hidden: int, filters: int, n_rbf: int, cutoff: float):
        super().__init__()
        self.cutoff = cutoff
        self.filter = nn.Sequential(
            nn.Linear(n_rbf, filters), _ShiftedSoftplus(), nn.Linear(filters, filters)
        )
        self.lin1 = nn.Linear(hidden, filters, bias=False)
        self.lin2 = nn.Linear(filters, hidden)
        self.act = _ShiftedSoftplus()
        self.lin3 = nn.Linear(hidden, hidden)

    def forward(
        self,
        h: torch.Tensor,
        rbf: torch.Tensor,
        d: torch.Tensor,
        pi: torch.Tensor,
        pj: torch.Tensor,
    ) -> torch.Tensor:
        c = 0.5 * (torch.cos(d * math.pi / self.cutoff) + 1.0) * (d < self.cutoff)
        w = self.filter(rbf) * c[:, None]
        x = self.lin1(h)
        agg = torch.zeros_like(x).index_add_(0, pi, x[pj] * w)
        out: torch.Tensor = self.lin3(self.act(self.lin2(agg)))
        return out


class SchNet(nn.Module):
    """Continuous-filter convolutions (PyG defaults except the cutoff)."""

    def __init__(  # type: ignore[no-untyped-def]
        self, hidden=128, filters=128, interactions=6, n_rbf=50, cutoff=5.0, mean=False
    ):
        super().__init__()
        self.mean, self.cutoff = mean, cutoff
        self.embed = nn.Embedding(10, hidden)
        self.register_buffer("centers", torch.linspace(0.0, cutoff, n_rbf))
        self.gamma = 0.5 / float(cutoff / (n_rbf - 1)) ** 2
        self.blocks = nn.ModuleList(
            _Interaction(hidden, filters, n_rbf, cutoff) for _ in range(interactions)
        )
        self.out = nn.Sequential(nn.Linear(hidden, hidden // 2), _ShiftedSoftplus())
        self.final = nn.Linear(hidden // 2, 1)

    def forward(self, b: dict[str, Any]) -> torch.Tensor:
        z, pos, pi, pj = b["x"][:, 0], b["pos"], b["pi"], b["pj"]
        d = (pos[pi] - pos[pj]).norm(dim=1)
        centers: torch.Tensor = self.centers  # type: ignore[assignment]
        rbf = torch.exp(-self.gamma * (d[:, None] - centers[None, :]) ** 2)
        h = self.embed(z)
        for block in self.blocks:
            h = h + block(h, rbf, d, pi, pj)
        atom = self.final(self.out(h))
        out: torch.Tensor = _readout(atom, b["mol"], b["n_mol"], self.mean).squeeze(1)
        return out


MODELS: dict[str, Callable[..., nn.Module]] = {"dmpnn": DMPNN, "schnet": SchNet}


def _batches(rows: np.ndarray, size: int, rng: np.random.Generator | None) -> Iterator[np.ndarray]:
    order = rows if rng is None else rows[rng.permutation(len(rows))]
    for k in range(0, len(order), size):
        yield order[k : k + size]


def device_for(name: str) -> torch.device:
    if name == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(name)


@dataclass
class TrainResult:
    predict: Callable[[np.ndarray], np.ndarray]
    history: list[dict[str, float]]
    best_epoch: int
    seconds: float
    device: str


def train(
    packed: Packed,
    train_rows: np.ndarray,
    y_train: np.ndarray,
    val_rows: np.ndarray,
    y_val: np.ndarray,
    params: dict[str, Any],
    mean_readout: bool,
    seed: int,
) -> TrainResult:
    """Adam + linear warm-up and exponential decay (Chemprop's Noam-like schedule).

    The target is standardized on the training set; per-atom baselines are
    the caller's job (the target is a residual of level 1 for energies).
    The epoch with the lowest validation MAE is kept (early stopping on
    validation only).
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = device_for(str(params.get("device", "auto")))
    model = MODELS[params["model"]](**params.get("arch", {}), mean=mean_readout).to(device)
    mu, sd = float(y_train.mean()), float(y_train.std() or 1.0)
    yt = torch.tensor((y_train - mu) / sd, dtype=torch.float32)
    epochs, bs = int(params["epochs"]), int(params["batch_size"])
    lr_max, lr_min = float(params["lr_max"]), float(params["lr_min"])
    warmup = max(1, int(params.get("warmup_epochs", 2)))
    opt = torch.optim.Adam(model.parameters(), lr=lr_max)
    steps_per_epoch = math.ceil(len(train_rows) / bs)
    decay = (lr_min / lr_max) ** (1.0 / max(1, (epochs - warmup) * steps_per_epoch))

    def predict(rows: np.ndarray) -> np.ndarray:
        model.eval()
        outs = []
        with torch.no_grad():
            for r in _batches(rows, 4 * bs, None):
                outs.append(model(batch(packed, r, device)).cpu().numpy())
        model.train()
        return np.concatenate(outs) * sd + mu if outs else np.zeros(0)

    best, best_epoch, best_state, history = math.inf, -1, None, []
    start, step = time.time(), 0
    for epoch in range(epochs):
        total = 0.0
        for k in _batches(np.arange(len(train_rows)), bs, rng):
            r = train_rows[k]
            if step < warmup * steps_per_epoch:
                lr = lr_max * (step + 1) / (warmup * steps_per_epoch)
            else:
                lr = lr_max * decay ** (step - warmup * steps_per_epoch)
            for g in opt.param_groups:
                g["lr"] = lr
            target = yt[torch.from_numpy(k)].to(device)
            loss = nn.functional.mse_loss(model(batch(packed, r, device)), target)
            opt.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            opt.step()
            total += float(loss.detach()) * len(r)
            step += 1
        val_mae = float(np.mean(np.abs(predict(val_rows) - y_val)))
        history.append(
            {"epoch": epoch, "train_mse_std": total / len(train_rows), "val_mae": val_mae}
        )
        if val_mae < best:
            best, best_epoch = val_mae, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    return TrainResult(predict, history, best_epoch, time.time() - start, str(device))
