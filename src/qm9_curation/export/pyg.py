"""PyTorch Geometric loader for the release (optional extra: ``uv sync --extra ml --extra pyg``).

Usage::

    from qm9_curation.export.pyg import QM9Curated

    ds = QM9Curated("pyg_cache", "data/release/qm9-curated-v1.0.0", tiers=("gold",))
    idx = ds.split_indices("scaffold_generic", seed=0)   # {"train": ..., "val": ..., "test": ...}
    train = ds[idx["train"]]

Each graph has ``z`` (atomic numbers), ``pos`` (Å), ``y`` (one row, the
requested targets in release units) and ``qm9_id``. Bonds are not stored:
build them with ``torch_geometric.transforms.RadiusGraph`` or use a model
that computes its own neighbours. The release Parquet files are read in
place; only the processed cache is written under ``root``, in a file named
after the tiers and targets, so different selections never collide.
"""

import hashlib
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch

with warnings.catch_warnings():
    # torch_geometric 2.8 still calls the deprecated torch.jit.script on import.
    warnings.filterwarnings("ignore", message=r".*torch\.jit\.script.*", category=FutureWarning)
    from torch_geometric.data import Data, InMemoryDataset

DEFAULT_TARGETS = ("gap_ev", "u0_atomization_ev")


class QM9Curated(InMemoryDataset):  # type: ignore[misc]
    def __init__(
        self,
        root: str | Path,
        release_dir: str | Path,
        tiers: Sequence[str] = ("gold",),
        targets: Sequence[str] = DEFAULT_TARGETS,
        transform: Callable[[Data], Data] | None = None,
        pre_transform: Callable[[Data], Data] | None = None,
        force_reload: bool = False,
    ) -> None:
        self.release_dir = Path(release_dir)
        self.tiers, self.targets = tuple(tiers), tuple(targets)
        tag = hashlib.sha256(repr((self.tiers, self.targets)).encode()).hexdigest()[:12]
        self._processed_name = f"qm9_curated_{tag}.pt"
        super().__init__(str(root), transform, pre_transform, force_reload=force_reload)
        self.load(self.processed_paths[0])
        self.qm9_ids = self._data.qm9_id.numpy()

    @property
    def raw_dir(self) -> str:
        return str(self.release_dir)

    @property
    def raw_file_names(self) -> list[str]:
        return ["molecules.parquet", "atoms.parquet", "splits.parquet"]

    @property
    def processed_file_names(self) -> list[str]:
        return [self._processed_name]

    def download(self) -> None:
        raise FileNotFoundError(
            f"Release files not found in {self.release_dir}; download the release first."
        )

    def process(self) -> None:
        mol = (
            pl.read_parquet(Path(self.raw_dir) / "molecules.parquet")
            .filter(pl.col("quality_tier").is_in(self.tiers))
            .select("qm9_id", *self.targets)
            .sort("qm9_id")
        )
        atoms = (
            pl.read_parquet(Path(self.raw_dir) / "atoms.parquet")
            .join(mol.select("qm9_id"), on="qm9_id", how="semi")
            .sort("qm9_id", "atom_idx")
        )
        ids = atoms["qm9_id"].to_numpy()
        z = torch.from_numpy(atoms["atomic_number"].to_numpy().astype(np.int64, copy=True))
        pos = torch.from_numpy(atoms.select("x_ang", "y_ang", "z_ang").to_numpy().copy())
        y = torch.from_numpy(mol.select(*self.targets).to_numpy().copy())
        starts = np.searchsorted(ids, mol["qm9_id"].to_numpy(), side="left")
        ends = np.searchsorted(ids, mol["qm9_id"].to_numpy(), side="right")
        graphs = []
        for i, (qid, a, b) in enumerate(zip(mol["qm9_id"].to_list(), starts, ends, strict=True)):
            g = Data(z=z[a:b], pos=pos[a:b].float(), y=y[i : i + 1].float(), qm9_id=int(qid))
            graphs.append(self.pre_transform(g) if self.pre_transform else g)
        self.save(graphs, self.processed_paths[0])

    def split_indices(self, split: str, seed: int = 0) -> dict[str, torch.Tensor]:
        """Dataset positions of the train / val / test molecules of a release split."""
        assigned = pl.read_parquet(Path(self.raw_dir) / "splits.parquet").filter(
            pl.col("split_name") == split, pl.col("seed") == seed
        )
        if assigned.is_empty():
            raise KeyError(f"No split {split!r} with seed {seed} in the release")
        order = pl.DataFrame({"qm9_id": self.qm9_ids, "pos": np.arange(len(self.qm9_ids))})
        rows = order.join(assigned.select("qm9_id", "subset"), on="qm9_id", how="inner")
        out: dict[str, Any] = {}
        for subset in ("train", "val", "test"):
            pos = rows.filter(pl.col("subset") == subset)["pos"].sort().to_numpy()
            out[subset] = torch.from_numpy(pos.astype(np.int64))
        return out
