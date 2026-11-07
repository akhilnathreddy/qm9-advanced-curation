# Phase 6 — features

All tables are in `data/processed/features/` (62 MB) and join on `qm9_id`.
Every column is described in `reports/features/feature_report.json`
(`dictionary`, 248 entries). Decisions: ADR 0010.

| Table | Rows | Columns |
|---|---|---|
| index | 133,885 | rdkit_features_available, unavailable_reason |
| composition | 133,885 | 7 counts |
| coulomb_eigenvalues | 133,885 | 29 |
| rdkit_descriptors | 133,863 | 184 |
| ecfp4 | 133,863 | sparse bits + counts (2,048-bit space) |
| maccs | 133,863 | sparse bits (167 keys) |
| graph_nodes | 2,407,363 | 8 atom features |
| graph_edges | 2,488,018 | 6 bond features + length |

Example — dense ECFP4 matrix for the Gold tier:

```python
import numpy as np
import polars as pl
from scipy.sparse import csr_matrix

mol = pl.read_parquet("data/processed/molecules.parquet").filter(pl.col("quality_tier") == "gold")
fp = pl.read_parquet("data/processed/features/ecfp4.parquet").join(
    mol.select("qm9_id"), on="qm9_id"
)
rows = np.repeat(np.arange(fp.height), fp["bits"].list.len().to_numpy())
X = csr_matrix(
    (fp["counts"].explode().to_numpy(), (rows, fp["bits"].explode().to_numpy())),
    shape=(fp.height, 2048),
)
```
