"""Spatial blocked k-fold CV + density subsampling. NEVER random CV (autocorrelation leaks)."""
from __future__ import annotations
import numpy as np


def density_subsample(n: int, pct: int, seed: int) -> np.ndarray:
    """Indices retained at the given density %. Deterministic per (n, pct, seed)."""
    rng = np.random.default_rng(seed)
    k = max(3, int(round(n * pct / 100.0)))
    return np.sort(rng.choice(n, size=min(k, n), replace=False))


def blocked_kfold(coords: np.ndarray, k: int, seed: int):
    """Yield (train_idx, test_idx) using spatial blocks. Falls back to verde if available,
    else a deterministic spatial-block partition by KMeans-like grid clustering.

    Blocks group nearby points together so a whole spatial neighbourhood is held out at once,
    preventing autocorrelated leakage across the train/test split.
    """
    n = len(coords)
    idx_all = np.arange(n)
    try:
        import verde as vd
        # square blocks sized so we get ~k spatially-coherent groups
        region = vd.get_region((coords[:, 0], coords[:, 1]))
        w = region[1] - region[0]
        h = region[3] - region[2]
        # spacing chosen so number of occupied blocks comfortably exceeds k
        spacing = max(w, h) / (2 * k + 1)
        kf = vd.BlockKFold(spacing=spacing, n_splits=k, shuffle=True, random_state=seed)
        for tr, te in kf.split(coords):
            yield idx_all[tr], idx_all[te]
        return
    except Exception:
        pass
    # Fallback: assign points to k spatial clusters, hold out one cluster per fold.
    from sklearn.cluster import KMeans
    labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(coords)
    for f in range(k):
        te = idx_all[labels == f]
        tr = idx_all[labels != f]
        if len(te) == 0 or len(tr) == 0:
            continue
        yield tr, te
