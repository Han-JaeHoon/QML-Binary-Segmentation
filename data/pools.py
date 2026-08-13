"""
data/pools.py — representation-INDEPENDENT center-coordinate pools.

Three disjoint pools per city, defined only by the common base D_corr13 and the
label, so Physical-4 and PCA-4 see EXACTLY the same training coordinates:

  positive           : eligible & y==1
  hard_negative      : eligible & y==0 & h_base > max(Q80_city, T_global)
  ordinary_negative  : eligible & y==0 & NOT hard        (= eligible neg minus hard)

  eligible = valid & interior   (interior keeps a full 3x3 patch: margin = 1)
  h_base(p) = mean over {B04,B05,B12,B08} of |ΔB^corr|(p)   (pre-clip magnitude)

T_global uses one vote per city: median over TRAIN cities of that city's
Q80(h_base | eligible, y==0). This module is fully deterministic; all
stochasticity lives in sampler.py.

Interfaces:
  compute_hardness(dcorr13) -> h_base[H,W]
  eligible_mask(valid, patch_size=3) -> mask[H,W]
  city_hard_quantile(dcorr13, labels, valid, q=0.8) -> float
  fit_global_hard_threshold(train_cities, dcorr13_by_city, labels_by_city, valid_by_city, q=0.8) -> float
  build_center_pools(dcorr13, labels, valid, T_global, city_quantile=0.8, patch_size=3) -> dict
"""
import numpy as np
from preprocess import PHYS_IDX

def compute_hardness(dcorr13):
    """h_base = mean of |ΔB^corr| over the 4 physical bands (unscaled)."""
    return dcorr13[..., PHYS_IDX].mean(-1)

def eligible_mask(valid, patch_size=3):
    """valid & interior; interior excludes a `margin`-wide border so every
    center admits a full patch_size x patch_size window."""
    m = patch_size // 2
    H, W = valid.shape
    interior = np.zeros_like(valid, dtype=bool)
    interior[m:H - m, m:W - m] = True
    return valid & interior

def city_hard_quantile(dcorr13, labels, valid, q=0.8, patch_size=3):
    """Q_q of h_base over this city's eligible, no-change pixels."""
    elig = eligible_mask(valid, patch_size)
    neg = elig & (labels == 0)
    h = compute_hardness(dcorr13)
    return float(np.quantile(h[neg], q)) if neg.any() else float("nan")

def fit_global_hard_threshold(train_cities, dcorr13_by_city, labels_by_city,
                              valid_by_city, q=0.8, patch_size=3):
    """One vote per city: median over TRAIN cities of Q_q(h_base | eligible neg)."""
    qs = [city_hard_quantile(dcorr13_by_city[c], labels_by_city[c],
                             valid_by_city[c], q, patch_size) for c in train_cities]
    return float(np.median(qs))

def build_center_pools(dcorr13, labels, valid, T_global, city_quantile=0.8, patch_size=3):
    """Partition eligible centers into positive / hard_negative / ordinary_negative."""
    elig = eligible_mask(valid, patch_size)
    h = compute_hardness(dcorr13)
    pos = elig & (labels == 1)
    neg = elig & (labels == 0)
    q_city = float(np.quantile(h[neg], city_quantile)) if neg.any() else float("inf")
    thr = max(q_city, T_global)
    hard = neg & (h > thr)
    ordinary = neg & ~hard
    return {
        "positive": np.argwhere(pos),
        "hard_negative": np.argwhere(hard),
        "ordinary_negative": np.argwhere(ordinary),
        "_q_city": q_city, "_threshold": thr,
    }

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse
    from splits import get_dev_split
    from preprocess import build_fold
    ap = argparse.ArgumentParser(); ap.add_argument("--data_dir", required=True)
    args = ap.parse_args()

    train, val = get_dev_split()
    fold = build_fold(train, val, args.data_dir)
    T_global = fit_global_hard_threshold(train, fold.dcorr13, fold.labels, fold.valid)
    print(f"T_global (median of per-city Q80) = {T_global:.4f}\n")

    hdr = f"{'city':12} {'elig':>8} {'pos':>7} {'neg':>8} {'Q80':>7} {'thr':>7} {'hard':>7} {'hard%':>6} {'ordin':>8}"
    print(hdr); print("-" * len(hdr))
    hard_sizes_train = []
    for c in train + val:
        pools = build_center_pools(fold.dcorr13[c], fold.labels[c], fold.valid[c], T_global)
        npos = len(pools["positive"]); nhard = len(pools["hard_negative"])
        nord = len(pools["ordinary_negative"]); nneg = nhard + nord
        tag = "" if c in train else " (val)"
        hp = 100 * nhard / nneg if nneg else 0.0
        print(f"{c:12} {npos+nneg:8d} {npos:7d} {nneg:8d} {pools['_q_city']:7.3f} "
              f"{pools['_threshold']:7.3f} {nhard:7d} {hp:6.1f} {nord:8d}{tag}")
        if c in train:
            hard_sizes_train.append(nhard)

    hs = np.array(hard_sizes_train)
    print("\nsanity checks (train cities):")
    print(f"  positive pool present in ALL cities?  "
          f"{all(len(build_center_pools(fold.dcorr13[c],fold.labels[c],fold.valid[c],T_global)['positive'])>0 for c in train)}")
    print(f"  hard-neg pool size  min/median/max = {hs.min()}/{int(np.median(hs))}/{hs.max()}")
    print(f"  cities with hard-neg < 50: {int((hs<50).sum())} / {len(hs)}")
