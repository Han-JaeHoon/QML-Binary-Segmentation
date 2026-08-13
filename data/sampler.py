"""
data/sampler.py — city-balanced stochastic 3x3 patch sampler.

Internal separation (so Physical-4 and PCA-4 can reuse identical coordinates):
    sample_index()          -> (city, category, row, col)     [representation-free]
    get_patch(city,r,c,rep) -> (X[3,3,4], Y[3,3])
    sample_batch(B)         -> (X[B,3,3,4], Y[B,3,3])

Sampling order (locked):  city uniform -> category (1:1:2) -> coordinate (with
replacement). All stochasticity lives here; the sampler is deterministic under a
fixed seed. `estimate_pixel_prevalence` uses its OWN rng and only touches
coordinates + labels, so it never consumes the training rng nor runs transforms.

Representation transform (physical/pca) is applied to the whole city ONCE and
cached as H x W x 4; sampling then just crops 3x3.
"""
import numpy as np
from preprocess import transform_physical4, transform_pca4

CATS = ["positive", "hard_negative", "ordinary_negative"]

class SpatialPatchSampler:
    def __init__(self, cities, pools_by_city, fold, representation="physical",
                 category_probs=(0.25, 0.25, 0.50), patch_size=3, seed=0):
        assert representation in ("physical", "pca")
        assert abs(sum(category_probs) - 1.0) < 1e-9
        self.cities = list(cities)
        self.pools = pools_by_city
        self.fold = fold
        self.representation = representation
        self.probs = np.asarray(category_probs)
        self.ps = patch_size
        self.m = patch_size // 2
        self.rng = np.random.RandomState(seed)
        self._xcache = {}   # (city, rep) -> [H,W,4]
        # only keep cities that actually have all needed pools non-empty enough
        self.cities = [c for c in self.cities if len(pools_by_city[c]["positive"]) > 0]

    # -- representation transform, cached per city --------------------------- #
    def _X(self, city, rep):
        key = (city, rep)
        if key not in self._xcache:
            D = self.fold.dcorr13[city]
            self._xcache[key] = (transform_physical4(D, self.fold.physical_tf) if rep == "physical"
                                 else transform_pca4(D, self.fold.pca_tf))
        return self._xcache[key]

    # -- coordinate sampling (representation-free) --------------------------- #
    def sample_index(self, rng=None):
        rng = rng if rng is not None else self.rng
        while True:
            city = self.cities[rng.randint(len(self.cities))]
            cat = CATS[rng.choice(3, p=self.probs)]
            coords = self.pools[city][cat]
            if len(coords):
                r, c = coords[rng.randint(len(coords))]
                return city, cat, int(r), int(c)
            # empty category for this city -> redraw (city-uniform preserved)

    # -- patch extraction ---------------------------------------------------- #
    def get_patch(self, city, r, c, representation=None):
        rep = representation or self.representation
        m = self.m
        X = self._X(city, rep)[r - m:r + m + 1, c - m:c + m + 1, :]
        Y = self.fold.labels[city][r - m:r + m + 1, c - m:c + m + 1]
        return X, Y.astype(np.int8)

    def sample_batch(self, batch_size, representation=None, rng=None):
        Xs, Ys = [], []
        for _ in range(batch_size):
            city, _, r, c = self.sample_index(rng)
            X, Y = self.get_patch(city, r, c, representation)
            Xs.append(X); Ys.append(Y)
        return np.stack(Xs), np.stack(Ys)


def estimate_pixel_prevalence(sampler, n_patches=50000, seed=12345):
    """Diagnostics only. Uses its OWN rng (does not touch the training rng) and
    samples coordinates + labels only (no representation transform)."""
    rng = np.random.RandomState(seed)
    cat_count = {k: 0 for k in CATS}
    city_count = {c: 0 for c in sampler.cities}
    m = sampler.m
    n_center_pos = 0
    n_pos_pixels = 0
    for _ in range(n_patches):
        city, cat, r, c = sampler.sample_index(rng)
        cat_count[cat] += 1
        city_count[city] += 1
        Y = sampler.fold.labels[city][r - m:r + m + 1, c - m:c + m + 1]
        n_center_pos += int(Y[m, m] == 1)
        n_pos_pixels += int(Y.sum())
    n_pix = n_patches * sampler.ps * sampler.ps
    freqs = np.array([city_count[c] for c in sampler.cities]) / n_patches
    return {
        "n_patches": n_patches,
        "category_freq": {k: cat_count[k] / n_patches for k in CATS},
        "city_freq_min": float(freqs.min()), "city_freq_max": float(freqs.max()),
        "pi_center": n_center_pos / n_patches,
        "pi_pixel": n_pos_pixels / n_pix,
    }


if __name__ == "__main__":
    import argparse
    from splits import get_dev_split
    from preprocess import build_fold
    from pools import build_center_pools, fit_global_hard_threshold
    ap = argparse.ArgumentParser(); ap.add_argument("--data_dir", required=True)
    args = ap.parse_args()

    train, val = get_dev_split()
    fold = build_fold(train, val, args.data_dir)
    T_global = fit_global_hard_threshold(train, fold.dcorr13, fold.labels, fold.valid)
    pools = {c: build_center_pools(fold.dcorr13[c], fold.labels[c], fold.valid[c], T_global)
             for c in train}

    smp = SpatialPatchSampler(train, pools, fold, representation="physical", seed=42)
    d = estimate_pixel_prevalence(smp, 50000)
    print(f"n_patches: {d['n_patches']:,}\n")
    print("category:")
    for k in CATS: print(f"  {k:16} {100*d['category_freq'][k]:5.1f}%")
    print(f"\ncity sampling:  min {100*d['city_freq_min']:.1f}%   max {100*d['city_freq_max']:.1f}%"
          f"   (uniform target {100/len(train):.1f}%)")
    print(f"\ncenter positive prevalence (pi_center): {100*d['pi_center']:.1f}%   (design ~25%)")
    print(f"pixel  positive prevalence (pi_pixel) : {100*d['pi_pixel']:.1f}%   <-- BCE weight basis")

    # same-coordinates-for-both-representations check
    s_phys = SpatialPatchSampler(train, pools, fold, "physical", seed=7)
    s_pca  = SpatialPatchSampler(train, pools, fold, "pca",      seed=7)
    idx_p = [s_phys.sample_index() for _ in range(1000)]
    idx_q = [s_pca.sample_index()  for _ in range(1000)]
    same = all(a == b for a, b in zip(idx_p, idx_q))
    Xp, Yp = SpatialPatchSampler(train, pools, fold, "physical", seed=1).sample_batch(64)
    Xq, Yq = SpatialPatchSampler(train, pools, fold, "pca",      seed=1).sample_batch(64)
    print(f"\nsame coordinates for physical/pca under same seed? {same}")
    print(f"X physical: shape {Xp.shape}  range [{Xp.min():.3f}, {Xp.max():.3f}]")
    print(f"X pca     : shape {Xq.shape}  range [{Xq.min():.3f}, {Xq.max():.3f}]")
    print(f"Y         : shape {Yp.shape}  values {sorted(np.unique(Yp).tolist())}")
    # determinism
    a = SpatialPatchSampler(train, pools, fold, "physical", seed=99).sample_batch(16)[0]
    b = SpatialPatchSampler(train, pools, fold, "physical", seed=99).sample_batch(16)[0]
    print(f"deterministic under fixed seed? {np.array_equal(a, b)}")
