"""
data/preprocess.py — fold-level transforms + common 13-band base.

Contract (locked interfaces):
  fit_band_normalization(train_cities, raw_root)      -> BandNormStats
  build_dcorr13(city, raw_root, band_stats)           -> (dcorr13[H,W,13], labels[H,W], valid[H,W])
  fit_physical_transform(train_dcorr_px)              -> PhysicalTransform
  transform_physical4(dcorr13, tf)                    -> X4 in [0,1]      [H,W,4]
  fit_pca_transform(train_dcorr_px)                   -> PCATransform
  transform_pca4(dcorr13, tf)                         -> X4 in [-1,1]     [H,W,4]
  build_fold(train_cities, val_cities, raw_root, ...) -> FoldArtifacts

Data-object semantics:
  D_corr13 = |ΔB^corr|_13   -- common PHYSICAL base (unscaled magnitude)
  X_physical4 in [0,1]^4    -- model-ready (angle θ=π·X)
  X_pca4    in [-1,1]^4     -- model-ready, SIGNED (angle θ=π·X)
ZZ strength s_i is NOT stored; computed on demand by the transform helpers,
so X stays decoupled from any specific quantum ansatz.

Leakage discipline:
  - per-pair median correction is UNSUPERVISED and per-image -> applied to every
    city (train/val/test) inside build_dcorr13.
  - band P1/P99, physical c_b, PCA basis, c_k^PC, c_norm are fit on TRAIN cities
    ONLY (they live in FoldArtifacts.* fitted from train_cities).
"""
from dataclasses import dataclass, field
import os
import numpy as np
from PIL import Image

BANDS = ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B10","B11","B12"]
BI = {b: i for i, b in enumerate(BANDS)}
PHYS_BANDS = ["B04", "B05", "B12", "B08"]
PHYS_IDX = [BI[b] for b in PHYS_BANDS]

def _img_dir(raw_root):
    return os.path.join(raw_root, "images", "Onera Satellite Change Detection dataset - Images")
def _lbl_dir(raw_root):
    return os.path.join(raw_root, "train_labels", "Onera Satellite Change Detection dataset - Train Labels")

def _load_bands(raw_root, city, sub):
    d = _img_dir(raw_root)
    return np.stack([np.array(Image.open(os.path.join(d, city, sub, f"{b}.tif")), dtype=np.float32)
                     for b in BANDS], -1)

def _load_label(raw_root, city):
    # NOTE: TIF encodes {1=no change, 2=change}, NOT the README's {0,1}.
    p = os.path.join(_lbl_dir(raw_root), city, "cm", f"{city}-cm.tif")
    return (np.array(Image.open(p)) == 2).astype(np.int8)

# --------------------------------------------------------------------------- #
# transforms (all fitted on TRAIN cities only)
# --------------------------------------------------------------------------- #
@dataclass
class BandNormStats:
    p1: np.ndarray   # [13]
    p99: np.ndarray  # [13]
    def normalize(self, raw13):
        return np.clip((raw13 - self.p1) / (self.p99 - self.p1), 0.0, 1.0)

@dataclass
class PhysicalTransform:
    c_b: np.ndarray          # [4] P99 of |dB^corr| for physical bands (train)
    bands: list = field(default_factory=lambda: list(PHYS_BANDS))

@dataclass
class PCATransform:
    mean: np.ndarray         # [13]
    scale: np.ndarray        # [13] per-band std (standardize before PCA)
    components: np.ndarray   # [4,13]
    c_pc: np.ndarray         # [4] P99(|z_k|) on train  -> per-component angle scale
    c_norm: float            # P99(||z||_2) on train    -> ZZ strength scale
    def _scores(self, dcorr13):
        # dcorr13: [...,13] -> z: [...,4]
        z = ((dcorr13 - self.mean) / self.scale) @ self.components.T
        return z

def fit_band_normalization(train_cities, raw_root):
    pool = []
    for c in train_cities:
        pool.append(_load_bands(raw_root, c, "imgs_1_rect").reshape(-1, 13))
        pool.append(_load_bands(raw_root, c, "imgs_2_rect").reshape(-1, 13))
    pool = np.vstack(pool)
    return BandNormStats(p1=np.percentile(pool, 1, 0), p99=np.percentile(pool, 99, 0))

def build_dcorr13(city, raw_root, band_stats):
    """Common physical base for a city: |ΔB^corr|_13 (unscaled), labels, valid mask."""
    y = _load_label(raw_root, city)
    H, W = y.shape
    t1 = _load_bands(raw_root, city, "imgs_1_rect")[:H, :W]
    t2 = _load_bands(raw_root, city, "imgs_2_rect")[:H, :W]
    valid = ~(((np.concatenate([t1, t2], -1)) == 0).any(-1))
    d = band_stats.normalize(t2) - band_stats.normalize(t1)     # ΔB on [0,1] bands
    med = np.median(d[valid], 0)                                # per-image, unsupervised
    dcorr13 = np.abs(d - med)
    return dcorr13, y, valid

def fit_physical_transform(train_dcorr_px):
    """train_dcorr_px: [N,13] pooled train pixels of |ΔB^corr|."""
    c_b = np.percentile(train_dcorr_px[:, PHYS_IDX], 99, 0)
    return PhysicalTransform(c_b=c_b)

def transform_physical4(dcorr13, tf):
    return np.clip(dcorr13[..., PHYS_IDX] / tf.c_b, 0.0, 1.0)

def fit_pca_transform(train_dcorr_px, n_components=4):
    from sklearn.decomposition import PCA
    mean = train_dcorr_px.mean(0)
    scale = train_dcorr_px.std(0) + 1e-8
    Z = (train_dcorr_px - mean) / scale
    pca = PCA(n_components=n_components).fit(Z)
    comps = pca.components_                     # [4,13]
    scores = Z @ comps.T                        # [N,4]
    c_pc = np.percentile(np.abs(scores), 99, 0)                 # [4]
    c_norm = float(np.percentile(np.linalg.norm(scores, axis=1), 99))
    return PCATransform(mean=mean, scale=scale, components=comps, c_pc=c_pc, c_norm=c_norm)

def transform_pca4(dcorr13, tf):
    z = tf._scores(dcorr13)
    return np.clip(z / tf.c_pc, -1.0, 1.0)      # signed, [-1,1]

def pca_zz_strength(dcorr13, tf):
    """ZZ strength for PCA branch: s = clip(||z||_2 / c_norm, 0, 1)."""
    z = tf._scores(dcorr13)
    r = np.linalg.norm(z, axis=-1)
    return np.clip(r / tf.c_norm, 0.0, 1.0)

def physical_zz_strength(x4):
    """ZZ strength for physical branch, per stage: s^(1)=sqrt((B04^2+B05^2)/2),
    s^(2)=sqrt((B12^2+B08^2)/2). x4 columns = [B04,B05,B12,B08]. Returns (s1,s2)."""
    s1 = np.sqrt((x4[..., 0] ** 2 + x4[..., 1] ** 2) / 2.0)
    s2 = np.sqrt((x4[..., 2] ** 2 + x4[..., 3] ** 2) / 2.0)
    return s1, s2

# --------------------------------------------------------------------------- #
# fold assembly
# --------------------------------------------------------------------------- #
@dataclass
class FoldArtifacts:
    train_cities: list
    val_cities: list
    band_stats: BandNormStats
    physical_tf: PhysicalTransform
    pca_tf: PCATransform
    dcorr13: dict          # city -> [H,W,13]
    labels: dict           # city -> [H,W]
    valid: dict            # city -> [H,W] bool

def build_fold(train_cities, val_cities, raw_root, fit_subsample=200_000, seed=0):
    """Fit all train-only transforms and build the common base for every city."""
    rng = np.random.RandomState(seed)
    band_stats = fit_band_normalization(train_cities, raw_root)

    dcorr13, labels, valid = {}, {}, {}
    for c in list(train_cities) + list(val_cities):
        dcorr13[c], labels[c], valid[c] = build_dcorr13(c, raw_root, band_stats)

    # pool TRAIN pixels only, to fit physical + PCA scales
    px = []
    for c in train_cities:
        v = valid[c]
        D = dcorr13[c][v]
        if len(D) > fit_subsample:
            D = D[rng.choice(len(D), fit_subsample, replace=False)]
        px.append(D)
    train_px = np.vstack(px)

    physical_tf = fit_physical_transform(train_px)
    pca_tf = fit_pca_transform(train_px)
    return FoldArtifacts(train_cities, val_cities, band_stats, physical_tf, pca_tf,
                         dcorr13, labels, valid)

# --------------------------------------------------------------------------- #
# smoke test: invariants + train-only fitting
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse
    from splits import get_dev_split
    ap = argparse.ArgumentParser(); ap.add_argument("--data_dir", required=True)
    args = ap.parse_args()

    train, val = get_dev_split()
    print(f"dev split: train={len(train)} cities, val={val}")
    fold = build_fold(train, val, args.data_dir)

    c = val[0]
    D = fold.dcorr13[c]
    Xp = transform_physical4(D, fold.physical_tf)
    Xq = transform_pca4(D, fold.pca_tf)
    s_pca = pca_zz_strength(D, fold.pca_tf)
    print(f"\n[{c}]  D_corr13 {D.shape}   labels {fold.labels[c].shape}")
    print(f"  X_physical4 {Xp.shape}  range [{Xp.min():.3f}, {Xp.max():.3f}]  "
          f"-> in[0,1]? {Xp.min()>=0 and Xp.max()<=1}")
    print(f"  X_pca4      {Xq.shape}  range [{Xq.min():.3f}, {Xq.max():.3f}]  "
          f"-> in[-1,1]? {Xq.min()>=-1 and Xq.max()<=1}")
    print(f"  pca ZZ strength range [{s_pca.min():.3f}, {s_pca.max():.3f}] -> in[0,1]? "
          f"{s_pca.min()>=0 and s_pca.max()<=1}")

    # train-only fitting check: drop one val city INTO train -> transforms must change
    train2 = train + [val[0]]; val2 = val[1:]
    fold2 = build_fold(train2, val2, args.data_dir)
    dP1  = np.abs(fold.band_stats.p99 - fold2.band_stats.p99).max()
    dcb  = np.abs(fold.physical_tf.c_b - fold2.physical_tf.c_b).max()
    dpca = np.abs(np.abs(fold.pca_tf.components) - np.abs(fold2.pca_tf.components)).max()
    print(f"\ntrain-only fitting check (fold vs fold+1 train city):")
    print(f"  band P99 max|Δ| = {dP1:.2f}   (should be > 0)")
    print(f"  physical c_b max|Δ| = {dcb:.4f}   (should be > 0)")
    print(f"  PCA components max|Δ| = {dpca:.4f}   (should be > 0)")
    ok = (Xp.min()>=0 and Xp.max()<=1 and Xq.min()>=-1 and Xq.max()<=1
          and D.shape[-1]==13 and dcb>0 and dpca>0)
    print(f"\nSMOKE TEST: {'PASS' if ok else 'FAIL'}")
