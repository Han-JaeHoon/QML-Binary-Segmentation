"""
Onera Satellite Change Detection (OSCD) -- Exploratory Data Analysis
===================================================================

Reproducible EDA for the "Quantum Change Detection in Satellite Earth
Observations" challenge (2026 Niels Bohr Quantum Summer School).

Goal of this script: understand the multispectral data well enough to decide,
on a *data-driven* basis, which features should be fed into a (few-qubit)
Quantum Machine Learning model.

Pipeline of analyses (see README for full reasoning):
  Step 0  Data hygiene (no-data mask) + T1<->T2 global radiometric shift
  Step 1  Per-band relevance: AUC of signed dB vs magnitude |dB|
  Step 2  Redundancy: correlation matrix + clustering of |dB|
  Step 3  Spectral-index change separability (NDVI/NDWI/NDBI/NDMIR)
  Step 4  PCA (intrinsic dimensionality) + leakage-free multivariate AUC

Everything is computed on the 14 labelled TRAIN cities only. Normalization
statistics (per-band P1/P99) are estimated on TRAIN (T1+T2 pooled) and frozen.

Usage:
    python eda.py --data_dir /path/to/OneraDataset
Outputs: results/*.png figures and results/RESULTS.md summary tables.
"""
import argparse, os, json
import numpy as np
from PIL import Image

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
BANDS = ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B10","B11","B12"]
TRAIN = ["aguasclaras","bercy","bordeaux","nantes","paris","rennes","saclay_e",
         "abudhabi","cupertino","pisa","beihai","hongkong","beirut","mumbai"]
BI = {b:i for i,b in enumerate(BANDS)}
RNG = np.random.RandomState(0)

def paths(data_dir):
    img = os.path.join(data_dir, "images",
                       "Onera Satellite Change Detection dataset - Images")
    lbl = os.path.join(data_dir, "train_labels",
                       "Onera Satellite Change Detection dataset - Train Labels")
    return img, lbl

def load_bands(img_dir, city, sub):
    """Load the 13 resampled (10 m) bands for one city/date -> (H,W,13) float32."""
    return np.stack([np.array(Image.open(os.path.join(img_dir, city, sub, f"{b}.tif")),
                              dtype=np.float32) for b in BANDS], -1)

def load_label(lbl_dir, city):
    """Change mask. NOTE: the *.tif rasters use {1=no change, 2=change},
    NOT the {0,1} stated in the dataset README -- verified against cm.png."""
    a = np.array(Image.open(os.path.join(lbl_dir, city, "cm", f"{city}-cm.tif")))
    return (a == 2).astype(np.int8)

# ----------------------------------------------------------------------------
# Load everything once: global shift + balanced sampled pixel table
# ----------------------------------------------------------------------------
def build_table(data_dir):
    img_dir, lbl_dir = paths(data_dir)
    sum_d = np.zeros(13); n_valid = 0
    ch_tot = nch_tot = anyzero = total = 0
    X1_l, X2_l, Y_l, C_l = [], [], [], []
    for cid, c in enumerate(TRAIN):
        t1 = load_bands(img_dir, c, "imgs_1_rect")
        t2 = load_bands(img_dir, c, "imgs_2_rect")
        y  = load_label(lbl_dir, c)
        H, W = y.shape; t1 = t1[:H, :W]; t2 = t2[:H, :W]
        both = np.concatenate([t1, t2], -1)
        valid = ~((both == 0).any(-1))                 # Step 0: no-data mask
        total += valid.size; anyzero += (~valid).sum()
        sum_d += (t2 - t1)[valid].sum(0); n_valid += valid.sum()
        ch  = valid & (y == 1); nch = valid & (y == 0)
        ch_tot += ch.sum(); nch_tot += nch.sum()
        ci = np.argwhere(ch); ni = np.argwhere(nch); k = len(ci)
        # keep all change pixels, sample >= as many no-change (min 20k) per city
        sel = ni[RNG.choice(len(ni), min(len(ni), max(k, 20000)), replace=False)]
        idx = np.vstack([ci, sel])
        X1_l.append(t1[idx[:,0], idx[:,1]]); X2_l.append(t2[idx[:,0], idx[:,1]])
        Y_l.append(np.concatenate([np.ones(k), np.zeros(len(sel))]))
        C_l.append(np.full(len(idx), cid))
    tab = dict(
        X1=np.vstack(X1_l), X2=np.vstack(X2_l),
        Y=np.concatenate(Y_l).astype(np.int8), CID=np.concatenate(C_l).astype(np.int8),
        global_shift=sum_d / n_valid,
        stats=dict(total=int(total), anyzero=int(anyzero),
                   change=int(ch_tot), nochange=int(nch_tot),
                   change_frac=float(ch_tot/(ch_tot+nch_tot))))
    return tab

def robust_params(pool):
    """Per-band robust min-max params from TRAIN (T1+T2 pooled)."""
    return np.percentile(pool, 1, 0), np.percentile(pool, 99, 0)

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True,
                    help="Path to OneraDataset (containing images/ and train_labels/)")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score, GroupKFold
    from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
    from scipy.spatial.distance import squareform

    print("Loading 14 TRAIN cities ...")
    T = build_table(args.data_dir)
    X1, X2, Y, CID = T["X1"], T["X2"], T["Y"], T["CID"]
    P1, P99 = robust_params(np.vstack([X1, X2]))
    norm = lambda x: np.clip((x - P1) / (P99 - P1), 0, 1)
    n1, n2 = norm(X1), norm(X2)
    dB  = n2 - n1
    adB = np.abs(dB)

    md = ["# OSCD EDA -- Results\n",
          "_Auto-generated by `eda.py`. Reasoning in the top-level README._\n"]

    # ---- Step 0 -----------------------------------------------------------
    s = T["stats"]
    md += ["## Step 0 -- Data hygiene & global shift\n",
           f"- No-data pixels (any zero band): {s['anyzero']:,} / {s['total']:,} "
           f"({100*s['anyzero']/s['total']:.2f}%) -> negligible\n",
           f"- Label encoding: TIF is `{{1=no change, 2=change}}` (README says 0/1)\n",
           f"- Class imbalance: change = {s['change']:,} / {s['change']+s['nochange']:,} "
           f"= **{100*s['change_frac']:.2f}%**\n",
           "\n| band | " + " | ".join(BANDS) + " |\n|" + "---|"*(len(BANDS)+1) + "\n"
           "| T2-T1 shift (DN) | " + " | ".join(f"{g:+.0f}" for g in T["global_shift"]) + " |\n"]

    # ---- Step 1: per-band AUC --------------------------------------------
    rows = []
    for i, b in enumerate(BANDS):
        a_d = max(roc_auc_score(Y, dB[:, i]), 1 - roc_auc_score(Y, dB[:, i]))
        a_a = roc_auc_score(Y, adB[:, i])
        rows.append((b, a_d, a_a))
    md += ["\n## Step 1 -- Per-band relevance (AUC)\n",
           "| band | AUC(dB) | AUC(|dB|) |\n|---|---|---|\n"]
    for b, ad, aa in rows:
        md.append(f"| {b} | {ad:.3f} | {aa:.3f} |\n")

    order = sorted(rows, key=lambda r: -r[2])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([r[0] for r in order], [r[2] for r in order], color="#3a7ca5", label="|dB|")
    ax.bar([r[0] for r in order], [r[1] for r in order], color="#d1495b",
           alpha=.7, width=.5, label="signed dB")
    ax.axhline(.5, ls="--", c="k", lw=.8); ax.set_ylim(.45, .85)
    ax.set_ylabel("ROC-AUC (change vs no-change)")
    ax.set_title("Step 1: single-band change separability"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{args.out}/step1_band_auc.png", dpi=130); plt.close(fig)

    # dB vs |dB| histogram for the best band (B04)
    j = BI["B04"]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for lab, col, name in [(0, "#8ecae6", "no-change"), (1, "#d1495b", "change")]:
        ax[0].hist(dB[Y==lab, j], bins=100, range=(-.4,.4), density=True, alpha=.6, color=col, label=name)
        ax[1].hist(adB[Y==lab, j], bins=100, range=(0,.4), density=True, alpha=.6, color=col, label=name)
    ax[0].set_title("B04 signed dB (overlaps -> weak)"); ax[1].set_title("B04 |dB| (separates -> strong)")
    for a in ax: a.legend(); a.set_yticks([])
    fig.tight_layout(); fig.savefig(f"{args.out}/step1_B04_hist.png", dpi=130); plt.close(fig)

    # ---- Step 2: correlation + clustering --------------------------------
    C = np.corrcoef(adB.T)
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(C, cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(13)); ax.set_xticklabels(BANDS, rotation=90)
    ax.set_yticks(range(13)); ax.set_yticklabels(BANDS)
    ax.set_title("Step 2: |dB| correlation across bands")
    fig.colorbar(im, fraction=.046); fig.tight_layout()
    fig.savefig(f"{args.out}/step2_corr.png", dpi=130); plt.close(fig)

    Dm = 1 - np.abs(C); np.fill_diagonal(Dm, 0)
    Z = linkage(squareform(Dm, checks=False), method="average")
    groups = {}
    for b, g in zip(BANDS, fcluster(Z, t=0.2, criterion="distance")):
        groups.setdefault(int(g), []).append(b)
    md += ["\n## Step 2 -- Redundancy groups (|corr| > 0.8)\n"]
    for g, mem in sorted(groups.items()):
        md.append(f"- {mem}\n")

    # ---- Step 3: spectral indices ----------------------------------------
    def ndi(X, a, b): return (X[:,BI[a]] - X[:,BI[b]]) / (X[:,BI[a]] + X[:,BI[b]] + 1e-6)
    def indices(X): return {"NDVI": ndi(X,"B08","B04"), "NDWI": ndi(X,"B03","B08"),
                            "NDBI": ndi(X,"B11","B08"), "NDMIR": ndi(X,"B11","B12")}
    i1, i2 = indices(X1.astype(np.float64)), indices(X2.astype(np.float64))  # on raw DN
    md += ["\n## Step 3 -- Spectral-index change separability (computed on RAW DN)\n",
           "| index | AUC(dIdx) | AUC(|dIdx|) |\n|---|---|---|\n"]
    for k in i1:
        di = i2[k] - i1[k]
        a_d = max(roc_auc_score(Y, di), 1 - roc_auc_score(Y, di))
        a_a = roc_auc_score(Y, np.abs(di))
        md.append(f"| {k} | {a_d:.3f} | {a_a:.3f} |\n")

    # ---- Step 4: PCA + leakage-free multivariate AUC ---------------------
    Zs = (adB - adB.mean(0)) / adB.std(0)
    pca = PCA().fit(Zs); ev = pca.explained_variance_ratio_
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.bar(range(1, 14), ev, color="#3a7ca5", label="per-PC")
    ax.plot(range(1, 14), np.cumsum(ev), "o-", color="#d1495b", label="cumulative")
    ax.axhline(.9, ls="--", c="k", lw=.8); ax.set_xlabel("principal component")
    ax.set_ylabel("explained variance"); ax.set_title("Step 4: PCA of |dB| (13-dim)"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{args.out}/step4_pca.png", dpi=130); plt.close(fig)

    gkf = GroupKFold(5)
    def cvauc(est, F):
        return cross_val_score(est, F, Y, cv=gkf, groups=CID, scoring="roc_auc").mean()
    lr = LogisticRegression(max_iter=500)
    # NOTE: PCA/standardization must be fit INSIDE each fold to stay leakage-free;
    # a globally-fit PCA lets the basis see the validation city's distribution.
    pca_pipe = make_pipeline(StandardScaler(), PCA(4), LogisticRegression(max_iter=500))
    sets = {
        "all 13 |dB|": (lr, adB),
        "PCA top-4 (per-fold, leakage-free)": (pca_pipe, adB),
        "compact 4 (B04,B05,B12,B08)": (lr, adB[:, [BI[b] for b in ["B04","B05","B12","B08"]]]),
        "compact 3 (B04,B05,B12)": (lr, adB[:, [BI[b] for b in ["B04","B05","B12"]]]),
        "single |dB(B04)|": (lr, adB[:, [BI["B04"]]]),
    }
    md += ["\n## Step 4 -- Intrinsic dimensionality & leakage-free multivariate AUC\n",
           f"- PCA cumulative variance: PC1={ev[0]:.2f}, top-4={np.cumsum(ev)[3]:.2f}, "
           f"top-6={np.cumsum(ev)[5]:.2f}\n",
           "\n| feature set | grouped-CV AUC (group=city) |\n|---|---|\n"]
    res = {}
    for name, (est, F) in sets.items():
        res[name] = cvauc(est, F); md.append(f"| {name} | {res[name]:.3f} |\n")

    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.barh(list(res.keys())[::-1], list(res.values())[::-1], color="#3a7ca5")
    ax.set_xlim(.75, .84); ax.set_xlabel("leakage-free ROC-AUC")
    ax.set_title("Step 4: 13 bands vs compact feature sets"); fig.tight_layout()
    fig.savefig(f"{args.out}/step4_multivariate_auc.png", dpi=130); plt.close(fig)

    # persist frozen normalization constants for the downstream pipeline
    with open(f"{args.out}/norm_params.json", "w") as f:
        json.dump({"bands": BANDS, "P1": P1.tolist(), "P99": P99.tolist()}, f, indent=2)

    with open(f"{args.out}/RESULTS.md", "w") as f:
        f.writelines(md)
    print("Done. See", args.out)

if __name__ == "__main__":
    main()
