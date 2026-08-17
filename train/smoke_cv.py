"""
train/smoke_cv.py — end-to-end wiring smoke for train/run_cv.py WITHOUT the raw
dataset.

Why this exists: `build_fold` is the only part of the CV runner that needs the
OSCD GeoTIFFs. Everything after it (representation transforms, centre pools,
sampler, paired training loop, stream checksums, exhaustive centre inference,
metrics, incremental writes) is data-shape-driven, so it can be exercised on
synthetic cities. This makes it possible to validate the runner while the real
dataset is busy with another job — or absent.

It fabricates 14 tiny cities with a PLANTED, learnable signal (a positive centre
label is correlated with a bump in the first principal direction), so a falling
train BCE here proves the optimizer is wired to the loss — nothing more. No
number produced here is a scientific result.

Everything downstream of the fake `build_fold` is the REAL committed code path,
including `fit_pca_transform`, `build_center_pools`, `SpatialPatchSampler`, the
paired-stream checksum assert and `predict_city_center`.

    python train/smoke_cv.py            # ~2-3 min, prints per-epoch BCE
"""
import os, sys, json, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "data"))
sys.path.insert(0, os.path.join(ROOT, "models"))
sys.path.insert(0, HERE)

from preprocess import (FoldArtifacts, fit_pca_transform, fit_physical_transform)
import run_cv


def synthetic_build_fold(train_cities, val_cities, raw_root, **kw):
    """Stand-in for preprocess.build_fold: same FoldArtifacts contract, random
    rasters, planted signal. Sizes vary per city like the real dataset does."""
    rng = np.random.RandomState(abs(hash(tuple(train_cities))) % (2**31))
    dcorr13, labels, valid = {}, {}, {}
    for i, c in enumerate(list(train_cities) + list(val_cities)):
        H, W = 46 + (i % 5) * 6, 52 + (i % 3) * 8
        D = np.abs(rng.randn(H, W, 13) * 0.15) + 0.05
        # planted signal: ~12% of pixels get a broad-band bump AND label 1
        y = (rng.rand(H, W) < 0.12)
        D[y] += 0.55
        D = D + rng.randn(H, W, 13) * 0.03            # noise on top of the bump
        dcorr13[c] = D
        labels[c] = y.astype(np.uint8)
        v = np.ones((H, W), dtype=bool); v[0, :] = v[-1, :] = v[:, 0] = v[:, -1] = False
        valid[c] = v
    px = np.vstack([dcorr13[c][valid[c]] for c in train_cities])
    return FoldArtifacts(list(train_cities), list(val_cities), None,
                         fit_physical_transform(px), fit_pca_transform(px),
                         dcorr13, labels, valid)


# a real Namespace, so vars(cfg) serializes exactly like the argparse runner
Cfg = lambda: argparse.Namespace(
    data_dir="(synthetic)", depth=2, tying="untied",
    lr=0.05, batch=16, steps_per_epoch=25, epochs=6,
    cheap_every=2, cheap_val_per_city=400, infer_batch=2048,
    n_splits=5, cv_seed=0, init_seed=0, stream_seed=0,
    folds=[0], resume=False, tag="smoke_cv_l2",
    out_dir=os.path.join(ROOT, "results", "runs"))


if __name__ == "__main__":
    run_cv.build_fold = synthetic_build_fold          # the ONLY substitution
    cfg = Cfg()
    print("=== SMOKE (synthetic cities) — wiring only, results are meaningless ===\n")
    summary = run_cv.run(cfg)

    rec = summary["folds"]["0"]
    m1, m2 = rec["arms"]["m1"], rec["arms"]["m2"]
    log = lambda k: [json.loads(l) for l in
                     open(os.path.join(cfg.out_dir, cfg.tag, f"fold0_{k}.jsonl"))][1:]
    checks = []

    def chk(label, cond, detail=""):
        checks.append(bool(cond))
        print(f"{label}  {detail}{'  ' if detail else ''}{'OK' if cond else 'FAIL'}")

    print("\n--- smoke assertions ---")
    chk("[S1] both arms 74 params", m1["n_params"] == 74 and m2["n_params"] == 74,
        f"{m1['n_params']}/{m2['n_params']}")
    chk("[S2] paired patch stream identical every epoch", rec["paired_stream_identical"])
    b1 = [e["train_BCE"] for e in log("m1")]
    b2 = [e["train_BCE"] for e in log("m2")]
    chk("[S3] M1 train BCE falls", b1[-1] < b1[0], f"{b1[0]:.4f} -> {b1[-1]:.4f}")
    chk("[S4] M2 train BCE falls", b2[-1] < b2[0], f"{b2[0]:.4f} -> {b2[-1]:.4f}")
    chk("[S5] identical start (same init + same first batch)",
        abs(b1[0] - b2[0]) < 0.25, f"|dBCE_ep1| {abs(b1[0]-b2[0]):.4f}")
    chk("[S6] every held-out city evaluated with full metric set",
        all(set(("AP", "roc_auc", "F1", "change_acc", "nochange_acc", "accuracy",
                 "prevalence", "tau")) <= set(v) for v in m1["per_city"].values()),
        f"cities {list(m1['per_city'])}")
    chk("[S7] fold + per-city deltas recorded",
        "delta_AP_fold" in rec and len(rec["delta_AP_per_city"]) == len(rec["val_cities"]))
    chk("[S8] incremental artefacts on disk",
        all(os.path.exists(os.path.join(cfg.out_dir, cfg.tag, f))
            for f in ("summary.json", "fold0_m1_final.npy", "fold0_m2_final.npy",
                      "fold0_oof.npz")))
    print(f"\nBCE per epoch  M1 {[round(x,4) for x in b1]}")
    print(f"BCE per epoch  M2 {[round(x,4) for x in b2]}")
    print(f"\n{sum(checks)}/{len(checks)} smoke checks passed")
    sys.exit(0 if all(checks) else 1)
