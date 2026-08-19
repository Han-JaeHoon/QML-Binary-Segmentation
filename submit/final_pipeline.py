"""
Submission pipeline — independent of the ablation sweep.

Three stages, runnable separately so nothing here waits on the experiments:

  threshold  pool the OOF probability maps of the chosen (kind, depth) over the
             5 city-grouped folds and pick ONE global tau* = argmax F1.
             Every labelled pixel appears exactly once, so this is an honest
             out-of-fold choice — the threshold never sees the hidden test set.
  train      refit the transforms on ALL 14 labelled cities and train the chosen
             architecture on them with the frozen protocol.
  predict    run the final model over the 10 hidden-label cities and write the
             deliverable: a pixel-aligned uint8 {0,255} PNG per city, plus the
             probability map per city (needed later for any threshold-free
             metric — see train/score_hidden_cities.py). Masks go to
             results/submission/masks_<kind>_L<depth>/, so running a comparison
             architecture cannot overwrite the frozen submission in masks/.

The 10 test cities have imagery but no labels, so `dcorr13_unlabeled` mirrors
preprocess.build_dcorr13 while taking the raster shape from the image instead of
the label. The per-pair median correction is label-free and per-image, so it
applies to the test cities exactly as it does in training.

Leakage discipline: tau* comes from out-of-fold predictions only, and is FROZEN
before touching the test cities — it is never re-selected on test.
"""
import os, sys, json, time, argparse
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "data"))
sys.path.insert(0, os.path.join(ROOT, "models"))
sys.path.insert(0, os.path.join(ROOT, "train"))

import pennylane as qml
from pennylane import numpy as pnp
from splits import TRAIN_CITIES, TEST_CITIES
import preprocess as pre
from preprocess import (BANDS, fit_band_normalization, build_dcorr13,
                        fit_physical_transform, fit_pca_transform,
                        transform_pca4, pca_zz_strength)
from pools import build_center_pools, fit_global_hard_threshold
from sampler import SpatialPatchSampler
import qml as qmodels
from inference import predict_city_center, evaluate_predictions, save_mask_png
from trainer import make_batch

LR, BATCH, STEPS, EPOCHS, SEED, INFER_BATCH = 0.02, 32, 320, 50, 0, 4096
OUT = os.path.join(ROOT, "results", "submission")
SWEEP = os.path.join(ROOT, "results", "runs", "p3_topology")


# --------------------------------------------------------------------------- #
def dcorr13_unlabeled(city, raw_root, band_stats):
    """|dB^corr|_13 for a city with no ground truth (shape taken from imagery)."""
    t1 = pre._load_bands(raw_root, city, "imgs_1_rect")
    t2 = pre._load_bands(raw_root, city, "imgs_2_rect")
    H = min(t1.shape[0], t2.shape[0]); W = min(t1.shape[1], t2.shape[1])
    t1, t2 = t1[:H, :W], t2[:H, :W]
    valid = ~(((np.concatenate([t1, t2], -1)) == 0).any(-1))
    d = band_stats.normalize(t2) - band_stats.normalize(t1)
    return np.abs(d - np.median(d[valid], 0)), valid


def final_transforms(raw_root, cities=TRAIN_CITIES, subsample=200_000, seed=SEED):
    """Fit band norm + PCA on ALL labelled cities (final mode)."""
    rng = np.random.RandomState(seed)
    band = fit_band_normalization(cities, raw_root)
    D, Y, V = {}, {}, {}
    for c in cities:
        D[c], Y[c], V[c] = build_dcorr13(c, raw_root, band)
    px = []
    for c in cities:
        d = D[c][V[c]]
        if len(d) > subsample:
            d = d[rng.choice(len(d), subsample, replace=False)]
        px.append(d)
    px = np.vstack(px)
    return band, fit_physical_transform(px), fit_pca_transform(px), D, Y, V


# --------------------------------------------------------------------------- #
def cmd_threshold(a):
    """tau* from pooled out-of-fold predictions (every labelled pixel once)."""
    os.makedirs(OUT, exist_ok=True)
    P, Yv = [], []
    used = []
    for f in range(5):
        p = os.path.join(SWEEP, f"{a.kind}_L{a.depth}_fold{f}_maps.npz")
        if not os.path.exists(p):
            print(f"  missing {os.path.basename(p)} — run that cell first"); continue
        z = np.load(p); used.append(f)
        cities = sorted({k.rsplit("_", 1)[0] for k in z.files})
        for c in cities:
            m = z[f"{c}_valid"]
            P.append(z[f"{c}_P"][m].ravel()); Yv.append(z[f"{c}_Y"][m].ravel())
    if not P:
        sys.exit("no OOF maps found for this (kind, depth)")
    P, Yv = np.concatenate(P), (np.concatenate(Yv) > 0).astype(int)
    m = evaluate_predictions(P, Yv, select_threshold=True)
    rec = {"kind": a.kind, "depth": a.depth, "folds_used": used,
           "n_pixels": int(P.size), "prevalence": float(Yv.mean()),
           "tau_final": m["tau"], "oof_AP": m["AP"], "oof_F1": m["F1"],
           "oof_change_acc": m["change_acc"], "oof_nochange_acc": m["nochange_acc"],
           "oof_accuracy": m["accuracy"], "oof_roc_auc": m["roc_auc"]}
    out = os.path.join(OUT, f"threshold_{a.kind}_L{a.depth}.json")
    json.dump(rec, open(out, "w"), indent=2)
    print(f"OOF over folds {used}: {P.size:,} px, prevalence {Yv.mean():.4f}")
    print(f"  tau* = {m['tau']:.4f}   AP {m['AP']:.4f}  F1 {m['F1']:.4f}  "
          f"ChangeAcc {m['change_acc']:.3f}  NoChangeAcc {m['nochange_acc']:.3f}")
    print(f"  -> {out}")


def cmd_train(a):
    """Train the chosen architecture on ALL 14 labelled cities."""
    os.makedirs(OUT, exist_ok=True)
    spec = qmodels.ModelSpec(a.kind, a.depth, "untied", "center_mean")
    print(f"final training: {spec.label} | {spec.n_params} params | 14 cities", flush=True)
    band, phys_tf, pca_tf, D, Y, V = final_transforms(a.data_dir)

    # make_batch indexes X/S/labels directly, so the sampler only needs a
    # fold-shaped carrier for its constructor.
    from types import SimpleNamespace
    F = SimpleNamespace(dcorr13=D, labels=Y, valid=V,
                        pca_tf=pca_tf, physical_tf=phys_tf)
    X = {c: transform_pca4(D[c], pca_tf).astype(np.float64) for c in TRAIN_CITIES}
    S = {c: pca_zz_strength(D[c], pca_tf).astype(np.float64) for c in TRAIN_CITIES}

    T_global = fit_global_hard_threshold(TRAIN_CITIES, D, Y, V)
    pools = {c: build_center_pools(D[c], Y[c], V[c], T_global) for c in TRAIN_CITIES}
    smp = SpatialPatchSampler(TRAIN_CITIES, pools, F, "pca", seed=SEED)

    forward = qmodels.build_model(spec)
    params = qmodels.init_params(spec, seed=SEED)
    opt = qml.AdamOptimizer(LR); rng = np.random.RandomState(SEED)
    t0 = time.time(); trace = []
    for epoch in range(1, EPOCHS + 1):
        losses = []
        for _ in range(STEPS):
            Xb, Sb, Yb, _ = make_batch(smp, X, S, Y, BATCH, rng, "center_mean")
            cost = lambda p: qmodels.bce_loss(p, Xb, Sb, Yb, forward)
            params, L = opt.step_and_cost(cost, params); losses.append(float(L))
        trace.append({"epoch": epoch, "train_BCE": float(np.mean(losses)),
                      "wall_time": time.time() - t0})
        if epoch % 10 == 0:
            print(f"  ep {epoch:3d}  BCE {trace[-1]['train_BCE']:.4f}  "
                  f"{trace[-1]['wall_time']:.0f}s", flush=True)

    tag = f"final_{a.kind}_L{a.depth}"
    np.savez_compressed(os.path.join(OUT, tag + "_model.npz"),
                        params=np.asarray(params),
                        band_p1=band.p1, band_p99=band.p99,
                        pca_mean=pca_tf.mean, pca_scale=pca_tf.scale,
                        pca_components=pca_tf.components, pca_c_pc=pca_tf.c_pc,
                        pca_c_norm=np.array([pca_tf.c_norm]))
    json.dump({"kind": a.kind, "depth": a.depth, "n_params": spec.n_params,
               "cities": TRAIN_CITIES, "trace": trace,
               "seconds": time.time() - t0}, open(os.path.join(OUT, tag + ".json"), "w"),
              indent=2)
    print(f"saved {tag}_model.npz  ({(time.time()-t0)/60:.0f} min)")


def cmd_predict(a):
    """Predict the 10 hidden cities and write uint8 {0,255} PNG masks."""
    tag = f"final_{a.kind}_L{a.depth}"
    mdl = np.load(os.path.join(OUT, tag + "_model.npz"))
    if a.tau is not None:
        # explicit tau: used to carry the frozen M1 operating point over to a
        # comparison architecture, which has no OOF threshold file of its own.
        tau = a.tau
    else:
        thr = json.load(open(os.path.join(OUT, f"threshold_{a.kind}_L{a.depth}.json")))
        tau = thr["tau_final"]
    print(f"predicting with FROZEN tau = {tau:.4f} (from OOF; never re-selected here)")

    band = pre.BandNormStats(p1=mdl["band_p1"], p99=mdl["band_p99"])
    pca_tf = pre.PCATransform(mean=mdl["pca_mean"], scale=mdl["pca_scale"],
                              components=mdl["pca_components"], c_pc=mdl["pca_c_pc"],
                              c_norm=float(mdl["pca_c_norm"][0]))
    spec = qmodels.ModelSpec(a.kind, a.depth, "untied", "center_mean")
    forward = qmodels.build_model(spec); params = mdl["params"]

    # per-(kind, depth) directory so a comparison run cannot overwrite the frozen
    # submission in results/submission/masks/ (M1 L3, written before this flag).
    mdir = a.mask_dir or os.path.join(OUT, f"masks_{a.kind}_L{a.depth}")
    os.makedirs(mdir, exist_ok=True)
    summary = []
    for c in TEST_CITIES:
        t = time.time()
        D, valid = dcorr13_unlabeled(c, a.data_dir, band)
        X = transform_pca4(D, pca_tf).astype(np.float64)
        S = pca_zz_strength(D, pca_tf).astype(np.float64)
        P = predict_city_center(forward, params, X, S, INFER_BATCH)
        png = os.path.join(mdir, f"{c}.png")
        mask = save_mask_png(P, tau, png)
        np.savez_compressed(os.path.join(mdir, f"{c}_prob.npz"), P=P.astype(np.float32),
                            valid=valid)
        frac = float((mask > 0).mean())
        summary.append({"city": c, "shape": list(mask.shape), "change_frac": frac,
                        "seconds": time.time() - t})
        print(f"  {c:12} {mask.shape}  change {100*frac:5.2f}%  ({time.time()-t:.0f}s)",
              flush=True)
    json.dump({"tau": tau, "kind": a.kind, "depth": a.depth,
               "mask_dir": os.path.relpath(mdir, ROOT), "cities": summary},
              open(os.path.join(OUT, f"predict_{a.kind}_L{a.depth}.json"), "w"), indent=2)

    # deliverable validation
    ok = True
    for s in summary:
        m = np.array(Image.open(os.path.join(mdir, f"{s['city']}.png")))
        ok &= (m.dtype == np.uint8 and set(np.unique(m)).issubset({0, 255})
               and list(m.shape) == s["shape"])
    print(f"\nvalidation: 10 masks, uint8, values subset {{0,255}}, shapes match  "
          f"{'OK' if ok else 'FAIL'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("threshold", "train", "predict"):
        s = sub.add_parser(name)
        s.add_argument("--kind", required=True)
        s.add_argument("--depth", type=int, required=True)
        if name != "threshold":
            s.add_argument("--data_dir", required=True)
        if name == "predict":
            s.add_argument("--tau", type=float, default=None,
                           help="frozen operating point; default: tau_final from the "
                                "OOF threshold file for this (kind, depth)")
            s.add_argument("--mask_dir", default=None,
                           help="output dir; default results/submission/masks_<kind>_L<depth>")
    a = ap.parse_args()
    {"threshold": cmd_threshold, "train": cmd_train, "predict": cmd_predict}[a.cmd](a)
