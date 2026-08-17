"""
P0 — the core experiment: M1(38, separable) vs M2-CZ(38, entangling),
centre branch, PCA-4, 5-fold city-grouped CV.

Question: do explicit inter-pixel QUANTUM INTERACTIONS help generalization,
at an identical parameter budget and identical everything-else?

Controls (verified in-process, per fold):
  * the fold is built ONCE and shared, so both models get the same
    train-only-fitted band norm / PCA basis / scales
  * identical initial parameters (same seed, same 38 values)
  * identical training patch stream (per-epoch crc32 checksums compared)
  * identical optimizer, budget (50 epochs x 320 steps x 32), plain BCE
  => the ONLY difference is the CZ layer.

Frozen before running (see step2_budget.py): 50 epochs, lr 0.02, batch 32,
320 steps/epoch. Evaluation uses the FINAL checkpoint; validation metrics are
never used to select a checkpoint.

Primary architecture metric: AP. F1*/ChangeAcc/NoChangeAcc/Accuracy are recorded
as challenge/diagnostic metrics (F1* picks its threshold on the same data, so it
is an optimistic best-operating-point number).

Every city is a held-out validation city exactly once, so results are saved at
both fold level and CITY level. City-level paired differences are DESCRIPTIVE
only: cities inside one fold share a trained model, so they are not independent
samples and no p-value should be computed from them.
"""
import os, sys, json, time, zlib
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "data"))
sys.path.insert(0, os.path.join(ROOT, "models"))
sys.path.insert(0, HERE)

from splits import get_grouped_folds
from preprocess import build_fold
from pools import build_center_pools, fit_global_hard_threshold
from sampler import SpatialPatchSampler
import qml as qmodels
from inference import (predict_city_center, evaluate_predictions,
                       make_fixed_val_coordinates, predict_coordinates)
from trainer import build_representation, make_batch

# ---- frozen configuration --------------------------------------------------
LR, BATCH, STEPS, EPOCHS = 0.02, 32, 320, 50
SEED = 0
CHEAP_EVERY = 5              # diagnostic trace only; never used for selection
CHEAP_PER_CITY = 3000
INFER_BATCH = 4096
KINDS = ["m1", "m2"]
OUT = os.path.join(ROOT, "results", "runs", "p0_5fold.json")


def train_one(kind, fold, smp, Xtr, Str, val_ctx, seed=SEED):
    spec = qmodels.ModelSpec(kind, 1, "untied", "center_mean")
    forward = qmodels.build_model(spec)
    params = qmodels.init_params(spec, seed=seed)
    init_vec = np.asarray(params).copy()
    opt = qml.AdamOptimizer(LR)
    rng = np.random.RandomState(seed)
    trace, checksums = [], []
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        losses, idxs = [], []
        for _ in range(STEPS):
            Xb, Sb, Yb, bidx = make_batch(smp, Xtr, Str, fold.labels,
                                          BATCH, rng, "center_mean")
            idxs.extend(bidx)
            cost = lambda p: qmodels.bce_loss(p, Xb, Sb, Yb, forward)
            params, L = opt.step_and_cost(cost, params)
            losses.append(float(L))
        checksums.append(zlib.crc32(repr(idxs).encode()))
        g = np.asarray(qml.grad(cost)(params))
        rec = {"epoch": epoch, "train_BCE": float(np.mean(losses)),
               "grad_norm": float(np.linalg.norm(g)),
               "wall_time": time.time() - t0}
        if epoch % CHEAP_EVERY == 0 or epoch == 1:
            ps, ys = [], []
            for c, (co, y, Xv, Sv) in val_ctx.items():
                ps.append(predict_coordinates(forward, np.asarray(params), Xv, Sv, co, INFER_BATCH))
                ys.append(y)
            m = evaluate_predictions(np.concatenate(ps), np.concatenate(ys), select_threshold=True)
            rec["cheap_AP"] = m["AP"]
        trace.append(rec)
        if epoch % 10 == 0:
            print(f"      [{kind}] ep {epoch:3d}  BCE {rec['train_BCE']:.4f}"
                  f"{'  cheapAP %.4f' % rec['cheap_AP'] if 'cheap_AP' in rec else ''}"
                  f"  {rec['wall_time']:.0f}s", flush=True)
    return spec, forward, params, init_vec, trace, checksums


def main(data_dir):
    folds = get_grouped_folds(5, seed=0)
    results = {"config": {"lr": LR, "batch": BATCH, "steps_per_epoch": STEPS,
                          "epochs": EPOCHS, "seed": SEED, "readout": "center_mean",
                          "representation": "pca", "n_params": 38},
               "folds": []}
    t_start = time.time()

    for fi, (train_cities, val_cities) in enumerate(folds):
        print(f"\n===== FOLD {fi}  val={val_cities} =====", flush=True)
        # one fold object -> both models share the SAME train-only transforms
        fold = build_fold(train_cities, val_cities, data_dir)
        T_global = fit_global_hard_threshold(train_cities, fold.dcorr13,
                                             fold.labels, fold.valid)
        pools = {c: build_center_pools(fold.dcorr13[c], fold.labels[c],
                                       fold.valid[c], T_global) for c in train_cities}
        smp = SpatialPatchSampler(train_cities, pools, fold, "pca", seed=SEED)
        Xtr, Str = build_representation(fold, train_cities, "pca")
        Xva, Sva = build_representation(fold, val_cities, "pca")
        val_ctx = {}
        for c in val_cities:
            co = make_fixed_val_coordinates(fold.labels[c], fold.valid[c],
                                            n=CHEAP_PER_CITY, seed=SEED + 1)
            val_ctx[c] = (co, fold.labels[c][co[:, 0], co[:, 1]].astype(int), Xva[c], Sva[c])

        entry = {"fold": fi, "train_cities": train_cities, "val_cities": val_cities,
                 "models": {}}
        keep = {}
        for kind in KINDS:
            print(f"    training {kind} ...", flush=True)
            spec, fwd, params, init_vec, trace, cks = train_one(
                kind, fold, smp, Xtr, Str, val_ctx)
            keep[kind] = (spec, fwd, params, init_vec, cks)
            entry["models"][kind] = {"n_params": spec.n_params, "trace": trace}

        # ---- paired-condition verification -------------------------------
        (s1, f1, p1, i1, c1), (s2, f2, p2, i2, c2) = keep["m1"], keep["m2"]
        same_init = bool(np.array_equal(i1, i2))
        same_stream = bool(c1 == c2)
        entry["paired_check"] = {"same_init": same_init, "same_stream": same_stream}
        print(f"    paired check: same_init={same_init}  same_stream={same_stream}", flush=True)
        assert same_init and same_stream, "paired condition violated"

        # ---- exhaustive evaluation of the FINAL checkpoints ---------------
        for kind, (spec, fwd, params, _, _) in keep.items():
            per_city, allp, ally = {}, [], []
            for c in val_cities:
                t = time.time()
                P = predict_city_center(fwd, np.asarray(params), Xva[c], Sva[c], INFER_BATCH)
                m = fold.valid[c]
                per_city[c] = evaluate_predictions(P, fold.labels[c],
                                                   select_threshold=True, mask=m)
                per_city[c]["seconds"] = time.time() - t
                allp.append(P[m].ravel().astype(np.float32))
                ally.append(fold.labels[c][m].ravel().astype(np.int8))
                print(f"      [{kind}] {c:11} AP {per_city[c]['AP']:.4f} "
                      f"F1* {per_city[c]['F1']:.4f} prev {per_city[c]['prevalence']:.4f} "
                      f"({per_city[c]['seconds']:.0f}s)", flush=True)
            pooled = evaluate_predictions(np.concatenate(allp), np.concatenate(ally),
                                          select_threshold=True)
            entry["models"][kind]["per_city"] = per_city
            entry["models"][kind]["fold_pooled"] = pooled

        d = (entry["models"]["m2"]["fold_pooled"]["AP"]
             - entry["models"]["m1"]["fold_pooled"]["AP"])
        entry["delta_AP_fold"] = d
        print(f"    FOLD {fi}: AP m1 {entry['models']['m1']['fold_pooled']['AP']:.4f} "
              f"| m2 {entry['models']['m2']['fold_pooled']['AP']:.4f} "
              f"| dAP {d:+.4f}", flush=True)

        results["folds"].append(entry)
        with open(OUT, "w") as f:           # incremental save
            json.dump(results, f, indent=2)

    # ---- summary ---------------------------------------------------------
    dfolds = [e["delta_AP_fold"] for e in results["folds"]]
    city_d = {}
    for e in results["folds"]:
        for c in e["val_cities"]:
            city_d[c] = (e["models"]["m2"]["per_city"][c]["AP"]
                         - e["models"]["m1"]["per_city"][c]["AP"])
    results["summary"] = {
        "delta_AP_folds": dfolds,
        "mean_delta_AP": float(np.mean(dfolds)),
        "std_delta_AP": float(np.std(dfolds)),
        "fold_wins_m2": int(sum(d > 0 for d in dfolds)),
        "city_delta_AP": city_d,
        "city_wins_m2": int(sum(v > 0 for v in city_d.values())),
        "total_hours": (time.time() - t_start) / 3600.0,
    }
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n=== SUMMARY ===")
    print(f"dAP per fold : {[round(d,4) for d in dfolds]}")
    print(f"mean {np.mean(dfolds):+.4f}  std {np.std(dfolds):.4f}  "
          f"M2 wins {results['summary']['fold_wins_m2']}/5 folds, "
          f"{results['summary']['city_wins_m2']}/14 cities")
    print(f"saved {OUT}   ({results['summary']['total_hours']:.1f} h)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run (2 epochs, 5 steps, 1 fold) to validate wiring")
    a = ap.parse_args()
    if a.smoke:
        EPOCHS, STEPS, CHEAP_EVERY, CHEAP_PER_CITY = 2, 5, 1, 300
        OUT = OUT.replace(".json", "_smoke.json")
        _orig = get_grouped_folds
        get_grouped_folds = lambda n=5, seed=0: _orig(n, seed)[:1]
    main(a.data_dir)
