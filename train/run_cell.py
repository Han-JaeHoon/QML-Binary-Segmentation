"""
Run ONE (kind, depth, fold) cell of the topology x capacity ablation.

One process per fold makes the sweep parallel-safe and incrementally freezable:
each cell writes its own file the moment it finishes, so the study can be cut off
at any time and still be reported from whatever completed.

Protocol is byte-identical to P0/P2 — same folds (get_grouped_folds(5, seed=0)),
PCA-4 fit on each fold's TRAIN cities only, centre-only 3x3->1, city-uniform
sampler at P:H:O = 1:1:2, plain BCE, Adam lr 0.02, batch 32, 320 steps/epoch,
50 epochs fixed, FINAL checkpoint evaluated (validation never selects it),
cheap-val every 5 epochs as a diagnostic only.

Writes results/runs/p3_topology/<kind>_L<depth>_fold<f>.json  (+ _maps.npz):
per-city and pooled AP / ROC-AUC / F1* / ChangeAcc / NoChangeAcc / Accuracy /
prevalence / tau*, the train-BCE trajectory, cheap-val diagnostics, the final
parameters, and the per-epoch training-stream crc32 (P0 stored only a boolean,
so recording the values here lets any later arm verify it saw the same stream).
"""
import os, sys, json, time, zlib, argparse
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

LR, BATCH, STEPS, EPOCHS = 0.02, 32, 320, 50   # frozen in Step 2; do not tune
SEED, CHEAP_EVERY, CHEAP_PER_CITY, INFER_BATCH = 0, 5, 3000, 4096
OUTDIR = os.path.join(ROOT, "results", "runs", "p3_topology")


def main(a):
    os.makedirs(OUTDIR, exist_ok=True)
    tag = f"{a.kind}_L{a.depth}_fold{a.fold}"
    out = os.path.join(OUTDIR, tag + ".json")
    os.makedirs(OUTDIR, exist_ok=True)
    if os.path.exists(out) and not a.force:
        print(f"{tag}: already done, skipping"); return

    global EPOCHS, STEPS
    if a.smoke:                      # wiring test only: 1 epoch, 2 steps, 1 val city
        EPOCHS, STEPS = 1, 2
        out = out.replace(".json", "_smoke.json")
    spec = qmodels.ModelSpec(a.kind, a.depth, "untied", "center_mean")
    train_cities, val_cities = get_grouped_folds(5, seed=0)[a.fold]
    if a.smoke:
        val_cities = val_cities[:1]
    print(f"=== {tag} | {spec.n_params} params | val={val_cities}", flush=True)

    fold = build_fold(train_cities, val_cities, a.data_dir)
    T_global = fit_global_hard_threshold(train_cities, fold.dcorr13, fold.labels, fold.valid)
    pools = {c: build_center_pools(fold.dcorr13[c], fold.labels[c], fold.valid[c], T_global)
             for c in train_cities}
    smp = SpatialPatchSampler(train_cities, pools, fold, "pca", seed=SEED)
    Xtr, Str = build_representation(fold, train_cities, "pca")
    Xva, Sva = build_representation(fold, val_cities, "pca")
    val_ctx = {}
    for c in val_cities:
        co = make_fixed_val_coordinates(fold.labels[c], fold.valid[c],
                                        n=CHEAP_PER_CITY, seed=SEED + 1)
        val_ctx[c] = (co, fold.labels[c][co[:, 0], co[:, 1]].astype(int))

    forward = qmodels.build_model(spec)
    params = qmodels.init_params(spec, seed=SEED)
    init_crc = zlib.crc32(np.asarray(params).tobytes())
    opt = qml.AdamOptimizer(LR)
    rng = np.random.RandomState(SEED)

    trace, checksums = [], []
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        losses, idxs = [], []
        for _ in range(STEPS):
            Xb, Sb, Yb, bidx = make_batch(smp, Xtr, Str, fold.labels, BATCH, rng, "center_mean")
            idxs.extend(bidx)
            cost = lambda p: qmodels.bce_loss(p, Xb, Sb, Yb, forward)
            params, L = opt.step_and_cost(cost, params)
            losses.append(float(L))
        checksums.append(zlib.crc32(repr(idxs).encode()))
        g = np.asarray(qml.grad(cost)(params))
        rec = {"epoch": epoch, "train_BCE": float(np.mean(losses)),
               "grad_norm": float(np.linalg.norm(g)), "wall_time": time.time() - t0}
        if epoch % CHEAP_EVERY == 0 or epoch == 1:
            ps, ys = [], []
            for c, (co, y) in val_ctx.items():
                ps.append(predict_coordinates(forward, np.asarray(params), Xva[c], Sva[c],
                                              co, INFER_BATCH))
                ys.append(y)
            rec["cheap_AP"] = evaluate_predictions(np.concatenate(ps), np.concatenate(ys),
                                                   select_threshold=True)["AP"]
        trace.append(rec)
        if epoch % 10 == 0:
            print(f"  [{tag}] ep {epoch:3d}  BCE {rec['train_BCE']:.4f}  "
                  f"{rec['wall_time']:.0f}s", flush=True)

    # ---- exhaustive evaluation of the FINAL checkpoint --------------------
    pn = np.asarray(params)
    per_city, allp, ally, maps = {}, [], [], {}
    for c in val_cities:
        t = time.time()
        P = predict_city_center(forward, pn, Xva[c], Sva[c], INFER_BATCH)
        m = fold.valid[c]
        per_city[c] = evaluate_predictions(P, fold.labels[c], select_threshold=True, mask=m)
        per_city[c]["seconds"] = time.time() - t
        allp.append(P[m].ravel().astype(np.float32))
        ally.append(fold.labels[c][m].ravel().astype(np.int8))
        maps[f"{c}_P"] = P.astype(np.float32)
        maps[f"{c}_Y"] = fold.labels[c].astype(np.int8)
        maps[f"{c}_valid"] = m
        print(f"  [{tag}] {c:11} AP {per_city[c]['AP']:.4f} F1* {per_city[c]['F1']:.4f} "
              f"prev {per_city[c]['prevalence']:.4f} ({per_city[c]['seconds']:.0f}s)", flush=True)
    pooled = evaluate_predictions(np.concatenate(allp), np.concatenate(ally),
                                  select_threshold=True)

    # derive from `out` (not `tag`) so a --smoke run cannot overwrite a real one
    np.savez_compressed(out.replace(".json", "_maps.npz"), **maps)
    json.dump({
        "tag": tag, "kind": a.kind, "depth": a.depth, "fold": a.fold,
        "n_params": spec.n_params, "readout": "center_mean", "representation": "pca",
        "config": {"lr": LR, "batch": BATCH, "steps_per_epoch": STEPS, "epochs": EPOCHS,
                   "seed": SEED, "tying": "untied"},
        "train_cities": train_cities, "val_cities": val_cities,
        "init_crc32": init_crc, "stream_crc32": checksums,
        "trace": trace, "per_city": per_city, "pooled": pooled,
        "final_params": np.asarray(params).tolist(),
        "total_seconds": time.time() - t0,
    }, open(out, "w"), indent=2)
    print(f"{tag}: pooled AP {pooled['AP']:.4f}  ({(time.time()-t0)/60:.0f} min)  -> {out}",
          flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--kind", required=True)
    ap.add_argument("--depth", type=int, required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    main(ap.parse_args())
