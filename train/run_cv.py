"""
train/run_cv.py — paired 5-fold city-grouped CV for an M1-vs-M2 comparison.

Written for P2 (L=2 untied, 74 params) but the depth/tying are arguments, so the
identical protocol can be replayed at L=1 (38 params) if a P0 rerun is ever needed.

PROTOCOL (fixed here so it cannot drift between the two arms):
  * centre branch 3x3x4 -> 1, PCA-4 representation, plain BCE (w_pos = 1)
  * folds from splits.get_grouped_folds(5, seed=cv_seed) — city-grouped, every
    labelled city is held out exactly once
  * per fold ONE FoldArtifacts object is built and SHARED by both arms, so
    normalization / PCA / hard-negative threshold are fit on that fold's train
    cities only and are bit-identical for M1 and M2
  * both arms start from the SAME initial parameter vector and consume the SAME
    patch stream (same sampler seed, same rng); the per-epoch stream checksum is
    stored for both and asserted equal at the end of the fold
  * fixed 50 epochs x 320 steps x batch 32, Adam lr 0.02 — no early stopping, no
    checkpoint selection: the FINAL parameters are what gets evaluated
  * cheap validation is computed every `cheap_every` epochs as a DIAGNOSTIC only;
    it never selects a checkpoint, an architecture, or a budget
  * held-out evaluation is exhaustive (stride-1 over every held-out city) with a
    per-city threshold swept on that city — F1* is therefore a best-operating-point
    DIAGNOSTIC, not an unbiased test F1. AP is the primary metric.

The only difference between the two arms is the fixed CZ grid.

Everything is written incrementally: each fold's record lands in its own
fold<i>.json before the next fold starts, so an interrupted run loses at most one
fold and `--resume` skips finished ones. Because the records are per-fold, several
processes may run disjoint `--folds` concurrently on different cores — folds are
independent by construction (their own fold object, own seeds, own arms), so
splitting them changes no result. summary.json is a merge of whatever is done.

Usage
    python train/run_cv.py --data_dir /path/to/OneraDataset \
        --depth 2 --tying untied --epochs 50 --steps_per_epoch 320 \
        --tag p2_l2 [--folds 0,1] [--resume]

    # or split across cores (identical results, shorter wall-clock):
    for f in 0 1 2 3 4; do python train/run_cv.py --data_dir ... --folds $f & done
"""
import os, sys, json, time, zlib, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "data"))
sys.path.insert(0, os.path.join(ROOT, "models"))
sys.path.insert(0, HERE)

import pennylane as qml
from pennylane import numpy as pnp
from splits import get_grouped_folds
from preprocess import build_fold
from pools import build_center_pools, fit_global_hard_threshold
from sampler import SpatialPatchSampler
import qml as qmodels
from inference import (predict_city_center, predict_coordinates,
                       evaluate_predictions, make_fixed_val_coordinates)
from trainer import build_representation, make_batch

ARMS = ("m1", "m2")          # separable vs entangling; identical in every other way


def train_arm(kind, cfg, fold, pools, Xtr, Str, Xva, Sva, val_coords, val_y, log_path):
    """One arm of one fold. Returns (final_params, per_epoch_records)."""
    spec = qmodels.ModelSpec(kind, cfg.depth, cfg.tying, "center_mean")
    forward = qmodels.build_model(spec)

    # identical across arms: same seed -> same vector (init_params depends only on
    # n_params and the seed, and the two arms have the same n_params)
    params = qmodels.init_params(spec, seed=cfg.init_seed)
    opt = qml.AdamOptimizer(cfg.lr)

    # identical across arms: the sampler never sees the model, and the stream is
    # driven by this rng alone
    smp = SpatialPatchSampler(fold.train_cities, pools, fold, "pca", seed=cfg.stream_seed)
    rng = np.random.RandomState(cfg.stream_seed)

    recs, t0 = [], time.time()
    with open(log_path, "w") as f:
        f.write(json.dumps({"kind": kind, "n_params": spec.n_params,
                            "label": spec.label, "config": vars(cfg),
                            "train_cities": list(fold.train_cities),
                            "val_cities": list(fold.val_cities)}) + "\n")
        for epoch in range(1, cfg.epochs + 1):
            losses, epoch_idx = [], []
            for _ in range(cfg.steps_per_epoch):
                Xb, Sb, Yb, bidx = make_batch(smp, Xtr, Str, fold.labels,
                                              cfg.batch, rng, "center_mean")
                epoch_idx.extend(bidx)
                cost = lambda p: qmodels.bce_loss(p, Xb, Sb, Yb, forward, 1.0)
                params, L = opt.step_and_cost(cost, params)
                losses.append(float(L))
            rec = {"epoch": epoch, "train_BCE": float(np.mean(losses)),
                   "stream_checksum": zlib.crc32(repr(epoch_idx).encode()),
                   "param_norm": float(np.linalg.norm(np.asarray(params))),
                   "wall_time": time.time() - t0}
            if epoch % cfg.cheap_every == 0 or epoch == cfg.epochs:
                pn = np.asarray(params)
                ps = [predict_coordinates(forward, pn, Xva[c], Sva[c],
                                          val_coords[c], cfg.infer_batch)
                      for c in fold.val_cities]
                ys = [val_y[c] for c in fold.val_cities]
                cv = evaluate_predictions(np.concatenate(ps), np.concatenate(ys),
                                          select_threshold=True)
                rec["cheap_AP"] = cv["AP"]          # DIAGNOSTIC ONLY
                rec["cheap_F1"] = cv["F1"]
            recs.append(rec)
            f.write(json.dumps(rec) + "\n"); f.flush()
            print(f"    [{kind}] ep {epoch:3d}  BCE {rec['train_BCE']:.4f}"
                  + (f"  cheapAP {rec['cheap_AP']:.4f}" if "cheap_AP" in rec else "")
                  + f"  {rec['wall_time']:.0f}s", flush=True)
    return params, recs


def evaluate_arm(kind, cfg, params, fold, Xva, Sva):
    """Exhaustive stride-1 evaluation of the FINAL parameters on every held-out
    city of this fold. Returns {city: metrics} plus the pooled-pixel scores."""
    spec = qmodels.ModelSpec(kind, cfg.depth, cfg.tying, "center_mean")
    forward = qmodels.build_model(spec)
    pn = np.asarray(params)
    per_city, allp, ally = {}, [], []
    for c in fold.val_cities:
        t = time.time()
        P = predict_city_center(forward, pn, Xva[c], Sva[c], cfg.infer_batch)
        m = fold.valid[c]
        met = evaluate_predictions(P, fold.labels[c], select_threshold=True, mask=m)
        met["seconds"] = time.time() - t
        per_city[c] = met
        allp.append(P[m].ravel()); ally.append(fold.labels[c][m].ravel())
        print(f"    [{kind}] {c:11} AP {met['AP']:.4f}  F1* {met['F1']:.4f}  "
              f"chAcc {met['change_acc']:.3f}  ({met['seconds']:.0f}s)", flush=True)
    pooled = evaluate_predictions(np.concatenate(allp), np.concatenate(ally),
                                  select_threshold=True)
    return per_city, pooled, np.concatenate(allp), np.concatenate(ally)


def merge_summary(out, cfg):
    """Rebuild summary.json from the per-fold records on disk.

    Each fold writes its OWN fold{i}.json, so several processes can run
    different --folds concurrently without clobbering a shared file; the merge
    is just a read of whatever is finished."""
    summary = {"config": vars(cfg), "folds": {}}
    for f in sorted(os.listdir(out)):
        if f.startswith("fold") and f.endswith(".json") and f != "summary.json":
            fi = f[4:-5]
            if fi.isdigit():
                summary["folds"][fi] = json.load(open(os.path.join(out, f)))
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
    return summary


def run(cfg):
    out = os.path.join(cfg.out_dir, cfg.tag)
    os.makedirs(out, exist_ok=True)

    folds = get_grouped_folds(cfg.n_splits, seed=cfg.cv_seed)
    want = cfg.folds if cfg.folds else list(range(len(folds)))
    spec_ref = qmodels.ModelSpec("m1", cfg.depth, cfg.tying, "center_mean")
    print(f"=== paired CV | depth {cfg.depth} {cfg.tying} | {spec_ref.n_params} params "
          f"per arm | folds {want} -> {out}\n")

    for fi in want:
        fold_path = os.path.join(out, f"fold{fi}.json")
        if cfg.resume and os.path.exists(fold_path) and json.load(open(fold_path)).get("done"):
            print(f"--- fold {fi}: already complete, skipping (resume)\n"); continue
        train_cities, val_cities = folds[fi]
        print(f"--- fold {fi}  held-out: {val_cities}", flush=True)
        t_fold = time.time()

        # ONE fold object, shared by both arms: train-only normalization + PCA
        fold = build_fold(train_cities, val_cities, cfg.data_dir)
        T_global = fit_global_hard_threshold(train_cities, fold.dcorr13,
                                             fold.labels, fold.valid)
        pools = {c: build_center_pools(fold.dcorr13[c], fold.labels[c],
                                       fold.valid[c], T_global)
                 for c in train_cities}
        Xtr, Str = build_representation(fold, train_cities, "pca")
        Xva, Sva = build_representation(fold, val_cities, "pca")
        val_coords, val_y = {}, {}
        for c in val_cities:
            co = make_fixed_val_coordinates(fold.labels[c], fold.valid[c],
                                            n=cfg.cheap_val_per_city, seed=cfg.init_seed + 1)
            val_coords[c] = co
            val_y[c] = fold.labels[c][co[:, 0], co[:, 1]].astype(int)

        rec = {"train_cities": train_cities, "val_cities": val_cities,
               "T_global": float(T_global), "arms": {}}
        checksums, pooled_scores = {}, {}
        for kind in ARMS:
            log_path = os.path.join(out, f"fold{fi}_{kind}.jsonl")
            params, epochs = train_arm(kind, cfg, fold, pools, Xtr, Str,
                                       Xva, Sva, val_coords, val_y, log_path)
            np.save(os.path.join(out, f"fold{fi}_{kind}_final.npy"), np.asarray(params))
            per_city, pooled, p_all, y_all = evaluate_arm(kind, cfg, params, fold, Xva, Sva)
            checksums[kind] = [e["stream_checksum"] for e in epochs]
            pooled_scores[kind] = (p_all, y_all)
            rec["arms"][kind] = {
                "n_params": qmodels.ModelSpec(kind, cfg.depth, cfg.tying,
                                              "center_mean").n_params,
                "train_BCE_first": epochs[0]["train_BCE"],
                "train_BCE_final": epochs[-1]["train_BCE"],
                "cheap_AP_final": epochs[-1].get("cheap_AP"),
                "per_city": per_city, "pooled": pooled,
                "train_seconds": epochs[-1]["wall_time"]}

        # PAIRED-CONTROL ASSERT: identical patch stream in both arms, every epoch
        same = checksums["m1"] == checksums["m2"]
        rec["paired_stream_identical"] = bool(same)
        rec["delta_AP_per_city"] = {c: rec["arms"]["m2"]["per_city"][c]["AP"]
                                    - rec["arms"]["m1"]["per_city"][c]["AP"]
                                    for c in val_cities}
        rec["delta_AP_fold"] = (rec["arms"]["m2"]["pooled"]["AP"]
                                - rec["arms"]["m1"]["pooled"]["AP"])
        rec["fold_seconds"] = time.time() - t_fold
        rec["done"] = True
        if not same:
            rec["done"] = False
            print(f"!!! fold {fi}: stream checksums DIFFER between arms — "
                  f"the paired control is broken, results not comparable", flush=True)

        # OOF scores for the pooled-across-folds AP
        np.savez_compressed(os.path.join(out, f"fold{fi}_oof.npz"),
                            **{f"{k}_p": pooled_scores[k][0] for k in ARMS},
                            y=pooled_scores["m1"][1])

        json.dump(rec, open(fold_path, "w"), indent=1)
        merge_summary(out, cfg)
        print(f"--- fold {fi} done in {rec['fold_seconds']/60:.1f} min | "
              f"pooled AP  M1 {rec['arms']['m1']['pooled']['AP']:.4f}  "
              f"M2 {rec['arms']['m2']['pooled']['AP']:.4f}  "
              f"dAP {rec['delta_AP_fold']:+.4f} | paired stream {same}\n", flush=True)

    summary = merge_summary(out, cfg)
    print(f"summary -> {os.path.join(out, 'summary.json')} "
          f"({len(summary['folds'])} fold records)")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--tying", type=str, default="untied")
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps_per_epoch", type=int, default=320)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--cheap_every", type=int, default=5)
    ap.add_argument("--cheap_val_per_city", type=int, default=3000)
    ap.add_argument("--infer_batch", type=int, default=4096)
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--cv_seed", type=int, default=0)
    ap.add_argument("--init_seed", type=int, default=0)
    ap.add_argument("--stream_seed", type=int, default=0)
    ap.add_argument("--folds", type=str, default="")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--tag", type=str, default="p2_l2_cv")
    ap.add_argument("--out_dir", type=str, default=os.path.join(ROOT, "results", "runs"))
    cfg = ap.parse_args()
    cfg.folds = [int(x) for x in cfg.folds.split(",") if x != ""]
    assert cfg.tying == "untied" or cfg.depth == 1, \
        "P2 is untied by definition — tied L2 keeps 38 params and is a different experiment"
    run(cfg)
