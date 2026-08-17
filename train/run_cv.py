"""
train/run_cv.py — paired 5-fold city-grouped CV for an M1-vs-M2 comparison.

Written for P2 (L=2 untied, 74 params) but depth/tying are arguments, so the
identical protocol can be replayed at L=1 (38 params) if a P0 rerun is ever needed.

PROTOCOL (fixed here so it cannot drift between the two arms):
  * centre branch 3x3x4 -> 1, PCA-4 representation, plain BCE (w_pos = 1)
  * folds from splits.get_grouped_folds(5, seed=cv_seed) — city-grouped, every
    labelled city is held out exactly once
  * per fold ONE FoldArtifacts object is built and SHARED by both arms, so
    normalization / PCA / hard-negative threshold are fit on that fold's train
    cities only and are bit-identical for M1 and M2
  * both arms start from the SAME initial parameter vector and consume the SAME
    patch stream (same sampler seed, same rng); per-epoch stream checksums are
    stored for both arms and compared before the fold is marked done
  * fixed 50 epochs x 320 steps x batch 32, Adam lr 0.02 — no early stopping, no
    checkpoint selection: the FINAL parameters are what gets evaluated
  * cheap validation every `cheap_every` epochs is a DIAGNOSTIC only; it never
    selects a checkpoint, an architecture, or a budget
  * held-out evaluation is exhaustive (stride-1 over every held-out city) with a
    per-city threshold swept on that city — F1* is therefore a best-operating-point
    DIAGNOSTIC, not an unbiased test F1. AP is the primary metric.

The only difference between the two arms is the fixed CZ grid.

OUTPUT LAYOUT (see docs/results_schema.md for the field-by-field schema)

    results/runs/<tag>/
      meta.json                  run-level provenance: full config, git commit,
                                 package versions, fold definitions + their hash
      fold<i>.json               per-fold record: per-city and pooled metrics for
                                 both arms, dAP, checkpoint digests, checksum status
      fold<i>_<arm>.jsonl        per-epoch log: train BCE, stream checksum,
                                 param norm, cheap-val diagnostics
      fold<i>_<arm>_final.npy    final parameter vector (also embedded in fold<i>.json)
      fold<i>_maps.npz           per held-out city, FULL RESOLUTION: p_m1, p_m2
                                 (float32 H x W), y (uint8), valid (bool) — every
                                 held-out pixel's OOF probability and ground truth,
                                 addressable by (city, row, col)
      summary.json               merge of the fold records present
      REPORT.md                  human-readable summary (train/report_cv.py --md)

Each fold writes its own fold<i>.json, so disjoint `--folds` may run concurrently
in separate processes: folds are independent by construction (own fold object, own
seeds, own arms), so splitting them across cores changes no result, only wall-clock.

Usage
    python train/run_cv.py --print_folds            # fold assignment + hash, no data needed
    python train/run_cv.py --data_dir /path/to/OneraDataset \
        --depth 2 --tying untied --epochs 50 --steps_per_epoch 320 --tag p2_l2_cv
    # split across cores (identical results):
    for f in 0 1 2 3 4; do python train/run_cv.py --data_dir ... --folds $f & done
"""
import os, sys, json, time, zlib, hashlib, platform, subprocess, argparse
from datetime import datetime, timezone
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "data"))
sys.path.insert(0, os.path.join(ROOT, "models"))
sys.path.insert(0, HERE)

import pennylane as qml
from pennylane import numpy as pnp
from splits import get_grouped_folds, TRAIN_CITIES
from preprocess import build_fold
from pools import build_center_pools, fit_global_hard_threshold
from sampler import SpatialPatchSampler
import qml as qmodels
from inference import (predict_city_center, predict_coordinates,
                       evaluate_predictions, make_fixed_val_coordinates)
from trainer import build_representation, make_batch

ARMS = ("m1", "m2")          # separable vs entangling; identical in every other way
SCHEMA_VERSION = "cv-1.0"


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def fold_table(n_splits, cv_seed):
    """Fold assignment as data: city -> fold index, plus a hash of the whole
    assignment so a later run (or P0) can be checked for identity mechanically."""
    folds = get_grouped_folds(n_splits, seed=cv_seed)
    assign = {c: fi for fi, (_, val) in enumerate(folds) for c in val}
    canon = json.dumps({c: assign[c] for c in sorted(assign)}, sort_keys=True)
    return folds, assign, hashlib.sha256(canon.encode()).hexdigest()[:16]


def provenance(cfg):
    def sh(*a):
        try:
            return subprocess.check_output(a, cwd=ROOT, text=True).strip()
        except Exception:
            return None
    import sklearn, PIL
    return {"schema_version": SCHEMA_VERSION, "written_at": now(),
            "git_commit": sh("git", "rev-parse", "HEAD"),
            "git_dirty": bool(sh("git", "status", "--porcelain")),
            "host": platform.node(), "python": platform.python_version(),
            "numpy": np.__version__, "pennylane": qml.version(),
            "sklearn": sklearn.__version__, "pillow": PIL.__version__,
            "config": vars(cfg)}


def sha_of(arr):
    return hashlib.sha256(np.ascontiguousarray(arr, dtype=np.float64).tobytes()).hexdigest()[:16]


def train_arm(kind, cfg, fold, pools, Xtr, Str, Xva, Sva, val_coords, val_y, log_path):
    """One arm of one fold. Returns (final_params, per_epoch_records)."""
    spec = qmodels.ModelSpec(kind, cfg.depth, cfg.tying, "center_mean")
    forward = qmodels.build_model(spec)

    # identical across arms: same seed -> same vector (init_params depends only on
    # n_params and the seed, and the two arms have the same n_params)
    params = qmodels.init_params(spec, seed=cfg.init_seed)
    init_sha = sha_of(np.asarray(params))
    opt = qml.AdamOptimizer(cfg.lr)

    # identical across arms: the sampler never sees the model, and the stream is
    # driven by this rng alone
    smp = SpatialPatchSampler(fold.train_cities, pools, fold, "pca", seed=cfg.stream_seed)
    rng = np.random.RandomState(cfg.stream_seed)

    recs, t0 = [], time.time()
    with open(log_path, "w") as f:
        f.write(json.dumps({"record": "header", "kind": kind, "label": spec.label,
                            "n_params": spec.n_params, "init_sha256": init_sha,
                            "train_cities": list(fold.train_cities),
                            "val_cities": list(fold.val_cities),
                            "config": vars(cfg), "started_at": now()}) + "\n")
        for epoch in range(1, cfg.epochs + 1):
            losses, epoch_idx = [], []
            for _ in range(cfg.steps_per_epoch):
                Xb, Sb, Yb, bidx = make_batch(smp, Xtr, Str, fold.labels,
                                              cfg.batch, rng, "center_mean")
                epoch_idx.extend(bidx)
                cost = lambda p: qmodels.bce_loss(p, Xb, Sb, Yb, forward, 1.0)
                params, L = opt.step_and_cost(cost, params)
                losses.append(float(L))
            rec = {"record": "epoch", "epoch": epoch,
                   "train_BCE": float(np.mean(losses)),
                   "train_BCE_last_step": float(losses[-1]),
                   "stream_checksum": zlib.crc32(repr(epoch_idx).encode()),
                   "param_norm": float(np.linalg.norm(np.asarray(params))),
                   "wall_time": time.time() - t0}
            if epoch % cfg.cheap_every == 0 or epoch == cfg.epochs:
                pn = np.asarray(params)
                ps = [predict_coordinates(forward, pn, Xva[c], Sva[c],
                                          val_coords[c], cfg.infer_batch)
                      for c in fold.val_cities]
                cv = evaluate_predictions(np.concatenate(ps),
                                          np.concatenate([val_y[c] for c in fold.val_cities]),
                                          select_threshold=True)
                rec["cheap_AP"] = cv["AP"]          # DIAGNOSTIC ONLY — never selects
                rec["cheap_F1"] = cv["F1"]
            recs.append(rec)
            f.write(json.dumps(rec) + "\n"); f.flush()
            print(f"    [{kind}] ep {epoch:3d}  BCE {rec['train_BCE']:.4f}"
                  + (f"  cheapAP {rec['cheap_AP']:.4f}" if "cheap_AP" in rec else "")
                  + f"  {rec['wall_time']:.0f}s", flush=True)
        f.write(json.dumps({"record": "footer", "finished_at": now(),
                            "final_sha256": sha_of(np.asarray(params))}) + "\n")
    return params, recs


def evaluate_arm(kind, cfg, params, fold, Xva, Sva):
    """Exhaustive stride-1 evaluation of the FINAL parameters on every held-out
    city. Returns per-city metrics, pooled metrics, and the full-resolution
    probability maps (so every held-out pixel stays addressable)."""
    spec = qmodels.ModelSpec(kind, cfg.depth, cfg.tying, "center_mean")
    forward = qmodels.build_model(spec)
    pn = np.asarray(params)
    per_city, maps, allp, ally = {}, {}, [], []
    for c in fold.val_cities:
        t = time.time()
        P = predict_city_center(forward, pn, Xva[c], Sva[c], cfg.infer_batch)
        m = fold.valid[c]
        met = evaluate_predictions(P, fold.labels[c], select_threshold=True, mask=m)
        met["seconds"] = time.time() - t
        per_city[c] = met
        maps[c] = P.astype(np.float32)
        allp.append(P[m].ravel()); ally.append(fold.labels[c][m].ravel())
        print(f"    [{kind}] {c:11} AP {met['AP']:.4f}  F1* {met['F1']:.4f}  "
              f"chAcc {met['change_acc']:.3f}  ({met['seconds']:.0f}s)", flush=True)
    pooled = evaluate_predictions(np.concatenate(allp), np.concatenate(ally),
                                  select_threshold=True)
    return per_city, pooled, maps


def merge_summary(out, cfg):
    """Rebuild summary.json from the per-fold records on disk (parallel-safe)."""
    summary = {"schema_version": SCHEMA_VERSION, "config": vars(cfg),
               "merged_at": now(), "folds": {}}
    for f in sorted(os.listdir(out)):
        if f.startswith("fold") and f.endswith(".json") and f[4:-5].isdigit():
            summary["folds"][f[4:-5]] = json.load(open(os.path.join(out, f)))
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
    return summary


def run(cfg):
    out = os.path.join(cfg.out_dir, cfg.tag)
    os.makedirs(out, exist_ok=True)
    folds, assign, fold_hash = fold_table(cfg.n_splits, cfg.cv_seed)
    meta = provenance(cfg)
    # `folds` is a run-slicing detail, not a protocol parameter: drop it so every
    # process (one per fold) writes byte-identical meta.json and races are harmless
    meta["config"] = {k: v for k, v in meta["config"].items() if k != "folds"}
    meta["fold_assignment"] = {c: assign[c] for c in TRAIN_CITIES}
    meta["fold_assignment_sha256"] = fold_hash
    meta["folds"] = [{"fold": i, "train": t, "val": v} for i, (t, v) in enumerate(folds)]
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)

    want = cfg.folds if cfg.folds else list(range(len(folds)))
    spec_ref = qmodels.ModelSpec("m1", cfg.depth, cfg.tying, "center_mean")
    print(f"=== paired CV | depth {cfg.depth} {cfg.tying} | {spec_ref.n_params} params "
          f"per arm | folds {want} | fold-assignment sha {fold_hash} -> {out}\n")

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

        rec = {"schema_version": SCHEMA_VERSION, "fold": fi,
               "train_cities": train_cities, "val_cities": val_cities,
               "fold_assignment_sha256": fold_hash,
               "T_global": float(T_global), "started_at": now(), "arms": {}}
        checksums, all_maps = {}, {}
        for kind in ARMS:
            log_path = os.path.join(out, f"fold{fi}_{kind}.jsonl")
            params, epochs = train_arm(kind, cfg, fold, pools, Xtr, Str,
                                       Xva, Sva, val_coords, val_y, log_path)
            ckpt = os.path.join(out, f"fold{fi}_{kind}_final.npy")
            np.save(ckpt, np.asarray(params))
            per_city, pooled, maps = evaluate_arm(kind, cfg, params, fold, Xva, Sva)
            checksums[kind] = [e["stream_checksum"] for e in epochs]
            all_maps[kind] = maps
            rec["arms"][kind] = {
                "label": qmodels.ModelSpec(kind, cfg.depth, cfg.tying, "center_mean").label,
                "n_params": qmodels.ModelSpec(kind, cfg.depth, cfg.tying,
                                              "center_mean").n_params,
                # final checkpoint, fully recorded: file, digest, and the vector
                "final_checkpoint": {
                    "path": os.path.relpath(ckpt, ROOT),
                    "sha256": sha_of(np.asarray(params)),
                    "param_norm": float(np.linalg.norm(np.asarray(params))),
                    "params": [float(x) for x in np.asarray(params)]},
                "train_BCE": [e["train_BCE"] for e in epochs],
                "train_BCE_first": epochs[0]["train_BCE"],
                "train_BCE_final": epochs[-1]["train_BCE"],
                "cheap_AP_final": epochs[-1].get("cheap_AP"),
                "stream_checksums": checksums[kind],
                "epoch_log": os.path.relpath(log_path, ROOT),
                "per_city": per_city, "pooled": pooled,
                "train_seconds": epochs[-1]["wall_time"]}

        # PAIRED-CONTROL: identical patch stream in both arms, every epoch
        same = checksums["m1"] == checksums["m2"]
        rec["paired_stream_identical"] = bool(same)
        rec["paired_stream_first_mismatch_epoch"] = None if same else next(
            (i + 1 for i, (a, b) in enumerate(zip(checksums["m1"], checksums["m2"])) if a != b),
            min(len(checksums["m1"]), len(checksums["m2"])) + 1)
        rec["same_initialization"] = (rec["arms"]["m1"]["final_checkpoint"]["sha256"] !=
                                      rec["arms"]["m2"]["final_checkpoint"]["sha256"])
        rec["delta_AP_per_city"] = {c: rec["arms"]["m2"]["per_city"][c]["AP"]
                                    - rec["arms"]["m1"]["per_city"][c]["AP"]
                                    for c in val_cities}
        rec["delta_AP_fold"] = (rec["arms"]["m2"]["pooled"]["AP"]
                                - rec["arms"]["m1"]["pooled"]["AP"])
        rec["fold_seconds"] = time.time() - t_fold
        rec["finished_at"] = now()
        rec["done"] = bool(same)
        if not same:
            print(f"!!! fold {fi}: stream checksums DIFFER between arms (epoch "
                  f"{rec['paired_stream_first_mismatch_epoch']}) — paired control "
                  f"broken, fold NOT marked done", flush=True)

        # every held-out pixel, full resolution and addressable by (city,row,col)
        payload = {}
        for c in val_cities:
            payload[f"{c}__p_m1"] = all_maps["m1"][c]
            payload[f"{c}__p_m2"] = all_maps["m2"][c]
            payload[f"{c}__y"] = fold.labels[c].astype(np.uint8)
            payload[f"{c}__valid"] = fold.valid[c]
        np.savez_compressed(os.path.join(out, f"fold{fi}_maps.npz"),
                            cities=np.array(val_cities), **payload)

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
    ap.add_argument("--data_dir", default="")
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
    ap.add_argument("--print_folds", action="store_true",
                    help="print the fold assignment + hash and exit (no data needed)")
    ap.add_argument("--tag", type=str, default="p2_l2_cv")
    ap.add_argument("--out_dir", type=str, default=os.path.join(ROOT, "results", "runs"))
    cfg = ap.parse_args()
    cfg.folds = [int(x) for x in cfg.folds.split(",") if x != ""]

    if cfg.print_folds:
        folds, assign, h = fold_table(cfg.n_splits, cfg.cv_seed)
        print(f"get_grouped_folds(n_splits={cfg.n_splits}, seed={cfg.cv_seed})")
        print(f"fold-assignment sha256[:16] = {h}\n")
        for i, (t, v) in enumerate(folds):
            print(f"  fold {i}  held-out ({len(v)}): {', '.join(v)}")
        print(f"\ncity -> fold: " + ", ".join(f"{c}:{assign[c]}" for c in TRAIN_CITIES))
        sys.exit(0)

    assert cfg.data_dir, "--data_dir is required (or use --print_folds)"
    assert cfg.tying == "untied" or cfg.depth == 1, \
        "P2 is untied by definition — tied L2 keeps 38 params and is a different experiment"
    run(cfg)
