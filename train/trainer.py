"""
train/trainer.py — optimization loop for the QML ladder.

Everything numeric here is a PILOT DEFAULT (config-driven), not a conclusion:
no held-out-city learning curve has been seen yet, so lr / epoch size /
validation cadence are starting points to be revised from the first curves.

An "epoch" is defined by STEPS, not by a pass over the data (there is no natural
pass: the sampler is an infinite stochastic stream):
    epoch = steps_per_epoch * batch  patches      (default 160*32 = 5120)

Validation
  every epoch      : fixed cheap-val coordinates (natural prevalence, ~3k/city)
                     -> monitoring + candidate checkpoints
  every N epochs   : exhaustive full-city overlap-averaged evaluation
                     -> authoritative; the BEST checkpoint is chosen on this
  Rationale: the model that looks best on 9k sampled pixels is not guaranteed to
  be best on the full city.

Logged per epoch: train_BCE, cheap AP / best-F1 / tau, grad_norm, param_norm,
lr, wall_time. At exhaustive checkpoints: per-city and pooled AP, F1,
ChangeAcc, NoChangeAcc, Accuracy, tau*.
"""
import os, sys, json, time
from dataclasses import dataclass, asdict, field
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "data"))
sys.path.insert(0, os.path.join(ROOT, "models"))
sys.path.insert(0, HERE)

from splits import get_dev_split
from preprocess import (build_fold, transform_pca4, transform_physical4,
                        pca_zz_strength, physical_zz_strength)
from pools import build_center_pools, fit_global_hard_threshold, eligible_mask
from sampler import SpatialPatchSampler
import qml as qmodels                       # models/qml.py
from inference import (predict_city, predict_coordinates, evaluate_predictions,
                       make_fixed_val_coordinates)


@dataclass
class TrainConfig:
    # model
    kind: str = "m3"
    depth: int = 1
    tying: str = "untied"
    representation: str = "pca"          # "pca" | "physical"
    # optimization (PILOT defaults)
    lr: float = 0.02
    batch: int = 32
    steps_per_epoch: int = 160           # epoch = 160*32 = 5120 patches
    epochs: int = 20
    w_pos: float = 1.0                   # 1.0 = plain BCE (sampler already ~78:22)
    # validation
    cheap_val_per_city: int = 3000
    exhaustive_every: int = 5
    exhaustive_cities: str = "smallest"  # "smallest" | "all" | "none"
    infer_batch: int = 4096
    # misc
    seed: int = 0
    tag: str = ""
    out_dir: str = os.path.join(ROOT, "results", "runs")


def build_representation(fold, cities, representation):
    """city -> (X[H,W,4], S[H,W] or [H,W,2])."""
    X, S = {}, {}
    for c in cities:
        D = fold.dcorr13[c]
        if representation == "pca":
            X[c] = transform_pca4(D, fold.pca_tf).astype(np.float64)
            S[c] = pca_zz_strength(D, fold.pca_tf).astype(np.float64)
        else:
            x4 = transform_physical4(D, fold.physical_tf).astype(np.float64)
            s1, s2 = physical_zz_strength(x4)
            X[c], S[c] = x4, np.stack([s1, s2], -1)
    return X, S


def make_batch(smp, X, S, labels, B, rng):
    Xb, Sb, Yb = [], [], []
    for _ in range(B):
        c, _, r, cc = smp.sample_index(rng)
        Xb.append(X[c][r-1:r+2, cc-1:cc+2])
        Sb.append(S[c][r-1:r+2, cc-1:cc+2])
        Yb.append(labels[c][r-1:r+2, cc-1:cc+2])
    return (pnp.array(np.array(Xb), requires_grad=False),
            pnp.array(np.array(Sb), requires_grad=False),
            pnp.array(np.array(Yb, dtype=float), requires_grad=False))


def run(cfg, data_dir):
    os.makedirs(cfg.out_dir, exist_ok=True)
    spec = qmodels.ModelSpec(cfg.kind, cfg.depth, cfg.tying)
    tag = cfg.tag or f"{cfg.kind}_L{cfg.depth}_{cfg.tying}_{cfg.representation}"
    log_path = os.path.join(cfg.out_dir, f"{tag}.jsonl")
    print(f"=== {spec.label} | {cfg.representation} | {spec.n_params} params -> {log_path}")

    # ---- data ----
    train_cities, val_cities = get_dev_split()
    fold = build_fold(train_cities, val_cities, data_dir)
    T_global = fit_global_hard_threshold(train_cities, fold.dcorr13, fold.labels, fold.valid)
    pools = {c: build_center_pools(fold.dcorr13[c], fold.labels[c], fold.valid[c], T_global)
             for c in train_cities}
    smp = SpatialPatchSampler(train_cities, pools, fold, cfg.representation, seed=cfg.seed)
    Xtr, Str = build_representation(fold, train_cities, cfg.representation)
    Xva, Sva = build_representation(fold, val_cities, cfg.representation)

    # fixed cheap-val coordinates (natural prevalence) + sanity print
    val_coords, val_y = {}, {}
    print(f"\ncheap-val ({cfg.cheap_val_per_city}/city, uniform over eligible):")
    for c in val_cities:
        co = make_fixed_val_coordinates(fold.labels[c], fold.valid[c],
                                        n=cfg.cheap_val_per_city, seed=cfg.seed + 1)
        val_coords[c] = co
        val_y[c] = fold.labels[c][co[:, 0], co[:, 1]].astype(int)
        el = eligible_mask(fold.valid[c])
        print(f"  {c:11} sampled prevalence {val_y[c].mean():.4f}   "
              f"city eligible {fold.labels[c][el].mean():.4f}")

    if cfg.exhaustive_cities == "all":
        ex_cities = list(val_cities)
    elif cfg.exhaustive_cities == "none":
        ex_cities = []
    else:
        ex_cities = [min(val_cities, key=lambda c: fold.labels[c].size)]
    print(f"exhaustive val on: {ex_cities or '(disabled)'}\n")

    # ---- model / optimizer ----
    forward = qmodels.build_model(spec)
    params = qmodels.init_params(spec, seed=cfg.seed)
    opt = qml.AdamOptimizer(cfg.lr)
    rng = np.random.RandomState(cfg.seed)

    def cheap_val(p):
        ps, ys = [], []
        pn = np.asarray(p)
        for c in val_cities:
            ps.append(predict_coordinates(forward, pn, Xva[c], Sva[c],
                                          val_coords[c], cfg.infer_batch))
            ys.append(val_y[c])
        return evaluate_predictions(np.concatenate(ps), np.concatenate(ys),
                                    select_threshold=True)

    def exhaustive_val(p):
        pn = np.asarray(p); per_city, allp, ally = {}, [], []
        for c in ex_cities:
            P = predict_city(forward, pn, Xva[c], Sva[c], cfg.infer_batch)
            m = fold.valid[c]
            per_city[c] = evaluate_predictions(P, fold.labels[c], select_threshold=True, mask=m)
            allp.append(P[m].ravel()); ally.append(fold.labels[c][m].ravel())
        pooled = evaluate_predictions(np.concatenate(allp), np.concatenate(ally),
                                      select_threshold=True) if allp else {}
        return per_city, pooled

    # Two checkpoints, on purpose:
    #  *_bestcheap.npy : best POOLED cheap-val AP (covers ALL val cities every
    #                    epoch) — the unbiased selector for cross-model comparisons
    #  *_best.npy      : best exhaustive AP (only over `ex_cities`; with
    #                    exhaustive_cities='smallest' this sees ONE city, so it is
    #                    a biased selector — do not use it to compare models)
    best = {"exhaustive_AP": -1.0, "epoch": -1}
    best_cheap = {"cheap_AP": -1.0, "epoch": -1}
    t0 = time.time()
    with open(log_path, "w") as f:
        f.write(json.dumps({"config": asdict(cfg), "n_params": spec.n_params,
                            "train_cities": train_cities, "val_cities": val_cities}) + "\n")

        for epoch in range(1, cfg.epochs + 1):
            losses = []
            for _ in range(cfg.steps_per_epoch):
                Xb, Sb, Yb = make_batch(smp, Xtr, Str, fold.labels, cfg.batch, rng)
                cost = lambda p: qmodels.bce_loss(p, Xb, Sb, Yb, forward, cfg.w_pos)
                params, L = opt.step_and_cost(cost, params)
                losses.append(float(L))
            g = np.asarray(qml.grad(cost)(params))          # last batch, for logging
            cv = cheap_val(params)
            rec = {"epoch": epoch, "train_BCE": float(np.mean(losses)),
                   "cheap_AP": cv["AP"], "cheap_best_F1": cv["F1"], "cheap_tau": cv["tau"],
                   "cheap_change_acc": cv["change_acc"],
                   "grad_norm": float(np.linalg.norm(g)),
                   "param_norm": float(np.linalg.norm(np.asarray(params))),
                   "lr": cfg.lr, "wall_time": time.time() - t0}
            if cv["AP"] > best_cheap["cheap_AP"]:
                best_cheap = {"cheap_AP": cv["AP"], "epoch": epoch, "F1": cv["F1"]}
                np.save(os.path.join(cfg.out_dir, f"{tag}_bestcheap.npy"), np.asarray(params))
            print(f"ep {epoch:3d}  BCE {rec['train_BCE']:.4f}  cheapAP {cv['AP']:.4f}  "
                  f"F1 {cv['F1']:.4f}  |g| {rec['grad_norm']:.3e}  {rec['wall_time']:.0f}s",
                  flush=True)

            if ex_cities and (epoch % cfg.exhaustive_every == 0 or epoch == cfg.epochs):
                te = time.time()
                per_city, pooled = exhaustive_val(params)
                rec["exhaustive"] = {"per_city": per_city, "pooled": pooled,
                                     "seconds": time.time() - te}
                print(f"      exhaustive: " + "  ".join(
                    f"{c}: AP {m['AP']:.4f} F1 {m['F1']:.4f} chAcc {m['change_acc']:.3f}"
                    for c, m in per_city.items()) + f"   ({time.time()-te:.0f}s)", flush=True)
                if pooled and pooled["AP"] > best["exhaustive_AP"]:
                    best = {"exhaustive_AP": pooled["AP"], "epoch": epoch,
                            "tau": pooled["tau"], "F1": pooled["F1"]}
                    np.save(os.path.join(cfg.out_dir, f"{tag}_best.npy"), np.asarray(params))
            f.write(json.dumps(rec) + "\n"); f.flush()

        f.write(json.dumps({"best_exhaustive": best, "best_cheap": best_cheap}) + "\n")
    print(f"\nbest exhaustive AP: {best}\nbest cheap AP    : {best_cheap}")
    return {"exhaustive": best, "cheap": best_cheap}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    for fld in ("kind", "tying", "representation", "exhaustive_cities", "tag"):
        ap.add_argument(f"--{fld}", type=str, default=None)
    for fld in ("depth", "batch", "steps_per_epoch", "epochs",
                "cheap_val_per_city", "exhaustive_every", "seed", "infer_batch"):
        ap.add_argument(f"--{fld}", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    a = ap.parse_args()
    cfg = TrainConfig(**{k: v for k, v in vars(a).items()
                         if k != "data_dir" and v is not None})
    run(cfg, a.data_dir)
