"""
train/eval_full.py — exhaustive full-city evaluation of a saved checkpoint on
ALL validation cities.

Reports, per city and aggregated:
  prevalence, AP, ROC-AUC, F1*, precision, ChangeAcc, NoChangeAcc, tau*

Two aggregations, because the val cities differ ~8x in prevalence
(paris 0.29% vs beihai/cupertino ~2.4%):
  macro : mean of per-city metrics (each city counts once)
  micro : all pixels pooled, ONE global tau* (the honest deployment setting;
          per-city tau* is an optimistic upper bound and is labelled as such)

Note on interpretation: AP and F1* are invariant to monotone rescaling of the
scores, so they are calibration-free measures of separability — a probability
miscalibration (e.g. from the train/val prevalence gap) cannot by itself explain
a low F1*.
"""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "data"))
sys.path.insert(0, os.path.join(ROOT, "models"))
sys.path.insert(0, HERE)

from splits import get_dev_split
from preprocess import build_fold
import qml as qmodels
from inference import predict_city, evaluate_predictions
from trainer import build_representation

KEYS = ["prevalence", "AP", "roc_auc", "F1", "precision", "change_acc", "nochange_acc", "tau"]


def main(data_dir, ckpt, kind, depth, tying, representation, out, infer_batch):
    spec = qmodels.ModelSpec(kind, depth, tying)
    params = np.load(ckpt)
    forward = qmodels.build_model(spec)
    print(f"{spec.label} | {representation} | {spec.n_params} params | ckpt {os.path.basename(ckpt)}")

    train_cities, val_cities = get_dev_split()
    fold = build_fold(train_cities, val_cities, data_dir)
    Xva, Sva = build_representation(fold, val_cities, representation)

    per_city, allp, ally = {}, [], []
    for c in val_cities:
        t = time.time()
        P = predict_city(forward, params, Xva[c], Sva[c], infer_batch)
        m = fold.valid[c]
        per_city[c] = evaluate_predictions(P, fold.labels[c], select_threshold=True, mask=m)
        allp.append(P[m].ravel()); ally.append(fold.labels[c][m].ravel())
        print(f"  {c:11} prev {per_city[c]['prevalence']:.4f}  AP {per_city[c]['AP']:.4f}  "
              f"ROC {per_city[c]['roc_auc']:.4f}  F1* {per_city[c]['F1']:.4f}  "
              f"prec {per_city[c]['precision']:.4f}  chAcc {per_city[c]['change_acc']:.3f}  "
              f"({time.time()-t:.0f}s)", flush=True)

    macro = {k: float(np.mean([per_city[c][k] for c in val_cities])) for k in KEYS}
    micro = evaluate_predictions(np.concatenate(allp), np.concatenate(ally),
                                 select_threshold=True)
    print(f"\n  macro (per-city mean, per-city tau* = optimistic):")
    print(f"    AP {macro['AP']:.4f}  ROC {macro['roc_auc']:.4f}  F1* {macro['F1']:.4f}  "
          f"prec {macro['precision']:.4f}  chAcc {macro['change_acc']:.3f}")
    print(f"  micro (pooled pixels, ONE global tau* = deployment):")
    print(f"    AP {micro['AP']:.4f}  ROC {micro['roc_auc']:.4f}  F1* {micro['F1']:.4f}  "
          f"prec {micro['precision']:.4f}  chAcc {micro['change_acc']:.3f}  "
          f"tau {micro['tau']:.3f}  prev {micro['prevalence']:.4f}")

    res = {"model": spec.label, "n_params": spec.n_params, "representation": representation,
           "checkpoint": os.path.basename(ckpt), "per_city": per_city,
           "macro": macro, "micro": micro}
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--kind", default="m3")
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--tying", default="untied")
    ap.add_argument("--representation", default="pca")
    ap.add_argument("--infer_batch", type=int, default=4096)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or a.ckpt.replace("_best.npy", "_fullval.json")
    main(a.data_dir, a.ckpt, a.kind, a.depth, a.tying, a.representation, out, a.infer_batch)
