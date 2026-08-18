"""
train/export_curves.py — collect every per-epoch curve and final held-out score of
the two paired CV runs into one JSON, for plotting.

    python train/export_curves.py                       # -> results/curves.json
    python train/export_curves.py --out /tmp/curves.json

Sources (both committed):
  L=1 · 38p   results/runs/p0_5fold.json            (P0 runner, folds[].models[].trace)
  L=2 · 74p   results/runs/p2_l2_cv/fold<i>_<arm>.jsonl + summary.json

The two runs used the identical fold assignment (sha c22242aede982d21), so a fold
index means the same held-out cities in both.

What is per-epoch and what is not — this is a design decision, not a gap:
  * per epoch  : train BCE (every epoch) and cheap-val AP (a diagnostic on fixed
                 coordinates, every epoch in P0 and every 5th in P2)
  * final only : AP / ROC-AUC / F1* / Change-Acc / No-change-Acc / Accuracy, from
                 the exhaustive stride-1 pass over each held-out city with the
                 FINAL checkpoint. Running that per epoch would cost tens of
                 minutes per city, and the protocol forbids using validation to
                 pick a checkpoint anyway.
"""
import os, sys, json, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
METRICS = ("AP", "roc_auc", "F1", "change_acc", "nochange_acc", "accuracy")
ARMS = ("m1", "m2")
CONST_BCE = 0.5601          # H(0.248), the centre-branch constant-predictor reference


def from_p0(path):
    """P0 runner format: folds[].models[{m1,m2}].{trace,per_city,fold_pooled}."""
    d = json.load(open(path))
    run = {"label": "L=1 · 38p", "n_params": 38, "source": os.path.relpath(path, ROOT),
           "folds": []}
    for f in d["folds"]:
        rec = {"fold": f["fold"], "val_cities": f["val_cities"], "arms": {}}
        for a in ARMS:
            M = f["models"][a]
            rec["arms"][a] = {
                "bce": [round(t["train_BCE"], 6) for t in M["trace"]],
                "cheap_ap": [[t["epoch"], round(t["cheap_AP"], 6)] for t in M["trace"]
                             if t.get("cheap_AP") is not None],
                "per_city": {c: {k: m[k] for k in METRICS} for c, m in M["per_city"].items()},
                "pooled": {k: M["fold_pooled"].get(k) for k in METRICS}}
        run["folds"].append(rec)
    return run


def from_cv(run_dir, label, n_params):
    """cv-1.0 format written by train/run_cv.py."""
    s = json.load(open(os.path.join(run_dir, "summary.json")))
    run = {"label": label, "n_params": n_params,
           "source": os.path.relpath(run_dir, ROOT), "folds": []}
    for k in sorted(s["folds"], key=int):
        F = s["folds"][k]
        rec = {"fold": int(k), "val_cities": F["val_cities"], "arms": {}}
        for a in ARMS:
            A = F["arms"][a]
            log = [json.loads(l) for l in open(os.path.join(run_dir, f"fold{k}_{a}.jsonl"))]
            ep = [r for r in log if r.get("record") == "epoch"]
            rec["arms"][a] = {
                "bce": [round(r["train_BCE"], 6) for r in ep],
                "cheap_ap": [[r["epoch"], round(r["cheap_AP"], 6)] for r in ep
                             if "cheap_AP" in r],
                "per_city": {c: {k2: m[k2] for k2 in METRICS} for c, m in A["per_city"].items()},
                "pooled": {k2: A["pooled"].get(k2) for k2 in METRICS}}
        run["folds"].append(rec)
    return run


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--p0", default=os.path.join(ROOT, "results", "runs", "p0_5fold.json"))
    ap.add_argument("--p2", default=os.path.join(ROOT, "results", "runs", "p2_l2_cv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "curves.json"))
    a = ap.parse_args()

    out = {"runs": {"L1": from_p0(a.p0),
                    "L2": from_cv(a.p2, "L=2 · 74p", 74)},
           "const_bce": CONST_BCE, "macro": {}}
    for rk, R in out["runs"].items():
        out["macro"][rk] = {}
        for arm in ARMS:
            cities = {c: m for f in R["folds"] for c, m in f["arms"][arm]["per_city"].items()}
            out["macro"][rk][arm] = {k: float(np.mean([m[k] for m in cities.values()]))
                                     for k in METRICS}
            out["macro"][rk][arm]["n_cities"] = len(cities)
    json.dump(out, open(a.out, "w"))

    print(f"{a.out}  ({os.path.getsize(a.out)/1024:.0f} KB)")
    for rk in out["runs"]:
        n = [len(f["arms"]["m1"]["bce"]) for f in out["runs"][rk]["folds"]]
        c = [len(f["arms"]["m1"]["cheap_ap"]) for f in out["runs"][rk]["folds"]]
        print(f"  {rk}: {len(n)} folds | BCE points/fold {n} | cheap-val points/fold {c}")
        for arm in ARMS:
            m = out["macro"][rk][arm]
            print(f"    {arm}: macro over {m['n_cities']} cities  "
                  + "  ".join(f"{k} {m[k]:.4f}" for k in METRICS))
