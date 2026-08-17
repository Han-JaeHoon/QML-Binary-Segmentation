"""
train/report_cv.py — turn one or two paired-CV summaries into the P2 result tables.

    python train/report_cv.py --l2 results/runs/p2_l2_cv/summary.json \
                             [--l1 results/runs/p0_l1_cv/summary.json]

Reports, in this order:
  1. per-fold AP for both arms, dAP_f = AP_f(M2) - AP_f(M1), fold win count
  2. fold mean +- std (descriptive spread across folds, NOT a confidence interval)
  3. per-held-out-city AP, dAP_c, city win count
  4. pooled out-of-fold AP (every labelled pixel scored exactly once, by the model
     of the fold that held its city out) + the official metric set
  5. if --l1 is given: the capacity/interaction decomposition
        d_int^L1  = AP(M2,L1) - AP(M1,L1)
        d_int^L2  = AP(M2,L2) - AP(M1,L2)
        d_depth^M1 = AP(M1,L2) - AP(M1,L1)
        d_depth^M2 = AP(M2,L2) - AP(M2,L1)
     and the pre-registered case label (A/B/C/D).

Interpretation guards, printed with the numbers so they travel with them:
  * AP is primary. F1* sweeps its threshold on the same held-out city, so it is a
    best-operating-point diagnostic — never an unbiased test F1.
  * The 14 cities are NOT 14 independent samples: cities inside one fold share a
    trained model. No p-value is computed over them, and none should be.
  * 5 folds is a small, dependent sample. Report direction and spread, not
    significance.
  * L2 raises depth AND parameter count together -> "increased depth/capacity
    under untied data re-uploading", never "data re-uploading works".
  * No quantum-advantage language: the parameter-matched classical baseline (37p
    conv) has not been run.
"""
import os, sys, json, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from inference import evaluate_predictions

ARMS = ("m1", "m2")
NAME = {"m1": "M1 (separable)", "m2": "M2 (CZ)"}


def load(path):
    s = json.load(open(path))
    folds = {int(k): v for k, v in s["folds"].items() if v.get("done")}
    return s, dict(sorted(folds.items()))


def pooled_oof(summary_path, folds):
    """Pool every fold's held-out pixels: each labelled pixel is scored once, by
    the model that never saw its city. AP over that pool is the single-number
    out-of-fold estimate."""
    d = os.path.dirname(summary_path)
    P = {a: [] for a in ARMS}; Y = []
    for fi in folds:
        f = os.path.join(d, f"fold{fi}_oof.npz")
        if not os.path.exists(f):
            return None
        z = np.load(f)
        for a in ARMS:
            P[a].append(z[f"{a}_p"])
        Y.append(z["y"])
    y = np.concatenate(Y)
    return {a: evaluate_predictions(np.concatenate(P[a]), y, select_threshold=True)
            for a in ARMS}


def fold_ap(folds):
    return {a: np.array([folds[fi]["arms"][a]["pooled"]["AP"] for fi in folds])
            for a in ARMS}


def city_ap(folds):
    out = {a: {} for a in ARMS}
    for fi, rec in folds.items():
        for a in ARMS:
            for c, m in rec["arms"][a]["per_city"].items():
                out[a][c] = m["AP"]
    return out


def report(tag, path):
    s, folds = load(path)
    if not folds:
        print(f"{tag}: no completed folds in {path}"); return None
    cfg = s["config"]
    n_par = folds[min(folds)]["arms"]["m1"]["n_params"]
    print(f"\n{'='*74}\n{tag}  |  depth {cfg.get('depth','?')} {cfg.get('tying','?')}  "
          f"|  {n_par} params/arm  |  {len(folds)}/{cfg.get('n_splits','?')} folds complete"
          f"\n{'='*74}")

    bad = [fi for fi, r in folds.items() if not r.get("paired_stream_identical")]
    print(f"paired patch stream identical in every fold: "
          f"{'YES' if not bad else 'NO — folds ' + str(bad) + ' are NOT comparable'}")

    ap = fold_ap(folds)
    d = ap["m2"] - ap["m1"]
    print(f"\n-- per-fold AP (pooled over that fold's held-out cities) --")
    print(f"{'fold':>5} {'held-out cities':38} {'M1':>8} {'M2':>8} {'dAP':>9}")
    for k, fi in enumerate(folds):
        print(f"{fi:>5} {','.join(folds[fi]['val_cities']):38} "
              f"{ap['m1'][k]:8.4f} {ap['m2'][k]:8.4f} {d[k]:+9.4f}")
    print(f"{'mean':>5} {'':38} {ap['m1'].mean():8.4f} {ap['m2'].mean():8.4f} {d.mean():+9.4f}")
    sd = lambda v: v.std(ddof=1) if len(v) > 1 else float("nan")
    print(f"{'std':>5} {'':38} {sd(ap['m1']):8.4f} {sd(ap['m2']):8.4f} "
          f"{sd(d):+9.4f}   (spread across folds, not a CI)")
    print(f"fold win count: M2 {int((d > 0).sum())} / M1 {int((d < 0).sum())} "
          f"of {len(d)}")

    cap = city_ap(folds)
    cities = list(cap["m1"])
    dc = {c: cap["m2"][c] - cap["m1"][c] for c in cities}
    print(f"\n-- per-held-out-city AP ({len(cities)} cities) --")
    print(f"{'city':14} {'prev':>7} {'M1':>8} {'M2':>8} {'dAP':>9}")
    prev = {c: m["prevalence"] for fi in folds
            for c, m in folds[fi]["arms"]["m1"]["per_city"].items()}
    for c in sorted(cities, key=lambda x: -dc[x]):
        print(f"{c:14} {prev[c]:7.4f} {cap['m1'][c]:8.4f} {cap['m2'][c]:8.4f} {dc[c]:+9.4f}")
    print(f"city win count: M2 {sum(v > 0 for v in dc.values())} / "
          f"M1 {sum(v < 0 for v in dc.values())} of {len(cities)}   "
          f"(descriptive only — cities within a fold share one model)")

    oof = pooled_oof(path, folds)
    if oof:
        print(f"\n-- pooled out-of-fold (every labelled pixel scored once) --")
        print(f"{'arm':16} {'AP':>8} {'ROC-AUC':>9} {'F1*':>8} {'ChangeAcc':>10} "
              f"{'NoChgAcc':>9} {'Acc':>8} {'tau*':>7}")
        for a in ARMS:
            m = oof[a]
            print(f"{NAME[a]:16} {m['AP']:8.4f} {m['roc_auc']:9.4f} {m['F1']:8.4f} "
                  f"{m['change_acc']:10.3f} {m['nochange_acc']:9.3f} {m['accuracy']:8.4f} "
                  f"{m['tau']:7.3f}")
        print(f"{'dAP':16} {oof['m2']['AP'] - oof['m1']['AP']:+8.4f}")
        print("F1*/ChangeAcc/NoChgAcc use a threshold swept on this same pool -> "
              "best-operating-point diagnostics, not unbiased test values.")
    return {"fold_ap": ap, "fold_d": d, "city_ap": cap, "oof": oof, "n_params": n_par}


def decompose(r1, r2):
    a = {("m1", "L1"): r1["fold_ap"]["m1"].mean(), ("m2", "L1"): r1["fold_ap"]["m2"].mean(),
         ("m1", "L2"): r2["fold_ap"]["m1"].mean(), ("m2", "L2"): r2["fold_ap"]["m2"].mean()}
    d_int_l1 = a[("m2", "L1")] - a[("m1", "L1")]
    d_int_l2 = a[("m2", "L2")] - a[("m1", "L2")]
    d_dep_m1 = a[("m1", "L2")] - a[("m1", "L1")]
    d_dep_m2 = a[("m2", "L2")] - a[("m2", "L1")]

    print(f"\n{'='*74}\ncapacity x interaction decomposition (fold-mean AP)\n{'='*74}")
    print(f"{'':16} {'L1 (38p)':>12} {'L2 (74p)':>12} {'d_depth':>12}")
    print(f"{NAME['m1']:16} {a[('m1','L1')]:12.4f} {a[('m1','L2')]:12.4f} {d_dep_m1:+12.4f}")
    print(f"{NAME['m2']:16} {a[('m2','L1')]:12.4f} {a[('m2','L2')]:12.4f} {d_dep_m2:+12.4f}")
    print(f"{'d_int':16} {d_int_l1:+12.4f} {d_int_l2:+12.4f}")

    w1, w2 = int((r1["fold_d"] > 0).sum()), int((r2["fold_d"] > 0).sum())
    n1, n2 = len(r1["fold_d"]), len(r2["fold_d"])
    if d_int_l1 > 0 and d_int_l2 > 0:
        case = ("A", "interaction benefit holds at both capacities")
    elif d_int_l1 <= 0 and d_int_l2 <= 0:
        case = ("C", "fixed CZ interaction was not a useful inductive bias in "
                     "either capacity regime")
    elif abs(d_int_l1) < 0.005 and d_int_l2 > 0:
        case = ("B", "interaction may need more variational capacity to be usable")
    else:
        case = ("D", "sign flips with capacity -> the interaction effect is "
                     "capacity-dependent; no blanket 'entanglement helps/hurts'")
    print(f"\npre-registered case: {case[0]} — {case[1]}")
    print(f"M2 fold wins: L1 {w1}/{n1}, L2 {w2}/{n2}")
    print("\nWording guards:")
    print("  * L2 changes depth AND parameter count together -> "
          "'increased depth/capacity under untied data re-uploading'.")
    print("  * tied-L2 at 38p previously gave no fitting gain, so a gain here is "
          "most likely a parameter-capacity effect.")
    print("  * no quantum-advantage claim before the 37-parameter classical conv.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--l2", required=True, help="summary.json of the L=2 (74p) CV")
    ap.add_argument("--l1", default="", help="summary.json of the L=1 (38p) CV (P0)")
    args = ap.parse_args()
    r2 = report("P2  L=2 untied", args.l2)
    r1 = report("P0  L=1", args.l1) if args.l1 else None
    if r1 and r2:
        decompose(r1, r2)
