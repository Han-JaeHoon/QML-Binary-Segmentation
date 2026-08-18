"""
Aggregate the P3 architecture x depth ablation into the final tables + CSV/JSON.

All nine cells come from ONE harness (run_cell.py), so the matrix is internally
consistent. P0/P2 numbers are kept separately and only used for a reproduction
cross-check, never mixed into the headline matrix.

Primary metric: AP. F1* is a best-operating-point diagnostic (its threshold is
swept on the very predictions it scores) and is reported as such.

Wording guard for the ring: it is a CZ ring-entangled HEA-style control, NOT a
geometry-agnostic or topology-only control — 6 of its 9 edges coincide with real
horizontal spatial neighbours and it uses 9 CZ/stage against the grid's 12.
"""
import json, glob, os, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SWEEP = os.path.join(ROOT, "results", "runs", "p3_topology")
OUT = os.path.join(ROOT, "results", "p3_matrix")
KINDS = [("m1", "M1 separable"), ("mring", "M_ring CZ ring"), ("m2", "M2 spatial grid")]
DEPTHS = [(1, 38), (2, 74), (3, 110)]
MET = ["AP", "roc_auc", "F1", "change_acc", "nochange_acc", "accuracy"]


def load():
    cells, cities, bce = collections.defaultdict(dict), collections.defaultdict(dict), collections.defaultdict(dict)
    for p in glob.glob(os.path.join(SWEEP, "*.json")):
        d = json.load(open(p))
        k = (d["kind"], d["depth"])
        cells[k][d["fold"]] = d["pooled"]
        bce[k][d["fold"]] = d["trace"][-1]["train_BCE"]
        for c, m in d["per_city"].items():
            cities[k][c] = m
    return cells, cities, bce


def pooled_oof(kind, depth):
    """One global ranking over every labelled pixel (each appears exactly once)."""
    import sys
    sys.path.insert(0, HERE)
    from inference import evaluate_predictions
    P, Y = [], []
    for f in range(5):
        p = os.path.join(SWEEP, f"{kind}_L{depth}_fold{f}_maps.npz")
        if not os.path.exists(p):
            return None
        z = np.load(p)
        for c in sorted({k.rsplit("_", 1)[0] for k in z.files}):
            m = z[f"{c}_valid"]
            P.append(z[f"{c}_P"][m].ravel()); Y.append(z[f"{c}_Y"][m].ravel())
    return evaluate_predictions(np.concatenate(P), (np.concatenate(Y) > 0).astype(int),
                                select_threshold=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    cells, cities, bce = load()

    print("=" * 74)
    print("ARCHITECTURE x DEPTH — mean fold AP (std), 5 city-grouped folds")
    print("=" * 74)
    print(f"{'':18}" + "".join(f"{f'L{L} ({p}p)':>18}" for L, p in DEPTHS))
    grid = {}
    for k, name in KINDS:
        row = ""
        for L, _ in DEPTHS:
            v = cells.get((k, L), {})
            if len(v) == 5:
                a = np.array([v[f]["AP"] for f in range(5)])
                grid[(k, L)] = a
                row += f"{a.mean():>11.4f} ({a.std():.3f})"
            else:
                row += f"{'incomplete':>18}"
        print(f"{name:18}{row}")

    print("\n" + "=" * 74)
    print("CAPACITY CURVE (mean fold AP)  +  depth effects")
    print("=" * 74)
    for k, name in KINDS:
        if all((k, L) in grid for L, _ in DEPTHS):
            m = [grid[(k, L)].mean() for L, _ in DEPTHS]
            print(f"  {name:18} {m[0]:.4f} -> {m[1]:.4f} -> {m[2]:.4f}   "
                  f"d(1->2) {m[1]-m[0]:+.4f}   d(2->3) {m[2]-m[1]:+.4f}   "
                  f"total {m[2]-m[0]:+.4f}")

    print("\n" + "=" * 74)
    print("PAIRED DIFFERENCES vs M1 (per fold; same folds, same init, same stream)")
    print("=" * 74)
    for L, p in DEPTHS:
        if ("m1", L) not in grid:
            continue
        print(f"  L{L} ({p}p):")
        for k, name in KINDS[1:]:
            if (k, L) not in grid:
                continue
            d = grid[(k, L)] - grid[("m1", L)]
            print(f"    {name:18} dAP {np.round(d,4).tolist()}  "
                  f"mean {d.mean():+.4f}  wins {int((d>0).sum())}/5")

    print("\n" + "=" * 74)
    print("CHALLENGE METRICS — macro over the 14 held-out cities")
    print("=" * 74)
    print(f"{'cell':22}" + "".join(f"{m:>14}" for m in MET))
    macro = {}
    for k, name in KINDS:
        for L, p in DEPTHS:
            cc = cities.get((k, L), {})
            if len(cc) != 14:
                continue
            row = {m: float(np.mean([cc[c][m] for c in cc])) for m in MET}
            macro[(k, L)] = row
            print(f"{k+'_L'+str(L):22}" + "".join(f"{row[m]:>14.4f}" for m in MET))

    print("\n" + "=" * 74)
    print("POOLED OUT-OF-FOLD (single global tau*, every labelled pixel once)")
    print("=" * 74)
    oof = {}
    for k, name in KINDS:
        for L, _ in DEPTHS:
            r = pooled_oof(k, L)
            if r:
                oof[f"{k}_L{L}"] = r
                print(f"  {k}_L{L:<4} AP {r['AP']:.4f}  ROC {r['roc_auc']:.4f}  "
                      f"F1* {r['F1']:.4f}  ChangeAcc {r['change_acc']:.3f}  "
                      f"NoChangeAcc {r['nochange_acc']:.3f}  Acc {r['accuracy']:.4f}  "
                      f"tau {r['tau']:.3f}")

    print("\n" + "=" * 74)
    print("FINAL train BCE (mean over folds)")
    print("=" * 74)
    for k, name in KINDS:
        vals = [np.mean(list(bce[(k, L)].values())) for L, _ in DEPTHS if len(bce.get((k, L), {})) == 5]
        if len(vals) == 3:
            print(f"  {name:18} L1 {vals[0]:.4f}   L2 {vals[1]:.4f}   L3 {vals[2]:.4f}")

    # machine-readable
    rows = []
    for (k, L), a in grid.items():
        rows.append({"kind": k, "depth": L, "n_params": 36 * L + 2,
                     "fold_AP": a.tolist(), "mean_AP": float(a.mean()),
                     "std_AP": float(a.std()),
                     "macro": macro.get((k, L)), "pooled_oof": oof.get(f"{k}_L{L}"),
                     "final_train_BCE": float(np.mean(list(bce[(k, L)].values())))})
    json.dump(rows, open(os.path.join(OUT, "matrix.json"), "w"), indent=2)
    with open(os.path.join(OUT, "matrix.csv"), "w") as f:
        f.write("kind,depth,n_params,mean_AP,std_AP,pooled_oof_AP,macro_F1,macro_change_acc,final_BCE\n")
        for r in sorted(rows, key=lambda x: (x["kind"], x["depth"])):
            o = r["pooled_oof"]; m = r["macro"]
            f.write(f"{r['kind']},{r['depth']},{r['n_params']},{r['mean_AP']:.6f},"
                    f"{r['std_AP']:.6f},{o['AP'] if o else ''},"
                    f"{m['F1'] if m else ''},{m['change_acc'] if m else ''},"
                    f"{r['final_train_BCE']:.6f}\n")
    print(f"\nsaved {OUT}/matrix.json and matrix.csv")
