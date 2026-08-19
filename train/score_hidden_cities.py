"""
train/score_hidden_cities.py — score the 10 hidden-label OSCD cities, the moment
their ground truth is available.

The 10 test cities are predict-only in this repo (`data/splits.py`,
`TEST_CITIES`): the pipeline writes a mask per city and stops, because no labels
ship with them. If the organisers release the test labels (or OSCD's separately
distributed test-label archive is permitted for the challenge), this script turns
those masks into the same metric table the 14 labelled cities get — per city,
macro over cities, and pooled over all test pixels.

    python train/score_hidden_cities.py \
        --label_root /path/to/OSCD_test_labels \
        --pred "M1 L3=results/submission/masks"

Comparing architectures on the hidden cities needs one prediction directory per
model (see `docs/hidden_city_evaluation.md` for how to produce them):

    python train/score_hidden_cities.py --label_root ... \
        --pred "M1 L3=results/submission/masks_m1_L3" \
        --pred "M2 L3=results/submission/masks_m2_L3"

Two input granularities, and the difference matters:

  <city>_prob.npz   probability map + valid mask -> the full table, including the
                    threshold-free **AP** (the primary metric) and ROC-AUC.
  <city>.png        the committed uint8 {0,255} deliverable -> already
                    thresholded, so only F1 / precision / Change-Accuracy at the
                    frozen tau can be recovered. AP and ROC-AUC are reported as
                    n/a rather than faked from a binary map.

Leakage discipline: tau is FROZEN (default: the out-of-fold tau* recorded in
`results/submission/threshold_*.json`). This script never selects a threshold on
test pixels — `evaluate_predictions(select_threshold=True)` is not reachable from
here. A per-city best-case F1 on hidden data would not be a test number.
"""
import os, sys, json, glob, argparse
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "data"))
sys.path.insert(0, HERE)

from splits import TEST_CITIES
from inference import evaluate_predictions

DEFAULT_TAU_JSON = os.path.join(ROOT, "results", "submission", "threshold_m1_L3.json")
KEYS = ["AP", "roc_auc", "F1", "precision", "change_acc", "nochange_acc", "accuracy"]


# --------------------------------------------------------------------------- #
# labels
# --------------------------------------------------------------------------- #
def find_label(label_root, city):
    """Locate a city's change mask under any of the layouts OSCD ships.

    Encoding follows preprocess._load_label: the TIF is {1 = no change,
    2 = change}, NOT {0, 1}. A PNG mask is {0, 255} and is thresholded at > 0.
    """
    pats = [
        os.path.join(label_root, "**", city, "cm", f"{city}-cm.tif"),
        os.path.join(label_root, "**", city, "cm", "cm.png"),
        os.path.join(label_root, "**", city, f"{city}-cm.tif"),
        os.path.join(label_root, "**", f"{city}-cm.tif"),
    ]
    for p in pats:
        hits = sorted(glob.glob(p, recursive=True))
        if hits:
            return hits[0]
    return None


def load_label(path):
    a = np.array(Image.open(path))
    if path.lower().endswith(".tif") or path.lower().endswith(".tiff"):
        u = set(np.unique(a).tolist())
        if not u.issubset({0, 1, 2}):
            raise ValueError(f"{path}: unexpected TIF label values {sorted(u)[:8]}")
        if u == {0, 1}:                      # already {0,1} — trust it, but say so
            print(f"    note: {os.path.basename(path)} is {{0,1}}-encoded, not {{1,2}}")
            return (a > 0).astype(np.int8)
        return (a == 2).astype(np.int8)
    return (a > 0).astype(np.int8)           # PNG {0,255}


# --------------------------------------------------------------------------- #
# predictions
# --------------------------------------------------------------------------- #
def load_prediction(pred_dir, city):
    """(scores, valid, is_probability). Prefers the probability map."""
    npz = os.path.join(pred_dir, f"{city}_prob.npz")
    if os.path.exists(npz):
        z = np.load(npz)
        P = z["P"].astype(np.float64)
        valid = z["valid"].astype(bool) if "valid" in z.files else np.ones(P.shape, bool)
        return P, valid, True
    png = os.path.join(pred_dir, f"{city}.png")
    if os.path.exists(png):
        m = np.array(Image.open(png))
        return (m > 0).astype(np.float64), np.ones(m.shape, bool), False
    return None, None, None


def crop_common(*arrays):
    H = min(a.shape[0] for a in arrays)
    W = min(a.shape[1] for a in arrays)
    return [a[:H, :W] for a in arrays]


# --------------------------------------------------------------------------- #
def score_model(name, pred_dir, label_root, tau, cities):
    per_city, pooled_p, pooled_y = {}, [], []
    binary_only = False
    for city in cities:
        lp = find_label(label_root, city)
        if lp is None:
            print(f"  {name}: {city:12} no label found under {label_root} — skipped")
            continue
        P, valid, is_prob = load_prediction(pred_dir, city)
        if P is None:
            print(f"  {name}: {city:12} no prediction in {pred_dir} — skipped")
            continue
        binary_only |= not is_prob
        Y = load_label(lp)
        P, valid, Y = crop_common(P, valid, Y)
        # a binary map is already thresholded: score it at 0.5, not at tau
        t = tau if is_prob else 0.5
        m = evaluate_predictions(P, Y, tau=t, mask=valid)
        if not is_prob:
            # a 0/1 score vector still yields *a* number from average_precision_score,
            # but it is the F1 operating point in disguise, not a ranking metric.
            m["AP"] = m["roc_auc"] = float("nan")
        m["is_probability"] = is_prob
        m["label_path"] = lp
        per_city[city] = m
        pooled_p.append(P[valid].ravel()); pooled_y.append(Y[valid].ravel())
        print(f"  {name}: {city:12} prev {m['prevalence']:.4f}  AP {m['AP'] if 'AP' in m else float('nan'):.4f}"
              f"  F1 {m['F1']:.4f}  ChangeAcc {m['change_acc']:.3f}"
              f"{'' if is_prob else '   (binary mask: AP/ROC-AUC not recoverable)'}")
    if not per_city:
        return None
    p, y = np.concatenate(pooled_p), np.concatenate(pooled_y)
    pooled = evaluate_predictions(p, y, tau=(0.5 if binary_only else tau))
    if binary_only:
        pooled["AP"] = pooled["roc_auc"] = float("nan")
    macro = {}
    for k in KEYS:                            # all-nan (binary masks) -> nan, no warning
        v = np.array([per_city[c].get(k, np.nan) for c in per_city], float)
        v = v[~np.isnan(v)]
        macro[k] = float(v.mean()) if v.size else float("nan")
    return {"name": name, "pred_dir": pred_dir, "tau": tau, "binary_only": binary_only,
            "n_cities": len(per_city), "per_city": per_city,
            "macro": macro, "pooled": pooled}


# --------------------------------------------------------------------------- #
def fmt(x, n=4):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{n}f}"


def write_report(results, metric, path):
    out, A = [], None
    A = out.append
    A("# Hidden-city evaluation (10 OSCD test cities)\n")
    A("Generated by [`train/score_hidden_cities.py`](../train/score_hidden_cities.py). "
      "These cities are held out of everything: they are absent from transform "
      "fitting, from training, and from the threshold choice. tau was frozen on "
      "out-of-fold predictions over the 14 labelled cities before any test city was "
      "touched, and is not re-selected here.\n")
    if any(r["binary_only"] for r in results):
        A("> ⚠️ At least one model was scored from **binary {0,255} masks**, which are "
          "already thresholded. AP and ROC-AUC cannot be recovered from those and are "
          "reported as n/a; only F1 / precision / Change-Accuracy at the frozen "
          "operating point are meaningful. Re-run the prediction stage keeping "
          "`<city>_prob.npz` for the threshold-free metrics.\n")

    names = [r["name"] for r in results]
    A(f"\n## Per-city {metric}\n")
    A("| city | prevalence | " + " | ".join(names) + " |")
    A("|" + "---|" * (2 + len(names)))
    for city in TEST_CITIES:
        vals = [r["per_city"].get(city, {}).get(metric) for r in results]
        if all(v is None for v in vals):
            continue
        prev = next((r["per_city"][city]["prevalence"] for r in results
                     if city in r["per_city"]), None)
        good = [v for v in vals if v is not None and not np.isnan(v)]
        best = max(good) if good else None
        A(f"| {city} | {fmt(prev)} | " +
          " | ".join(f"**{fmt(v)}**" if best is not None and v == best else fmt(v)
                     for v in vals) + " |")
    A("| **macro (cities)** |  | " +
      " | ".join(fmt(r["macro"].get(metric)) for r in results) + " |")
    A("| **pooled (all px)** |  | " +
      " | ".join(fmt(r["pooled"].get(metric)) for r in results) + " |")

    A("\n## Full metric table\n")
    A("| model | cities | " + " | ".join(KEYS) + " |")
    A("|" + "---|" * (2 + len(KEYS)))
    for r in results:
        A(f"| {r['name']} macro | {r['n_cities']} | " +
          " | ".join(fmt(r["macro"].get(k)) for k in KEYS) + " |")
        A(f"| {r['name']} pooled | {r['n_cities']} | " +
          " | ".join(fmt(r["pooled"].get(k)) for k in KEYS) + " |")

    if len(results) > 1:
        A(f"\n## Paired comparison on the hidden cities ({metric})\n")
        A("| comparison | mean Δ | median Δ | wins | cities lost |")
        A("|---|---|---|---|---|")
        base = results[0]
        for r in results[1:]:
            shared = [c for c in TEST_CITIES
                      if c in base["per_city"] and c in r["per_city"]]
            d = np.array([base["per_city"][c][metric] - r["per_city"][c][metric]
                          for c in shared], float)
            d = d[~np.isnan(d)]
            if not d.size:
                continue
            lost = [c for c in shared
                    if base["per_city"][c][metric] < r["per_city"][c][metric]]
            A(f"| {base['name']} − {r['name']} | {fmt(d.mean())} | "
              f"{fmt(float(np.median(d)))} | {int((d > 0).sum())}/{d.size} | "
              f"{', '.join(lost) if lost else '—'} |")
        A("\nThese 10 cities are scored by **one** model each (the 14-city refit), so "
          "unlike the CV table there is no fold structure — each city is a single "
          "independent draw from a different domain. Read the win count, and expect "
          "the city-to-city spread to dwarf the model-to-model difference (Finding 3).\n")

    open(path, "w").write("\n".join(out) + "\n")
    return path


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label_root", required=True,
                    help="directory holding the test-city change masks (searched recursively)")
    ap.add_argument("--pred", action="append", required=True, metavar="NAME=DIR",
                    help="prediction directory, repeatable; first one is the baseline "
                         "in the paired comparison")
    ap.add_argument("--tau", type=float, default=None,
                    help="frozen operating point; default: tau_final from --threshold_json")
    ap.add_argument("--threshold_json", default=DEFAULT_TAU_JSON)
    ap.add_argument("--metric", default="AP", choices=KEYS)
    ap.add_argument("--cities", nargs="*", default=TEST_CITIES)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "submission",
                                                  "hidden_city_scores.json"))
    ap.add_argument("--report", default=os.path.join(ROOT, "docs",
                                                     "results_hidden_cities.md"))
    a = ap.parse_args()

    tau = a.tau
    if tau is None:
        if not os.path.exists(a.threshold_json):
            sys.exit(f"no --tau given and {a.threshold_json} is missing")
        tau = json.load(open(a.threshold_json))["tau_final"]
    print(f"frozen tau = {tau:.4f} (never re-selected on test)")

    results = []
    for spec in a.pred:
        if "=" not in spec:
            sys.exit(f"--pred expects NAME=DIR, got {spec!r}")
        name, d = spec.split("=", 1)
        if not os.path.isdir(d):
            sys.exit(f"prediction directory not found: {d}")
        r = score_model(name.strip(), d, a.label_root, tau, a.cities)
        if r:
            results.append(r)
    if not results:
        sys.exit("nothing scored — check --label_root and --pred")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"tau": tau, "metric": a.metric, "results": results},
              open(a.out, "w"), indent=2, default=float)
    write_report(results, a.metric, a.report)

    print()
    for r in results:
        print(f"{r['name']:12} {r['n_cities']} cities   macro {a.metric} "
              f"{fmt(r['macro'].get(a.metric))}   pooled {a.metric} "
              f"{fmt(r['pooled'].get(a.metric))}")
    print(f"-> {a.out}\n-> {a.report}")


if __name__ == "__main__":
    main()
