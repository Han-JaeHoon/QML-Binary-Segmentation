"""
train/adapt_p0.py — bring the P0 (L=1, 38p) 5-fold result into the cv-1.0 schema.

P0 was run on a different machine with a different runner, so its numbers reach
this repo only as a written report. This adapter freezes those numbers into a
`summary.json` that `train/report_cv.py --l1` can read, so the L1-vs-L2
decomposition is computed by the same code path as everything else instead of by
hand.

PROVENANCE, stated in the output file itself: every value here is TRANSCRIBED
from the P0 run report, not read from P0 artefacts. Two consequences:

  * per-fold pooled AP for M2 is reconstructed as AP(M1) + dAP_f, because the
    report gave M1's pooled AP and the delta rather than M2's directly;
  * per-city AP is present only for the 12 cities the report listed
    individually (bordeaux and beihai were reported as deltas only), so
    city-level L1 tables cover 12 of 14. Fold-level tables are complete.

Replace this file with a real converter the moment the P0 artefacts are
available — `--from_json` takes a dump of the same fields if the numbers ever
arrive as data rather than prose.
"""
import os, sys, json, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "data"))
from splits import get_grouped_folds

SOURCE = ("TRANSCRIBED from the P0 run report (chat message, 2026-08-18); "
          "not read from P0 artefacts. M2 per-fold AP = M1 AP + dAP_f.")

# per-fold pooled AP for M1, and dAP_f = AP(M2) - AP(M1), in fold order
FOLD_M1 = [0.2519, 0.0224, 0.1299, 0.1222, 0.0784]
FOLD_DAP = [-0.0072, -0.0013, +0.0045, -0.0047, -0.0078]

# per-city: (M1 AP or None if the report gave only the delta, dAP, prevalence)
CITY = {
    "cupertino":   (0.3201, +0.0123, 0.0237),
    "mumbai":      (0.1510, -0.0170, 0.0256),
    "nantes":      (0.2721, -0.0649, 0.0114),
    "saclay_e":    (0.0114, +0.0024, 0.0099),
    "pisa":        (0.0290, -0.0022, 0.0164),
    "aguasclaras": (0.1249, -0.0556, 0.0164),
    "paris":       (0.0279, +0.0046, 0.0029),
    "bercy":       (0.0414, +0.0077, 0.0074),
    "rennes":      (0.4557, -0.0461, 0.0258),
    "hongkong":    (0.1518, -0.0021, 0.0356),
    "abudhabi":    (0.0726, +0.0017, 0.0376),
    "beirut":      (0.1725, -0.0049, 0.0269),
    "bordeaux":    (None,   -0.0026, 0.0100),
    "beihai":      (None,   -0.0343, 0.0249),
}

# 14-city macro means as reported (kept for cross-checking the fold-mean view)
MACRO = {"m1": {"AP": 0.1419, "roc_auc": 0.806, "F1": 0.1994, "change_acc": 0.347,
                "nochange_acc": 0.926, "accuracy": 0.915},
         "m2": {"AP": 0.1276, "roc_auc": 0.800, "F1": 0.1877, "change_acc": 0.319,
                "nochange_acc": 0.950, "accuracy": 0.938}}
TRAIN_BCE = {"m1": 0.4665, "m2": 0.4745}          # mean final train BCE over folds


def build(out_dir, tag="p0_l1_cv_transcribed"):
    d = os.path.join(out_dir, tag)
    os.makedirs(d, exist_ok=True)
    folds = get_grouped_folds(5, seed=0)
    summary = {"schema_version": "cv-1.0", "source": SOURCE,
               "config": {"depth": 1, "tying": "untied", "n_splits": 5,
                          "epochs": 50, "steps_per_epoch": 320, "batch": 32,
                          "lr": 0.02, "readout": "center_mean",
                          "representation": "pca"},
               "macro_as_reported": MACRO,
               "train_BCE_final_mean": TRAIN_BCE,
               "folds": {}}
    for fi, (train_cities, val_cities) in enumerate(folds):
        m1p, dap = FOLD_M1[fi], FOLD_DAP[fi]
        arms = {}
        for a, ap in (("m1", m1p), ("m2", m1p + dap)):
            per_city = {}
            for c in val_cities:
                base, cd, prev = CITY[c]
                v = None if base is None else (base if a == "m1" else base + cd)
                per_city[c] = {"AP": v, "prevalence": prev,
                               "transcribed": True, "complete": v is not None}
            arms[a] = {"label": f"{a.upper()} L=1 [center]", "n_params": 38,
                       "per_city": per_city, "pooled": {"AP": round(ap, 6)},
                       "source": SOURCE}
        summary["folds"][str(fi)] = {
            "schema_version": "cv-1.0", "fold": fi, "source": SOURCE,
            "train_cities": train_cities, "val_cities": val_cities,
            "arms": arms,
            "delta_AP_fold": dap,
            "delta_AP_per_city": {c: CITY[c][1] for c in val_cities},
            # the P0 report states same_init and same_stream passed in every fold
            "paired_stream_identical": True, "done": True}
    p = os.path.join(d, "summary.json")
    json.dump(summary, open(p, "w"), indent=1)
    print(f"wrote {p}")
    print(f"  source: {SOURCE}")
    print(f"  fold AP  M1 {[round(x,4) for x in FOLD_M1]}")
    print(f"  fold dAP    {[round(x,4) for x in FOLD_DAP]}")
    miss = [c for c, v in CITY.items() if v[0] is None]
    print(f"  per-city AP missing for {miss} (deltas only in the report)")
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=os.path.join(ROOT, "results", "runs"))
    ap.add_argument("--tag", default="p0_l1_cv_transcribed")
    a = ap.parse_args()
    build(a.out_dir, a.tag)
