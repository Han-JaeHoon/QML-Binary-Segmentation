"""
train/check_data.py — pre-flight check on a dataset directory, BEFORE a long run.

A 5-fold P2 run is many hours; this catches a wrong layout, a missing city or the
label-encoding trap in seconds instead of at fold 3.

    python train/check_data.py --data_dir /path/to/OneraDataset

Expected layout (what data/preprocess.py reads):

    <data_dir>/
      images/Onera Satellite Change Detection dataset - Images/
        <city>/imgs_1_rect/{B01..B12,B8A}.tif        # 13 single-band uint16 GeoTIFFs
        <city>/imgs_2_rect/{B01..B12,B8A}.tif
      train_labels/Onera Satellite Change Detection dataset - Train Labels/
        <city>/cm/<city>-cm.tif                      # {1 = no change, 2 = change}

Checked, per labelled city:
  * both date folders present with all 13 bands
  * label file present, and its values are the {1,2} encoding (NOT {0,1}) —
    using the README's {0,1} silently marks every pixel as change
  * label / image shapes consistent, resulting change prevalence plausible
  * a valid-pixel mask that is not empty

Reports the total pixel count too, which is what the exhaustive-evaluation time
scales with.
"""
import os, sys, argparse
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "data"))
from preprocess import BANDS, _img_dir, _lbl_dir
from splits import TRAIN_CITIES, TEST_CITIES


def check_city(root, city, labelled=True):
    row = {"city": city, "ok": True, "notes": []}
    d = _img_dir(root)
    for sub in ("imgs_1_rect", "imgs_2_rect"):
        p = os.path.join(d, city, sub)
        if not os.path.isdir(p):
            row["ok"] = False; row["notes"].append(f"missing {sub}"); return row
        miss = [b for b in BANDS if not os.path.exists(os.path.join(p, f"{b}.tif"))]
        if miss:
            row["ok"] = False; row["notes"].append(f"{sub}: missing {','.join(miss)}")
    if not row["ok"]:
        return row

    b04 = np.array(Image.open(os.path.join(d, city, "imgs_1_rect", "B04.tif")))
    row["img_shape"] = tuple(b04.shape)
    row["pixels"] = int(b04.size)

    if not labelled:
        return row

    lp = os.path.join(_lbl_dir(root), city, "cm", f"{city}-cm.tif")
    if not os.path.exists(lp):
        row["ok"] = False; row["notes"].append("missing label tif"); return row
    lab = np.array(Image.open(lp))
    vals = set(np.unique(lab).tolist())
    row["label_shape"] = tuple(lab.shape)
    row["label_values"] = sorted(vals)
    if not vals <= {1, 2}:
        row["ok"] = False
        row["notes"].append(f"label values {sorted(vals)} — expected {{1,2}}")
    y = (lab == 2)
    row["prevalence"] = float(y.mean())
    if not (0.0 < row["prevalence"] < 0.5):
        row["ok"] = False
        row["notes"].append(f"implausible prevalence {row['prevalence']:.4f}")
    if lab.shape != b04.shape:
        row["notes"].append(f"label {lab.shape} vs image {b04.shape} "
                            f"(preprocess crops images to the label — expected)")
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--test_cities", action="store_true",
                    help="also check the 10 unlabelled test cities")
    a = ap.parse_args()

    print(f"data_dir: {a.data_dir}")
    for p in (_img_dir(a.data_dir), _lbl_dir(a.data_dir)):
        print(f"  {'OK  ' if os.path.isdir(p) else 'MISS'} {p}")

    rows = [check_city(a.data_dir, c) for c in TRAIN_CITIES]
    print(f"\n{'city':14} {'image':>14} {'label':>14} {'values':>10} {'prev':>8}  notes")
    total = 0
    for r in rows:
        total += r.get("pixels", 0)
        print(f"{r['city']:14} {str(r.get('img_shape','-')):>14} "
              f"{str(r.get('label_shape','-')):>14} {str(r.get('label_values','-')):>10} "
              f"{r.get('prevalence', float('nan')):8.4f}  "
              f"{'; '.join(r['notes']) if r['notes'] else ''}")
    bad = [r["city"] for r in rows if not r["ok"]]
    print(f"\n{len(rows)-len(bad)}/{len(rows)} labelled cities OK"
          + (f" — PROBLEMS: {bad}" if bad else ""))
    print(f"total labelled pixels: {total/1e6:.2f} M "
          f"(exhaustive evaluation cost scales with this)")

    if a.test_cities:
        t = [check_city(a.data_dir, c, labelled=False) for c in TEST_CITIES]
        tbad = [r["city"] for r in t if not r["ok"]]
        print(f"{len(t)-len(tbad)}/{len(t)} test cities OK"
              + (f" — missing: {tbad}" if tbad else ""))

    sys.exit(1 if bad else 0)
