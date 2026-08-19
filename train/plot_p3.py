"""Figures for the P3 matrix: capacity curves and the per-city spread."""
import json, glob, os, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SWEEP = os.path.join(ROOT, "results", "runs", "p3_topology")
OUT = os.path.join(ROOT, "results", "p3_matrix")
KINDS = [("m1", "M1  separable", "#1f77b4", "o"),
         ("mring", "M_ring  CZ ring", "#ff7f0e", "s"),
         ("m2", "M2  spatial CZ grid", "#2ca02c", "^")]
DEPTHS = [(1, 38), (2, 74), (3, 110)]
PREV = 0.0229

cells, cities = collections.defaultdict(dict), collections.defaultdict(dict)
for p in glob.glob(os.path.join(SWEEP, "*.json")):
    d = json.load(open(p))
    cells[(d["kind"], d["depth"])][d["fold"]] = d["pooled"]["AP"]
    for c, m in d["per_city"].items():
        cities[(d["kind"], d["depth"])][c] = m

fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))

# --- capacity curves ------------------------------------------------------
xs = [p for _, p in DEPTHS]
for k, name, col, mk in KINDS:
    m = [np.mean(list(cells[(k, L)].values())) for L, _ in DEPTHS]
    s = [np.std(list(cells[(k, L)].values())) / np.sqrt(5) for L, _ in DEPTHS]
    ax[0].errorbar(xs, m, yerr=s, marker=mk, color=col, label=name,
                   capsize=4, lw=2, ms=8)
ax[0].axhline(PREV, ls="--", c="gray", lw=1)
ax[0].text(40, PREV * 1.15, f"chance (prevalence {100*PREV:.2f}%)", color="gray", fontsize=9)
ax[0].set_xticks(xs); ax[0].set_xlabel("trainable parameters")
ax[0].set_ylabel("mean fold AP  (5 city-grouped folds)")
ax[0].set_title("Capacity curve — only the separable model\nconverts parameters into accuracy")
ax[0].legend(frameon=False); ax[0].grid(alpha=.25); ax[0].set_ylim(0, 0.20)

# --- per-city spread of the best model ------------------------------------
best = cities[("m1", 3)]
order = sorted(best, key=lambda c: best[c]["AP"])
apv = [best[c]["AP"] for c in order]
prv = [best[c]["prevalence"] for c in order]
y = np.arange(len(order))
ax[1].barh(y, apv, color="#1f77b4", label="AP (M1 L3)")
ax[1].plot(prv, y, "k.", ms=9, label="chance level = prevalence")
ax[1].set_yticks(y); ax[1].set_yticklabels(order, fontsize=9)
ax[1].set_xlabel("Average Precision")
ax[1].set_title(f"Same model, 14 held-out cities:\n"
                f"{max(apv) / max(min(apv), 1e-9):.1f}x spread is the dominant effect")
ax[1].legend(frameon=False, loc="lower right"); ax[1].grid(alpha=.25, axis="x")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "p3_summary.png"), dpi=140, bbox_inches="tight")
print("saved", os.path.join(OUT, "p3_summary.png"))
