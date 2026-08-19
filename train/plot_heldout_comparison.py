"""
train/plot_heldout_comparison.py — the two figures that answer "which cities are
these numbers on, and is the model comparison fair?"

  city_split.png              the 24-city split: 14 labelled cities that every
                              metric comes from (coloured by CV fold), 10
                              test cities, which this project has never scored.
  heldout_city_comparison.png M1 vs M2 vs M_ring on each of the 14 held-out
                              cities, plus the paired per-city differences.

Both are built from committed artifacts (`results/runs/p3_topology/*.json` and
`data/splits.py`) — no dataset, no retraining. Numbers match
`docs/results_heldout_city_comparison.md`, which the same fold records generate.

    python train/plot_heldout_comparison.py [--depth 3] [--metric AP]
"""
import os, sys, json, glob, argparse, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "data"))

from splits import TRAIN_CITIES, TEST_CITIES

SWEEP = os.path.join(ROOT, "results", "runs", "p3_topology")
OUT = os.path.join(ROOT, "results", "p3_matrix")
# colours kept identical to plot_p3.py so the figures read as one set
KINDS = [("m1", "M1  separable", "#1f77b4", "o"),
         ("m2", "M2  spatial CZ grid", "#2ca02c", "^"),
         ("mring", "M_ring  CZ ring", "#ff7f0e", "s")]
FOLD_COLORS = ["#4c78a8", "#72b7b2", "#54a24b", "#eeca3b", "#e45756"]
PREVALENCE = 0.0229


def load(sweep=SWEEP):
    """{(kind, depth): {city: metrics}} and {city: fold}."""
    cells, fold_of = collections.defaultdict(dict), {}
    for p in sorted(glob.glob(os.path.join(sweep, "*_fold*.json"))):
        d = json.load(open(p))
        for c, m in d["per_city"].items():
            cells[(d["kind"], d["depth"])][c] = m
            fold_of.setdefault(c, d["fold"])
            if fold_of[c] != d["fold"]:
                raise ValueError(f"{c}: fold assignment differs between cells — "
                                 "the comparison would not be paired")
    return cells, fold_of


# --------------------------------------------------------------------------- #
def fig_city_split(fold_of, path):
    """Why there is no metric for 10 of the 24 cities."""
    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.set_xlim(0, 10.6); ax.set_ylim(-0.45, 6.2); ax.axis("off")

    def chip(x, y, text, face, edge, fg="black", sub=None):
        ax.add_patch(FancyBboxPatch((x, y), 1.75, 0.62, boxstyle="round,pad=0.02,rounding_size=0.09",
                                    fc=face, ec=edge, lw=1.4))
        ax.text(x + 0.875, y + (0.36 if sub else 0.31), text, ha="center",
                va="center", fontsize=9.5, color=fg)
        if sub:
            ax.text(x + 0.875, y + 0.15, sub, ha="center", va="center",
                    fontsize=7.2, color=fg, alpha=.85)

    ax.text(0, 5.85, "OSCD: 24 cities", fontsize=14, fontweight="bold")

    # --- labelled -----------------------------------------------------------
    ax.text(0, 5.35, "14 labelled cities  —  every reported metric comes from here",
            fontsize=11.5, fontweight="bold", color="#1a1a1a")
    ax.text(0, 5.05, "5-fold city-grouped CV: each city is held out IN FULL exactly once "
                     "and scored by a model trained without it.",
            fontsize=9.2, color="#444")
    for i, c in enumerate(TRAIN_CITIES):
        r, k = divmod(i, 5)
        f = fold_of.get(c, 0)
        chip(0.05 + k * 1.95, 4.15 - r * 0.82, c, FOLD_COLORS[f] + "40",
             FOLD_COLORS[f], sub=f"held out in fold {f}")

    # --- hidden -------------------------------------------------------------
    ax.text(0, 1.72, "10 test cities  —  never trained, tuned or selected on; NOT SCORED here",
            fontsize=11.5, fontweight="bold", color="#8a1a1a")
    ax.text(0, 1.42, "OSCD publishes labels for these separately; this project never used them. "
                     "So far the only output is one predicted {0,255} mask per city.",
            fontsize=9.2, color="#444")
    for i, c in enumerate(TEST_CITIES):
        r, k = divmod(i, 5)
        chip(0.05 + k * 1.95, 0.52 - r * 0.82, c, "#dddddd", "#999999", fg="#333")

    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
def fig_comparison(cells, fold_of, depth, metric, path):
    """Per-city model comparison + paired differences, on held-out cities only."""
    present = [k for k in KINDS if (k[0], depth) in cells]
    if not present:
        sys.exit(f"no cells at depth {depth}")
    n_params = json.load(open(glob.glob(os.path.join(
        SWEEP, f"{present[0][0]}_L{depth}_fold*.json"))[0]))["n_params"]

    base = cells[("m1", depth)]
    order = sorted(TRAIN_CITIES, key=lambda c: base[c][metric])   # worst at bottom
    y = np.arange(len(order))

    fig, ax = plt.subplots(1, 2, figsize=(13.6, 5.6),
                           gridspec_kw={"width_ratios": [1.25, 1]})

    # --- panel A: three circuits on the same held-out city -------------------
    for i, c in enumerate(order):
        vals = [cells[(k, depth)][c][metric] for k, *_ in present]
        ax[0].plot([min(vals), max(vals)], [i, i], color="#bbbbbb", lw=1.2, zorder=1)
    for kind, name, col, mk in present:
        v = [cells[(kind, depth)][c][metric] for c in order]
        ax[0].scatter(v, y, color=col, marker=mk, s=62, zorder=3, label=name,
                      edgecolor="white", lw=.7)
    prev = [base[c]["prevalence"] for c in order]
    ax[0].scatter(prev, y, marker="|", color="black", s=70, zorder=4,
                  label="chance level = city prevalence")
    ax[0].set_yticks(y); ax[0].set_yticklabels(order, fontsize=9.5)
    ax[0].set_xlabel(f"{metric} on the held-out city")
    ax[0].set_title(f"Same held-out city, three circuits  ({n_params} parameters)\n"
                    "every point is a city the model never trained on",
                    fontsize=11.5)
    ax[0].legend(frameon=False, fontsize=9, loc="lower right")
    ax[0].grid(axis="x", alpha=.25)
    ax[0].grid(axis="y", alpha=.12, ls="-")
    ax[0].set_axisbelow(True)
    ax[0].set_xlim(left=0)

    lo, hi = base[order[0]][metric], base[order[-1]][metric]
    ax[0].annotate(f"city spread for one model:\n{lo:.3f} … {hi:.3f}  ({hi/max(lo,1e-9):.1f}×)",
                   xy=(hi, len(order) - 1), xytext=(hi * 0.42, len(order) - 3.4),
                   fontsize=9, color="#333",
                   arrowprops=dict(arrowstyle="->", color="#666", lw=1))

    # --- panel B: paired differences vs M1 ----------------------------------
    ax[1].axvline(0, color="black", lw=1.1, zorder=2)
    lines, biggest_gap = [], 0.0
    for kind, name, col, mk in present:
        if kind == "m1":
            continue
        d = np.array([base[c][metric] - cells[(kind, depth)][c][metric] for c in order])
        biggest_gap = max(biggest_gap, float(np.abs(d).max()))
        ax[1].scatter(d, y, color=col, marker=mk, s=62, zorder=3,
                      edgecolor="white", lw=.7, label=f"M1 − {name.split('  ')[0]}")
        ax[1].axvline(d.mean(), color=col, ls="--", lw=1.2, alpha=.8, zorder=1)
        lines.append(f"M1 − {name.split('  ')[0]}:  {int((d > 0).sum())}/{len(d)} cities, "
                     f"mean Δ {d.mean():+.3f}")
    ax[1].set_yticks(y); ax[1].set_yticklabels([])
    ax[1].set_xlabel(f"paired per-city difference in {metric}   "
                     "(right of 0 = M1 better)")
    ax[1].set_title("Paired differences — identical folds, initialisation\n"
                    "and patch stream, so the entangler is the only difference",
                    fontsize=11.5)
    ax[1].legend(frameon=False, fontsize=9, loc="lower right")
    ax[1].grid(axis="x", alpha=.25)
    ax[1].grid(axis="y", alpha=.12, ls="-")   # same row order as panel A
    ax[1].set_axisbelow(True)
    ax[1].text(0.02, 0.985, "\n".join(lines), transform=ax[1].transAxes, va="top",
               fontsize=9.2, bbox=dict(fc="white", ec="#cccccc", boxstyle="round,pad=0.4"))

    ax[1].text(0.5, -0.155, f"city spread ({hi - lo:.3f}) dwarfs every per-city "
                            f"architecture gap (largest {biggest_gap:.3f})",
               transform=ax[1].transAxes, ha="center", fontsize=9.2, color="#8a1a1a")

    fig.suptitle("Model comparison on cities the models never saw — "
                 "5-fold city-grouped CV over the 14 labelled cities",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--metric", default="AP")
    ap.add_argument("--outdir", default=OUT)
    a = ap.parse_args()

    cells, fold_of = load()
    os.makedirs(a.outdir, exist_ok=True)
    p1 = fig_city_split(fold_of, os.path.join(a.outdir, "city_split.png"))
    p2 = fig_comparison(cells, fold_of, a.depth, a.metric,
                        os.path.join(a.outdir, "heldout_city_comparison.png"))
    print(f"-> {p1}\n-> {p2}")


if __name__ == "__main__":
    main()
