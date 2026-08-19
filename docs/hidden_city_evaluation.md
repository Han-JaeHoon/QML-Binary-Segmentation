# Evaluating on the 10 test cities — outstanding work

Raised in review: *"the 5-fold CV is over the training cities only — isn't the
point of the 10 validation cities to compare the two models on them? The results
section only reports the 14."*

**The review is right, and the answer is yes: we should evaluate on the 10, and
have not.** An earlier version of this note claimed their ground truth was
withheld. That was wrong — **OSCD publishes labels for the test cities**, as a
separate download alongside the train labels
([dataset page](https://rcdaudt.github.io/oscd/); `torchgeo`'s OSCD loader
exposes a `test` split with masks). This project simply never downloaded or used
them: `data/preprocess.py` only ever resolves
`train_labels/… - Train Labels`, and no code path in the repo references test
labels at all.

So the 10 cities are unscored, not unscorable. Nothing in the project is trained,
tuned or selected on them — which is why the model selection below stands on its
own — but a test-set number is genuinely missing and this note says what it takes
to produce one.

---

## 1. The split, precisely

![the 24-city split](../results/p3_matrix/city_split.png)

| | cities | labels used here | role here |
|---|---|---|---|
| train | 14 | yes | transform fitting, training, 5-fold city-grouped CV, threshold choice |
| test | 10 | **no — never downloaded** | predicted only; not scored, and not used for any decision |

What exists for the 10 cities today is the deliverable and its sanity check:

- 10 uint8 `{0,255}` PNG masks (`results/submission/masks/`)
- per-city predicted change fraction (`results/submission/predict_m1_L3.json`)
- a threshold-transfer check: the frozen out-of-fold operating point implies a
  4.99 % positive rate, the test masks average 5.19 %

That check confirms τ carried over without mis-scaling. It is **not** an accuracy
measurement — it never sees a label.

## 2. What the 14-city protocol already gives you

The 5-fold CV is **city-grouped**, not a random pixel split: each of the 14
labelled cities is held out in full exactly once and scored by a model trained
without it. So every number in the results section is already a *leave-city-out*
number — cross-city generalization measured on 14 held-out cities, with matched
pairing across architectures (same folds, same init, same patch stream). This is
the right basis for *choosing* between architectures; it does not replace a
test-set number for *reporting*.

![held-out city comparison](../results/p3_matrix/heldout_city_comparison.png)

*Left: the same held-out city scored by all three circuits. Right: the paired
per-city difference against M1 — dashed lines are the means. Every point is a
city that model never trained on.*

[`results_heldout_city_comparison.md`](results_heldout_city_comparison.md) is
that comparison written out per city — M1 vs M2 vs M_ring, one row per held-out
city, with paired differences and win counts. Regenerate both with:

```bash
python train/compare_heldout_cities.py            # tables,  no dataset, no GPU, ~1 s
python train/plot_heldout_comparison.py           # figures, no dataset, no GPU, ~2 s
```

At L3 (110 params) it reports M1 ahead on macro AP 0.1748 vs 0.1358 (M2) and
0.1226 (M_ring), winning 11/14 and 14/14 held-out cities respectively — while the
city-to-city spread for a single model is 0.023…0.458, about 20×. The binding
constraint is which city you test on, not which circuit you use.

## 3. Scoring the 10 — how

Download the OSCD **test** labels (a separate, small archive from the same
source as the train labels — masks only, no imagery). Then
`train/score_hidden_cities.py` produces the same table for the 10 cities:

```bash
python train/score_hidden_cities.py \
    --label_root /path/to/test_labels \
    --pred "M1 L3=results/submission/masks"
```

It searches `--label_root` recursively for `<city>/cm/<city>-cm.tif` (and the
PNG/flat variants), decodes `{1 = no change, 2 = change}` exactly as
`preprocess._load_label` does, crops predictions and labels to their common
extent, and writes `results/submission/hidden_city_scores.json` plus
`docs/results_hidden_cities.md`.

**τ is frozen.** The default operating point is `tau_final` from
`results/submission/threshold_m1_L3.json`, chosen out-of-fold before any test
city was touched; the script has no code path that selects a threshold on test
pixels. That property is what makes a test number meaningful once we have one.

### Two granularities, and why it matters

| input | metrics recoverable |
|---|---|
| `<city>_prob.npz` (probability map) | everything, including **AP** (primary) and ROC-AUC |
| `<city>.png` (the committed `{0,255}` mask) | F1 / precision / Change-Accuracy at the frozen τ only |

The masks are already thresholded, so AP and ROC-AUC cannot be reconstructed from
them and are reported as `n/a` rather than computed from a binary vector (which
would silently return the operating point dressed up as a ranking metric). The
probability maps are covered by `.gitignore` (`*.npz`) and are **not** in the
repo — they exist only on the machine that ran the pipeline. If that machine
still has `results/submission/masks/*_prob.npz`, point `--pred` at it and the full
table comes out in under a minute with no recomputation.

### Comparing architectures on the 10 cities

Only M1 L3 was ever trained on all 14 and run over the test cities, so a
model-vs-model comparison there needs the comparison architecture put through the
same two stages. `--tau` carries the frozen M1 operating point over to a model
that has no out-of-fold threshold file of its own, and masks now land in
`masks_<kind>_L<depth>/` so nothing overwrites the frozen submission:

```bash
python submit/final_pipeline.py train   --kind m2 --depth 3 --data_dir /path/to/OneraDataset
python submit/final_pipeline.py predict --kind m2 --depth 3 --data_dir /path/to/OneraDataset \
       --tau 0.5808
python train/score_hidden_cities.py --label_root /path/to/test_labels \
    --pred "M1 L3=results/submission/masks_m1_L3" \
    --pred "M2 L3=results/submission/masks_m2_L3"
```

## 4. Cost

Measured from the committed run records (`results/submission/final_m1_L3.json`,
`predict_m1_L3.json`, `results/runs/p3_topology/*_fold*.json`) on the machine that
produced them; the entangled circuits are scaled by their per-fold time ratio
against M1 (M2 ×1.31, M_ring ×1.22).

| stage | M1 L3 | M2 L3 | M_ring L3 |
|---|---|---|---|
| final train, 14 cities, 50 epochs | 35 min (measured) | ~46 min | ~43 min |
| predict 10 cities, stride-1 | 52 min (measured) | ~68 min | ~63 min |
| **per model** | **~1.5 h** | **~1.9 h** | **~1.8 h** |

Scoring itself is under a minute. Three models end to end is ~5 h sequentially,
or ~2 h if the three are run as separate processes. M1 has to be re-run too: its
weights (`results/submission/*_model.npz`) and probability maps are gitignored and
absent from the repo.

Doing this at every depth instead of L3 only would be 9 models, ~15 h sequential —
not worth it: depth is already covered by the capacity sweep on the 14 cities.

## 5. One discipline point

The submitted model was selected on out-of-fold AP over the 14 labelled cities and
is frozen. When test-city scores are produced, report them as a final evaluation —
do not re-pick the architecture, depth or threshold on them. Selecting on the test
cities would turn the only unbiased number in the project into another validation
number, and the poster's claims are all stated against the CV protocol.
