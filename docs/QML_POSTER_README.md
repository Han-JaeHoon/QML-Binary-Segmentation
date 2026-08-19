# QML section — everything needed to write the poster

Self-contained reference for the quantum side of *Quantum Change Detection in
Satellite Earth Observations*. Sections 1–4 map onto the four things the
challenge asks every submission to document; sections 5–8 give the protocol,
metrics and results; section 10 lists what may and may not be claimed.

Every number below is produced by the committed code and stored in
[`../results/p3_matrix/matrix.csv`](../results/p3_matrix/matrix.csv) and
`../results/runs/p3_topology/*.json`.

---

## 0. One paragraph

We built a 9-qubit variational quantum classifier that labels each pixel of a
Sentinel-2 image pair as urban change or not, using a 3×3 spatial patch of
median-corrected spectral-change features. To test whether the *quantum* part
contributes, we compared three circuits that are identical in every respect
except their entangler — none, a CZ ring, and a CZ grid matched to the 2D pixel
layout — at three parameter budgets (38 / 74 / 110), under 5-fold city-grouped
cross-validation with paired controls (same folds, same initialisation, same
training stream). The separable circuit won at every budget, and it was the only
one that converted extra parameters into accuracy. The best model reaches
**AP 0.110 out-of-fold against a 0.023 chance level (4.8× lift)**, with
**F1 0.199** at its operating point. The dominant source of variation is not the
circuit but which city is held out: the same model scores AP 0.023 on one city
and 0.458 on another.

---

## 1. Data preprocessing and resampling

**Source.** OSCD / Onera, Sentinel-2 MSI. 24 cities, **14 with labels** (train)
and 10 hidden (test). 13 bands × 2 dates per city.

**Resampling.** We use the dataset's `imgs_*_rect` products, in which every band
— native 10 m (B02–04, B08), 20 m (B05–07, B8A, B11, B12) and 60 m (B01, B09,
B10) — is already resampled to a common **10 m grid** and co-registered, so all
26 rasters share one pixel lattice. No further resampling is applied.

**Feature construction** (identical for every model):

```
1. robust per-band normalisation   B̃ = clip((B − P1)/(P99 − P1), 0, 1)
                                   P1/P99 fitted on TRAIN cities only
2. temporal difference             ΔB = B̃(T2) − B̃(T1)
3. per-pair median correction      ΔBᶜᵒʳʳ = ΔB − medianₚ(ΔB)     ← per image, label-free
4. magnitude                       D = |ΔBᶜᵒʳʳ|                   (13 channels)
5. dimensionality reduction        z = PCA₄(D),  u = clip(z/c_pc, −1, 1)
6. patch                           3×3 neighbourhood → 3×3×4 input
```

Why each step, from the EDA (all city-grouped):

| step | evidence |
|---|---|
| use magnitude \|ΔB\|, not signed ΔB | signed ΔB has AUC ≈ 0.50 per band; \|ΔB\| reaches 0.79 |
| per-pair median correction | per-city median ΔB differs by ~1500 DN (B8A), so raw \|ΔB\| partly encodes *city identity*; correcting lifts AUC **0.811 → 0.864** |
| spatial patch instead of single pixel | mean-pooled \|ΔBᶜᵒʳʳ\|: 1×1 0.864 → 3×3 0.884 → 5×5 0.896 → 7×7 0.902 |
| PCA-4 | on the common 13-band base, PCA-4 0.865 ≈ all-13 0.864 > physical-4 {B04,B05,B12,B08} 0.855 |
| spectral indices dropped | ΔNDVI/ΔNDBI/ΔNDWI reach only 0.56–0.62 — normalised-difference indices divide out the brightness magnitude that carries the signal |

**Leakage discipline.** The median correction uses only each image's own pixels
and no labels, so it applies to train, validation and hidden-test cities alike.
Everything fitted — band P1/P99, the PCA basis, the scaling constants — is fitted
on **training cities only** and re-fitted inside every CV fold.

**Class imbalance.** Change is **2.29 %** of pixels. Training patches are drawn
by a city-balanced sampler: *city uniform → category → coordinate*, with
categories positive : hard-negative : ordinary-negative = **1 : 1 : 2**. A hard
negative is an unchanged pixel whose change magnitude exceeds
`max(Q80_city, T_global)` — "changed a lot but not urban", the case the task is
really about. The sampler yields a **24.8 %** positive rate at the patch centre,
so plain BCE is used; applying the raw 1:43 imbalance weight on top would
double-correct.

---

## 2. Quantum encoding strategy

**One qubit per spatial pixel.** A 3×3 patch maps to 9 qubits; qubit *q* carries
pixel *q*:

```
q0 q1 q2
q3 q4 q5     (q4 = the pixel being classified)
q6 q7 q8
```

**Angle encoding in two spectral stages.** Each pixel has 4 features, which do
not fit one qubit at once, so they are uploaded in two stages separated by a
trainable layer (data re-uploading):

```
E1 :  RY(π·u₁) RZ(π·u₂)        stage-1 features
E2 :  RY(π·u₃) RZ(π·u₄)        stage-2 features
```

**Why angle and not amplitude encoding.** The EDA showed the leading signal is
the *overall magnitude* of change (PC1 alone explains 59 % of the variance of
|ΔB|). Amplitude encoding normalises `‖x‖` away and would discard exactly that;
angle encoding preserves it. Amplitude encoding was therefore not used.

**Readout.** All nine qubits are measured and combined by a **parameter-free**
average, then a shared affine calibration:

```
z_q = ⟨Z_q⟩ ,   m = (1/9) Σ_q z_q ,   p = σ(a·m + b)
```

Measuring only the centre qubit would leave 32 of the 36 mixer parameters
structurally dead in the separable model (a product state makes ⟨Z₄⟩ depend on
pixel 4 alone), which would make the "same parameter count" comparison a fiction.
The mean-pool readout keeps every parameter connected to the output.

---

## 3. QNN architectures

All three share the same encoding, the same trainable single-qubit mixers, the
same re-uploading structure and the same readout. **The entangler is the only
difference.** One depth block is:

```
E1 → ENT → V1 → E2 → ENT → V2        V = RY(θ) RX(θ) on every qubit
```

`V` must not commute with Z: the entanglers are diagonal, so `[Z_iZ_j, Z_i] = 0`
and a diagonal layer alone would be invisible to a Z measurement.

| model | entangler | edges/stage | figure |
|---|---|---|---|
| **M1 — separable** | none | 0 | `results/circuits/circuit_m1.png` |
| **M_ring — CZ ring** | CZ on `(0,1)(1,2)…(7,8)(8,0)` | 9 | `results/circuits/circuit_mring.png` |
| **M2 — spatial CZ grid** | CZ on the 12 nearest-neighbour edges of the 3×3 pixel lattice | 12 | `results/circuits/circuit_m2.png` |

All entanglers are **fixed and non-trainable**; CZ is diagonal and commuting, so
each layer is order-free.

**What M1 is and is not.** With the mean-pool readout M1 still *reads* all nine
pixels — it combines them **additively, with no interaction terms**. Verified
numerically: the mixed second difference of the pre-sigmoid score is
`|I_ij| ≈ 5×10⁻¹⁶` for M1 (machine zero) versus `≈10⁻²` for M_ring and M2. So the
ladder is **separable/additive → interacting/entangled**, not "no spatial
context → spatial context".

**Depth.** `L ∈ {1,2,3}` repeats the whole block with **untied** parameters (each
block has its own mixers; the same 4 features are re-uploaded each time).

> **A fourth circuit, M3, exists but is not in the headline comparison.**
> M3 replaces CZ with a *data-dependent* `IsingZZ(γ·s_i·s_j)` coupling
> (`results/circuits/circuit_m3.png`). A diagnostic showed that with the current
> scaling the rotation angle is below 0.1 rad for ~90 % of neighbour pairs — the
> gate is near-identity almost everywhere — so M3 as configured has never had a
> fair test and is reported only as a design note.

---

## 4. Parameter counting (needed for the classical comparison)

Per depth block: 9 qubits × 2 stages × 2 rotations = **36** trainable mixer
angles. Plus the shared readout `(a, b)` = 2. Entanglers contribute **0**.

```
N(L) = 36·L + 2      →     L1: 38      L2: 74      L3: 110
```

This is **identical for M1, M_ring and M2** at each depth — verified in
`train/accept_mring.py`, which also reads the executed circuit tape to confirm
the entangler is applied 2/4/6 times at L1/L2/L3 (9-edge ring → 18/36/54 CZ;
12-edge grid → 24/48/72 CZ).

**Classical counterpart** (to be run by the classical teammate): a single 3×3
same-padding convolution, 4 input channels → 1 output,
`4·3·3 + 1 = 37 parameters`, on the same PCA-4 features, same folds, same
sampler, same loss, same threshold procedure. That gives the headline
**38-parameter quantum vs 37-parameter classical** comparison at L1.

Report it as *"same input, same receptive field, near-identical trainable
parameter budget"* — **not** "identical except quantumness": the quantum mixers
are position-dependent while a convolution kernel is weight-shared.

---

## 5. Training protocol (identical for all cells)

| item | value |
|---|---|
| optimiser | Adam, lr = 0.02 |
| batch | 32 patches |
| epoch | 320 steps = 10,240 patches (the sampler is an infinite stream, so an "epoch" is defined by steps) |
| epochs | 50, fixed |
| loss | plain BCE on the centre-pixel label |
| simulator | PennyLane `default.qubit`, backprop |
| seed | 0 |
| checkpoint | the **final** epoch — validation never selects a checkpoint |

The budget was fixed **before** the comparison, from a convergence diagnostic on
a development split, and was chosen on *training* loss only (choosing it on
validation would leak, because the development validation cities reappear as
CV validation cities). Both the criterion and the fallback were written down
before the runs.

**Paired controls, asserted in-process for every fold:** the fold is built once
and shared, so the arms use the same fold-fitted transforms; `same_init` (the
38/74/110 initial values are identical); `same_stream` (the per-epoch CRC32 of
the sampled `(city,row,col)` sequence matches). **The entangler is therefore the
only difference between arms.** As an external check, re-running M1 L1 in the new
harness reproduced the original run's per-epoch training loss to four decimals.

**Cost.** 45 cells (9 configurations × 5 folds), ≈ 21 CPU-hours, run through an
8-worker pool.

---

## 6. Evaluation protocol and metrics

**Cross-validation.** 5-fold **city-grouped**: each of the 14 labelled cities is
a held-out validation city exactly once. Never a random pixel split — the EDA
showed random splits are optimistic (a land-cover feature scored 0.59 under
random CV and 0.53, i.e. chance, under city-grouped CV).

**Why the 10 test cities are not the validation set.** They have no labels
(`data/splits.py` — `TEST_CITIES`, predict-only), so they cannot score anything;
the model comparison has to live on the 14 labelled cities. Because the CV holds
out whole cities, that comparison is still cross-city: each model is scored on
cities it never trained on, 14 of them, paired across architectures (identical
folds, initialisation and patch stream). See
[`hidden_city_evaluation.md`](hidden_city_evaluation.md).

**Inference.** Stride-1 over the whole held-out city at the natural 2.29 %
prevalence; reflect-padding supplies context at borders, and outputs are
accumulated only inside the original raster.

**Metrics.**

| metric | role |
|---|---|
| **AP** (Average Precision, `average_precision_score`) | **primary**, threshold-free, chance level = prevalence = 0.0229 |
| F1\* | best-operating-point **diagnostic** — its threshold is swept on the very pixels it scores, so it is optimistic, not an unbiased test value |
| Change-Accuracy | recall on changed pixels |
| No-change-Accuracy | specificity |
| Accuracy | reported because the challenge asks for it, but **not** a headline: predicting "no change" everywhere already scores 97.7 % |
| ROC-AUC | secondary; over-optimistic under extreme imbalance |
| train BCE | optimisation sanity check against the constant-predictor baseline H(0.248) = 0.560 |

**Statistics.** 5 folds, and cities inside a fold share a trained model, so the
14 cities are not independent samples. Differences are reported as paired
per-fold values and win counts. **No p-values** — with 5 folds a signed-rank test
cannot reach p < 0.05 even in principle.

---

## 7. Results

> **Which cities these numbers are on.** Every metric in this section is over the
> **14 labelled cities**, each one held out in full exactly once by the
> city-grouped CV — so they are leave-city-out numbers, not in-sample ones. The
> other **10 cities carry no ground truth** (`data/splits.py` — `TEST_CITIES`,
> predict-only), so no accuracy can be computed on them here; their deliverable is
> a predicted mask per city, plus a threshold-transfer check (§9). Per-city,
> model-vs-model tables are in
> [`results_heldout_city_comparison.md`](results_heldout_city_comparison.md);
> what it would take to score the 10 hidden cities if their labels are released
> is in [`hidden_city_evaluation.md`](hidden_city_evaluation.md).

### 7.1 Architecture × depth (mean fold AP, 5 city-grouped folds)

| | L1 · 38p | L2 · 74p | L3 · 110p |
|---|---|---|---|
| **M1 separable** | 0.1210 ± 0.076 | 0.1573 ± 0.083 | **0.1728 ± 0.089** |
| M_ring CZ ring | 0.1095 ± 0.067 | 0.1102 ± 0.049 | 0.1127 ± 0.055 |
| M2 spatial CZ grid | 0.1177 ± 0.075 | 0.1091 ± 0.053 | 0.1329 ± 0.056 |

Per-fold AP (folds 0–4):

```
m1_L1     0.2519 0.0224 0.1299 0.1222 0.0784
m1_L2     0.2944 0.0350 0.1578 0.1602 0.1393
m1_L3     0.3084 0.0395 0.1508 0.1473 0.2180
mring_L1  0.2250 0.0206 0.1151 0.1098 0.0771
mring_L2  0.1805 0.0277 0.1141 0.1083 0.1202
mring_L3  0.2046 0.0318 0.1204 0.1104 0.0962
m2_L1     0.2447 0.0211 0.1344 0.1175 0.0706
m2_L2     0.1975 0.0310 0.0991 0.1022 0.1159
m2_L3     0.2005 0.0335 0.1247 0.1684 0.1376
```

### 7.2 Challenge metrics — macro over the 14 held-out cities

| model | AP | ROC-AUC | F1\* | Change-Acc | No-change-Acc | Accuracy |
|---|---|---|---|---|---|---|
| **M1 L3** | **0.1748** | 0.8374 | **0.2406** | 0.3378 | 0.9683 | 0.9564 |
| M1 L2 | 0.1616 | 0.8355 | 0.2309 | 0.3440 | 0.9648 | 0.9529 |
| M1 L1 | 0.1419 | 0.8058 | 0.1994 | 0.3469 | 0.9262 | 0.9147 |
| M2 L3 | 0.1358 | 0.8311 | 0.2014 | 0.3250 | 0.9608 | 0.9494 |
| M2 L1 | 0.1276 | 0.8000 | 0.1877 | 0.3189 | 0.9498 | 0.9378 |
| M_ring L3 | 0.1226 | 0.8264 | 0.1855 | 0.3459 | 0.9472 | 0.9364 |
| M_ring L1 | 0.1220 | 0.8034 | 0.1813 | 0.3618 | 0.9170 | 0.9061 |
| M2 L2 | 0.1182 | 0.8223 | 0.1858 | 0.3343 | 0.9533 | 0.9416 |
| M_ring L2 | 0.1145 | 0.8229 | 0.1824 | 0.3441 | 0.9432 | 0.9327 |

### 7.3 Pooled out-of-fold (one global threshold, every labelled pixel once)

| model | AP | ROC-AUC | F1\* | Change-Acc | No-change-Acc | Accuracy | τ\* |
|---|---|---|---|---|---|---|---|
| **M1 L3** | **0.1096** | 0.8137 | 0.1990 | 0.316 | 0.956 | 0.9417 | 0.581 |
| M1 L2 | 0.1010 | 0.8049 | 0.1886 | 0.324 | 0.950 | 0.9362 | 0.550 |
| M2 L3 | 0.0941 | 0.8108 | 0.1479 | 0.270 | 0.944 | 0.9289 | 0.522 |
| M_ring L2 | 0.0797 | 0.8018 | 0.1421 | 0.358 | 0.914 | 0.9012 | 0.475 |
| M_ring L3 | 0.0793 | 0.8052 | 0.1410 | 0.463 | 0.880 | 0.8709 | 0.443 |
| M1 L1 | 0.0764 | 0.8025 | 0.1480 | 0.384 | 0.911 | 0.8988 | 0.454 |
| M2 L2 | 0.0769 | 0.7971 | 0.1387 | 0.311 | 0.926 | 0.9115 | 0.492 |
| M_ring L1 | 0.0715 | 0.7991 | 0.1348 | 0.398 | 0.895 | 0.8831 | 0.440 |
| M2 L1 | 0.0711 | 0.7940 | 0.1346 | 0.402 | 0.893 | 0.8816 | 0.442 |

The pooled figure is lower than the fold mean because one global ranking has to
hold across cities whose score scales differ.

### 7.4 What the numbers mean in practice

At the submission operating point (M1 L3, τ = 0.581), per **1,000,000 pixels**:

| | pixels |
|---|---|
| actually changed | 22,900 |
| flagged by the model | 50,229 |
| ├ correct | **7,283** → precision **14.5 %** |
| └ false alarm | **42,946** |
| missed | 15,617 → recall **31.6 %** |

**AP 0.110 against a 0.023 chance level = 4.8× lift.** The model ranks pixels
usefully (ROC-AUC 0.81–0.84) but is not sharp enough to produce a clean binary
mask at this imbalance.

*Scale reference:* a purely classical linear probe on all 13 median-corrected
bands (from the EDA, city-grouped) reaches ROC-AUC ≈ 0.864 pixel-wise and ≈ 0.902
with 7×7 pooling. A 110-parameter quantum circuit at ROC-AUC ≈ 0.84 is in the
same range but has not beaten the unconstrained classical probe.

### 7.5 Per-city results, best model (M1 L3)

| city | prevalence | AP | lift | ROC-AUC | F1\* | Change-Acc |
|---|---|---|---|---|---|---|
| rennes | 2.58 % | 0.4575 | 17.7× | 0.961 | 0.505 | 0.573 |
| cupertino | 2.37 % | 0.4213 | 17.8× | 0.960 | 0.466 | 0.555 |
| beihai | 2.49 % | 0.2638 | 10.6× | 0.889 | 0.333 | 0.374 |
| beirut | 2.69 % | 0.2237 | 8.3× | 0.880 | 0.312 | 0.399 |
| mumbai | 2.56 % | 0.2229 | 8.7× | 0.851 | 0.269 | 0.341 |
| nantes | 1.14 % | 0.2045 | 18.0× | 0.937 | 0.293 | 0.379 |
| hongkong | 3.56 % | 0.1697 | 4.8× | 0.789 | 0.255 | 0.332 |
| bercy | 0.74 % | 0.1297 | 17.6× | 0.819 | 0.217 | 0.235 |
| aguasclaras | 1.64 % | 0.1024 | 6.2× | 0.768 | 0.187 | 0.211 |
| bordeaux | 1.00 % | 0.0804 | 8.0× | 0.855 | 0.154 | 0.167 |
| abudhabi | 3.76 % | 0.0679 | 1.8× | 0.653 | 0.124 | 0.259 |
| pisa | 1.64 % | 0.0496 | 3.0× | 0.730 | 0.114 | 0.235 |
| paris | 0.29 % | 0.0307 | 10.7× | 0.943 | 0.075 | 0.561 |
| saclay_e | 0.99 % | 0.0232 | 2.3× | 0.687 | 0.065 | 0.109 |

### 7.6 Final training loss (mean over folds; constant-predictor baseline 0.560)

| | L1 | L2 | L3 |
|---|---|---|---|
| M1 | 0.4665 | 0.4358 | **0.4299** |
| M_ring | 0.4754 | 0.4519 | 0.4502 |
| M2 | 0.4745 | 0.4507 | 0.4479 |

---

## 8. The three findings (poster body)

### Finding 1 — adding entanglement did not help, at any budget or topology

Paired per-fold differences against M1 (same folds, same initialisation, same
training stream — the entangler is the only difference):

| | 38p | 74p | 110p |
|---|---|---|---|
| M_ring − M1 | −0.0114 (0/5 wins) | −0.0472 (0/5) | −0.0601 (0/5) |
| M2 − M1 | −0.0033 (1/5) | −0.0482 (0/5) | −0.0399 (1/5) |

The entangled arms lose **28 of 30** paired comparisons, and they are also worse
on the **training** objective (§7.6) — so this is not a fit-versus-generalisation
trade; both axes are worse. Crucially, the entanglers *do* create genuine
inter-pixel interaction, established **before** training
(`|I_ij| ≈ 10⁻²` vs M1's `10⁻¹⁶`), which rules out the obvious objection that
"the entangler did nothing".

### Finding 2 — only the separable circuit converts parameters into accuracy

38 → 110 parameters: **M1 +0.052**, M_ring +0.003, M2 +0.015. The gap between M1
and the entangled arms widens from 0.012 at 38p to 0.060 at 110p — extra capacity
makes the entangled circuits *relatively worse*, not better.

### Finding 3 — the held-out city matters far more than the circuit

Same model, 14 cities: **AP 0.023 (saclay_e) → 0.458 (rennes), a ~20× spread**,
an order of magnitude larger than any architecture difference. Cross-city domain
shift, not circuit design, is the binding constraint — consistent with the EDA,
where land-cover features that separated urban from natural change under random
CV collapsed to chance under city-grouped CV.

---

## 9. Submission

**Model:** M1, depth 3, **110 parameters** (leads every aggregation).

1. τ\* = **0.5808** chosen from pooled out-of-fold predictions over all
   **6,516,692** labelled pixels (each exactly once), then frozen.
2. Retrained on all 14 labelled cities with the frozen protocol; band
   normalisation and PCA re-fitted on those 14. Final train BCE 0.4327.
3. Predicted the 10 hidden cities → pixel-aligned **uint8 {0,255} PNG** masks,
   validated for dtype, values and shape.

Threshold transfer check: the OOF operating point implies a 4.99 % positive rate;
the test masks average 5.19 %, so τ carried over without silently mis-scaling.

Predicted change fraction per city: brasilia 1.00, montpellier 4.80, norcia
10.83, rio 4.06, saclay_w 10.82, valencia 5.59, dubai 6.57, lasvegas 5.62,
milano 0.55, chongqing 2.09 (%).

---

## 10. Wording guards — what may and may not be claimed

**Safe to write:**
- "Under matched trainable-parameter budgets (38/74/110), adding a fixed CZ
  entangler — ring or spatially aligned grid — did not improve cross-city
  generalisation in this setup."
- "Only the separable circuit converted additional parameters into held-out
  performance."
- "Performance differences across separable, ring-entangled and spatial-grid
  circuits show sensitivity to entangler structure at a fixed parameter budget."
- "A 110-parameter quantum circuit extracts real change signal, 4.8× above the
  chance level."

**Must not be written:**
- *quantum advantage* — the parameter-matched classical comparison is still
  outstanding.
- *entanglement is useless* / *spatial entanglement is universally harmful* —
  two fixed entanglers, one encoding, one readout, one task, one seed per fold.
- *pure topology effect* / *geometry-agnostic control* for M_ring vs M2 — on a
  3×3 raster the ring shares **6 of its 9 edges** with real horizontal
  neighbours, and uses 9 gates/stage against the grid's 12. It is an
  architecture-sensitivity control, not a topology-only one.
- any significance claim (p < 0.05) — see §6.
- treating **Accuracy** as the headline result.
- mixing the earlier dense-branch (3×3→3×3) capacity plateau into the
  interpretation of these centre-branch L3 numbers; they are different
  architectures.

---

## 11. Figures

| file | content |
|---|---|
| `results/p3_matrix/p3_summary.png` | capacity curves + per-city spread (main results figure) |
| `results/circuits/circuit_m1.png` | M1 circuit, barrier-separated stages |
| `results/circuits/circuit_mring.png` | M_ring circuit |
| `results/circuits/circuit_m2.png` | M2 circuit |
| `results/circuits/circuit_m3.png` | M3 (design note only, not in the comparison) |
| `results/step1_band_auc.png` | \|ΔB\| vs signed ΔB per band (motivates the feature) |
| `results/part3_corrected_sweep.png` | median correction × spatial context |
| `results/p3_matrix/matrix.csv` | all numbers, machine-readable |

---

## 12. Reproducing

```bash
pip install -r requirements.txt                       # numpy<2 required

python train/accept_mring.py                          # architecture acceptance
train/run_p3_sweep.sh /path/to/OneraDataset 8         # 25-cell sweep
python train/report_p3.py                             # tables + matrix.csv
python train/plot_p3.py                               # figures

python submit/final_pipeline.py threshold --kind m1 --depth 3
python submit/final_pipeline.py train     --kind m1 --depth 3 --data_dir /path/to/OneraDataset
python submit/final_pipeline.py predict   --kind m1 --depth 3 --data_dir /path/to/OneraDataset
```

---

## 13. Generative-AI disclosure (required by the challenge)

Analysis code, experiment harnesses and these write-ups were produced with
assistance from a generative AI coding assistant (Anthropic Claude). The
assistant proposed the model ladder, the paired-control design and the metric
policy. Every number reported here is generated by the committed scripts and is
reproducible from the raw dataset; claims were checked against the data rather
than accepted as asserted. Several AI-proposed conclusions were tested and
**overturned** during the work — the confounding of depth with parameter count in
an early capacity sweep, the incorrect claim that a calibration mismatch could
explain a low F1\*, and the assumption that an index-order CZ ring is a
geometry-agnostic control (it shares 6 of 9 edges with the spatial grid). These
corrections are recorded in the repository history and in `docs/HANDOFF.md`.
