# Project Handoff — read this first

Self-contained state of the project for someone (or some assistant) picking it up
cold. Covers what the task is, every design decision **and why**, what has been
measured, which earlier conclusions were **wrong and got corrected**, how to run
everything, and what to do next.

---

## 0. TL;DR

- **Task:** pixel-level urban-change detection on Sentinel-2 image pairs (OSCD
  dataset), to be solved with a **Quantum Machine Learning** pipeline, compared
  against a classical model with **no more trainable parameters**.
- **Main model:** **M3** — a 9-qubit "Spatial ZZ" variational circuit mapping a
  `3×3×4` patch to a `3×3` probability map. 1 qubit ↔ 1 spatial pixel.
- **Where we are:** full pipeline works end-to-end (data → circuit → training →
  full-city evaluation). A capacity sweep (38/74/110 params) is done.
- **Best number so far** (dev 11/3 split, 3 held-out cities, pooled, single global
  threshold): **AP 0.147, F1\* 0.302, ROC-AUC 0.899** at 74 params.
- **The central experiment is DONE** ([`results_p0_entanglement.md`](results_p0_entanglement.md)):
  M1 (separable, 38p) vs M2-CZ (entangling, 38p), 5-fold city-grouped, all
  controls asserted (same fold transforms, same init, same patch stream).
  **Result: mean ΔAP = −0.0033 ± 0.0045, M2 wins 1/5 folds and 5/14 cities.**
  Fixed CZ entanglement produces genuine interactions (verified `|I_ij| ~ 1e-3`
  vs M1's `1e-16`) but gives **no useful inductive bias** at this budget — an
  honest negative result, with the "the entangler did nothing" objection ruled
  out by construction. **M1(38) is therefore the QML model for the headline
  comparison against the 37-param classical conv.**
- **Division of labour:** a teammate builds the classical baseline; this repo is
  the quantum side (training, gradients, runtime, ablations).

---

## 1. The challenge

Given a registered pair of 13-band Sentinel-2 images of the same place at two
dates (T₁, T₂), label **every pixel**:

- `255` = urban / structural change
- `0` = no change, **or natural change** (e.g. seasonal vegetation)

The hard part is that second clause: it is not "find what changed", it is "find
what changed **and is urban**". Deliverables require documenting preprocessing,
quantum encoding, QNN architecture, and a **parameter-constrained classical
comparison** (same input features, `N_classical ≤ N_QML`). Metrics: Accuracy,
Change-Accuracy, No-change-Accuracy, F1. Generative-AI use must be disclosed.

---

## 2. Dataset facts (and one trap)

**Onera Satellite Change Detection (OSCD)**, Sentinel-2 MSI.

| | |
|---|---|
| Cities | 24 — **14 with labels** (train), 10 hidden (test) |
| Per city | 13 bands × 2 dates, co-registered at 10 m in `imgs_*_rect/` |
| Size | variable, 385×241 … 1070×1180 (**not** a fixed 600×600) |
| Storage | one single-band **uint16** GeoTIFF per band; value = reflectance × 10000 |
| Change prevalence | **2.29 %** of pixels |

> ### ⚠️ TRAP: label encoding
> `<city>-cm.tif` encodes **`{1 = no change, 2 = change}`**, *not* the `{0,1}`
> stated in the dataset README. Using the README convention silently marks
> **every** pixel as change. This repo uses `change = (tif == 2)`. Verified
> against `cm.png`. This bug was hit once already — don't hit it again.

Bands look black in image viewers because uint16 data occupies the bottom ~2 % of
the 0–65535 display range; the data is fine, it just needs a contrast stretch.

The dataset is **not committed** (size + CC-BY-NC-SA / Copernicus terms). Point
`--data_dir` at an extracted `OneraDataset` folder containing `images/` and
`train_labels/`.

---

## 3. What the EDA established (and why each choice was made)

Reproduce with `eda.py`, `eda_spatial.py`, `eda_spatial2.py`,
`eda_representation.py`. Details in [`../README.md`](../README.md).

1. **Use magnitude, not direction.** Per band, `AUC(signed ΔB) ≈ 0.5` while
   `AUC(|ΔB|)` reaches 0.79. A *nonlinear* model can recover 0.79 from signed ΔB
   (it re-derives the magnitude), and `[ΔB, |ΔB|]` does **not** beat `|ΔB|` alone
   → feed `|ΔB|` explicitly. *Wording discipline: "direction has no
   univariate/linear relevance", not "direction is meaningless".*
2. **13 bands ≈ 5 independent axes.** Correlation clustering of `|ΔB|` gives
   {B02,B03,B04} / {B06,B07,B08,B8A} / {B11,B12} / B05 / isolated B01,B09,B10.
   B09 and B10 are useless here on evidence (AUC ≤ 0.55).
3. **Spectral-index *deltas* are weaker** (ΔNDVI/ΔNDBI/ΔNDWI, AUC 0.56–0.62) than
   raw `|ΔB|` — normalized-difference indices divide out the very brightness
   magnitude that carries the signal. Dropped from the primary input.
4. **Per-pair median correction is a big lever.** Per-city median ΔB varies
   enormously (B8A spread ~1500 DN), so raw `|ΔB|` partly encodes *city identity*.
   Subtracting each image's own median (`ΔBᶜᵒʳʳ`, label-free, applied to
   train/val/test alike) lifts cross-city AUC **0.811 → 0.864**.
5. **Spatial context is a real, separate lever.** Mean-pooled `|ΔBᶜᵒʳʳ|`:
   1×1 0.864 → 3×3 0.884 → 5×5 0.896 → 7×7 0.902. Complementary to (4), roughly
   additive. **This is why the model is a patch model, not a pixel model.**
6. **Pixel spectrum cannot separate urban from natural change across cities.**
   Within "changed a lot" pixels, land-cover state (`NDVI/NDBI` at T₁,T₂) gives
   AUC 0.59 under random CV but **0.53 under city-grouped CV** — chance. So that
   distinction must come from spatial *structure*. NDVI/NDBI are therefore **not**
   in the primary input (optional ablation only).
7. **Representation:** on the common median-corrected 13-band base, 5-fold
   city-grouped linear probe gives **PCA-4 0.865 ≈ All-13 0.864 > Physical-4
   {B04,B05,B12,B08} 0.855**. Default is **PCA-4**; Physical-4 is kept as the
   interpretable branch. Margin is inside fold std, so this is a default, not a
   settled fact.

**Reference numbers to keep in mind** (classical, from EDA, city-grouped):
pixel-level ROC-AUC ≈ 0.864, median-corrected 7×7 linear probe ≈ 0.902. These are
*references*, not bars the QML must clear — the challenge comparison is against a
**parameter-matched** classical model.

---

## 4. Data pipeline (`data/`)

Full detail: [`data_pipeline.md`](data_pipeline.md).

```
T1,T2 → robust band norm [0,1] (train P1/P99)  → ΔB
      → per-pair median correction (unsupervised) → |ΔB^corr|_13   ← COMMON BASE
      → ┬ Physical-4 : {B04,B05,B12,B08}, x = clip(|·|/c_b, 0,1)  → X ∈ [0,1]^4
        └ PCA-4      : z = PCA(base),     u = clip(z/c_pc, -1,1)  → X ∈ [-1,1]^4  (signed)
      → 3×3 patch  →  model  →  3×3 probabilities
```

**The fork is deliberate**: both branches share the same 13-band base, the same
center coordinates and the same labels, so a Physical-vs-PCA comparison isolates
only the encoding.

**Leakage discipline (enforced in code, verified by a smoke test):** the per-pair
median is label-free and per-image, so it applies to every city; but band
`P1/P99`, `c_b`, the PCA basis, `c_pc`, `c_norm` are fit on **train cities only**
and change when a city moves from val into train.

**Center pools** (`pools.py`) are representation-independent, disjoint:
`positive` / `hard_negative` / `ordinary_negative`, where
`hard = y==0 & h_base > max(Q80_city, T_global)`, `h_base` = mean `|ΔBᶜᵒʳʳ|` over
the 4 physical bands, and `T_global` = **median over train cities** of each city's
Q80 (one vote per city, so big cities don't set the floor). Dev stats:
`T_global = 0.0726`; every city has positives; hard-neg pool 9.5k–245k; the floor
correctly trims quiet cities (rennes 5.1 %, mumbai 6.5 %) instead of admitting
barely-changed pixels.

**Sampler** (`sampler.py`): **city uniform → category 1:1:2 → coordinate**, with
replacement. Measured on 50k draws: categories 24.8/25.0/50.2 %, city frequency
8.8–9.3 % (uniform target 9.1 %), **π_pixel = 21.8 %**.

**Loss decision:** because the sampler already yields ≈78:22, a raw imbalance
weight (`w₊ ≈ 43`) would double-correct. **Plain BCE** is the default; if recall
is poor, use a *mild* `w₊ ≈ 3.6` derived from the measured `π_pixel`.

---

## 5. Models (`models/qml.py`)

**Contract:** `forward(params, X, S) -> P`
- `X (B,3,3,4)` angle features, `S (B,3,3)` (or `(B,3,3,2)` per-stage) change
  strength, `P (B,3,3)` probabilities.
- **Why `S` is a separate argument:** in the PCA branch it is *not* recoverable
  from `X`. `X = clip(z/c_pc, −1, 1)` is clipped per component, while
  `S = clip(‖z‖₂/c_norm, 0, 1)` uses the unclipped score norm.

**Grid** (qubit ↔ pixel) and the 12 nearest-neighbour edges:

```
0 1 2     H: (0,1)(1,2)(3,4)(4,5)(6,7)(7,8)
3 4 5     V: (0,3)(3,6)(1,4)(4,7)(2,5)(5,8)
6 7 8
```

**One cycle** (M1/M2/M3 share this skeleton):
`E1 → [entangler] → V1 → E2 → [entangler] → V2`, where
`E1 = RY(π·x_B04)RZ(π·x_B05)`, `E2 = RY(π·x_B12)RZ(π·x_B08)` (PCA: the 4 PCs in
order), and `V = RY(θ)RX(θ)` per qubit (**must** be non-commuting with Z, because
`[Z_iZ_j, Z_i] = 0` — a diagonal entangler alone is invisible to a Z measurement).
Readout `p = σ(a·⟨Z⟩ + b)` with `a,b` **shared** across the 9 pixels.

| model | entangler | params (L=1) |
|---|---|---|
| **M0** | 4-qubit *pixel* VQC, ring CNOT over feature qubits, applied to each pixel independently | 13 |
| **M1** | none (9 independent position-wise VQCs + shared scalar calibration) | 38 |
| **M2** | fixed NN CNOT — **order pinned**: all 6 horizontal, then all 6 vertical, control = lower index (CNOTs don't commute; M3's ZZ gates do, so only M2 needs this) | 38 |
| **M3** | data-dependent `IsingZZ(γ·sᵢ·sⱼ)`, γ = π/2 **fixed, not trainable** | 38 |

`depth=L` repeats the cycle; `tying="tied"` reuses one mixer block (params stay
38), `"untied"` gives each cycle its own (`L·36 + 2`). **M4 ≡ M3 with depth=2.**

**Verified structurally** (perturb a corner pixel, read the centre):
M1 `max|Δp| = 1e-16` (independent) · M2 `2.4e-2` · M3 `2.2e-3` (both coupled;
corner→centre is 2 hops, reached because there are two entangling layers per
cycle).

**Parameter-matched headline comparison:** **M3 L=1 (38) vs classical 3×3
same-pad conv 4→1 (4·3·3+1 = 37)**. Same input, same receptive field. Report it as
*"same input, same receptive field, nearly identical parameter budget"* — **not**
"identical except quantumness": M3's mixer is position-dependent while a conv
kernel is weight-shared.

---

## 6. Training & evaluation (`train/`)

- `trainer.py` — config-driven loop. An **"epoch" is defined by steps**
  (`steps_per_epoch × batch`), because the sampler is an infinite stream.
  Pilot defaults: `lr=0.02, batch=32, steps_per_epoch=160 (5120 patches), epochs=20`.
- **Two validation paths:**
  - *cheap*, every epoch: **fixed** coordinates drawn **uniformly over eligible
    pixels** (natural prevalence, ~3k/city — **not** the 1:1:2 training mixture),
    so metric changes reflect the model, not the sample. Reads the centre pixel
    only (a proxy).
  - *exhaustive*, periodically: stride-1 over the whole city with overlap
    averaging — authoritative.
- **Two checkpoints, on purpose:** `*_bestcheap.npy` (best **pooled** cheap-val AP
  over all val cities — the unbiased selector for comparing models) and
  `*_best.npy` (best exhaustive AP; **biased** when `exhaustive_cities='smallest'`
  because it sees one city — do not compare models with it).
- `inference.py` — `predict_city` (reflect-pad for **input context only**;
  accumulate outputs **only inside the original H×W**: `P = S(p)/C(p)`),
  `evaluate_predictions`, `make_fixed_val_coordinates`.
  **Primary threshold-free metric is Average Precision (AP)** = sklearn
  `average_precision_score` (named explicitly; not trapezoidal PR-AUC). `τ*` is
  taken from `precision_recall_curve` candidates maximizing F1, **on validation
  only** — the API rejects passing `tau` and `select_threshold` together.
  7 acceptance tests pass, including an **echo test** that reconstructs
  `X[...,0]` exactly and thus validates the patch→pixel offset arithmetic.
- `eval_full.py` — exhaustive evaluation of a checkpoint on **all** val cities,
  reporting per-city, **macro** (per-city mean, per-city τ* → optimistic) and
  **micro** (pooled pixels, **one global τ*** → deployment-realistic).

**Cost:** ~0.65 ms/patch forward, ~1.37 ms/patch forward+grad (L=1, simulator,
batched). A 5120-patch epoch ≈ 16 s at L=1. A full-city exhaustive eval is
minutes to tens of minutes per city and scales with depth.

**Splits** (`data/splits.py`): fixed **dev 11/3** (val = paris, cupertino, beihai)
for engineering; `get_grouped_folds` for 5-fold **city-grouped CV** (this is *not*
leave-one-region-out — true LORO would be 14 folds); `TEST_CITIES` for the final
predict-only mode. Splits are injected into `build_fold`, so dev / CV / final all
run the *same* preprocessing code.

---

## 7. Results so far

### 7.1 Circuit sanity (M3)
Gradients reach all 38 params (36/36 mixer nonzero), outputs finite, loss falls.
*Wording: "no obvious gradient-vanishing in this smoke test"* — a landscape-level
barren-plateau claim would need gradient variance across seeds/depths.

### 7.2 Fixed-batch capacity probe — and the correction that followed
On one fixed 32-patch batch (real labels, lr=0.1, 300 steps, one seed):

| config | params | loss |
|---|---|---|
| L=1 | 38 | 0.471 |
| **L=2 tied** | **38** | **0.504** ← *no gain at fixed params* |
| L=2 untied | 74 | 0.189 |
| L=3 untied | 110 | 0.070 |
| L=1 per-pixel calibration | 54 | 0.456 |

> **An earlier conclusion here was wrong.** The first sweep varied `L` with
> *untied* mixers, so depth and parameter count were confounded, and it was
> written up as "depth is the capacity knob". Adding the **tied** control showed
> the opposite: at a fixed 38-parameter budget the extra cycle gives **no** gain.
> The fitting gain tracks **parameter count**, not re-uploading depth, in this
> setup. Consequences: M4's advantage is a *capacity* result, not evidence that
> "re-uploading works"; and the 74-param model is **not** parameter-matched to the
> 37-param conv, so **M3 L=1 stays the sole headline model**.
> Also: per-pixel calibration doesn't help → keep the **shared** `(a,b)` readout.

### 7.3 Capacity sweep on real data (the main experiment so far)

Identical protocol, checkpoints selected on pooled cheap-val AP.
Full write-up: [`results_capacity_sweep.md`](results_capacity_sweep.md).

| micro (pooled, one global τ*) | L1 (38) | **L2 (74)** | L3 (110) |
|---|---|---|---|
| AP | 0.1108 | **0.1466** | 0.1436 |
| F1\* | 0.1888 | **0.3017** | 0.3003 |
| ROC-AUC | 0.8649 | 0.8985 | 0.9067 |
| ChangeAcc | 0.439 | 0.614 | 0.662 |

Per city, AP / F1\*:

| city | prevalence | L1 | L2 | L3 |
|---|---|---|---|---|
| paris | 0.29 % | 0.018 / 0.041 | 0.037 / 0.091 | 0.043 / 0.104 |
| cupertino | 2.37 % | 0.269 / 0.359 | 0.258 / 0.454 | 0.261 / 0.467 |
| beihai | 2.49 % | 0.069 / 0.129 | 0.099 / 0.224 | 0.096 / 0.217 |

**Findings.**
- **L1 → L2:** train BCE only −0.016, but validation improves a lot
  (micro AP +32 %, F1\* +60 %) and **every** city improves.
- **L2 → L3:** saturation in **both** train and validation; the differences lie
  inside the epoch-to-epoch cheap-val spread. **L3 > L2 is not established.**
- So: capacity **was** a binding constraint at 38 params, and is **no longer**
  binding beyond ≈74.
- **Cross-city spread persists at every capacity** (L3: F1\* 0.467 / 0.217 / 0.104;
  cupertino and beihai differ ~2.7× in AP despite near-identical prevalence).
  **Domain generalization is a separate axis that capacity does not fix.**
- The fixed-batch probe **overstated** the capacity gap (0.471→0.189 there vs
  0.4815→0.4659 on the real stream): it rewards memorizing 32 patches, while
  training sees 5120 fresh patches per epoch.
- Curiosity: on cupertino **AP is flat across all three** while F1\* rises —
  capacity improved the operating point, not the ranking.

### 7.4 The diagnostic that sets the next direction

Distribution of the ZZ phase `φ = γ·sᵢ·sⱼ` (γ = π/2), NN pairs, 5 train cities:

| P50 | P75 | P90 | P99 | mean | max | > 0.1 rad |
|---|---|---|---|---|---|---|
| 0.044 | 0.061 | 0.096 | 1.361 | 0.085 | 1.571 | **9.6 %** |

**For ~90 % of neighbour pairs the ZZ rotation is < 0.1 rad — effectively
near-identity.** Cause: `s = clip(‖z‖₂/c_norm, 0, 1)` with `c_norm = P99(‖z‖)`
makes a typical `s ≈ 0.17`, so `sᵢsⱼ ≈ 0.03`. So M3's "data-dependent spatial
coupling" is a sparse, high-threshold interaction that fires only where *both*
pixels changed strongly — and M3 may be operating close to M1 over most of the map.

---

## 7.5 Center branch (3×3 → 1) — what changed with it

- **Readout:** `p = σ(a·mean_q⟨Z_q⟩ + b)`, a **parameter-free** aggregation over
  all 9 qubits. Measuring only the centre qubit would leave 32 of 36 mixer
  parameters structurally dead in M1 (its state is a product state, so
  `⟨Z_centre⟩` depends on the centre pixel alone) and make "38 vs 38" a fake
  parameter match.
- **M1 is therefore NOT a "no spatial context" model.** It reads all 9 pixels but
  combines them **additively** with no interaction terms. The right framing is
  **separable/additive vs interacting/entangling** — which is a sharper question
  than "does spatial information help", and it lines up neatly with the classical
  twin (a 3×3 conv is also additive, but linear per pixel).
- **The training target distribution changed.** In the 3×3→3×3 branch the loss
  used all 9 patch labels, giving `π_pixel ≈ 21.8 %`. Center-only takes the
  centre label alone, and the sampler picks categories **by centre**, so
  `π_train ≈ 25 %` (measured `π_centre = 24.8 %`). Constant-predictor BCE
  baselines therefore differ:

  | branch | target prior | constant-predictor BCE |
  |---|---|---|
  | 3×3→3×3 (dense) | 21.8 % | **0.524** |
  | 3×3→1 (centre) | 24.8 % | **0.562** |

  **Never compare BCE values across the two branches** — the supervised target
  distributions are different. Compare each against its own baseline above.
- Note also that centre training runs at a ~25 % prior while cheap-val runs at the
  natural ~2.2 %. AP depends on prevalence, so cheap-val AP is comparable
  *between models* (same val set) but not to training BCE.

## 8. Interpretation discipline (please keep to this)

These phrasings were agreed after several over-claims were caught:

- Not "no barren plateau" → **"no obvious gradient-vanishing in this smoke test"**.
- Not "capacity ceiling" → **"fitting limitation under the tested setup"** (single
  optimizer/lr/seed; real labels, not a random-label memorization test).
- Tied-L2 showing no gain does **not** mean re-uploading is useless — only that
  parameter-fixed depth gave no fitting gain in this setup (a reused mixer may
  also be harder to optimize).
- **AP and F1\* are invariant to monotone rescaling of scores.** So a
  calibration/prevalence mismatch **cannot by itself** explain a low F1\* once τ*
  has been swept. (An earlier claim that it could was wrong.)
- ROC-AUC is a **secondary** metric here; at 2.29 % prevalence report **AP, F1,
  Change-Accuracy**. Don't compare a single city's ROC-AUC to the EDA's
  city-grouped pooled figure as if they were the same quantity.
- L2/L3 are **capacity ablations**, never the headline. Headline = **M3 L=1 (38)
  vs 3×3 conv (37)**.
- Validate model *rankings* under 5-fold city-grouped CV before any final claim;
  everything so far is a single seed on a single 11/3 split.

---

## 9. How to run

```bash
pip install -r requirements.txt          # numpy<2 required (pennylane 0.45 warns)
```

```bash
# EDA (regenerates figures + results/RESULTS.md)
python eda.py --data_dir /path/to/OneraDataset

# module smoke tests (each is self-checking)
python data/preprocess.py --data_dir /path/to/OneraDataset
python data/pools.py      --data_dir /path/to/OneraDataset
python data/sampler.py    --data_dir /path/to/OneraDataset
python models/qml.py
python train/inference.py

# train one model
python train/trainer.py --data_dir /path/to/OneraDataset \
  --kind m3 --depth 1 --tying untied --representation pca \
  --lr 0.02 --batch 32 --steps_per_epoch 160 --epochs 20 \
  --cheap_val_per_city 3000 --exhaustive_cities none --tag my_run

# exhaustive 3-city evaluation of a checkpoint
python train/eval_full.py --data_dir /path/to/OneraDataset \
  --ckpt results/runs/my_run_bestcheap.npy --kind m3 --depth 1 --tying untied
```

Artifacts land in `results/runs/`: `<tag>.jsonl` (per-epoch log),
`<tag>_fullval.json` (exhaustive metrics), `*_bestcheap.npy` / `*_best.npy`
(weights, git-ignored).

---

## 10. Next steps, in priority order

1. **M1 vs M2 vs M3 at a fixed capacity (L=2 untied, 74 params). ← do this first.**
   The project's central claim — a *data-dependent* NN ZZ beats *no* spatial
   mixing and beats a *fixed* CNOT coupling — has **never been tested**; all runs
   to date are M3. The φ diagnostic (§7.4) makes it urgent: if ZZ is near-identity
   for 90 % of pixels, M3 might not separate from M1.
   Cost ≈ 2 trainings (~25 min) + 2 exhaustive evals (~70 min).
2. **γ and `s`-normalization ablation** (γ ∈ {0.5, 1, π/2}; normalize `s` by a
   lower percentile than P99). Directly motivated by §7.4, and it changes **no**
   parameter count, so the parameter-matched comparison survives.
3. Connectivity (4- vs 8-neighbour) and representation (Physical-4 vs PCA-4).
4. Do **not** add more depth — it has saturated.
5. Before final claims: rerun the winning configuration under **5-fold
   city-grouped CV**, then final mode (fit transforms on all 14 labelled cities,
   predict the 10 hidden ones, save masks as `{0,255}` PNG / `<city>-cm.tif`).
6. Coordinate with the classical teammate so the comparison is exact: **same
   PCA-4 features, same folds, same sampler rule, plain BCE, same full-city
   validation, same τ*-selection procedure, same metrics.**

---

## 11. Known caveats

- Single seed, single dev split for every result so far.
- Exhaustive evaluation is expensive and scales with depth; one L1 run logged
  anomalous timings (12215 s / 10101 s) from an external machine slowdown —
  metrics are deterministic and unaffected, but ignore those two timings.
- `models/qml.py` is imported as `import qml as qmodels` while PennyLane is
  `import pennylane as qml`. It works (module name vs bound name), but be careful
  when adding imports.
- M0 is defined and unit-tested but has **not** been trained; `inference.py`'s
  overlap routine is for 3×3→3×3 patch models and would need a trivial separate
  path for M0's 1×1→1.
- Generative AI (Anthropic Claude) assisted the analysis, code and write-ups.
  Every number in this repo is reproduced by the committed scripts; the challenge
  requires disclosing this and describing how claims were independently checked.
