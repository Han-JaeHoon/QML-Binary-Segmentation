# Training-Data Pipeline

How raw OSCD imagery becomes model-ready `3×3×4 → 3×3` training samples for the
QML model ladder. Implemented in [`data/`](../data); every design choice below is
grounded in the EDA (top-level `README.md`) and verified by the smoke tests each
module ships with.

Modules: [`splits.py`](../data/splits.py) · [`preprocess.py`](../data/preprocess.py)
· [`pools.py`](../data/pools.py) · [`sampler.py`](../data/sampler.py).

---

## 0. Design principles (from the EDA)

- **Magnitude, corrected.** Signal lives in `|ΔB^corr|` (median-corrected temporal
  magnitude), not signed `ΔB` and not spectral-index deltas.
- **Spatial context is essential.** A pixel-only linear probe plateaus at ~0.82;
  median-corrected `7×7` reaches ~0.90. So samples are patches, not pixels.
- **Cross-city generalization.** Everything is fit/validated **by city**
  (leave-region-out), never by random pixel split (which is optimistic).
- **Severe imbalance (2.29 % change).** Random patches are almost all no-change,
  so we sample deliberately (positive / hard-negative / ordinary) with a
  city-balanced stochastic sampler rather than materializing millions of patches.

---

## 1. Common base → two branches

The pipeline forks **after** a shared 13-band base, so Physical-4 and PCA-4 see
identical coordinates and only the encoded features differ.

```
T1,T2 → robust band norm[0,1] (train P1/P99) → ΔB
      → per-pair median correction (unsupervised)  → |ΔB^corr|_13   ← COMMON BASE
      → ┬ Physical-4: {B04,B05,B12,B08}, x=clip(|·|/c_b, 0,1)      → X ∈ [0,1]^4
        └ PCA-4:      z=PCA(base), u_k=clip(z_k/c_k^PC, -1,1)       → X ∈ [-1,1]^4 (signed)
```

**Leakage discipline (enforced in code):**
- *per-pair median correction* uses only each image's own pixels, no labels →
  applied to **every** city (train/val/test), exactly as at test time.
- band `P1/P99`, physical `c_b`, PCA basis, `c_k^PC`, `c_norm` are fit on
  **TRAIN cities only**. The smoke test confirms they change when a city moves
  from val into train.

**Representation decision (median-corrected, 5-fold city-grouped, linear probe):**

| representation | ROC-AUC |
|---|---|
| All-13 | 0.864 |
| **PCA-4** | **0.865** |
| Physical-4 {B04,B05,B12,B08} | 0.855 |

PCA-4 equals All-13 (4 PCs preserve the 13-band linear info) and beats Physical-4
in 4/5 folds, though the +0.010 margin is within the fold std (~0.046). **Default
main = PCA-4**; Physical-4 is kept as the interpretability branch, and the final
Physical-vs-PCA call is deferred to the nonlinear/spatial VQC under 5-fold.
Reproduce: [`eda_representation.py`](../eda_representation.py).

---

## 2. Center pools (representation-independent)

Each city's eligible centers are partitioned into three **disjoint** pools by the
common base only, so both branches train on the same coordinates:

```
eligible  = valid & interior          (interior keeps a full 3×3 window)
h_base(p) = mean_{B04,B05,B12,B08} |ΔB^corr|(p)      (pre-clip magnitude)
positive          = eligible & y==1
hard_negative     = eligible & y==0 & h_base > max(Q80_city, T_global)
ordinary_negative = eligible & y==0 & NOT hard        (= eligible neg − hard)
```

`T_global` = **median over train cities** of each city's `Q80(h_base | eligible,
y==0)` — one vote per city, so large cities don't dominate the floor.

**Dev 11/3 pool statistics** (`T_global = 0.0726`) confirm the hybrid threshold
behaves as intended:
- positive pool present in **every** city; hard-neg size min/median/max =
  **9,469 / 46,777 / 244,839**; **0/11** cities starved by the floor.
- in **quiet** cities the floor cuts hard% below 20 % (rennes 5.1 %, mumbai 6.5 %,
  nantes 12.9 %) — removing barely-changed pixels that the naive top-20 % would
  have mislabelled as "hard"; in **active** cities (abudhabi, pisa, beirut, beihai)
  the threshold stays at the city's own Q80 (hard% = 20 %).

---

## 3. City-balanced stochastic sampler

Order (locked): **city uniform → category (1:1:2) → coordinate (with
replacement)**. Internally split so coordinates are representation-free:
`sample_index() → get_patch(city,r,c,rep) → sample_batch(B)`. Each city's
`H×W×4` transform is cached once; sampling just crops `3×3`.

Category `P:H:O = 25:25:50`. `estimate_pixel_prevalence` uses its **own rng**
(never consumes the training rng) and touches only coordinates + labels.

**Dev 11/3, 50k virtual samples:**

| diagnostic | value | target |
|---|---|---|
| category (pos/hard/ord) | 24.8 / 25.0 / 50.2 % | 1:1:2 |
| city sampling (min–max) | 8.8 – 9.3 % | 9.1 % uniform |
| `π_center` | 24.8 % | ~25 % by design |
| **`π_pixel`** | **21.8 %** | measured, not assumed |

`π_center = 24.8 %` but `π_pixel = 21.8 %` (barely lower) ⇒ urban change is
**spatially connected**: positive-centered patches carry many positive neighbours,
so the 9-pixel supervision stays positive-rich.

Verified invariants: identical coordinates for Physical/PCA under the same seed;
`X_phys ∈ [0,1]`, `X_pca ∈ [-1,1]`, `Y ∈ {0,1}^{3×3}`; deterministic under a
fixed seed.

---

## 4. Loss decision

The sampler already brings the effective training distribution to **≈ 78:22**
(no-change : change). So a raw imbalance weight `w₊ ≈ 97.7/2.29 ≈ 43` would be a
gross double-correction.

**Start with plain BCE.** If change recall is low, add a *mild* `w₊ ≈ 78.2/21.8 ≈
3.6` computed from the **actual sampled** `π_pixel` — never from the raw 2.29 %.

`L = (1/9) Σᵢ BCE(yᵢ, pᵢ)` over the 9 patch pixels; **focal loss** is an ablation.

---

## 5. Train vs validation vs final

| mode | cities | transforms fit on | sampling |
|---|---|---|---|
| **dev** | fixed 11 / 3 | 11 train | stochastic (train) / exhaustive (val) |
| **CV** | 5-fold grouped | each fold's train | same |
| **final** | 14 train → 10 test | all 14 | predict-only on test |

- **Train** = sampled distribution (1:1:2). **Validation** = *natural* full-image
  distribution: stride-1 over the whole held-out city, average overlapping
  predictions per pixel, then metrics at the true 2.29 % prevalence.
- **Metrics:** F1, Change-Accuracy, PR-AUC (primary); No-change-Acc, Accuracy
  (secondary); ROC-AUC (EDA/representation only).
- **Borders:** train centers restricted to the interior; inference uses reflect
  padding so border pixels are still predicted.

---

## Status

| stage | state |
|---|---|
| EDA (Part 1–3) + representation check | ✅ |
| Model design (M0–M4) + M3 circuit | ✅ |
| Data: preprocess / pools / sampler | ✅ (smoke tests pass) |
| Training + evaluation infrastructure | ✅ (`models/qml.py`, `train/`) |
| M3 capacity sweep 38/74/110 | ✅ [results](results_capacity_sweep.md) |
| M1 vs M2 vs M3 at fixed capacity | ⬜ next — the core claim is still untested |
| Classical baseline (teammate) | ⬜ |
