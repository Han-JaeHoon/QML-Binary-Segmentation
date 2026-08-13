# Model Ladder — M0 → M4 (Spatial Quantum Change Detector)

Design document for the QML models used in the OSCD urban-change-detection
challenge. Every model is grounded in the EDA (see top-level `README.md`):
the signal lives in **change magnitude** (`|ΔB|`, not signed) and in **local
spatial structure** (per-pixel spectrum alone plateaus at AUC ≈ 0.82, spatial
context lifts a linear probe to ≈ 0.90). So the models encode
**magnitude on single qubits** and **spatial co-change through two-qubit
interactions**.

The ladder is built so that **M1, M2, M3 all have exactly 38 trainable
parameters**. Each rung therefore isolates *one* design factor, and each has a
**parameter-matched classical twin** as required by the challenge rule
(same input features, same receptive field, no more trainable parameters).

---

## 1. Shared pipeline (identical for every model)

```
T1, T2  (13-band Sentinel-2, imgs_*_rect, 10 m)
  ↓ robust per-band normalization      (clip to [P1,P99] on TRAIN, → [0,1])
  ↓ temporal difference                 ΔB = B̃_T2 − B̃_T1
  ↓ per-pair median correction          ΔBᶜᵒʳʳ = ΔB − medianₚ(ΔB)     (unsupervised)
  ↓ magnitude                            |ΔBᶜᵒʳʳ|
  ↓ keep 4 bands                         {B04, B05, B12, B08}
  ↓ frozen scaling                       x_b = clip(|ΔBᶜᵒʳʳ_b| / c_b, 0, 1),  c_b = P99 on TRAIN
  ↓ patch extraction                     3×3×4 (M1–M4) or 1×1×4 (M0)
  → model → per-pixel change probability → overlap-average → threshold τ → {0,255}
```

**Feature choice** `{B04,B05,B12,B08}` = the highest-relevance representative of
each redundancy group (visible / red-edge / SWIR / NIR). `PCA-4` is an ablation.
`c_b` (the `|ΔBᶜᵒʳʳ|` P99 per band) is **frozen on TRAIN and never re-fit per test
image** — re-fitting would erase the magnitude signal.

---

## 2. Shared building blocks

| block | definition | trainable? |
|---|---|---|
| **Angle encode** `E` | `RY(π·x_a) RZ(π·x_b)` per qubit, for a band pair `(a,b)` | no (data) |
| **Change strength** `s` | `s = √((x_a² + x_b²)/2) ∈ [0,1]` per qubit | no (data) |
| **Spatial ZZ** (M3/M4) | `R_ZZ(γ · sᵢ sⱼ)` on 12 NN edges, `γ = π/2` **fixed** | no (data) |
| **Spatial CNOT** (M2) | `CNOT` on 12 NN edges (fixed, data-independent) | no |
| **Mixer** `V` | `RY(θ) RX(θ)` per qubit (non-commuting with Z) | **2 / qubit** |
| **Readout** | `zᵢ = ⟨Zᵢ⟩` → `pᵢ = σ(a·zᵢ + b)`, `a,b` shared across pixels | **2** |

**Grid & nearest-neighbour edges** (qubit ↔ spatial pixel):
```
0 1 2      H: (0,1)(1,2)(3,4)(4,5)(6,7)(7,8)
3 4 5      V: (0,3)(3,6)(1,4)(4,7)(2,5)(5,8)
6 7 8      → 12 NN couplings, no all-to-all
```
Why a mixer after ZZ/CNOT: `[ZᵢZⱼ, Zᵢ] = 0`, so a diagonal entangler alone is
invisible to a `Z` measurement. `RY`/`RX` rotate it into an observable basis.

**Core idea in one line:** *single-qubit angle encodes how much each pixel
changed; data-dependent nearest-neighbour ZZ encodes how strongly neighbouring
pixels changed together.*

---

## 3. The ladder at a glance

| model | what it adds | circuit (one cycle) | qubits | QML params | classical twin (params) |
|---|---|---|---|---|---|
| **M0** | quantum baseline, **no spatial** | `E → CNOTring → V` (4q) | 4 | **13** | MLP 4→2→1 (13) |
| **M1** | 9-qubit repr, **no spatial mixing** | `E1 → V1 → E2 → V2` | 9 | **38** | 1×1 conv 4→1 (5) † |
| **M2** | **fixed** spatial entanglement | `E1 → CNOT_NN → V1 → E2 → CNOT_NN → V2` | 9 | **38** | 3×3 conv 4→1 (37) |
| **M3** | **data-dependent** spatial ZZ *(main)* | `E1 → ZZ1 → V1 → E2 → ZZ2 → V2` | 9 | **38** | 3×3 conv 4→1 (37) |
| **M4** | **re-uploading** (L=2) | `[E1→ZZ1→V1→E2→ZZ2→V2]×2` | 9 | **74** / 38 ‡ | 3×3 conv, 2 layers |

† M1 has no spatial receptive field, so its honest twin is a per-pixel logistic
(5 params); it is kept only to measure the value of spatial *interaction*, not to
win the parameter-matched comparison.
‡ M4 untied mixers = 74; **tied** mixers (shared across cycles) = 38 — run both to
separate re-uploading expressivity from the extra parameters.

**Parameter arithmetic**
- Mixer per stage = `9 qubits × 2 (RY,RX) = 18`; two stages per cycle = `36`.
- `+ 2` shared calibration `(a,b)` → **38** for one cycle (M1/M2/M3).
- M0: encode 4 features on 4 qubits, mixer `4×2=8`, linear head `w(4)+b(1)=5` → **13**.
- M4 untied: `2 cycles × 36 + 2 = 74`; tied: `36 + 2 = 38`.

---

## 4. Per-model detail

### M0 — Pixel VQC (quantum baseline)
- **Input** 1×1×4 (single pixel, 4 bands). **4 qubits**, one band per qubit.
- **Circuit** `RY(π x_b)` per qubit → ring `CNOT` (spectral, fixed) → mixer `RY,RX`
  → readout `w·⟨Z⟩ + b` (linear head over 4 qubits).
- **Purpose** establish the *no-spatial-context* quantum performance. Everything
  above this is the value of spatial modelling.
- **Twin** MLP `4→2→1` (13 params).

### M1 — Angle-only 9-to-9 VQC (spatial representation, no mixing)
- **Circuit** `E1 → V1 → E2 → V2` on 9 qubits, **no entangler**.
- Qubits are independent ⇒ this is 9 per-pixel VQCs sharing mixer weights and a
  joint readout. **No information crosses between pixels.**
- **Purpose** baseline that has the 9-qubit representation but *not* spatial
  interaction. `M1 → M2` measures whether spatial entanglement helps at all.

### M2 — CNOT Spatial VQC (fixed entanglement)
- **Circuit** `E1 → CNOT_NN → V1 → E2 → CNOT_NN → V2`.
- Spatial entanglement is **data-independent** (same CNOT pattern for every patch).
- **Purpose** `M2 → M3` isolates the value of making the spatial interaction
  **depend on the data** (co-change strength) versus a fixed coupling.

### M3 — Spatial ZZ Re-uploading VQC (**main model**)
- **Circuit** `E1 → ZZ1(sᵢsⱼ) → V1 → E2 → ZZ2(sᵢsⱼ) → V2`, `γ=π/2` fixed, `L=1`.
- ZZ angle `π/2 · sᵢ sⱼ` grows when two neighbours *both* changed strongly.
- **Twin** 3×3 same-pad conv `4→1` = **37 params**, identical `3×3×4 → 3×3` I/O.
- Diagram: [`results/m3_circuit.png`](../results/m3_circuit.png); code:
  [`circuits/m3_spatial_zz.py`](../circuits/m3_spatial_zz.py).

### M4 — M3 + genuine re-uploading (L=2)
- **Circuit** repeat the full M3 cycle twice (same features re-encoded).
- Run **untied (74)** and **tied (38)** mixers: if tied-L2 ≳ L1 the gain is
  re-uploading expressivity, not parameters.

---

## 5. Training protocol (all models)

- **Patch sampling.** Change is only **2.29 %** of pixels, so random patches are
  almost all no-change. Sample patches **centred on** (a) all urban-change pixels,
  (b) *hard* negatives — high-`|ΔBᶜᵒʳʳ|` but label 0 (changed-a-lot-but-not-urban),
  (c) ordinary no-change. Hard-negative *features* don't generalize across cities,
  but hard-negative *sampling* still shapes the decision boundary.
- **Loss.** Weighted BCE over the 9 patch pixels,
  `L = (1/9) Σ WBCE(yᵢ, pᵢ)`, positive class up-weighted by inverse frequency;
  compare **focal loss** as ablation.
- **Optimizer.** Adam + parameter-shift gradients (9 qubits, simulator).
- **Validation.** **City-grouped / leave-region-out** only (never random pixel
  split — EDA showed random CV is optimistic: hard-neg 0.59→0.53). All fitted
  transforms (`c_b`, any PCA, threshold `τ`) are chosen on TRAIN folds only.
- **Inference.** Stride-1 overlapping 3×3 patches; average each pixel's
  predictions across the patches covering it; threshold `τ` chosen on validation;
  save `1→255, 0→0`.

## 6. Evaluation metrics

Primary: **F1, Change-Accuracy, PR-AUC** (evaluated at the true 2.29 % prevalence,
not on the balanced training sample). Secondary: No-change-Accuracy, Accuracy.
ROC-AUC retained only for EDA/representation comparison.

## 7. Ablation matrix

| axis | settings | isolates |
|---|---|---|
| spatial interaction | M1 (none) → M2 (CNOT) → M3 (ZZ) | value & type of entanglement |
| coupling strength | `γ ∈ {0.5, 1, π/2}` (M3) | ZZ feature-map scale |
| depth / re-uploading | M3 (L=1) → M4 tied (L=2) → M4 untied (L=2) | re-uploading vs params |
| mixer sharing | independent (38) vs shared (6) | translation-symmetry price |
| features | `{B04,B05,B12,B08}` vs `PCA-4` | physical vs data-driven |
| correction | raw `|ΔB|` vs median-corrected | domain-shift removal |
| loss | weighted-BCE vs focal | imbalance handling |

## 8. What each comparison proves

- **M0 vs M1/M2/M3** → is spatial modelling worth it? (EDA says yes.)
- **M1 vs M2** → does spatial *entanglement* help beyond a joint readout?
- **M2 vs M3** → does *data-dependent* coupling beat fixed coupling? (the core claim)
- **M3 vs M4** → does re-uploading add expressivity at fixed parameters?
- **M3 (38) vs 3×3 conv (37)** → the headline **parameter-matched** QML-vs-classical
  result the challenge asks for.

> Disclosure: model design was developed with an AI assistant; all parameter
> counts and circuit structure are reproduced by `circuits/m3_spatial_zz.py`, and
> every performance claim will be validated under city-grouped CV before reporting.
