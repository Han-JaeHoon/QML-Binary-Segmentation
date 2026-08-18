# P0 — Does explicit quantum inter-pixel interaction help?

**The central experiment of this project.** M1 (separable, 38 params) vs
M2-CZ (entangling, 38 params), 5-fold city-grouped CV, 4.6 h of compute.
Raw results: [`../results/runs/p0_5fold.json`](../results/runs/p0_5fold.json).
Driver: [`../train/run_p0_5fold.py`](../train/run_p0_5fold.py).

---

## The question

Both models are 9-qubit circuits on a `3×3×4` patch predicting the **centre**
pixel, with a parameter-free readout `p = σ(a·mean_q⟨Z_q⟩ + b)`:

- **M1** — `E1 → V1 → E2 → V2`, no two-qubit gates. Its pre-sigmoid score is
  *exactly additive*, `g = (a/9)Σ_q f_q(x_q) + b` (verified: max mixed
  finite-difference interaction `|I_ij| = 5.6e-16`). It is **not** a "no spatial
  context" model — it reads all 9 pixels, but combines them without interaction
  terms.
- **M2** — the same circuit plus a fixed **CZ** layer on the 12 nearest-neighbour
  edges, before each mixer. CZ is diagonal and mutually commuting, so the layer
  is order-free. Genuine interaction confirmed: `|I_ij| = 5.8e-3 … 1.05e-2` on NN
  pairs.

So `M1 → M2` isolates exactly one thing: **the presence of inter-pixel
interaction terms**, at an identical parameter budget.

## Controls (asserted in-process, every fold — all passed)

| control | how |
|---|---|
| same preprocessing | the fold is built **once**; both models use the same train-only band norm / PCA basis / scales |
| same initial parameters | `same_init = True` (identical 38 values) |
| same training data stream | `same_stream = True` (per-epoch crc32 of the sampled `(city,row,col)` sequence) |
| same optimizer & budget | Adam lr 0.02, batch 32, 320 steps/epoch, 50 epochs — frozen before the run |
| no checkpoint selection | the **final** checkpoint is evaluated; validation never picks a checkpoint |

Evaluation is exhaustive (stride-1 over the whole held-out city, natural
prevalence). Every one of the 14 labelled cities is a held-out validation city
**exactly once**.

---

## Results

### Fold level (primary)

| fold | held-out cities | M1 AP | M2 AP | ΔAP |
|---|---|---|---|---|
| 0 | cupertino, mumbai, nantes | 0.2519 | 0.2447 | −0.0072 |
| 1 | saclay_e, pisa, aguasclaras | 0.0224 | 0.0211 | −0.0013 |
| 2 | paris, bercy, rennes | 0.1299 | **0.1344** | **+0.0045** |
| 3 | hongkong, abudhabi, beirut | 0.1222 | 0.1175 | −0.0047 |
| 4 | bordeaux, beihai | 0.0784 | 0.0706 | −0.0078 |

**mean ΔAP = −0.0033 ± 0.0045 · M2 wins 1/5 folds.**

### City level (descriptive)

| city | prev | M1 AP | M2 AP | ΔAP |
|---|---|---|---|---|
| cupertino | 2.37 % | 0.3201 | 0.3324 | **+0.0123** |
| bercy | 0.74 % | 0.0414 | 0.0491 | +0.0077 |
| paris | 0.29 % | 0.0279 | 0.0325 | +0.0046 |
| saclay_e | 0.99 % | 0.0114 | 0.0138 | +0.0024 |
| abudhabi | 3.76 % | 0.0726 | 0.0743 | +0.0017 |
| hongkong | 3.56 % | 0.1518 | 0.1498 | −0.0021 |
| pisa | 1.64 % | 0.0290 | 0.0268 | −0.0022 |
| bordeaux | 1.00 % | 0.0273 | 0.0247 | −0.0026 |
| beirut | 2.69 % | 0.1725 | 0.1676 | −0.0049 |
| mumbai | 2.56 % | 0.1510 | 0.1340 | −0.0170 |
| beihai | 2.49 % | 0.1295 | 0.0952 | −0.0343 |
| rennes | 2.58 % | 0.4557 | 0.4096 | −0.0461 |
| aguasclaras | 1.64 % | 0.1249 | 0.0693 | −0.0556 |
| nantes | 1.14 % | 0.2721 | 0.2072 | **−0.0649** |

**M2 wins 5/14 cities.** Cities inside a fold share a trained model, so these are
**not** independent samples — descriptive only, **no p-value is computed**. (With
5 folds a signed-rank test cannot reach p<0.05 anyway, so none is reported.)

### Challenge metrics (macro over the 14 held-out cities)

| model | AP | ROC-AUC | F1\* | ChangeAcc | NoChangeAcc | Accuracy | final train BCE |
|---|---|---|---|---|---|---|---|
| **M1** | **0.1419** | **0.806** | **0.1994** | **0.347** | 0.926 | 0.915 | **0.4665** |
| M2 | 0.1276 | 0.800 | 0.1877 | 0.319 | **0.950** | **0.938** | 0.4745 |

> M2 leads on **Accuracy and NoChangeAcc**, but that is an operating-point
> artefact: its `τ*` lands more conservative, trading recall for specificity. At
> 2.29 % prevalence Accuracy is nearly free to inflate. The threshold-free **AP**
> and the **F1\*** both favour M1, as does train BCE. This is precisely why
> Accuracy is not the headline metric here.

---

## Reading

**Verdict — the pre-registered "result B".**

> Fixed CZ entanglement introduces **genuine** inter-pixel interactions
> (`|I_ij| ~ 10⁻³` vs M1's `10⁻¹⁶`), but under this task, this 38-parameter
> budget and this training setup those interactions **do not provide a useful
> inductive bias**. Every axis agrees in direction: 4/5 folds, 9/14 cities,
> macro AP, F1\*, and train BCE.

Because the interaction was *verified to exist* before training, the objection
"the entangler did nothing, hence no difference" is ruled out.

**The effect is small but directionally consistent.** Mean ΔAP = −0.0033 against
per-fold APs spanning 0.022–0.252; it is not a large effect.

**The informative part is the asymmetry:**

| | n | mean | best/worst |
|---|---|---|---|
| M2 gains | 5 | +0.0057 | +0.0123 |
| M2 losses | 9 | −0.0255 | −0.0649 |

M2 **never gains much but sometimes loses a lot**, and
`corr(ΔAP, M1's AP) = −0.49`: **the better the separable model does on a city,
the more the CZ layer costs.** Prevalence explains nothing here
(`corr(ΔAP, prevalence) = −0.003`). A plausible reading — untested — is that
fixed, data-independent coupling mixes neighbouring pixels indiscriminately, and
that mixing is most damaging exactly where the per-pixel signal was already
clean.

**What this does NOT say:**
- Not "entanglement is useless for change detection" — one fixed coupling (CZ),
  one budget (38), one encoding (PCA-4), one topology (4-neighbour), one seed
  per fold.
- Not a statement about **data-dependent** coupling (M3). The φ diagnostic
  showed the current M3 ZZ is near-identity for ~90 % of neighbour pairs
  (P50 φ = 0.044 rad), so M3 as configured has never been given a fair test.

---

## Consequences

1. **M1 (38 params) is the QML model to carry into the classical comparison**, not
   M2. Headline: **M1(38) vs 3×3 conv(37)** — same input, same receptive field,
   nearly identical trainable-parameter budget.
2. Adding entanglement is **not** the lever here. Do not spend remaining time
   deepening or widening the entangler.
3. The dominant difficulty remains **cross-city generalization**, unchanged from
   the EDA onward: per-city AP spans 0.011 (saclay_e, ~chance at 1.15× lift) to
   0.456 (rennes) for the *same* model.
4. If time allows, the one quantum-side question still worth asking is whether an
   **actually-on** data-dependent coupling (M3-active, φ rescaled so
   P50 ≈ 0.2–0.3 rad) behaves differently from fixed CZ. That is a separate
   question from the one answered here.
