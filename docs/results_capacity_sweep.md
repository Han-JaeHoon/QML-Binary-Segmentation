# Capacity Sweep — does more QML capacity buy cross-city generalization?

**Question.** Does increasing M3 from 38 → 74 → 110 trainable parameters improve
*cross-city generalization*, or only *training fit*?

**Protocol (identical across all three).** dev 11/3 city split · PCA-4 · γ=π/2 ·
Adam lr=0.02 · batch 32 · 160 steps/epoch (5120 patches) · 20 epochs · plain BCE ·
seed 0 · same city-balanced 1:1:2 sampler · same fixed natural-prevalence
cheap-val coordinates. Checkpoints selected on **pooled cheap-val AP** over all
three val cities (never on a single city). L2-*tied* is excluded — it already
served as the depth-vs-parameters diagnostic.

> These are **capacity-scaling ablations, not headline models.** L2/L3 are not
> parameter-matched to the classical baseline; the challenge headline comparison
> remains **M3 L=1 (38 params) vs classical 3×3 conv (37 params)**.

---

## 1. Sweep results

### Training / cheap-val

| model | params | BCE min | BCE (last 5) | best cheap AP | @ep | cheap F1\* | AP (last 5) |
|---|---|---|---|---|---|---|---|
| L1 | 38 | 0.4815 | 0.4890 | 0.0781 | 16 | 0.159 | 0.0671 |
| **L2** | 74 | 0.4659 | 0.4717 | **0.1051** | 10 | 0.227 | 0.0973 |
| L3 | 110 | 0.4636 | 0.4689 | 0.1096 | 14 | 0.257 | 0.1002 |

### Exhaustive full-city (bestcheap checkpoints, overlap-averaged)

Per city — AP / F1\*:

| city | prevalence | L1 (38) | L2 (74) | L3 (110) |
|---|---|---|---|---|
| paris | 0.29 % | 0.018 / 0.041 | 0.037 / 0.091 | 0.043 / 0.104 |
| cupertino | 2.37 % | 0.269 / 0.359 | 0.258 / 0.454 | 0.261 / 0.467 |
| beihai | 2.49 % | 0.069 / 0.129 | 0.099 / 0.224 | 0.096 / 0.217 |

Aggregates:

| aggregation | metric | L1 (38) | L2 (74) | L3 (110) |
|---|---|---|---|---|
| **micro** (pooled, one global τ\*) | **AP** | 0.1108 | **0.1466** | 0.1436 |
| | **F1\*** | 0.1888 | **0.3017** | 0.3003 |
| | ROC-AUC | 0.8649 | 0.8985 | 0.9067 |
| | precision | 0.120 | 0.200 | 0.194 |
| | ChangeAcc | 0.439 | 0.614 | 0.662 |
| macro (per-city mean) | AP | 0.1187 | 0.1314 | 0.1334 |
| | F1\* | 0.1765 | 0.2564 | 0.2628 |

---

## 2. Train vs validation

**L1 → L2.** Train BCE falls only **0.016** (0.4815 → 0.4659), yet validation
improves substantially and consistently: micro AP **+32 %** (0.111 → 0.147),
micro F1\* **+60 %** (0.189 → 0.302), and **every** city improves in F1\*.

**L2 → L3.** Both saturate. Train BCE −0.002; micro AP 0.147 → 0.144 and micro
F1\* 0.302 → 0.300 (macro nudges up: AP 0.131 → 0.133, F1\* 0.256 → 0.263). The
differences sit inside the epoch-to-epoch cheap-val spread (L2 0.088–0.105,
L3 0.083–0.110). **L3 > L2 is not established** by these runs.

**A note that revises an earlier diagnostic.** In the fixed-batch test the same
38 → 74 change moved the loss 0.471 → 0.189. On the real stream it moves it
0.016. The fixed-batch setting rewards *memorizing* 32 patches; training sees
5120 fresh patches per epoch and cannot memorize. That diagnostic therefore
**overstated** the capacity difference, and its ordering should not be read as a
prediction of generalization.

---

## 3. Which interpretation (A–D)?

- **A** (L1 < L2 < L3 in train *and* val) — holds for **L1 → L2 only**.
- **B** (train improves, val plateaus) — **no**: at L2 → L3 *train also* plateaus,
  so this is not a case of fitting capacity failing to transfer.
- **C** (intermediate optimum, L3 declines) — **closest**, but L3 does not
  decline; it is flat. So: *capacity was a binding constraint at 38 parameters
  and is no longer binding beyond ≈74.*
- **D** (only specific cities improve) — **no** for the L1 → L2 step (all three
  improve). **However**, cross-city spread persists at every capacity: at L3,
  F1\* is 0.467 (cupertino) vs 0.217 (beihai) vs 0.104 (paris), and cupertino/beihai
  still differ ~2.7× in AP despite near-identical prevalence.

**Conclusion.** Extra capacity helps up to ~74 parameters and then saturates in
both fit and generalization. It **lifts all cities but does not compress the
cross-city spread** — domain generalization is a separate axis that capacity does
not address.

A curiosity worth noting: on cupertino, **AP is flat across all three models**
(0.269 / 0.258 / 0.261) while **F1\* rises** (0.359 → 0.454 → 0.467). Capacity
improved the best operating point there without improving overall ranking.

---

## 4. Diagnostic: is the spatial ZZ actually engaging?

Distribution of the ZZ phase φ = γ·sᵢ·sⱼ (γ=π/2) over nearest-neighbour pairs,
5 train cities, PCA branch:

| P50 | P75 | P90 | P99 | mean | max | fraction > 0.1 rad |
|---|---|---|---|---|---|---|
| 0.044 | 0.061 | 0.096 | 1.361 | 0.085 | 1.571 | **9.6 %** |

**For ~90 % of neighbour pairs the ZZ rotation is < 0.1 rad — effectively
near-identity**, with a small heavy tail that saturates near π/2. So the
data-dependent coupling behaves like a sparse, high-threshold interaction: it
engages only where *both* pixels changed strongly. This is consistent with the
structural check at init (M3's neighbour→centre influence was ~10× weaker than
M2's CNOT) and means **M3 is operating close to M1 (no spatial mixing) over most
of the map**. Cause: `s = clip(‖z‖₂/c_norm, 0, 1)` with `c_norm = P99(‖z‖)` makes
typical `s ≈ 0.17`, so `sᵢsⱼ ≈ 0.03`.

---

## 5. What to run next

**Not more depth** — it has saturated.

1. **M1 vs M2 vs M3 at fixed capacity (L=2 untied, 74 params). Highest priority.**
   The project's central claim — that a *data-dependent* nearest-neighbour ZZ beats
   no spatial mixing (M1) and beats a *fixed* CNOT coupling (M2) — is still
   **untested**: every run so far has been M3. The φ diagnostic makes this urgent,
   because if ZZ is near-identity for 90 % of pixels, M3 may not separate from M1.
2. **γ / s-normalization ablation** (γ ∈ {0.5, 1, π/2}, and normalizing `s` by a
   lower percentile than P99). Directly motivated by the φ distribution; changes no
   parameter count, so it stays parameter-matched.
3. Only then: connectivity (4- vs 8-neighbour) and representation (Physical-4 vs
   PCA-4).

---

## Caveats

- **Timing anomaly.** L1's cupertino/beihai exhaustive evaluations logged 12215 s /
  10101 s, ~10× the depth-scaled expectation and inconsistent with L2/L3, whose
  paris timings scale correctly with depth (131 / 256 / 377 s ≈ 1:2:3). This was an
  external machine slowdown during that window. Metrics are deterministic and
  unaffected; those two timings should not be used.
- Single seed, single dev split (11/3). Model ranking should be confirmed under
  5-fold city-grouped CV before any final claim.
- τ\* is selected on validation; per-city τ\* (macro) is optimistic, the pooled
  single-τ\* micro row is the deployment-realistic number.
