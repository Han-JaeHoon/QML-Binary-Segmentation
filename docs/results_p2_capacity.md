# P2 — does the interaction effect change with capacity?

**Question.** P0 answered "does a fixed nearest-neighbour CZ coupling help at 38
parameters?" with *no*. The obvious rebuttal was **capacity**: the earlier M3
sweep showed 38p was a binding fitting constraint and that useful capacity
saturates near ~74p, so perhaps the entangler had no room to be useful. P2 tests
that by repeating the identical paired comparison at **L=2 untied, 74 parameters**.

**Answer.** The rebuttal does not survive. At 74 parameters M1 wins **5/5 folds**,
and the interaction penalty is roughly **ten times larger** than at 38 parameters.

---

## The four numbers

Fold-mean AP over the 5 city-grouped folds (both runs use the identical fold
assignment, sha256 `c22242aede982d21`):

| | L1 · 38p | L2 · 74p | Δ_depth |
|---|---:|---:|---:|
| **M1** (separable) | 0.1210 | 0.1603 | **+0.0393** |
| **M2** (fixed CZ) | 0.1177 | 0.1283 | **+0.0106** |
| **Δ_int** = M2 − M1 | **−0.0033** | **−0.0320** | −0.0287 |

Same picture under the other aggregation (mean of the 14 per-city APs):
M1 0.1419 → 0.1617, M2 0.1276 → 0.1320, Δ_int −0.0143 → −0.0297.

**Pre-registered case: C** — *the fixed CZ interaction was not a useful inductive
bias in either capacity regime.*

> ### ⚠️ Two different A/B/C schemes exist — do not mix them
> The first plan labelled outcomes **A** = M2 wins / **B** = M1 wins / **C** =
> mixed by fold. The P2 brief later defined a *different* four-way scheme over
> **both** capacities: **A** = M2 wins at L1 and L2, **B** = tie at L1 then M2
> wins at L2, **C** = M1 wins at both, **D** = the sign flips with capacity.
> The P0 report's verdict "결과 B" is the **first** scheme (M1 wins). In the
> P2 scheme the joint result is **C**. Writing "B" in a document that also cites
> the P2 scheme would read as "entanglement needed more capacity" — the opposite
> of what happened.

## What the extra 36 parameters bought

The separable model converts capacity into held-out AP about **3.7× more
effectively** than the entangling one (+0.0393 vs +0.0106 fold-mean). So the CZ
grid does not merely fail to add signal; under this budget it also absorbs part
of the capacity that M1 turns into generalization. M2 is worse on the **fit** axis
too, in 5/5 folds at L2 (final train BCE higher by +0.0162 on average), matching
the direction of the L1 run (0.4745 vs 0.4665).

| | L1 · 38p | L2 · 74p |
|---|---|---|
| fold wins (M1 : M2) | 4 : 1 | **5 : 0** |
| city wins (M1 : M2) | 9 : 5 | **12 : 2** |
| mean Δ_int | −0.0033 | −0.0320 |

## The asymmetry replicates — and sharpens

P0 observed that CZ's rare gains are small while its losses are large, and that
the damage concentrates where the model works best. Both hold at 74 parameters,
more strongly:

| | L1 · 38p | L2 · 74p |
|---|---:|---:|
| M2 max / mean **gain** | +0.0123 / +0.0057 | +0.0219 / +0.0125 |
| M2 max / mean **loss** | −0.0649 / −0.0255 | −0.1258 / −0.0367 |
| corr(Δ_int, M1's AP) | −0.50 (12 cities) | **−0.76** (14 cities) |

At 74p the correlation is strong enough to state plainly: **the better the
separable model does on a city, the more the CZ layer costs there.**

## What did *not* replicate: the per-city story

Per-city Δ_int signs agree between L1 and L2 in only **9 of 14** cities:

| flipped | Δ_int L1 | Δ_int L2 |
|---|---:|---:|
| cupertino | +0.0123 | −0.1258 |
| bercy | +0.0077 | −0.0497 |
| paris | +0.0046 | −0.0051 |
| abudhabi | +0.0017 | −0.0026 |
| nantes | −0.0649 | +0.0219 |

The P0 narrative floated "CZ helps in low-prevalence cities, hurts in
high-prevalence ones". That does not survive: paris (0.29 %) and bercy (0.74 %)
were M2's two clearest low-prevalence wins at 38p and both flip to M1 at 74p,
while nantes flips the other way. **Per-city Δ_int is not stable across capacity,
so no per-city or prevalence-conditioned claim should be made.** Only the
aggregate direction is stable — and it points the same way at both capacities.

## Caveats that belong with these numbers

1. **Δ_int is strong evidence; Δ_depth is weaker.** Δ_int is measured *within* a
   run: same fold object, same initialization, same patch stream (per-epoch
   checksums identical in every fold of both runs), only the CZ layer differs.
   Δ_depth compares two runs executed on different machines with different
   runners, so it inherits any protocol difference between them; treat the
   capacity gain as approximate.
2. **The P0 numbers here are transcribed** from the P0 run report, not read from
   P0 artefacts (`train/adapt_p0.py` states this in the file it writes, and marks
   the two cities whose per-city AP was reported only as a delta). Replace it
   with a real converter when the artefacts land.
3. Single seed per arm per capacity; 5 folds is a small, dependent sample. Report
   direction and spread, not significance.
4. L=2 raises depth **and** parameter count together — "increased depth/capacity
   under untied data re-uploading", never "data re-uploading works". The earlier
   tied-L2 control (38p, no gain) is what makes the parameter-capacity reading
   the likely one.
5. No advantage language until the 37-parameter classical convolution runs.

## Reproduce

```bash
python train/adapt_p0.py                       # freeze the transcribed P0 numbers
python train/report_cv.py \
    --l2 results/runs/p2_l2_cv/summary.json \
    --l1 results/runs/p0_l1_cv_transcribed/summary.json
```

P2 raw artefacts are in `results/runs/p2_l2_cv/` — per-fold records, per-epoch
logs for both arms, final checkpoints, and `fold<i>_maps.npz` carrying every
held-out pixel's out-of-fold probability for both arms with ground truth. Schema:
[`results_schema.md`](results_schema.md).
