# P2 — M1 vs M2-CZ at 74 parameters — paired 5-fold city-grouped CV

*generated 2026-08-18T00:51:06 from `results/runs/p2_l2_cv/summary.json`*

## Setup

- **Arms**: M1 (separable) vs M2 (CZ), **74 trainable parameters each** — the only difference is the fixed CZ grid
- **Circuit**: depth 2 untied, centre branch 3x3x4 -> 1, PCA-4
- **Budget**: 50 epochs x 320 steps x batch 32, Adam lr 0.02, plain BCE, final checkpoint evaluated
- **Folds**: 5/5 complete, assignment sha256 `c22242aede982d21`
- **Provenance**: commit `c32da1c157` (dirty), pennylane 0.45.0, numpy 1.26.4
- **Paired control**: patch stream identical in every fold — **YES**

## Per-fold AP (pooled over that fold's held-out cities)

| fold | held-out cities | M1 | M2 | dAP |
|---|---|---:|---:|---:|
| 0 | cupertino, mumbai, nantes | 0.3055 | 0.2381 | -0.0674 |
| 1 | saclay_e, pisa, aguasclaras | 0.0335 | 0.0320 | -0.0015 |
| 2 | paris, bercy, rennes | 0.1765 | 0.1236 | -0.0529 |
| 3 | hongkong, abudhabi, beirut | 0.1467 | 0.1318 | -0.0149 |
| 4 | bordeaux, beihai | 0.1393 | 0.1159 | -0.0234 |
| **mean** | | **0.1603** | **0.1283** | **-0.0320** |
| std | | 0.0975 | 0.0733 | 0.0273 |

Fold win count: **M2 0 / M1 5** of 5. The std is the spread across folds, not a confidence interval.

## Per-held-out-city AP

| city | prevalence | M1 | M2 | dAP |
|---|---:|---:|---:|---:|
| nantes | 0.0114 | 0.1440 | 0.1659 | +0.0219 |
| saclay_e | 0.0099 | 0.0263 | 0.0294 | +0.0031 |
| aguasclaras | 0.0164 | 0.0811 | 0.0805 | -0.0006 |
| abudhabi | 0.0376 | 0.0651 | 0.0625 | -0.0026 |
| pisa | 0.0164 | 0.0396 | 0.0346 | -0.0049 |
| paris | 0.0029 | 0.0392 | 0.0340 | -0.0051 |
| beirut | 0.0269 | 0.2119 | 0.1995 | -0.0125 |
| beihai | 0.0249 | 0.1517 | 0.1385 | -0.0133 |
| bordeaux | 0.0100 | 0.0848 | 0.0436 | -0.0412 |
| bercy | 0.0074 | 0.1044 | 0.0547 | -0.0497 |
| mumbai | 0.0256 | 0.2268 | 0.1694 | -0.0574 |
| rennes | 0.0258 | 0.4386 | 0.3802 | -0.0584 |
| hongkong | 0.0356 | 0.2062 | 0.1368 | -0.0694 |
| cupertino | 0.0237 | 0.4442 | 0.3184 | -0.1258 |

City win count: **M2 2 / M1 12** of 14. Descriptive only — cities inside one fold share a trained model, so these are not independent samples and carry no p-value.

## Pooled out-of-fold (every labelled pixel scored exactly once)

| arm | AP | ROC-AUC | F1* | ChangeAcc | NoChangeAcc | Accuracy | tau* |
|---|---:|---:|---:|---:|---:|---:|---:|
| M1 (separable) | 0.0959 | 0.8069 | 0.1808 | 0.319 | 0.948 | 0.9338 | 0.556 |
| M2 (CZ) | 0.0882 | 0.8042 | 0.1564 | 0.266 | 0.950 | 0.9344 | 0.521 |
| **dAP** | **-0.0077** | | | | | | |

F1*, ChangeAcc and NoChangeAcc use a threshold swept on this same pool, so they are best-operating-point diagnostics, not unbiased test values. **AP is the primary metric.**

## Reading guard

- 5 folds is a small, dependent sample: report direction and spread, not significance.
- L=2 raises depth **and** parameter count together, so any gain is "increased depth/capacity under untied data re-uploading", never "data re-uploading works".
- No quantum-advantage language: the parameter-matched 37-parameter classical convolution has not been run.

## Files

| file | contents |
|---|---|
| `meta.json` | config, git commit, package versions, fold assignment + hash |
| `fold<i>.json` | per-city and pooled metrics for both arms, dAP, checkpoint digests, checksum status |
| `fold<i>_<arm>.jsonl` | per-epoch train BCE, stream checksum, param norm, cheap-val diagnostics |
| `fold<i>_<arm>_final.npy` | final parameter vector |
| `fold<i>_maps.npz` | every held-out pixel: OOF probability per arm, ground truth, valid mask |
