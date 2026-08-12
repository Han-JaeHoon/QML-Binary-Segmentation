# OSCD Exploratory Data Analysis for Quantum Change Detection

Data-driven EDA for the **Quantum Change Detection in Satellite Earth Observations**
challenge (2026 Niels Bohr Quantum Summer School, SDU Odense).

The task: pixel-level **binary change detection** on multi-temporal Sentinel-2
imagery — flag *urban* growth/structural change (label `255`) while ignoring
natural variation such as seasonal vegetation (label `0`). The end goal is a
**Quantum Machine Learning (QML)** pipeline, where the number of input features
directly costs qubits/parameters. This repository answers the question that
comes *before* modelling:

> **Which features actually carry the change signal, so we can feed a small,
> well-chosen set into a few-qubit circuit?**

All analysis is on the 14 labelled **train** cities. It is fully reproduced by
[`eda.py`](eda.py); numeric outputs are in [`results/RESULTS.md`](results/RESULTS.md).

---

## Dataset

**Onera Satellite Change Detection (OSCD)** — Sentinel-2 MSI, Copernicus.

| | |
|---|---|
| Cities | 24 (14 train w/ labels, 10 test w/ hidden labels) |
| Per city | 13 spectral bands × 2 dates (T1, T2), co-registered to 10 m (`imgs_*_rect`) |
| Image size | variable, 385×241 … 1070×1180 |
| Labels | `<city>-cm.tif` change mask |
| Change prevalence | **2.29 %** of pixels (severe imbalance) |

> ⚠️ **Label encoding gotcha:** the `*.tif` rasters encode `{1 = no change,
> 2 = change}`, **not** the `{0, 1}` stated in the dataset README. Verified
> against `cm.png`. Using the README convention silently labels every pixel as
> change. This repo uses `change = (tif == 2)`.

The dataset itself is **not** committed (size + CC-BY-NC-SA / Copernicus terms).
Download it from the [challenge / OSCD source](https://rcdaudt.github.io/oscd/)
and point `--data_dir` at the extracted `OneraDataset` folder.

---

## Method & findings

Normalization is fixed as **per-band robust min-max**: clip to the per-band
1st/99th percentile (estimated on **train, T1+T2 pooled**, then frozen) and
scale to `[0,1]`. Heavy bright tails (clouds/water push maxima to ~10× P99) make
plain min-max crush typical pixels, so robust clipping is used throughout.
Frozen constants: [`results/norm_params.json`](results/norm_params.json).

### Step 0 — Data hygiene & global shift
No-data pixels are negligible (0.01 %). A mild global radiometric shift exists
(T2 ~3–5 % darker, strongest in NIR/Red-Edge) — a change-independent baseline to
keep in mind. AUC (rank-based) is unaffected by it.

### Step 1 — Per-band relevance: `dB` vs `|dB|`
![band AUC](results/step1_band_auc.png)

Single-band separability, measured as ROC-AUC (imbalance-robust) of the
temporal difference.

**Direction is meaningless; magnitude is the signal.** For every band
`AUC(signed dB) ≈ 0.5`, while `AUC(|dB|)` reaches **0.79**. Urban change goes in
both spectral directions depending on the city, but *"changed a lot"* is
consistent. → **features must be `|dB|`, not signed `dB`.**

![B04 histogram](results/step1_B04_hist.png)

Best bands: **B04 (Red) 0.79**, B05 0.77, B03, B02, then SWIR B12/B11. The
atmospheric bands **B09 (water vapour)** and **B10 (cirrus)** are useless here
(AUC ≤ 0.55; B10's `|dB|` is even below chance) — now dropped on *evidence*, not
just physical prior.

### Step 2 — Redundancy: correlation & grouping of `|dB|`
![correlation](results/step2_corr.png)

The 13 bands collapse to **~5 independent axes** (|corr| > 0.8):
`{B02,B03,B04}` visible · `{B06,B07,B08,B8A}` NIR/Red-Edge · `{B11,B12}` SWIR ·
`B05` Red-Edge-1 (bridge) · isolated `B01/B09/B10`. Keeping the highest-AUC
representative per group is enough.

### Step 3 — Spectral-index change (counter-intuitive)
NDVI/NDWI/NDBI/NDMIR *changes* are **weaker** (AUC 0.56–0.62) than raw `|dB|`
(0.72–0.79). Normalized-difference indices divide out overall brightness — which
is exactly the magnitude signal we need. → indices are **not** used in the
baseline feature set. "Physically plausible" ≠ "discriminative on this data".

### Step 4 — Intrinsic dimensionality & leakage-free validation
![PCA](results/step4_pca.png)
![multivariate AUC](results/step4_multivariate_auc.png)

PCA on `|dB|`: PC1 alone explains **59 %**, top-4 reach 85 %. Multivariate AUC
(logistic regression, **GroupKFold by city → no train/test leakage**):

| feature set | AUC |
|---|---|
| all 13 `|dB|` | 0.812 |
| PCA top-4 | **0.822** |
| compact 4 (B04,B05,B12,B08) | 0.805 |
| compact 3 (B04,B05,B12) | 0.808 |
| single `|dB(B04)|` | 0.795 |

**Compressing 13 → 3–4 features loses essentially nothing.** The per-pixel
spectral ceiling is ~0.82 — beating it requires **spatial context (4×4 patches)**,
not more bands.

---

## Recommendation for the QML pipeline

- **Input features:** per-band robust-normalized **`|dB|`** for a small set —
  **`{B04, B05, B12}` (+`B08`)** or **PCA top-4** → 4-qubit angle encoding.
- **Exclude** signed `dB` and spectral-index deltas (no / weak signal).
- **Drop** B01/B09/B10 (low relevance *and* redundant/isolated).
- Give the parameter-matched classical baseline the **same** features.
- The real accuracy gains live in **spatial patches**, not extra bands.

```
raw T1/T2 → robust per-band norm ([0,1]) → |dB| → {B04,B05,B12(,B08)} / PCA-4
          → 4×4 patch → PQC   vs   param-matched MLP
```

---

## Reproduce

```bash
pip install -r requirements.txt
python eda.py --data_dir /path/to/OneraDataset
```

Regenerates every figure in `results/` and `results/RESULTS.md`. Seeded
(`RandomState(0)`) for deterministic sampling.

## Disclosure

Analysis code and this write-up were prepared with assistance from a generative
AI coding assistant (Anthropic Claude). Every numeric result is produced by the
committed [`eda.py`](eda.py) and is independently reproducible from the raw
dataset; findings (e.g. the `{1,2}` label encoding, the `|dB|` vs `dB` result)
were verified against the data rather than taken on assertion.
