# CV results schema (`cv-1.0`)

Output contract of [`train/run_cv.py`](../train/run_cv.py). Written so a result can
be re-derived, re-plotted or re-analysed months later without rerunning anything,
and so a second experiment (P0 at L=1, a classical baseline) can be joined to it
by an adapter rather than by re-reading logs.

Everything lands under `results/runs/<tag>/`. Each fold is written the moment it
finishes, so an interrupted run keeps its completed folds and `--resume` continues.

```
results/runs/<tag>/
  meta.json                  run-level provenance
  fold<i>.json               per-fold record  (the analysis unit)
  fold<i>_<arm>.jsonl        per-epoch log, one JSON object per line
  fold<i>_<arm>_final.npy    final parameter vector
  fold<i>_maps.npz           every held-out pixel: OOF probability + label
  summary.json               merge of the fold<i>.json present
  REPORT.md                  human-readable summary (train/report_cv.py)
```

`<arm>` is `m1` (separable) or `m2` (fixed CZ grid). Both arms always have the same
parameter count; the CZ layer is non-trainable.

---

## `meta.json`

| field | meaning |
|---|---|
| `schema_version` | `"cv-1.0"` |
| `written_at` | UTC ISO-8601 |
| `git_commit`, `git_dirty` | code state that produced the run |
| `host`, `python`, `numpy`, `pennylane`, `sklearn`, `pillow` | environment |
| `config` | every CLI argument, verbatim (depth, tying, lr, batch, steps_per_epoch, epochs, seeds, …) |
| `fold_assignment` | `{city: fold_index}` for all 14 labelled cities |
| `fold_assignment_sha256` | 16-hex digest of the canonicalized assignment — **compare this across experiments to prove they used the same folds** |
| `folds` | `[{fold, train: [...], val: [...]}, …]` |

## `fold<i>.json`

The analysis unit. One record per fold.

| field | meaning |
|---|---|
| `fold`, `train_cities`, `val_cities` | fold identity |
| `fold_assignment_sha256` | same digest as `meta.json`, repeated so the record stands alone |
| `T_global` | hard-negative threshold fitted on this fold's train cities |
| `paired_stream_identical` | `true` iff both arms consumed the same patch stream in **every** epoch |
| `paired_stream_first_mismatch_epoch` | `null` when identical, else the first epoch that differs |
| `delta_AP_per_city` | `{city: AP(M2) − AP(M1)}` |
| `delta_AP_fold` | pooled `AP(M2) − AP(M1)` over the fold's held-out pixels |
| `fold_seconds`, `started_at`, `finished_at` | timing |
| `done` | `true` only if the paired-stream check passed — **a fold without `done` must not enter an analysis** |
| `arms.<arm>` | see below |

`arms.<arm>`:

| field | meaning |
|---|---|
| `label`, `n_params` | e.g. `"M2 L=2 untied [center]"`, `74` |
| `final_checkpoint.path` / `.sha256` / `.param_norm` / `.params` | the evaluated checkpoint — file path, digest, norm, and the parameter vector itself (74 floats), so a result can be reproduced without the `.npy` |
| `train_BCE` | list of per-epoch mean train BCE |
| `train_BCE_first`, `train_BCE_final` | convenience |
| `cheap_AP_final` | last cheap-val AP — **diagnostic only**, never used for selection |
| `stream_checksums` | per-epoch CRC32 of the sampled `(city,row,col)` list |
| `epoch_log` | path of the matching `.jsonl` |
| `per_city.<city>` | full metric set on that held-out city (below) |
| `pooled` | same metric set over the fold's held-out pixels pooled |
| `train_seconds` | wall-clock of the training loop |

Metric set (from `inference.evaluate_predictions`): `n`, `prevalence`, `AP`,
`roc_auc`, `tau`, `F1`, `precision`, `change_acc` (recall on change),
`nochange_acc` (specificity), `accuracy`, plus `seconds` for the per-city entries.

> `tau` is swept on the same pixels it is scored on, so `F1` / `change_acc` /
> `nochange_acc` are **best-operating-point diagnostics**, not unbiased test
> values. `AP` is the primary, threshold-free metric.

## `fold<i>_<arm>.jsonl`

One JSON object per line, three record types:

- `{"record": "header", …}` — arm, label, `n_params`, `init_sha256`, city lists, config, `started_at`
- `{"record": "epoch", …}` — `epoch`, `train_BCE`, `train_BCE_last_step`,
  `stream_checksum`, `param_norm`, `wall_time`, and on cheap-val epochs
  `cheap_AP` / `cheap_F1`
- `{"record": "footer", …}` — `finished_at`, `final_sha256`

Filter on `record == "epoch"` to get a learning curve.

## `fold<i>_maps.npz`

Every held-out pixel of the fold, at full resolution, addressable by
`(city, row, col)`:

| key | dtype / shape | meaning |
|---|---|---|
| `cities` | `str[k]` | held-out cities in this fold |
| `<city>__p_m1` | `float32 [H,W]` | M1 out-of-fold probability |
| `<city>__p_m2` | `float32 [H,W]` | M2 out-of-fold probability |
| `<city>__y` | `uint8 [H,W]` | ground truth, `1 = change` |
| `<city>__valid` | `bool [H,W]` | pixels with valid reflectance in both dates |

Pooled out-of-fold vectors are rebuilt from these (mask by `valid`), so the
per-pixel arrays and the headline AP can never disagree:

```python
z = np.load("fold0_maps.npz")
c = str(z["cities"][0]); v = z[f"{c}__valid"]
p, y = z[f"{c}__p_m2"][v], z[f"{c}__y"][v]
```

Each labelled pixel appears in exactly one fold, so concatenating all folds gives
one out-of-fold prediction per pixel of the whole labelled set — scored by a model
that never saw that pixel's city.

## `summary.json`

`{schema_version, config, merged_at, folds: {"<i>": <fold record>}}`. Rebuilt from
the `fold<i>.json` files on every fold completion, so several processes can run
disjoint `--folds` concurrently without fighting over one file.

---

## Joining a second experiment

To compare two runs (e.g. P2 at 74 params against P0 at 38):

1. check `meta.fold_assignment_sha256` is equal — otherwise the folds differ and
   fold-level differences are not comparable;
2. join on `fold` for `Δ_int` (interaction, within a run) and across runs for
   `Δ_depth` (capacity);
3. `train/report_cv.py --l2 <p2>/summary.json --l1 <p0>/summary.json` does both and
   prints the pre-registered A/B/C/D case label.

If the other experiment used a different runner, write an adapter that emits this
schema rather than teaching the report script a second format.
