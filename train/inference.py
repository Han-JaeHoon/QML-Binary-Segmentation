"""
train/inference.py — model-agnostic evaluation for the 3x3 -> 3x3 patch models.

Scope: patch models (M1/M2/M3, any depth/tying). M0 is 1x1 -> 1 and needs a
trivial path instead of this overlap routine; not generalized here on purpose.

Two evaluation paths
  cheap  : fixed coordinates drawn UNIFORMLY over eligible pixels (natural
           prevalence, NOT the 1:1:2 training mixture), same coords every epoch
           so metric moves reflect the model, not the sample. Reads the CENTRE
           pixel of each patch (no overlap averaging) -> a proxy.
  exhaustive : stride-1 over the whole city with overlap averaging -> the
           authoritative number. Cost note: ~0.65 ms/patch, so a 1070x1180 city
           (~1.26M centres) takes ~15 min; run it periodically, not every epoch.

Border handling (important):
  input context  -> reflect-pad by 1 so every original pixel can be a centre
  output accumul -> only positions inside the ORIGINAL HxW are accumulated
  S(p) = sum_k p_k(p),  C(p) = #patches covering p,  P(p) = S(p)/C(p)

Threshold-free metric is **Average Precision (AP)** = sklearn
`average_precision_score` (chosen and named explicitly; it is not identical to
trapezoidal `auc(recall, precision)`). tau* is selected from the
`precision_recall_curve` candidate thresholds by maximizing F1 — on validation
ONLY. The API separates prediction from metrics so test-time threshold
re-selection is hard to do by accident.
"""
import numpy as np
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_recall_curve)


# --------------------------------------------------------------------------- #
# prediction
# --------------------------------------------------------------------------- #
def _reflect_pad(A, m=1):
    pw = [(m, m), (m, m)] + [(0, 0)] * (A.ndim - 2)
    return np.pad(A, pw, mode="reflect")


def predict_city(forward, params, X_city, S_city, batch_size=4096, dtype=np.float32):
    """Stride-1 overlap-averaged full-city prediction.
    X_city (H,W,4); S_city (H,W) or (H,W,2)  ->  P (H,W) in [0,1]."""
    H, W = X_city.shape[:2]
    Xp, Sp = _reflect_pad(X_city), _reflect_pad(S_city)

    centers = [(r, c) for r in range(H) for c in range(W)]
    preds = np.empty((H * W, 3, 3), dtype=dtype)
    for i0 in range(0, len(centers), batch_size):
        chunk = centers[i0:i0 + batch_size]
        Xb = np.stack([Xp[r:r + 3, c:c + 3] for (r, c) in chunk])
        Sb = np.stack([Sp[r:r + 3, c:c + 3] for (r, c) in chunk])
        preds[i0:i0 + len(chunk)] = np.asarray(forward(params, Xb, Sb), dtype=dtype)
    preds = preds.reshape(H, W, 3, 3)

    acc = np.zeros((H, W), dtype=np.float64)
    cnt = np.zeros((H, W), dtype=np.float64)
    for dr in range(3):
        for dc in range(3):
            # centre (r,c) writes its (dr,dc) output to (r+dr-1, c+dc-1);
            # keep only writes landing inside the ORIGINAL domain.
            r0, r1 = max(0, 1 - dr), min(H, H + 1 - dr)
            c0, c1 = max(0, 1 - dc), min(W, W + 1 - dc)
            acc[r0 + dr - 1:r1 + dr - 1, c0 + dc - 1:c1 + dc - 1] += preds[r0:r1, c0:c1, dr, dc]
            cnt[r0 + dr - 1:r1 + dr - 1, c0 + dc - 1:c1 + dc - 1] += 1.0
    assert cnt.min() > 0, "some pixel was never predicted"
    return acc / cnt


def make_fixed_val_coordinates(labels, valid, n=10000, seed=0, patch_size=3):
    """Fixed cheap-val centres, UNIFORM over eligible pixels -> natural prevalence.
    Deterministic given (labels, valid, n, seed)."""
    m = patch_size // 2
    H, W = labels.shape
    interior = np.zeros_like(valid, dtype=bool)
    interior[m:H - m, m:W - m] = True
    elig = np.argwhere(valid & interior)
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(elig), size=min(n, len(elig)), replace=False)
    return elig[idx]


def predict_coordinates(forward, params, X_city, S_city, coords, batch_size=4096):
    """Centre-pixel prediction at given coordinates (cheap-val path, no overlap
    averaging). Returns p (N,) aligned with `coords`."""
    out = np.empty(len(coords), dtype=np.float64)
    for i0 in range(0, len(coords), batch_size):
        ch = coords[i0:i0 + batch_size]
        Xb = np.stack([X_city[r - 1:r + 2, c - 1:c + 2] for (r, c) in ch])
        Sb = np.stack([S_city[r - 1:r + 2, c - 1:c + 2] for (r, c) in ch])
        out[i0:i0 + len(ch)] = np.asarray(forward(params, Xb, Sb))[:, 1, 1]
    return out


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def evaluate_predictions(P, Y, tau=None, select_threshold=False, mask=None):
    """Metrics from probabilities. Either supply `tau` OR set select_threshold=True
    (validation only) — never both, so test-time re-selection is hard by accident."""
    if select_threshold and tau is not None:
        raise ValueError("pass tau OR select_threshold=True, not both")
    p = np.asarray(P).ravel() if mask is None else np.asarray(P)[mask].ravel()
    y = np.asarray(Y).ravel() if mask is None else np.asarray(Y)[mask].ravel()
    y = (y > 0).astype(int)

    out = {"n": int(y.size), "prevalence": float(y.mean())}
    if y.min() == y.max():                       # degenerate (one class only)
        out.update(AP=float("nan"), roc_auc=float("nan"))
    else:
        out["AP"] = float(average_precision_score(y, p))       # primary, threshold-free
        out["roc_auc"] = float(roc_auc_score(y, p))            # secondary

    if select_threshold:
        prec, rec, thr = precision_recall_curve(y, p)
        f1 = np.divide(2 * prec * rec, prec + rec,
                       out=np.zeros_like(prec), where=(prec + rec) > 0)
        k = int(np.argmax(f1[:-1])) if len(thr) else 0         # last point has no thr
        tau = float(thr[k]) if len(thr) else 0.5
    elif tau is None:
        tau = 0.5

    yhat = (p >= tau).astype(int)
    tp = int(((yhat == 1) & (y == 1)).sum()); fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum()); tn = int(((yhat == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    out.update(
        tau=float(tau),
        F1=float(2 * prec * rec / (prec + rec)) if prec + rec else 0.0,
        precision=float(prec),
        change_acc=float(rec),                                  # recall on y=1
        nochange_acc=float(tn / (tn + fp)) if tn + fp else 0.0,  # specificity
        accuracy=float((tp + tn) / y.size),
    )
    return out


# --------------------------------------------------------------------------- #
# acceptance tests (synthetic — machinery is model/data agnostic)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    rng = np.random.RandomState(0)
    H, W = 17, 23
    X = rng.uniform(-1, 1, (H, W, 4)); S = rng.uniform(0, 1, (H, W))

    # 1) constant model -> constant map
    const = lambda params, Xb, Sb: np.full((len(Xb), 3, 3), 0.7)
    P = predict_city(const, None, X, S)
    # tolerance is float32-scale: preds are stored float32 to halve memory
    print(f"[1] constant model      -> max|P-0.7| = {np.abs(P-0.7).max():.2e}  "
          f"{'OK' if np.abs(P-0.7).max() < 1e-6 else 'FAIL'}")

    # 2) alignment: model echoes each patch position's own feature ->
    #    overlap average must reconstruct X[...,0] EXACTLY (offset math check)
    echo = lambda params, Xb, Sb: np.asarray(Xb)[..., 0]
    P = predict_city(echo, None, X, S)
    err = np.abs(P - X[..., 0]).max()
    print(f"[2] alignment (echo)    -> max|P-X0| = {err:.2e}  {'OK' if err < 1e-6 else 'FAIL'}")

    # 3) every pixel covered (borders included)
    cnt = np.zeros((H, W))
    for dr in range(3):
        for dc in range(3):
            r0, r1 = max(0, 1-dr), min(H, H+1-dr); c0, c1 = max(0, 1-dc), min(W, W+1-dc)
            cnt[r0+dr-1:r1+dr-1, c0+dc-1:c1+dc-1] += 1
    print(f"[3] coverage C(p)       -> min {int(cnt.min())} (corner), max {int(cnt.max())} "
          f"(interior)  {'OK' if cnt.min() > 0 else 'FAIL'}")

    # 4) batched == unbatched
    noisy = lambda params, Xb, Sb: 1/(1+np.exp(-(np.asarray(Xb)[...,0]+np.asarray(Sb))))
    Pa = predict_city(noisy, None, X, S, batch_size=7)
    Pb = predict_city(noisy, None, X, S, batch_size=100000)
    print(f"[4] batch invariance    -> max|Δ| = {np.abs(Pa-Pb).max():.2e}  "
          f"{'OK' if np.abs(Pa-Pb).max() < 1e-12 else 'FAIL'}")

    # 5) tau* deterministic + AP sane
    Y = (rng.uniform(size=(H, W)) < 0.05).astype(int)
    Pp = np.clip(0.3 * Y + rng.uniform(0, 0.6, (H, W)), 0, 1)
    m1 = evaluate_predictions(Pp, Y, select_threshold=True)
    m2 = evaluate_predictions(Pp, Y, select_threshold=True)
    print(f"[5] tau* deterministic  -> {m1['tau']:.6f} == {m2['tau']:.6f}  "
          f"{'OK' if m1['tau'] == m2['tau'] else 'FAIL'}   AP={m1['AP']:.3f} F1={m1['F1']:.3f}")

    # 6) cheap-val coords fixed + natural prevalence (no 1:1:2 balancing)
    valid = np.ones((H, W), bool)
    lab = (rng.uniform(size=(H, W)) < 0.10).astype(int)
    c1 = make_fixed_val_coordinates(lab, valid, n=200, seed=3)
    c2 = make_fixed_val_coordinates(lab, valid, n=200, seed=3)
    prev_all = lab[1:-1, 1:-1].mean(); prev_val = lab[c1[:,0], c1[:,1]].mean()
    print(f"[6] cheap-val coords    -> identical {np.array_equal(c1,c2)}   "
          f"prevalence val {prev_val:.3f} vs eligible {prev_all:.3f} (natural)")

    # 7) API guard: tau and select_threshold are mutually exclusive
    try:
        evaluate_predictions(Pp, Y, tau=0.5, select_threshold=True); print("[7] guard FAIL")
    except ValueError:
        print("[7] guard               -> tau + select_threshold rejected  OK")
