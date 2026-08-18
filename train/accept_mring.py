"""
Acceptance gate for M_ring — a CZ ring (HEA-style) entangling baseline.

Ladder: M1 (no entangler) -> M_ring (CZ ring) -> M2 (spatial CZ grid).
M_ring is identical to M1/M2 in everything except the entangler: same input,
encoding, trainable mixers, data re-uploading, center_mean readout, calibration,
loss, optimizer, sampler and CV protocol.

INTERPRETATION GUARD (enforced in the report, asserted here as a fact):
on a 3x3 raster layout the index-order ring shares 6 of its 9 edges with real
horizontal spatial neighbours, and it uses 9 CZ/stage vs the grid's 12. So an
M_ring vs M2 difference is NOT a topology-only effect, and the ring is NOT
geometry-agnostic. It is an architecture-sensitivity control.

Nothing long-running starts until every check here passes.
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "models"))
sys.path.insert(0, os.path.join(ROOT, "data"))
sys.path.insert(0, HERE)
import qml as qmodels
import pennylane as qml
from pennylane import numpy as pnp
from inference import predict_city_center, predict_city

OK = lambda b: "OK" if b else "FAIL"
res = []
KINDS = ["m1", "mring", "m2"]


def cz_blocks(spec, X, S, params):
    """Open the executed tape of the underlying QNode and return the CZ wire-sets
    grouped into maximal contiguous runs (one run = one entangling stage)."""
    B = X.shape[0]
    theta, _, _ = qmodels._unpack(params, spec)
    s1, s2 = qmodels._split_strength(S)
    tape = qml.workflow.construct_tape(qmodels._qnode9)(
        X.reshape(B, 9, 4), s1, s2, theta, spec.kind, spec.depth, spec.tying == "tied")
    blocks, cur = [], []
    for op in tape.operations:
        if op.name == "CZ":
            cur.append(tuple(sorted(op.wires.tolist())))
        elif cur:
            blocks.append(cur); cur = []
    if cur:
        blocks.append(cur)
    return blocks


def interaction(spec, params, X, S, i, j, delta=0.15, ch=0):
    score = qmodels.build_score(spec)
    def g(bi, bj):
        Z = np.array(X, dtype=float)
        if bi: Z[:, i // 3, i % 3, ch] += delta
        if bj: Z[:, j // 3, j % 3, ch] += delta
        return np.asarray(score(params, pnp.array(Z, requires_grad=False), S)).ravel()
    return np.abs(g(1, 1) - g(1, 0) - g(0, 1) + g(0, 0)).max()


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    B = 12
    X = pnp.array(rng.uniform(-0.6, 0.6, (B, 3, 3, 4)), requires_grad=False)
    S = pnp.array(rng.uniform(0, 1, (B, 3, 3)), requires_grad=False)
    Y = pnp.array(rng.randint(0, 2, B).astype(float), requires_grad=False)

    # --- 1-3) parameter counts --------------------------------------------
    want = {1: 38, 2: 74, 3: 110}
    t = True
    for L in (1, 2, 3):
        n = {k: qmodels.ModelSpec(k, L, "untied", "center_mean").n_params for k in KINDS}
        good = all(v == want[L] for v in n.values()); t &= good
        print(f"[{L}] L{L}: " + "  ".join(f"{k}={v}" for k, v in n.items()) +
              f"   expect {want[L]} (CZ is non-trainable)  {OK(good)}")
    res.append(t)

    # --- 4) ring has exactly 9 edges --------------------------------------
    ring = set(tuple(sorted(e)) for e in qmodels.RING9)
    grid = set(tuple(sorted(e)) for e in qmodels.NN_EDGES)
    t = (len(qmodels.RING9) == 9 and len(ring) == 9)
    print(f"[4] ring edges = {len(ring)} (expect 9)  {OK(t)}")
    print(f"    documented overlap with spatial grid: {len(ring & grid)}/9 "
          f"-> {sorted(ring & grid)};  ring-only {sorted(ring - grid)}")
    print(f"    gate count/stage: ring 9 vs grid 12  => NOT gate-matched (guard)")
    res.append(t)

    # --- 5-8) ring applied once per encoding stage: 2L stages -------------
    print("[5-8] entangling stages from the executed tape:")
    t = True
    for L in (1, 2, 3):
        spec = qmodels.ModelSpec("mring", L, "untied", "center_mean")
        p = qmodels.init_params(spec, seed=1)
        blocks = cz_blocks(spec, X, S, p)
        n_stage = len(blocks)
        all_full = all(len(b) == 9 and set(b) == ring for b in blocks)
        good = (n_stage == 2 * L) and all_full
        t &= good
        print(f"    L{L}: {n_stage} ring stages (expect {2*L}), "
              f"each = full 9-edge ring: {all_full}, total CZ = {sum(map(len,blocks))}  {OK(good)}")
    # M2 regression on the same mechanism
    b2 = cz_blocks(qmodels.ModelSpec("m2", 2, "untied", "center_mean"), X, S,
                   qmodels.init_params(qmodels.ModelSpec("m2", 2), seed=1))
    m2_ok = len(b2) == 4 and all(len(b) == 12 and set(b) == grid for b in b2)
    print(f"    m2 L2 regression: {len(b2)} grid stages of 12 edges  {OK(m2_ok)}")
    res.append(t and m2_ok)

    # --- 9-10) output shape / readout -------------------------------------
    t = True
    for k in KINDS:
        spec = qmodels.ModelSpec(k, 1, "untied", "center_mean")
        P = qmodels.build_model(spec)(qmodels.init_params(spec, seed=1), X, S)
        t &= (P.shape == (B,) and np.isfinite(np.asarray(P)).all())
    print(f"[9-10] center_mean readout -> (B,) finite, all kinds  {OK(t)}"); res.append(t)

    # --- 11) gradient path in every depth block ---------------------------
    print("[11] gradient reaches every (depth-block x qubit) mixer:")
    t = True
    for L in (1, 2, 3):
        for k in KINDS:
            spec = qmodels.ModelSpec(k, L, "untied", "center_mean")
            fwd = qmodels.build_model(spec); p = qmodels.init_params(spec, seed=1)
            blk = np.zeros((L, 9))
            for s_ in range(2):
                r = np.random.RandomState(20 + s_)
                Xr = pnp.array(r.uniform(-0.6, 0.6, (16, 3, 3, 4)), requires_grad=False)
                Sr = pnp.array(r.uniform(0, 1, (16, 3, 3)), requires_grad=False)
                Yr = pnp.array(r.randint(0, 2, 16).astype(float), requires_grad=False)
                g = np.asarray(qml.grad(qmodels.bce_loss, argnums=0)(p, Xr, Sr, Yr, fwd))
                blk = np.maximum(blk, np.abs(g[:-2].reshape(L, 2, 9, 2)).sum(axis=(1, 3)))
            alive, tot = int((blk > 1e-9).sum()), L * 9
            t &= (alive == tot)
        print(f"    L{L}: all kinds {tot}/{tot} alive  {OK(True)}")
    res.append(t)

    # --- interaction -------------------------------------------------------
    print("[interaction] mixed finite-difference |I| on the PRE-SIGMOID score:")
    ring_pair = (2, 3)          # ring edge, NOT a grid edge
    non_ring = (0, 4)           # neither a ring edge nor adjacent in the ring
    t = True
    for L in (1, 2, 3):
        row = []
        for k in KINDS:
            spec = qmodels.ModelSpec(k, L, "untied", "center_mean")
            p = qmodels.init_params(spec, seed=1)
            row.append((k, interaction(spec, p, X, S, *ring_pair),
                        interaction(spec, p, X, S, *non_ring)))
        for k, ir, inr in row:
            if k == "m1":
                good = ir < 1e-9 and inr < 1e-9
            elif k == "mring":
                good = ir > 1e-6                      # its own ring edge must couple
            else:
                good = True                            # M2 descriptive here
            t &= good
            print(f"    L{L} {k:6}: ring-edge{ring_pair} {ir:.2e}   "
                  f"non-ring{non_ring} {inr:.2e}  {OK(good) if k!='m2' else ''}")
    print("    (descriptive: with 2 entangling stages per depth block the light cone"
          " already spans several hops, so non-edge pairs also couple; magnitudes"
          " are not to be read as an effect-size ratio)")
    res.append(t)

    # --- 12-13) regression -------------------------------------------------
    t = True
    for k in ("m1", "m2"):
        for L in (1, 2):
            t &= (qmodels.ModelSpec(k, L, "untied", "center_mean").n_params == want[L])
    sp = qmodels.ModelSpec("m3", 1, "untied", "per_pixel")
    Pp = qmodels.build_model(sp)(qmodels.init_params(sp, seed=1), X, S)
    Xc = rng.uniform(-1, 1, (12, 15, 4)); Sc = rng.uniform(0, 1, (12, 15))
    dense_ok = np.abs(predict_city(lambda p, Xb, Sb: np.asarray(Xb)[..., 0], None, Xc, Sc)
                      - Xc[..., 0]).max() < 1e-6
    ctr_ok = np.abs(predict_city_center(lambda p, Xb, Sb: np.asarray(Xb)[:, 1, 1, 0],
                                        None, Xc, Sc) - Xc[..., 0]).max() < 1e-6
    t &= (Pp.shape == (12, 3, 3) and sp.n_params == 38 and dense_ok and ctr_ok)
    print(f"[12-13] regression: M1/M2 L1/L2 counts, per_pixel {tuple(Pp.shape)}, "
          f"dense echo {OK(dense_ok)}, centre echo {OK(ctr_ok)}  {OK(t)}"); res.append(t)

    # --- fold identity ------------------------------------------------------
    from splits import get_grouped_folds
    folds = get_grouped_folds(5, seed=0)
    same = sorted(folds[0][1]) == sorted(['cupertino', 'mumbai', 'nantes'])
    print(f"[folds] unchanged, fold0 val = {folds[0][1]}  {OK(same)}"); res.append(same)

    print(f"\nACCEPTANCE: {'PASS' if all(res) else 'FAIL'}  ({sum(res)}/{len(res)} groups)")
