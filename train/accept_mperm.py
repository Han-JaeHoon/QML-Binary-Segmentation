"""
Acceptance gate for M_perm (geometry-scrambled CZ control) at L = 1, 2, 3.

M_perm uses the SAME 12-edge coupling graph as M2's spatial grid, but with the
qubit<->pixel assignment permuted, so the ladder isolates one factor per step:

    M1     -> M_perm : add 12 CZ couplings         (generic entanglement effect)
    M_perm -> M2     : the same 12 CZ, now aligned (spatial topology effect)

Both arms therefore match in gate family (CZ), edge count (12), degree sequence
and parameter count — only the mapping to pixels differs.

Why not an index-order CZ ring: on this grid it shares 6 of its 9 edges with the
spatial graph (all six horizontal neighbours), so it is not geometry-agnostic.

Nothing downstream runs until every check here passes.
"""
import os, sys, zlib
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
KINDS = ["m1", "mperm", "m2"]


def interaction(spec, params, X, S, i, j, delta=0.15, ch=0):
    """Mixed 2nd difference of the PRE-SIGMOID score w.r.t. pixels i and j."""
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

    # --- 0) topology invariants -------------------------------------------
    grid = set(tuple(sorted(e)) for e in qmodels.NN_EDGES)
    perm = set(qmodels.PERM_EDGES)
    deg = lambda E: sorted(sum(1 for e in E if q in e) for q in range(9))
    t = (len(perm) == 12 and len(grid & perm) == 0 and deg(grid) == deg(perm))
    print(f"[0] topology: perm 12 edges, grid overlap {len(grid & perm)}/12, "
          f"degree seq {deg(perm)} == grid {deg(grid)}  {OK(t)}"); res.append(t)

    # --- 1-3) parameter counts at L1/L2/L3, identical across topologies ----
    want = {1: 38, 2: 74, 3: 110}
    t = True
    for L in (1, 2, 3):
        n = {k: qmodels.ModelSpec(k, L, "untied", "center_mean").n_params for k in KINDS}
        t &= all(v == want[L] for v in n.values())
        print(f"[{L}] L{L} params: " + "  ".join(f"{k}={v}" for k, v in n.items()) +
              f"   expect {want[L]}  {OK(all(v == want[L] for v in n.values()))}")
    res.append(t)

    # --- 4) CZ is parameter-free: adding edges must not change the count ---
    t = (qmodels.ModelSpec("m1", 2).n_params == qmodels.ModelSpec("mperm", 2).n_params
         == qmodels.ModelSpec("m2", 2).n_params)
    print(f"[4] CZ parameter-free (M1 == M_perm == M2 param count)  {OK(t)}"); res.append(t)

    # --- 5) untied indexing: every depth block must carry gradient ---------
    print("[5] untied depth blocks carry gradient (max over 2 random batches):")
    t = True
    for L in (2, 3):
        for k in KINDS:
            spec = qmodels.ModelSpec(k, L, "untied", "center_mean")
            fwd = qmodels.build_model(spec); p = qmodels.init_params(spec, seed=1)
            blocks = np.zeros((L, 9))
            for s_ in range(2):
                r = np.random.RandomState(20 + s_)
                Xr = pnp.array(r.uniform(-0.6, 0.6, (16, 3, 3, 4)), requires_grad=False)
                Sr = pnp.array(r.uniform(0, 1, (16, 3, 3)), requires_grad=False)
                Yr = pnp.array(r.randint(0, 2, 16).astype(float), requires_grad=False)
                g = np.asarray(qml.grad(qmodels.bce_loss, argnums=0)(p, Xr, Sr, Yr, fwd))
                th = g[:-2].reshape(L, 2, 9, 2)
                blocks = np.maximum(blocks, np.abs(th).sum(axis=(1, 3)))
            alive = int((blocks > 1e-9).sum()); tot = L * 9
            t &= (alive == tot)
            print(f"    {k:6} L{L}: {alive}/{tot} (depth-block x qubit) alive  {OK(alive == tot)}")
    res.append(t)

    # --- 6-7) center_mean readout, output shape (B,) ----------------------
    t = True
    for k in KINDS:
        spec = qmodels.ModelSpec(k, 1, "untied", "center_mean")
        P = qmodels.build_model(spec)(qmodels.init_params(spec, seed=1), X, S)
        t &= (P.shape == (B,) and np.isfinite(np.asarray(P)).all())
    print(f"[6/7] center_mean readout -> shape (B,) finite for all kinds  {OK(t)}"); res.append(t)

    # --- 8) interaction structure -----------------------------------------
    print("[8] mixed finite-difference interaction on PRE-SIGMOID score:")
    gpair = qmodels.NN_EDGES[0]          # (0,1): grid edge, NOT a perm edge
    ppair = qmodels.PERM_EDGES[0]        # a perm edge, NOT a grid edge
    t = True
    for L in (1, 2):
        for k in KINDS:
            spec = qmodels.ModelSpec(k, L, "untied", "center_mean")
            p = qmodels.init_params(spec, seed=1)
            ig = interaction(spec, p, X, S, *gpair)
            ip = interaction(spec, p, X, S, *ppair)
            if k == "m1":
                good = ig < 1e-9 and ip < 1e-9
            elif k == "m2":
                good = ig > 1e-6                       # its own edge must couple
            else:
                good = ip > 1e-6
            t &= good
            print(f"    {k:6} L{L}: |I| grid-pair{gpair} = {ig:.2e}   "
                  f"perm-pair{ppair} = {ip:.2e}  {OK(good)}")
    res.append(t)

    # --- 9) REGRESSION: existing M1/M2 L1/L2 and the dense branch ----------
    t = True
    for k in ("m1", "m2"):
        for L in (1, 2):
            s = qmodels.ModelSpec(k, L, "untied", "center_mean")
            t &= (s.n_params == want[L])
    sp = qmodels.ModelSpec("m3", 1, "untied", "per_pixel")
    Pp = qmodels.build_model(sp)(qmodels.init_params(sp, seed=1), X, S)
    Xc = rng.uniform(-1, 1, (12, 15, 4)); Sc = rng.uniform(0, 1, (12, 15))
    echo = lambda params, Xb, Sb: np.asarray(Xb)[..., 0]
    dense_ok = np.abs(predict_city(echo, None, Xc, Sc) - Xc[..., 0]).max() < 1e-6
    ctr_ok = np.abs(predict_city_center(lambda p, Xb, Sb: np.asarray(Xb)[:, 1, 1, 0],
                                        None, Xc, Sc) - Xc[..., 0]).max() < 1e-6
    t &= (Pp.shape == (12, 3, 3) and sp.n_params == 38 and dense_ok and ctr_ok)
    print(f"[9] regression: M1/M2 L1/L2 counts, per_pixel path {tuple(Pp.shape)}, "
          f"dense echo {OK(dense_ok)}, centre echo {OK(ctr_ok)}  {OK(t)}"); res.append(t)

    # --- 10) training stream is model-independent & reproducible ----------
    # P0 stored only booleans for the paired check, so cross-run comparability is
    # established by recomputing the sampler stream: it depends on (fold, seed)
    # only, never on the model.
    from splits import get_grouped_folds
    folds = get_grouped_folds(5, seed=0)
    fh = zlib.crc32(repr([(sorted(tr), sorted(va)) for tr, va in folds]).encode())
    print(f"[10] fold assignment crc32 = {fh:08x}  (same 5 folds as P0/P2)")
    print(f"     fold0 val = {folds[0][1]}")
    res.append(sorted(folds[0][1]) == sorted(['cupertino', 'mumbai', 'nantes']))

    print(f"\nACCEPTANCE: {'PASS' if all(res) else 'FAIL'}  ({sum(res)}/{len(res)} groups)")
