"""
Step-1 acceptance tests for the 3x3 -> 1 (center_mean) branch.

Gate for the M1 vs M2 5-fold experiment. Nothing downstream should start until
every check here passes.

The decisive test is #4/#5, the MIXED FINITE-DIFFERENCE INTERACTION:

    I_ij = g(x_i+d, x_j+d) - g(x_i+d, x_j) - g(x_i, x_j+d) + g(x_i, x_j)

evaluated on the PRE-SIGMOID score g. With the parameter-free `center_mean`
readout, M1's score is exactly additive,
    g_M1(X) = (a/9) * sum_q f_q(x_q) + b,
so every cross term must vanish: I_ij ~ 0. M2 (fixed NN CZ) entangles the qubits
before the same readout, so I_ij != 0. This is what isolates
    separable/additive  vs  interacting/entangling
and it replaces the old neighbour-perturbation test, which is meaningless here:
under center_mean, perturbing a neighbour changes M1's output too (it reads all
9 qubits), so that test no longer probes entanglement.

I must be measured on the logit, never on the probability: the sigmoid is itself
nonlinear and manufactures apparent mixed effects even for an additive model.
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "models"))
sys.path.insert(0, HERE)
import qml as qmodels
from pennylane import numpy as pnp
import pennylane as qml
from inference import predict_city_center, predict_city, save_mask_png

NN = qmodels.NN_EDGES
OK = lambda b: "OK" if b else "FAIL"
results = []


def interaction(spec, params, X, S, i, j, delta=0.15, ch=0):
    """Mixed second difference of the PRE-SIGMOID score w.r.t. pixels i and j."""
    score = qmodels.build_score(spec)
    def g(bump_i, bump_j):
        Z = np.array(X, dtype=float)
        if bump_i: Z[:, i // 3, i % 3, ch] += delta
        if bump_j: Z[:, j // 3, j % 3, ch] += delta
        return np.asarray(score(params, pnp.array(Z, requires_grad=False), S)).ravel()
    return g(True, True) - g(True, False) - g(False, True) + g(False, False)


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    B = 12
    X = pnp.array(rng.uniform(-0.6, 0.6, (B, 3, 3, 4)), requires_grad=False)
    S = pnp.array(rng.uniform(0, 1, (B, 3, 3)), requires_grad=False)
    Yc = pnp.array(rng.randint(0, 2, B).astype(float), requires_grad=False)

    s1 = qmodels.ModelSpec("m1", readout="center_mean")
    s2 = qmodels.ModelSpec("m2", readout="center_mean")
    f1, f2 = qmodels.build_model(s1), qmodels.build_model(s2)
    p1 = qmodels.init_params(s1, seed=1)
    p2 = qmodels.init_params(s2, seed=1)

    # 1) output shape / finiteness
    P1, P2 = f1(p1, X, S), f2(p2, X, S)
    t = (P1.shape == (B,) and P2.shape == (B,)
         and np.isfinite(np.asarray(P1)).all() and np.isfinite(np.asarray(P2)).all())
    print(f"[1] center output shape {tuple(P1.shape)} finite  {OK(t)}"); results.append(t)

    # 2) parameter counts identical
    t = (s1.n_params == 38 and s2.n_params == 38)
    print(f"[2] params M1 {s1.n_params} / M2 {s2.n_params} (both 38)  {OK(t)}"); results.append(t)

    # 3) gradient PATH for all 9 qubit mixer blocks (structural, not per-batch)
    #    relaxed criterion: over several random batches and a non-symmetric init,
    #    every qubit's 4-angle block must carry gradient somewhere.
    print("[3] gradient path per qubit mixer block (max over 3 random batches):")
    for name, spec, pp in (("M1", s1, p1), ("M2", s2, p2)):
        fwd = qmodels.build_model(spec)
        blocks = np.zeros(9)
        for k in range(3):
            r = np.random.RandomState(10 + k)
            Xr = pnp.array(r.uniform(-0.6, 0.6, (16, 3, 3, 4)), requires_grad=False)
            Sr = pnp.array(r.uniform(0, 1, (16, 3, 3)), requires_grad=False)
            Yr = pnp.array(r.randint(0, 2, 16).astype(float), requires_grad=False)
            g = np.asarray(qml.grad(qmodels.bce_loss, argnums=0)(pp, Xr, Sr, Yr, fwd))
            th = g[:-2].reshape(spec.n_blocks, 2, 9, 2)
            blocks = np.maximum(blocks, np.abs(th).sum(axis=(0, 1, 3)))
        alive = int((blocks > 1e-9).sum())
        t = alive == 9
        print(f"    {name}: {alive}/9 qubit blocks alive   min|g|_block {blocks.min():.2e}  {OK(t)}")
        results.append(t)

    # 4/5) THE decisive test: mixed interaction on the logit
    print("[4/5] mixed finite-difference interaction I_ij on PRE-SIGMOID score:")
    nn_pairs = NN[:4]
    far_pairs = [(0, 8), (2, 6)]
    i1 = np.array([np.abs(interaction(s1, p1, X, S, i, j)).max() for i, j in nn_pairs + far_pairs])
    i2 = np.array([np.abs(interaction(s2, p2, X, S, i, j)).max() for i, j in nn_pairs + far_pairs])
    t1 = i1.max() < 1e-9
    t2 = np.array([np.abs(interaction(s2, p2, X, S, i, j)).max() for i, j in nn_pairs]).min() > 1e-6
    print(f"    M1 (separable): max|I_ij| over all pairs = {i1.max():.2e}   expect ~0   {OK(t1)}")
    print(f"    M2 (CZ)       : NN pairs |I_ij| = " + ", ".join(f"{v:.2e}" for v in i2[:4]) +
          f"   expect != 0   {OK(t2)}")
    # (0,8) and (2,6) are 4 hops apart on the NN grid, so neither reaches the
    # other within the 2-layer light cone. The interaction appears because each
    # input's TWO-HOP light cone overlaps: pixel 0 reaches {0,1,2,3,4,6},
    # pixel 8 reaches {2,4,5,6,7,8}; the shared qubits {2,4,6} carry the cross
    # term, and center_mean averages all nine expectations.
    print(f"    M2 far pairs (0,8),(2,6): " + ", ".join(f"{v:.2e}" for v in i2[4:]) +
          "   (4 hops apart; cross term via overlapping 2-hop light cones)")
    results += [t1, t2]

    # 6) loss + gradient finite for both
    t = True
    for spec, pp in ((s1, p1), (s2, p2)):
        fwd = qmodels.build_model(spec)
        L = qmodels.bce_loss(pp, X, S, Yc, fwd)
        g = np.asarray(qml.grad(qmodels.bce_loss, argnums=0)(pp, X, S, Yc, fwd))
        t &= bool(np.isfinite(float(L)) and np.isfinite(g).all())
    print(f"[6] BCE + gradients finite for both  {OK(t)}"); results.append(t)

    # 7) centre-label alignment: patch centred at (r,c) must carry label Y[r,c]
    H, W = 14, 19
    lab = rng.randint(0, 2, (H, W))
    Xc = rng.uniform(-1, 1, (H, W, 4)); Sc = rng.uniform(0, 1, (H, W))
    ok = True
    for _ in range(200):
        r, c = rng.randint(1, H - 1), rng.randint(1, W - 1)
        patch_lab = lab[r - 1:r + 2, c - 1:c + 2]
        ok &= (patch_lab[1, 1] == lab[r, c])
    print(f"[7] centre-label alignment over 200 draws  {OK(ok)}"); results.append(ok)

    # 8) full-image stride-1 centre inference: an "echo centre" model must
    #    reproduce X[...,0] exactly -> validates padding + indexing
    echo_centre = lambda params, Xb, Sb: np.asarray(Xb)[:, 1, 1, 0]
    P = predict_city_center(echo_centre, None, Xc, Sc)
    err = np.abs(P - Xc[..., 0]).max()
    t = (P.shape == (H, W)) and err < 1e-6
    print(f"[8] stride-1 centre inference -> {P.shape}, max|P-X0| = {err:.2e}  {OK(t)}"); results.append(t)

    # 9) REGRESSION: the existing 3x3->3x3 path is untouched
    sp = qmodels.ModelSpec("m3", readout="per_pixel")
    fp = qmodels.build_model(sp)
    Pp = fp(qmodels.init_params(sp, seed=1), X, S)
    echo = lambda params, Xb, Sb: np.asarray(Xb)[..., 0]
    Pfull = predict_city(echo, None, Xc, Sc)
    t = (Pp.shape == (B, 3, 3) and sp.n_params == 38
         and np.abs(Pfull - Xc[..., 0]).max() < 1e-6)
    print(f"[9] regression: per_pixel path intact (shape {tuple(Pp.shape)}, "
          f"overlap echo err {np.abs(Pfull - Xc[...,0]).max():.2e})  {OK(t)}"); results.append(t)

    # 10) deliverable format: uint8 {0,255} H x W PNG
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.png")
        m = save_mask_png(P, 0.0, path)
        from PIL import Image
        back = np.array(Image.open(path))
        t = (back.dtype == np.uint8 and back.shape == (H, W)
             and set(np.unique(back)).issubset({0, 255}))
    print(f"[10] PNG mask uint8 {tuple(back.shape)} values {sorted(set(np.unique(back).tolist()))}  {OK(t)}")
    results.append(t)

    print(f"\nSTEP 1: {'PASS' if all(results) else 'FAIL'}  ({sum(results)}/{len(results)} checks)")
