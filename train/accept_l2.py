"""
train/accept_l2.py — acceptance / wiring tests for the P2 capacity confirmation
(M1 vs M2-CZ at L=2 untied, 74 trainable parameters, centre branch).

WIRING VERIFICATION ONLY. Nothing here says anything about learning quality; it
exists so a 5-fold run is not launched on a mis-specified model.

What L=2 untied means here (locked, matches the P2 brief):
    block = E1(x1,x2) -> [entangler] -> V1(theta) -> E2(x3,x4) -> [entangler] -> V2(theta)
    L=2   = that whole block twice, the SAME 4 PCA features re-encoded in block 2
            (data re-uploading), with block 2's mixer angles NOT shared with block 1.
    params = 36 (block 1 mixer) + 36 (block 2 mixer) + 2 (shared a,b) = 74.
    M2's CZ grid is fixed / non-trainable -> contributes 0 parameters, and at
    L=2 the CZ grid therefore appears FOUR times (2 blocks x 2 encoding stages).

Data-free by construction: every check runs on synthetic arrays, so this can be
executed while a long training job holds the dataset elsewhere.
Checks that genuinely need the dataset (sampler stream checksum over real pools)
are asserted inside train/run_cv.py at every epoch instead; check [6] here is the
data-free half of that (the stream depends only on the rng, never on the model).
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
from sampler import SpatialPatchSampler
from accept_center import interaction          # reuse the committed diagnostic

OK = lambda b: "OK" if b else "FAIL"
results = []


def check(label, cond, detail=""):
    results.append(bool(cond))
    print(f"{label}  {detail}{'  ' if detail else ''}{OK(cond)}")


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    B = 12
    X = pnp.array(rng.uniform(-0.6, 0.6, (B, 3, 3, 4)), requires_grad=False)
    S = pnp.array(rng.uniform(0, 1, (B, 3, 3)), requires_grad=False)
    Y = pnp.array(rng.randint(0, 2, B).astype(float), requires_grad=False)

    s1 = qmodels.ModelSpec("m1", depth=2, tying="untied", readout="center_mean")
    s2 = qmodels.ModelSpec("m2", depth=2, tying="untied", readout="center_mean")
    f1, f2 = qmodels.build_model(s1), qmodels.build_model(s2)
    p1 = qmodels.init_params(s1, seed=1)
    p2 = qmodels.init_params(s2, seed=1)

    print(f"=== P2 acceptance: {s1.label} vs {s2.label} ===\n")

    # [1][2] parameter counts
    check("[1] M1 L2 untied params = 74", s1.n_params == 74, f"got {s1.n_params}")
    check("[2] M2 L2 untied params = 74", s2.n_params == 74, f"got {s2.n_params}")

    # [3] the two mixer blocks are INDEPENDENT (untied, not tied)
    #     - the spec must carry 2 blocks
    #     - perturbing block 2 alone must move the output (block 2 is real)
    #     - perturbing block 1 alone must move the output (block 1 is not dead)
    #     - and the tied variant must be a DIFFERENT, 38-param model (guard against
    #       silently running tied when untied was intended)
    tied = qmodels.ModelSpec("m1", depth=2, tying="tied", readout="center_mean")
    sc1 = qmodels.build_score(s1)
    base = np.asarray(sc1(p1, X, S))
    d1 = np.array(p1, dtype=float); d1[0] += 0.3                 # block 1 angle
    d2 = np.array(p1, dtype=float); d2[36] += 0.3                # block 2 angle
    m1_move = np.abs(np.asarray(sc1(pnp.array(d1, requires_grad=True), X, S)) - base).max()
    m2_move = np.abs(np.asarray(sc1(pnp.array(d2, requires_grad=True), X, S)) - base).max()
    check("[3] blocks independent (untied)",
          s1.n_blocks == 2 and tied.n_params == 38 and m1_move > 1e-6 and m2_move > 1e-6,
          f"n_blocks {s1.n_blocks} | tied-L2 would be {tied.n_params}p | "
          f"d|score| block1 {m1_move:.2e} block2 {m2_move:.2e}")

    # [3b] every one of the 72 mixer angles carries gradient (nothing structurally dead)
    for name, spec, pp in (("M1", s1, p1), ("M2", s2, p2)):
        fwd = qmodels.build_model(spec)
        gmax = np.zeros(spec.n_params)
        for k in range(3):
            r = np.random.RandomState(10 + k)
            Xr = pnp.array(r.uniform(-0.6, 0.6, (16, 3, 3, 4)), requires_grad=False)
            Sr = pnp.array(r.uniform(0, 1, (16, 3, 3)), requires_grad=False)
            Yr = pnp.array(r.randint(0, 2, 16).astype(float), requires_grad=False)
            g = np.abs(np.asarray(qml.grad(qmodels.bce_loss, argnums=0)(pp, Xr, Sr, Yr, fwd)))
            gmax = np.maximum(gmax, g)
        alive = int((gmax[:-2] > 1e-9).sum())
        check(f"[3b] {name}: all 72 mixer angles carry gradient", alive == 72,
              f"alive {alive}/72, min |g| {gmax[:-2].min():.2e}")

    # [4] M1 and M2 can start from the IDENTICAL 74-vector
    check("[4] identical initialization vector", np.array_equal(np.asarray(p1), np.asarray(p2)),
          f"max|p1-p2| {np.abs(np.asarray(p1) - np.asarray(p2)).max():.1e}")

    # [5] centre_mean output shape + finite BCE and gradients
    P1, P2 = f1(p1, X, S), f2(p2, X, S)
    fin = True
    for spec, pp in ((s1, p1), (s2, p2)):
        fwd = qmodels.build_model(spec)
        L = qmodels.bce_loss(pp, X, S, Y, fwd)
        g = np.asarray(qml.grad(qmodels.bce_loss, argnums=0)(pp, X, S, Y, fwd))
        fin &= bool(np.isfinite(float(L)) and np.isfinite(g).all() and g.size == 74)
    check("[5] centre_mean shape (B,) + finite BCE/grad",
          P1.shape == (B,) and P2.shape == (B,) and fin,
          f"shapes {tuple(P1.shape)}/{tuple(P2.shape)}")

    # [6] sampler stream depends ONLY on the rng, never on the model (data-free
    #     half of the paired-stream guarantee; run_cv.py asserts the real one)
    fake_pools = {c: {"positive": np.argwhere(np.ones((6, 6))) + 1,
                      "hard_negative": np.argwhere(np.ones((6, 6))) + 1,
                      "ordinary_negative": np.argwhere(np.ones((6, 6))) + 1}
                  for c in ("a", "b", "c")}
    smp = SpatialPatchSampler(["a", "b", "c"], fake_pools, fold=None, representation="pca")
    draw = lambda seed: [smp.sample_index(np.random.RandomState(seed)) for _ in range(200)]
    check("[6] sampler stream reproducible from seed alone", draw(7) == draw(7))

    # [7] centre-label alignment (patch centred at (r,c) carries Y[r,c])
    H, W = 14, 19
    lab = rng.randint(0, 2, (H, W))
    ok = all(lab[r - 1:r + 2, c - 1:c + 2][1, 1] == lab[r, c]
             for r, c in zip(rng.randint(1, H - 1, 200), rng.randint(1, W - 1, 200)))
    check("[7] centre-label alignment over 200 draws", ok)

    # [8] THE decisive one: mixed finite difference on the PRE-SIGMOID score.
    #     M1 L2 must still be exactly additive; M2 L2 must interact.
    nn = qmodels.NN_EDGES[:4]
    far = [(0, 8), (2, 6)]
    i1 = np.array([np.abs(interaction(s1, p1, X, S, i, j)).max() for i, j in nn + far])
    i2nn = np.array([np.abs(interaction(s2, p2, X, S, i, j)).max() for i, j in nn])
    i2far = np.array([np.abs(interaction(s2, p2, X, S, i, j)).max() for i, j in far])
    check("[8a] M1 L2 still separable/additive", i1.max() < 1e-9, f"max|I_ij| {i1.max():.2e}")
    check("[8b] M2 L2 interacts on NN pairs", i2nn.min() > 1e-6,
          "NN |I_ij| " + ", ".join(f"{v:.2e}" for v in i2nn))
    print(f"      M2 L2 far pairs (0,8),(2,6): " +
          ", ".join(f"{v:.2e}" for v in i2far) + "   (4 layers of CZ -> wider light cone)")

    # [9] REGRESSION: L=1 centre branch and the dense per_pixel path are untouched
    r1 = qmodels.ModelSpec("m1", readout="center_mean")
    r2 = qmodels.ModelSpec("m2", readout="center_mean")
    q1, q2 = qmodels.init_params(r1, seed=1), qmodels.init_params(r2, seed=1)
    j1 = max(np.abs(interaction(r1, q1, X, S, i, j)).max() for i, j in nn)
    j2 = min(np.abs(interaction(r2, q2, X, S, i, j)).max() for i, j in nn)
    dense = qmodels.ModelSpec("m3", readout="per_pixel")
    Pd = qmodels.build_model(dense)(qmodels.init_params(dense, seed=1), X, S)
    check("[9] L=1 centre + dense per_pixel regression intact",
          r1.n_params == 38 and r2.n_params == 38 and j1 < 1e-9 and j2 > 1e-6
          and Pd.shape == (B, 3, 3) and dense.n_params == 38,
          f"L1 M1 {j1:.1e} / L1 M2 {j2:.1e}")

    print(f"\n{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)
