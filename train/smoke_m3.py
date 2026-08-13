"""
QML training smoke test — M3 (9-to-9 Spatial ZZ VQC), PCA-4 features, dev 11/3.

Checks (per the plan), NOT performance:
  1. circuit outputs finite for a real sampler batch
  2. BCE computes over the 9-pixel targets
  3. gradients actually flow into all 38 trainable params
  4. loss decreases over a few dozen steps
  5. wall-clock per patch (fwd, and fwd+grad) -> training budget

M3 encoding uses PCA-4 (signed, θ=π·u); ZZ strength s = pca_zz_strength (‖z‖/c_norm),
same s for both ZZ stages (PCA design). Mirrors circuits/m3_spatial_zz.py structure.
Uses default.qubit + backprop for fast simulation (parameter-shift cost noted below).
"""
import os, sys, time
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "data"))
from splits import get_dev_split
from preprocess import build_fold, transform_pca4, pca_zz_strength
from pools import build_center_pools, fit_global_hard_threshold
from sampler import SpatialPatchSampler

# --- M3 circuit (PCA branch), differentiable in theta -------------------------
H_EDGES = [(0,1),(1,2),(3,4),(4,5),(6,7),(7,8)]
V_EDGES = [(0,3),(3,6),(1,4),(4,7),(2,5),(5,8)]
NN_EDGES = H_EDGES + V_EDGES
GAMMA = np.pi / 2
dev = qml.device("default.qubit", wires=9)

@qml.qnode(dev, interface="autograd", diff_method="backprop")
def m3(u, s, theta):
    """u:(B,9,4) angle feats in [-1,1]; s:(B,9) ZZ strength in [0,1];
    theta:(2,9,2) trainable mixer. Batched via broadcasting."""
    for q in range(9):                                   # E1
        qml.RY(np.pi * u[:, q, 0], wires=q); qml.RZ(np.pi * u[:, q, 1], wires=q)
    for (i, j) in NN_EDGES:                               # ZZ1
        qml.IsingZZ(GAMMA * s[:, i] * s[:, j], wires=[i, j])
    for q in range(9):                                   # V1
        qml.RY(theta[0, q, 0], wires=q); qml.RX(theta[0, q, 1], wires=q)
    for q in range(9):                                   # E2
        qml.RY(np.pi * u[:, q, 2], wires=q); qml.RZ(np.pi * u[:, q, 3], wires=q)
    for (i, j) in NN_EDGES:                               # ZZ2
        qml.IsingZZ(GAMMA * s[:, i] * s[:, j], wires=[i, j])
    for q in range(9):                                   # V2
        qml.RY(theta[1, q, 0], wires=q); qml.RX(theta[1, q, 1], wires=q)
    return [qml.expval(qml.PauliZ(q)) for q in range(9)]

def forward(u, s, theta, a, b):
    z = pnp.stack(m3(u, s, theta)).T          # (B,9) in [-1,1]
    return 1.0 / (1.0 + pnp.exp(-(a * z + b)))  # (B,9) probs

def bce(theta, a, b, u, s, y):
    p = forward(u, s, theta, a, b)
    eps = 1e-7
    return -pnp.mean(y * pnp.log(p + eps) + (1 - y) * pnp.log(1 - p + eps))

# --- data ---------------------------------------------------------------------
def main(data_dir):
    train, val = get_dev_split()
    print(f"dev split: train={len(train)}, val={val}\nbuilding fold ...")
    fold = build_fold(train, val, data_dir)
    T_global = fit_global_hard_threshold(train, fold.dcorr13, fold.labels, fold.valid)
    pools = {c: build_center_pools(fold.dcorr13[c], fold.labels[c], fold.valid[c], T_global)
             for c in train}
    smp = SpatialPatchSampler(train, pools, fold, representation="pca", seed=42)
    Xp = {c: transform_pca4(fold.dcorr13[c], fold.pca_tf) for c in train}
    Sm = {c: pca_zz_strength(fold.dcorr13[c], fold.pca_tf) for c in train}

    rng = np.random.RandomState(0)
    def batch(B):
        U, S, Y = [], [], []
        for _ in range(B):
            c, _, r, cc = smp.sample_index(rng)
            U.append(Xp[c][r-1:r+2, cc-1:cc+2, :].reshape(9, 4))
            S.append(Sm[c][r-1:r+2, cc-1:cc+2].reshape(9))
            Y.append(fold.labels[c][r-1:r+2, cc-1:cc+2].reshape(9))
        return (pnp.array(np.array(U), requires_grad=False),
                pnp.array(np.array(S), requires_grad=False),
                pnp.array(np.array(Y, dtype=float), requires_grad=False))

    # trainable params: theta(2,9,2)=36 + a + b = 38
    theta = pnp.array(0.1 * rng.randn(2, 9, 2), requires_grad=True)
    a = pnp.array(1.0, requires_grad=True); b = pnp.array(0.0, requires_grad=True)
    n_params = theta.size + 2
    print(f"trainable params: {n_params}")

    # ---- check 1-3: finite outputs, BCE, gradients ----
    u, s, y = batch(16)
    p = forward(u, s, theta, a, b)
    print(f"\n[1] outputs finite? {bool(np.isfinite(np.asarray(p)).all())}  "
          f"p range [{float(p.min()):.3f},{float(p.max()):.3f}]  shape {p.shape}")
    L0 = bce(theta, a, b, u, s, y)
    print(f"[2] BCE computes: {float(L0):.4f}")
    gt, ga, gb = qml.grad(bce, argnums=[0, 1, 2])(theta, a, b, u, s, y)
    gt = np.asarray(gt)
    print(f"[3] grad finite? {np.isfinite(gt).all() and np.isfinite(float(ga)) and np.isfinite(float(gb))}  "
          f"|grad theta| mean {np.abs(gt).mean():.4e} max {np.abs(gt).max():.4e}  "
          f"nonzero {int((np.abs(gt)>1e-9).sum())}/36  ga={float(ga):.3e} gb={float(gb):.3e}")

    # ---- check 4: loss decreases over steps ----
    opt = qml.AdamOptimizer(0.05)
    print("\n[4] short training (batch=32):")
    for step in range(40):
        u, s, y = batch(32)
        (theta, a, b, u, s, y), L = opt.step_and_cost(bce, theta, a, b, u, s, y)
        if step % 8 == 0 or step == 39:
            print(f"   step {step:3d}  loss {float(L):.4f}")

    # ---- check 5: timing ----
    print("\n[5] wall-clock (default.qubit + backprop):")
    for B in (100, 500):
        u, s, y = batch(B)
        t = time.time(); _ = forward(u, s, theta, a, b); tf = time.time() - t
        t = time.time(); _ = qml.grad(bce, argnums=[0,1,2])(theta, a, b, u, s, y); tg = time.time() - t
        print(f"   {B:4d} patches:  forward {tf:.3f}s ({1000*tf/B:.2f} ms/patch)  "
              f"fwd+grad {tg:.3f}s ({1000*tg/B:.2f} ms/patch)")
    print("\nNote: backprop timing is the SIMULATOR cost. On real-hardware-style "
          "parameter-shift, gradient cost scales with #params (~2*38 evals).")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--data_dir", required=True)
    main(ap.parse_args().data_dir)
