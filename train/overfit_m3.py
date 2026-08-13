"""
Fixed-batch overfit test for M3 (strong capacity / gradient-path unit test).

Show ONE fixed batch of 32 patches repeatedly for N steps; the loss should drop
clearly (target < 0.2). If it does, the circuit has the capacity and a working
gradient path to fit the target mapping — a stronger check than the stochastic
smoke test. Reuses the M3 circuit from smoke_m3.py.
"""
import os, sys
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "data"))
sys.path.insert(0, HERE)
from smoke_m3 import forward, bce                       # M3 circuit + loss
from splits import get_dev_split
from preprocess import build_fold, transform_pca4, pca_zz_strength
from pools import build_center_pools, fit_global_hard_threshold
from sampler import SpatialPatchSampler

def main(data_dir):
    train, val = get_dev_split()
    print("building fold ...")
    fold = build_fold(train, val, data_dir)
    T = fit_global_hard_threshold(train, fold.dcorr13, fold.labels, fold.valid)
    pools = {c: build_center_pools(fold.dcorr13[c], fold.labels[c], fold.valid[c], T) for c in train}
    smp = SpatialPatchSampler(train, pools, fold, "pca", seed=42)
    Xp = {c: transform_pca4(fold.dcorr13[c], fold.pca_tf) for c in train}
    Sm = {c: pca_zz_strength(fold.dcorr13[c], fold.pca_tf) for c in train}

    rng = np.random.RandomState(0)
    U, S, Y = [], [], []
    for _ in range(32):                                  # ONE fixed batch
        c, _, r, cc = smp.sample_index(rng)
        U.append(Xp[c][r-1:r+2, cc-1:cc+2, :].reshape(9, 4))
        S.append(Sm[c][r-1:r+2, cc-1:cc+2].reshape(9))
        Y.append(fold.labels[c][r-1:r+2, cc-1:cc+2].reshape(9))
    u = pnp.array(np.array(U), requires_grad=False)
    s = pnp.array(np.array(S), requires_grad=False)
    y = pnp.array(np.array(Y, float), requires_grad=False)
    print(f"fixed batch: {u.shape[0]} patches, positive-pixel frac {float(y.mean()):.3f}")

    theta = pnp.array(0.1 * rng.randn(2, 9, 2), requires_grad=True)
    a = pnp.array(1.0, requires_grad=True); b = pnp.array(0.0, requires_grad=True)
    opt = qml.AdamOptimizer(0.1)
    L0 = float(bce(theta, a, b, u, s, y))
    print(f"step   0  loss {L0:.4f}")
    for step in range(1, 201):
        (theta, a, b, u, s, y), L = opt.step_and_cost(bce, theta, a, b, u, s, y)
        if step % 25 == 0:
            print(f"step {step:3d}  loss {float(L):.4f}")
    Lf = float(bce(theta, a, b, u, s, y))
    print(f"\nfinal loss {Lf:.4f}  (start {L0:.4f})  -> {'PASS (<0.2)' if Lf < 0.2 else 'did not reach 0.2'}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--data_dir", required=True)
    main(ap.parse_args().data_dir)
