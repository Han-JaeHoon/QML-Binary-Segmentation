"""
Decisive control: separate DEPTH (re-uploading) from PARAMETER COUNT.

Same fixed 32-patch batch, three configs:
  L=1            , untied : 38 params   (M3 main)
  L=2 mixer TIED , 38 params            <- same params as L=1, deeper
  L=2 untied     , 74 params            (M4)

L1(38) -> L2_tied(38)   : pure re-uploading/depth effect at FIXED parameters
L2_tied(38) -> L2_untied(74) : effect of the extra parameters at FIXED depth
"""
import os, sys
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "data")); sys.path.insert(0, HERE)
from splits import get_dev_split
from preprocess import build_fold, transform_pca4, pca_zz_strength
from pools import build_center_pools, fit_global_hard_threshold
from sampler import SpatialPatchSampler

NN_EDGES = [(0,1),(1,2),(3,4),(4,5),(6,7),(7,8)] + [(0,3),(3,6),(1,4),(4,7),(2,5),(5,8)]
GAMMA = np.pi/2
dev = qml.device("default.qubit", wires=9)

@qml.qnode(dev, interface="autograd", diff_method="backprop")
def circuit(u, s, theta, L, tied):
    """theta: (n_blocks,2,9,2). tied -> n_blocks=1 reused for every cycle."""
    for l in range(L):
        t = theta[0] if tied else theta[l]
        for q in range(9): qml.RY(np.pi*u[:,q,0], wires=q); qml.RZ(np.pi*u[:,q,1], wires=q)
        for i,j in NN_EDGES: qml.IsingZZ(GAMMA*s[:,i]*s[:,j], wires=[i,j])
        for q in range(9): qml.RY(t[0,q,0], wires=q); qml.RX(t[0,q,1], wires=q)
        for q in range(9): qml.RY(np.pi*u[:,q,2], wires=q); qml.RZ(np.pi*u[:,q,3], wires=q)
        for i,j in NN_EDGES: qml.IsingZZ(GAMMA*s[:,i]*s[:,j], wires=[i,j])
        for q in range(9): qml.RY(t[1,q,0], wires=q); qml.RX(t[1,q,1], wires=q)
    return [qml.expval(qml.PauliZ(q)) for q in range(9)]

def run(name, L, tied, u, s, y, steps=300, seed=1):
    rng = np.random.RandomState(seed)
    nb = 1 if tied else L
    theta = pnp.array(0.1*rng.randn(nb,2,9,2), requires_grad=True)
    a = pnp.array(1.0, requires_grad=True); b = pnp.array(0.0, requires_grad=True)
    def cost(theta, a, b):
        z = pnp.stack(circuit(u, s, theta, L, tied)).T
        p = 1/(1+pnp.exp(-(a*z+b))); eps = 1e-7
        return -pnp.mean(y*pnp.log(p+eps) + (1-y)*pnp.log(1-p+eps))
    opt = qml.AdamOptimizer(0.1); L0 = float(cost(theta,a,b))
    for _ in range(steps): (theta,a,b),_ = opt.step_and_cost(cost, theta, a, b)
    print(f"  {name:26} params {theta.size+2:3d}   {L0:.3f} -> {float(cost(theta,a,b)):.3f}", flush=True)

def main(data_dir):
    train, val = get_dev_split(); print("building fold ...", flush=True)
    fold = build_fold(train, val, data_dir)
    T = fit_global_hard_threshold(train, fold.dcorr13, fold.labels, fold.valid)
    pools = {c: build_center_pools(fold.dcorr13[c], fold.labels[c], fold.valid[c], T) for c in train}
    smp = SpatialPatchSampler(train, pools, fold, "pca", seed=42)
    Xp = {c: transform_pca4(fold.dcorr13[c], fold.pca_tf) for c in train}
    Sm = {c: pca_zz_strength(fold.dcorr13[c], fold.pca_tf) for c in train}
    rng = np.random.RandomState(0); U,S,Y = [],[],[]
    for _ in range(32):
        c,_,r,cc = smp.sample_index(rng)
        U.append(Xp[c][r-1:r+2, cc-1:cc+2, :].reshape(9,4))
        S.append(Sm[c][r-1:r+2, cc-1:cc+2].reshape(9))
        Y.append(fold.labels[c][r-1:r+2, cc-1:cc+2].reshape(9))
    u = pnp.array(np.array(U), requires_grad=False); s = pnp.array(np.array(S), requires_grad=False)
    y = pnp.array(np.array(Y, float), requires_grad=False)
    print(f"fixed 32-patch batch, pos frac {float(y.mean()):.3f}, const-baseline BCE ~0.63\n", flush=True)
    run("L=1 (38, main M3)",      1, False, u, s, y)
    run("L=2 TIED (38)",          2, True,  u, s, y)
    run("L=2 untied (74, M4)",    2, False, u, s, y)

if __name__ == "__main__":
    import argparse; ap = argparse.ArgumentParser(); ap.add_argument("--data_dir", required=True)
    main(ap.parse_args().data_dir)
