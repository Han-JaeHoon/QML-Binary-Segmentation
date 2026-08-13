"""
Capacity diagnostic: why does M3 (L=1, shared calibration) plateau ~0.46 on a
fixed 32-patch batch? Locate the bottleneck by varying capacity on the SAME batch:
  A) L=1, shared (a,b)          -- 38 params   (baseline)
  B) L=2, shared (a,b)          -- 74 params   (more depth / re-uploading)
  C) L=1, per-pixel (a_i,b_i)   -- 54 params   (readout expressivity)
  D) L=3, shared (a,b)          -- 110 params  (even more depth)
If B/D drop far below A -> depth/capacity limited (expected). If only C drops ->
the shared readout is the bottleneck. If nothing reaches <0.2 -> architectural.
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

H_EDGES=[(0,1),(1,2),(3,4),(4,5),(6,7),(7,8)]; V_EDGES=[(0,3),(3,6),(1,4),(4,7),(2,5),(5,8)]
NN_EDGES=H_EDGES+V_EDGES; GAMMA=np.pi/2
dev=qml.device("default.qubit",wires=9)

@qml.qnode(dev, interface="autograd", diff_method="backprop")
def circuit(u,s,theta,L):
    for l in range(L):
        for q in range(9): qml.RY(np.pi*u[:,q,0],wires=q); qml.RZ(np.pi*u[:,q,1],wires=q)
        for i,j in NN_EDGES: qml.IsingZZ(GAMMA*s[:,i]*s[:,j],wires=[i,j])
        for q in range(9): qml.RY(theta[l,0,q,0],wires=q); qml.RX(theta[l,0,q,1],wires=q)
        for q in range(9): qml.RY(np.pi*u[:,q,2],wires=q); qml.RZ(np.pi*u[:,q,3],wires=q)
        for i,j in NN_EDGES: qml.IsingZZ(GAMMA*s[:,i]*s[:,j],wires=[i,j])
        for q in range(9): qml.RY(theta[l,1,q,0],wires=q); qml.RX(theta[l,1,q,1],wires=q)
    return [qml.expval(qml.PauliZ(q)) for q in range(9)]

def run(name,L,per_pixel,u,s,y,rng,steps=300):
    theta=pnp.array(0.1*rng.randn(L,2,9,2),requires_grad=True)
    if per_pixel: a=pnp.array(np.ones(9),requires_grad=True); b=pnp.array(np.zeros(9),requires_grad=True)
    else:         a=pnp.array(1.0,requires_grad=True);        b=pnp.array(0.0,requires_grad=True)
    def cost(theta,a,b):
        z=pnp.stack(circuit(u,s,theta,L)).T
        p=1/(1+pnp.exp(-(a*z+b))); eps=1e-7
        return -pnp.mean(y*pnp.log(p+eps)+(1-y)*pnp.log(1-p+eps))
    opt=qml.AdamOptimizer(0.1); L0=float(cost(theta,a,b))
    for _ in range(steps): (theta,a,b),_=opt.step_and_cost(cost,theta,a,b)
    nparams=theta.size+(np.size(a)+np.size(b))
    print(f"  {name:28} params {nparams:3d}  {L0:.3f} -> {float(cost(theta,a,b)):.3f}")

def main(data_dir):
    train,val=get_dev_split(); print("building fold ...")
    fold=build_fold(train,val,data_dir)
    T=fit_global_hard_threshold(train,fold.dcorr13,fold.labels,fold.valid)
    pools={c:build_center_pools(fold.dcorr13[c],fold.labels[c],fold.valid[c],T) for c in train}
    smp=SpatialPatchSampler(train,pools,fold,"pca",seed=42)
    Xp={c:transform_pca4(fold.dcorr13[c],fold.pca_tf) for c in train}
    Sm={c:pca_zz_strength(fold.dcorr13[c],fold.pca_tf) for c in train}
    rng=np.random.RandomState(0); U,S,Y=[],[],[]
    for _ in range(32):
        c,_,r,cc=smp.sample_index(rng)
        U.append(Xp[c][r-1:r+2,cc-1:cc+2,:].reshape(9,4)); S.append(Sm[c][r-1:r+2,cc-1:cc+2].reshape(9))
        Y.append(fold.labels[c][r-1:r+2,cc-1:cc+2].reshape(9))
    u=pnp.array(np.array(U),requires_grad=False); s=pnp.array(np.array(S),requires_grad=False)
    y=pnp.array(np.array(Y,float),requires_grad=False)
    print(f"fixed 32-patch batch, pos frac {float(y.mean()):.3f}, const-baseline BCE ~0.63\n")
    run("A) L=1 shared",1,False,u,s,y,np.random.RandomState(1))
    run("B) L=2 shared",2,False,u,s,y,np.random.RandomState(1))
    run("C) L=1 per-pixel calib",1,True,u,s,y,np.random.RandomState(1))
    run("D) L=3 shared",3,False,u,s,y,np.random.RandomState(1))

if __name__=="__main__":
    import argparse; ap=argparse.ArgumentParser(); ap.add_argument("--data_dir",required=True)
    main(ap.parse_args().data_dir)
