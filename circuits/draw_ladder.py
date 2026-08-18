"""
Draw the model-ladder circuits with qml.Barrier separating every stage.

DRAWING ONLY. This rebuilds the ansatz in a separate qnode rather than touching
models/qml.py, so the production circuit and the tape-inspection checks in
train/accept_mring.py stay exactly as they are. Barriers are only_visual=True,
so they carry no operational meaning either way.

One depth block is  E1 | ENT | V1 | E2 | ENT | V2  where
    E1  RY(pi*x1) RZ(pi*x2)   angle encoding, spectral stage 1
    E2  RY(pi*x3) RZ(pi*x4)   angle encoding, spectral stage 2
    V   RY(theta) RX(theta)   trainable mixer (must not commute with Z)
    ENT the entangler, fixed and non-trainable:
          M1      none                       (separable)
          M_ring  CZ ring, 9 edges           (HEA-style control)
          M2      CZ on 12 spatial NN edges  (task-aligned grid)
          M3      IsingZZ(gamma*s_i*s_j)     (data-dependent, gamma=pi/2)

Qubit q <-> pixel q of the 3x3 patch:   0 1 2 / 3 4 5 / 6 7 8
Readout: <Z_q> on all nine, then p = sigma(a * mean_q<Z_q> + b).
"""
import os, sys
import numpy as np
import pennylane as qml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "models"))
import qml as qmodels

dev = qml.device("default.qubit", wires=9)
GAMMA = qmodels.GAMMA

ENTANGLERS = {
    "m1":    ("M1 — separable (no entangler)",            None),
    "mring": ("M_ring — CZ ring (HEA-style control)",     qmodels.RING9),
    "m2":    ("M2 — spatial CZ grid (12 NN edges)",       qmodels.NN_EDGES),
    "m3":    ("M3 — data-dependent IsingZZ (gamma=pi/2)", qmodels.NN_EDGES),
}


def _bar():
    qml.Barrier(wires=range(9), only_visual=True)


@qml.qnode(dev)
def circuit(x, s, theta, kind, L):
    edges = ENTANGLERS[kind][1]
    for l in range(L):
        for stage in (0, 1):
            f0, f1 = (0, 1) if stage == 0 else (2, 3)
            for q in range(9):                                   # E
                qml.RY(np.pi * x[q, f0], wires=q)
                qml.RZ(np.pi * x[q, f1], wires=q)
            _bar()
            if edges is not None:                                # ENT
                for (i, j) in edges:
                    if kind == "m3":
                        qml.IsingZZ(GAMMA * s[i] * s[j], wires=[i, j])
                    else:
                        qml.CZ(wires=[i, j])
                _bar()
            for q in range(9):                                   # V
                qml.RY(theta[l, stage, q, 0], wires=q)
                qml.RX(theta[l, stage, q, 1], wires=q)
            _bar()
    return [qml.expval(qml.PauliZ(q)) for q in range(9)]


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.RandomState(0)
    x = rng.uniform(-0.6, 0.6, (9, 4))
    s = rng.uniform(0.2, 0.9, 9)
    L = 1
    theta = rng.uniform(0, 2 * np.pi, (L, 2, 9, 2))

    outdir = os.path.join(ROOT, "results", "circuits")
    os.makedirs(outdir, exist_ok=True)

    for kind in ("m1", "m2", "m3", "mring"):
        title, edges = ENTANGLERS[kind]
        n_edges = 0 if edges is None else len(edges)
        fig, ax = qml.draw_mpl(circuit, decimals=2, fontsize=11)(x, s, theta, kind, L)
        fig.suptitle(f"{title}   |   L=1, 38 trainable params   |   "
                     f"{n_edges} entangling gates/stage", y=1.01, fontsize=13)
        p = os.path.join(outdir, f"circuit_{kind}.png")
        fig.savefig(p, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {p}")

        # text form, useful for quick diffing
        txt = qml.draw(circuit, max_length=200)(x, s, theta, kind, L)
        open(os.path.join(outdir, f"circuit_{kind}.txt"), "w").write(txt)
