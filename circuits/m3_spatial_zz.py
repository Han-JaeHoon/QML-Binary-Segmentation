"""
M3 -- 9-to-9 Spatial ZZ Re-uploading VQC (structure / diagram only).

3x3 patch of median-corrected |dB| for 4 bands {B04,B05,B12,B08}
-> 9 qubits (1 qubit <-> 1 spatial pixel)
-> per-cycle: [E1 -> ZZ1 -> V1 -> E2 -> ZZ2 -> V2]
-> measure <Z_i> on all 9 qubits -> 3x3 probability map.

Qubit grid / index layout:
    0 1 2
    3 4 5
    6 7 8

Nearest-neighbour edges (horizontal + vertical only, 12 total):
    (0,1)(1,2)(3,4)(4,5)(6,7)(7,8)   horizontal
    (0,3)(3,6)(1,4)(4,7)(2,5)(5,8)   vertical
"""
import numpy as np
import pennylane as qml

# ---- grid topology -------------------------------------------------------
H_EDGES = [(0,1),(1,2),(3,4),(4,5),(6,7),(7,8)]
V_EDGES = [(0,3),(3,6),(1,4),(4,7),(2,5),(5,8)]
NN_EDGES = H_EDGES + V_EDGES              # 12 nearest-neighbour couplings
BANDS = ["B04","B05","B12","B08"]         # feature order per pixel
GAMMA = np.pi / 2                         # ZZ feature-map coupling (FIXED, not trainable)

dev = qml.device("default.qubit", wires=9)

# --- FIXED input scaling (frozen on TRAIN cities, applied unchanged to val/test) -----
# feature into the circuit:  x_b = clip(|dB_corr_b| / c_b, 0, 1),  c_b = P99 on TRAIN.
# Do NOT re-fit c_b per test image -- that would erase the change magnitude the EDA
# found to be the key signal. c_b is a placeholder here; set from results/norm at train.
C_SCALE = {"B04": 0.35, "B05": 0.35, "B12": 0.30, "B08": 0.30}   # example P99(|dB_corr|)

def rescale(abs_dB_corr, band):
    """Map |dB_corr| -> [0,1] with a frozen per-band scale (train-derived)."""
    return np.clip(abs_dB_corr / C_SCALE[band], 0.0, 1.0)

def strength(a, b):
    """Per-pixel change strength in [0,1]: s = sqrt((a^2 + b^2)/2), a,b in [0,1]."""
    return np.sqrt((a**2 + b**2) / 2.0)

def _encode(x_pair):
    """Angle stage: RY(pi*a)RZ(pi*b) on each qubit for one (band_a, band_b) pair."""
    a, b = x_pair                          # each shape (9,)
    for q in range(9):
        qml.RY(np.pi * a[q], wires=q)
        qml.RZ(np.pi * b[q], wires=q)

def _zz(strength):
    """Data-dependent nearest-neighbour ZZ: angle = gamma * s_i * s_j (fixed gamma)."""
    for (i, j) in NN_EDGES:
        qml.IsingZZ(GAMMA * strength[i] * strength[j], wires=[i, j])

def _mixer(theta):
    """Trainable non-commuting mixer: RY,RX per qubit. theta shape (9,2)."""
    for q in range(9):
        qml.RY(theta[q, 0], wires=q)
        qml.RX(theta[q, 1], wires=q)

@qml.qnode(dev)
def circuit(x, params, L=1):
    """
    x      : (9,4) median-corrected |dB| features, rescaled to [0,1]
             columns = [B04, B05, B12, B08]
    params : (L,2,9,2) trainable mixer angles  (L cycles, 2 mixers, 9 qubits, {RY,RX})
    """
    s1 = strength(x[:,0], x[:,1])           # change strength in [0,1], stage 1 (B04,B05)
    s2 = strength(x[:,2], x[:,3])           # change strength in [0,1], stage 2 (B12,B08)
    bar = lambda: qml.Barrier(wires=range(9), only_visual=True)
    for l in range(L):
        _encode((x[:,0], x[:,1]))           # E1  : B04,B05
        bar()
        _zz(s1)                             # ZZ1 : spatial coupling on stage-1 strength
        bar()
        _mixer(params[l,0])                 # V1  : trainable mixer
        bar()
        _encode((x[:,2], x[:,3]))           # E2  : B12,B08
        bar()
        _zz(s2)                             # ZZ2
        bar()
        _mixer(params[l,1])                 # V2
        bar()
    return [qml.expval(qml.PauliZ(q)) for q in range(9)]   # 3x3 map

def param_count(L=1, share_mixer=False):
    """QML trainable params. Mixer + shared calibration (a,b)."""
    per_mixer = 2 if share_mixer else 9*2   # shared: 2 angles; independent: 18
    mixers = L*2*per_mixer
    calib = 2                               # p=sigma(a*z+b), shared across 9 pixels
    return mixers + calib

if __name__ == "__main__":
    rng = np.random.RandomState(0)
    x = rng.rand(9,4)                       # dummy patch in [0,1]
    L = 1
    params = rng.uniform(0, 2*np.pi, size=(L,2,9,2))

    print("="*70)
    print("M3: 9-to-9 Spatial ZZ Re-uploading VQC  (L=1)")
    print("="*70)
    print(qml.draw(circuit, max_length=140)(x, params, L))
    print("\nOutput:", np.round(circuit(x, params, L), 3))
    print("\nFROZEN config: gamma=pi/2 | s=sqrt((a^2+b^2)/2) in [0,1] | "
          "x=clip(|dB_corr|/c_b,0,1) | mixer=independent | L=1")
    print("\nParameter budget (QML trainable):")
    for L_ in (1,2):
        for share in (False, True):
            tag = "shared-mixer" if share else "independent-mixer"
            main = "  <-- MAIN" if (L_==1 and not share) else ""
            print(f"  L={L_}  {tag:18}: {param_count(L_,share)} params{main}")
    print("\nParameter-matched classical baseline:")
    print("  3x3 same-pad conv, 4->1 channels: 4*3*3 + 1 = 37 params  (<= 38 QML)")
    print("  same input/output: 3x3x4 -> 3x3.  MLP is a secondary baseline.")

    # matplotlib diagram
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = qml.draw_mpl(circuit, decimals=2, fontsize=9)(x, params, L)
    fig.suptitle("M3: 9-to-9 Spatial ZZ Re-uploading VQC (L=1)", y=1.02)
    fig.savefig("results/m3_circuit.png", dpi=140, bbox_inches="tight")
    print("\nsaved results/m3_circuit.png")
