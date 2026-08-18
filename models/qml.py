"""
models/qml.py — unified QML model builders for the M0–M3 ladder.

Contract (locked):
    forward(params, X, S) -> P
      X : (B,3,3,4)  angle-encoding features   (physical: [0,1] | pca: [-1,1])
      S : (B,3,3)    per-pixel change strength in [0,1], OR
          (B,3,3,2)  per-stage strength (physical branch: s1 from B04/B05,
                                         s2 from B12/B08)
      P : (B,3,3)    per-pixel change probability

S is passed explicitly because it is NOT recoverable from X in the PCA branch:
X = clip(z/c_pc, -1, 1) is clipped per component, while S = clip(||z||_2/c_norm, 0, 1)
uses the unclipped score norm. Physical branch could derive it, but both use the
same interface so trainer/inference stay model-agnostic.

Models (all share: angle encoding in two spectral stages, RY/RX mixer, readout):
  M0  4 qubits, ONE pixel at a time (applied to all 9 independently) — no spatial
      context. Ring CNOT over the 4 feature qubits. 13 params.
  M1  9 qubits (1 qubit <-> 1 pixel), NO entangler — 9 independent position-wise
      VQCs + shared scalar calibration. 38 params (L=1).
  M2  M1 + fixed nearest-neighbour CNOT entangler (data-independent).
      Order pinned: all 6 horizontal edges, then all 6 vertical; control = lower
      index (CNOTs do not commute, so this must be fixed for a fair M2<->M3).
  M3  M1 + data-dependent nearest-neighbour IsingZZ(gamma * s_i * s_j), gamma=pi/2
      fixed (feature map, not trainable). ZZ gates commute pairwise -> order-free.

Depth/tying (M1/M2/M3): `depth=L` repeats the whole cycle; `tying="tied"` reuses
one mixer block across cycles (params stay 38), `"untied"` gives each cycle its
own block (L*36+2). M4 == M3 with depth=2.
"""
from dataclasses import dataclass
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

# --- grid topology (qubit index <-> spatial pixel) ---------------------------
#   0 1 2
#   3 4 5
#   6 7 8
H_EDGES = [(0,1),(1,2),(3,4),(4,5),(6,7),(7,8)]
V_EDGES = [(0,3),(3,6),(1,4),(4,7),(2,5),(5,8)]
NN_EDGES = H_EDGES + V_EDGES          # 12 nearest-neighbour couplings
RING4 = [(0,1),(1,2),(2,3),(3,0)]     # M0 spectral ring
GAMMA = np.pi / 2                     # fixed ZZ feature-map coupling

# --- CZ ring entangler (M_ring, HEA-style control) --------------------------
# A plain 9-qubit CZ ring in index order, applied after each encoding stage —
# the standard "hardware-efficient ansatz" entangler. It sits between the
# separable model and the task-inspired spatial grid as an architecture-
# sensitivity control.
#
# INTERPRETATION GUARD — this ring is NOT a geometry-agnostic or "pure topology"
# control on a 3x3 raster layout:
#   * 6 of its 9 edges coincide with real horizontal spatial neighbours
#     (0,1)(1,2)(3,4)(4,5)(6,7)(7,8); only (2,3),(5,6),(8,0) are non-spatial.
#   * it uses 9 CZ per stage vs the grid's 12, so gate count is not matched.
# Therefore an M_ring vs M2 difference must NOT be read as a topology-only
# effect. Call it a ring-topology entangling baseline.
RING9 = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,0)]

dev9 = qml.device("default.qubit", wires=9)
dev4 = qml.device("default.qubit", wires=4)


@dataclass
class ModelSpec:
    kind: str = "m3"            # "m0"|"m1"|"m2"(grid CZ)|"mring"(CZ ring)|"m2cnot"|"m3"
    depth: int = 1              # L (ignored for m0)
    tying: str = "untied"       # "tied" | "untied" (ignored for m0, L=1)
    readout: str = "per_pixel"  # "per_pixel" -> P (B,3,3) | "center_mean" -> P (B,)

    def __post_init__(self):
        assert self.kind in ("m0", "m1", "m2", "mring", "m2cnot", "m3")
        assert self.tying in ("tied", "untied")
        assert self.readout in ("per_pixel", "center_mean")
        assert self.depth >= 1

    @property
    def n_blocks(self):
        return 1 if self.tying == "tied" else self.depth

    @property
    def n_params(self):
        if self.kind == "m0":
            return 4 * 2 + 4 + 1                 # mixer 8 + readout w(4)+b(1) = 13
        return self.n_blocks * 2 * 9 * 2 + 2     # mixer blocks + shared (a,b)

    @property
    def label(self):
        if self.kind == "m0":
            return "M0 pixel-VQC"
        t = "" if self.depth == 1 else f" L={self.depth} {self.tying}"
        r = "" if self.readout == "per_pixel" else " [center]"
        return f"{self.kind.upper()}{t}{r}"


# --- entanglers -------------------------------------------------------------
def _entangle(kind, s):
    if kind == "m1":
        return                                   # no spatial mixing (separable)
    if kind == "m2":
        # Fixed CZ on the 12 NN edges. CZ is diagonal, so the gates mutually
        # commute -> the layer is ORDER-FREE and has no control/target choice.
        # This removes the arbitrary ordering CNOT would force, and matches the
        # structural form of M3's IsingZZ (also diagonal) for a fair M2<->M3.
        for (i, j) in NN_EDGES: qml.CZ(wires=[i, j])
        return
    if kind == "mring":
        # HEA-style CZ ring in index order; diagonal & commuting -> order-free.
        # See the RING9 interpretation guard: 6/9 edges coincide with horizontal
        # spatial neighbours, and 9 != 12 gates, so this is not a topology-only
        # control.
        for (i, j) in RING9: qml.CZ(wires=[i, j])
        return
    if kind == "m2cnot":                         # legacy: order matters -> pinned
        for (i, j) in H_EDGES: qml.CNOT(wires=[i, j])
        for (i, j) in V_EDGES: qml.CNOT(wires=[i, j])
        return
    for (i, j) in NN_EDGES:                      # m3: data-dependent, commuting
        qml.IsingZZ(GAMMA * s[:, i] * s[:, j], wires=[i, j])


@qml.qnode(dev9, interface="autograd", diff_method="backprop")
def _qnode9(u, s1, s2, theta, kind, L, tied):
    """u:(B,9,4) s*:(B,9) theta:(n_blocks,2,9,2) -> 9 expvals."""
    for l in range(L):
        t = theta[0] if tied else theta[l]
        for q in range(9):                                   # E1
            qml.RY(np.pi * u[:, q, 0], wires=q); qml.RZ(np.pi * u[:, q, 1], wires=q)
        _entangle(kind, s1)                                  # ZZ1 / CNOT / none
        for q in range(9):                                   # V1
            qml.RY(t[0, q, 0], wires=q); qml.RX(t[0, q, 1], wires=q)
        for q in range(9):                                   # E2
            qml.RY(np.pi * u[:, q, 2], wires=q); qml.RZ(np.pi * u[:, q, 3], wires=q)
        _entangle(kind, s2)                                  # ZZ2 / CNOT / none
        for q in range(9):                                   # V2
            qml.RY(t[1, q, 0], wires=q); qml.RX(t[1, q, 1], wires=q)
    return [qml.expval(qml.PauliZ(q)) for q in range(9)]


@qml.qnode(dev4, interface="autograd", diff_method="backprop")
def _qnode4(x, theta):
    """M0: one pixel, 4 feature qubits. x:(N,4) theta:(4,2) -> 4 expvals."""
    for q in range(4):
        qml.RY(np.pi * x[:, q], wires=q)
    for (i, j) in RING4:
        qml.CNOT(wires=[i, j])
    for q in range(4):
        qml.RY(theta[q, 0], wires=q); qml.RX(theta[q, 1], wires=q)
    return [qml.expval(qml.PauliZ(q)) for q in range(4)]


# --- params -----------------------------------------------------------------
def init_params(spec, seed=0, scale=0.1):
    """Flat trainable vector (so optimizers/grad stay trivial)."""
    rng = np.random.RandomState(seed)
    if spec.kind == "m0":
        v = np.concatenate([scale * rng.randn(8), np.ones(4), np.zeros(1)])
    else:
        v = np.concatenate([scale * rng.randn(spec.n_blocks * 36), [1.0], [0.0]])
    assert v.size == spec.n_params
    return pnp.array(v, requires_grad=True)


def _unpack(params, spec):
    if spec.kind == "m0":
        return params[:8].reshape(4, 2), params[8:12], params[12]
    theta = params[:-2].reshape(spec.n_blocks, 2, 9, 2)
    return theta, params[-2], params[-1]


def _split_strength(S):
    """(B,3,3) -> same s for both stages; (B,3,3,2) -> per-stage."""
    B = S.shape[0]
    if S.ndim == 4:
        return S[..., 0].reshape(B, 9), S[..., 1].reshape(B, 9)
    s = S.reshape(B, 9)
    return s, s


# --- public API -------------------------------------------------------------
def build_score(spec):
    """Returns score(params, X, S) -> PRE-SIGMOID logit.
    per_pixel   -> (B,3,3)
    center_mean -> (B,)   logit = a * mean_q<Z_q> + b

    Use this (not the probability) for interaction diagnostics: the sigmoid is
    itself nonlinear and would manufacture apparent mixed effects even for a
    strictly additive model.
    """
    def score(params, X, S):
        B = X.shape[0]
        if spec.kind == "m0":
            theta, w, b = _unpack(params, spec)
            x = X.reshape(B * 9, 4)                       # each pixel independently
            z = pnp.stack(_qnode4(x, theta)).T            # (B*9,4)
            return (pnp.sum(z * w, axis=1) + b).reshape(B, 3, 3)
        theta, a, b = _unpack(params, spec)
        u = X.reshape(B, 9, 4)
        s1, s2 = _split_strength(S)
        z = pnp.stack(_qnode9(u, s1, s2, theta, spec.kind,
                              spec.depth, spec.tying == "tied")).T   # (B,9)
        if spec.readout == "center_mean":
            # parameter-FREE aggregation over all 9 qubits, so every mixer
            # parameter reaches the output even when there is no entangler.
            return a * pnp.mean(z, axis=1) + b                        # (B,)
        return (a * z + b).reshape(B, 3, 3)
    return score


def build_model(spec):
    """Returns forward(params, X, S) -> P.
    per_pixel -> (B,3,3);  center_mean -> (B,)"""
    score = build_score(spec)
    def forward(params, X, S):
        return 1.0 / (1.0 + pnp.exp(-score(params, X, S)))
    return forward


def bce_loss(params, X, S, Y, forward, w_pos=1.0):
    """Mean BCE over all 9 pixels. w_pos=1 -> plain BCE (the agreed default)."""
    p = forward(params, X, S)
    eps = 1e-7
    return -pnp.mean(w_pos * Y * pnp.log(p + eps) + (1 - Y) * pnp.log(1 - p + eps))


# --- smoke test -------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.RandomState(0)
    X = pnp.array(rng.uniform(-1, 1, (8, 3, 3, 4)), requires_grad=False)
    S = pnp.array(rng.uniform(0, 1, (8, 3, 3)), requires_grad=False)
    S2 = pnp.array(rng.uniform(0, 1, (8, 3, 3, 2)), requires_grad=False)
    Y = pnp.array(rng.randint(0, 2, (8, 3, 3)).astype(float), requires_grad=False)

    specs = [ModelSpec("m0"), ModelSpec("m1"), ModelSpec("m2"), ModelSpec("m3"),
             ModelSpec("m3", depth=2, tying="tied"),
             ModelSpec("m3", depth=2, tying="untied")]
    print(f"{'model':22} {'params':>6}  {'P shape':>10}  {'finite':>6}  {'grad nonzero':>13}")
    for spec in specs:
        fwd = build_model(spec)
        p = init_params(spec, seed=1)
        P = fwd(p, X, S)
        g = np.asarray(qml.grad(bce_loss, argnums=0)(p, X, S, Y, fwd))
        print(f"{spec.label:22} {spec.n_params:6d}  {str(tuple(P.shape)):>10}  "
              f"{str(bool(np.isfinite(np.asarray(P)).all())):>6}  "
              f"{int((np.abs(g)>1e-9).sum()):>6}/{spec.n_params}")

    # per-stage strength path (physical branch)
    fwd = build_model(ModelSpec("m3"))
    P2 = fwd(init_params(ModelSpec("m3")), X, S2)
    print(f"\nper-stage S (B,3,3,2) accepted: shape {tuple(P2.shape)}, "
          f"finite {bool(np.isfinite(np.asarray(P2)).all())}")

    # structural check: does a NEIGHBOUR pixel influence the centre prediction?
    # M1 must be spatially independent; M2/M3 must not be.
    # NOTE: use a small ADDITIVE perturbation, not a sign flip. CZ transmits
    # <Z_r> = cos(pi*u_r) to its neighbours, and cos is EVEN, so negating a
    # pixel's features is invisible through the CZ coupling channel (verified:
    # negation gives 1e-16 for M2 while +0.15 gives 1.6e-2).
    print("\nneighbour-influence on centre pixel (perturb pixel (0,1), +0.15 on ch0):")
    for spec in [ModelSpec("m1"), ModelSpec("m2"), ModelSpec("m3")]:
        f = build_model(spec); p = init_params(spec, seed=1)
        Xa = np.array(X, dtype=float); Xb = Xa.copy()
        Xb[:, 0, 1, 0] += 0.15                           # direct NN of the centre
        Sa = np.array(S, dtype=float); Sb = Sa.copy(); Sb[:, 0, 1] = np.clip(Sb[:, 0, 1] + 0.15, 0, 1)
        c_a = np.asarray(f(p, pnp.array(Xa, requires_grad=False),
                             pnp.array(Sa, requires_grad=False)))[:, 1, 1]
        c_b = np.asarray(f(p, pnp.array(Xb, requires_grad=False),
                             pnp.array(Sb, requires_grad=False)))[:, 1, 1]
        d = float(np.abs(c_a - c_b).max())
        expect = "independent" if spec.kind == "m1" else "coupled"
        ok = (d < 1e-12) if spec.kind == "m1" else (d > 1e-6)
        print(f"  {spec.label:6} max|Δp_centre| = {d:.3e}   expect {expect:12} "
              f"{'OK' if ok else 'FAIL'}")
