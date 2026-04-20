"""
Fast Batched Unitary Sampler
============================
Replaces the Qiskit-per-sample loop with a single batched matrix product.

Key idea
--------
Instead of calling Qiskit N times (each time rebuilding + simulating the circuit),
we parse the circuit ONCE into an ordered list of gate operations, then evaluate
all N unitaries simultaneously using batched PyTorch tensor operations.

Qubit ordering
--------------
Qiskit uses LITTLE-ENDIAN convention: qubit 0 is the RIGHTMOST (least significant)
tensor factor. So for n=3:

    |q2 q1 q0⟩  →  index = 4*q2 + 2*q1 + q0

The embed functions below account for this by reversing the qubit index when
computing Kronecker structure: qubit k occupies position (n-1-k) from the left.

Supported gates
---------------
Parametric : Rx, Ry, Rz, P (phase), U1
Fixed      : CX/CNOT, H, X, Y, Z, S, T, SWAP, Id

Usage
-----
    from fast_sampler import CircuitCompiler
    from qiskit.circuit.library import EfficientSU2

    circuit  = EfficientSU2(4, reps=2, entanglement='linear')
    compiler = CircuitCompiler.from_qiskit(circuit)

    # numpy array (N, d, d)
    Us = compiler.sample(n_samples=500, seed=42)

    # or as a torch tensor (stays on device for frame potential)
    Us_t = compiler.sample_torch(n_samples=500, device=torch.device('cpu'), seed=42)
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterExpression


# ─────────────────────────────────────────────────────────────────────────────
# Batched single-qubit gate matrices
# Each function: theta (N,) float32  →  (N, 2, 2) complex64
# ─────────────────────────────────────────────────────────────────────────────

def _rx_batch(theta: torch.Tensor) -> torch.Tensor:
    # Cast to complex so arithmetic with 1j works without dtype errors
    t = theta.to(torch.complex64) / 2
    c = torch.cos(t)                          # (N,) complex64
    s = torch.sin(t)
    # [[cos, -i·sin], [-i·sin, cos]]
    row0 = torch.stack([c,       -1j * s], dim=1)
    row1 = torch.stack([-1j * s,  c     ], dim=1)
    return torch.stack([row0, row1], dim=1)   # (N, 2, 2)

def _ry_batch(theta: torch.Tensor) -> torch.Tensor:
    t = theta.to(torch.complex64) / 2
    c = torch.cos(t)
    s = torch.sin(t)
    # [[cos, -sin], [sin, cos]]
    row0 = torch.stack([c, -s], dim=1)
    row1 = torch.stack([s,  c], dim=1)
    return torch.stack([row0, row1], dim=1)

def _rz_batch(theta: torch.Tensor) -> torch.Tensor:
    t = theta.to(torch.complex64) / 2
    # [[e^{-it}, 0], [0, e^{it}]]
    ep = torch.exp( 1j * t)
    em = torch.exp(-1j * t)
    z  = torch.zeros_like(ep)
    row0 = torch.stack([em, z ], dim=1)
    row1 = torch.stack([z,  ep], dim=1)
    return torch.stack([row0, row1], dim=1)

def _phase_batch(theta: torch.Tensor) -> torch.Tensor:
    t = theta.to(torch.complex64)
    one = torch.ones_like(t)
    z   = torch.zeros_like(t)
    e   = torch.exp(1j * t)
    # [[1, 0], [0, e^{it}]]
    row0 = torch.stack([one, z], dim=1)
    row1 = torch.stack([z,   e], dim=1)
    return torch.stack([row0, row1], dim=1)

_PARAMETRIC_FN = {
    "rx": _rx_batch,
    "ry": _ry_batch,
    "rz": _rz_batch,
    "p" : _phase_batch,
    "u1": _phase_batch,   # u1(θ) ≡ P(θ) up to global phase
}


# ─────────────────────────────────────────────────────────────────────────────
# Fixed gate matrices  (Qiskit little-endian, standard computational basis)
# ─────────────────────────────────────────────────────────────────────────────

def _permute_2q(M: torch.Tensor) -> torch.Tensor:
    """
    Reorder a 4×4 two-qubit gate matrix from standard |q0 q1⟩ ordering
    to the ordering used by _apply_2q_gate_batch:
        slot 0 → (q0=0, q1=0)  index = base
        slot 1 → (q0=0, q1=1)  index = base | s1   ← q1 varies first
        slot 2 → (q0=1, q1=0)  index = base | s0
        slot 3 → (q0=1, q1=1)  index = base | s0 | s1

    Standard ordering is [|00⟩, |01⟩, |10⟩, |11⟩] = slots [0,1,2,3].
    Our ordering is       [|00⟩, |10⟩, |01⟩, |11⟩] = perm  [0,2,1,3].
    We apply the permutation to both rows and columns.
    """
    perm = torch.tensor([0, 2, 1, 3])
    return M[perm][:, perm]


def _cx_matrix() -> torch.Tensor:
    # CX in little-endian standard ordering: control=q0 (bit 0), target=q1 (bit 1)
    # Flips q1 when q0=1:
    #   |00⟩→|00⟩, |01⟩→|11⟩, |10⟩→|10⟩, |11⟩→|01⟩
    M = torch.tensor(
        [[1, 0, 0, 0],
         [0, 0, 0, 1],
         [0, 0, 1, 0],
         [0, 1, 0, 0]], dtype=torch.complex64)
    return _permute_2q(M)

_FIXED_2x2 = {
    "h"  : torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64) / math.sqrt(2),
    "x"  : torch.tensor([[0, 1], [1,  0]], dtype=torch.complex64),
    "y"  : torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64),
    "z"  : torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64),
    "s"  : torch.tensor([[1, 0], [0,  1j]], dtype=torch.complex64),
    "sdg": torch.tensor([[1, 0], [0, -1j]], dtype=torch.complex64),
    "t"  : torch.tensor([[1, 0], [0, math.cos(math.pi/4) + 1j*math.sin(math.pi/4)]],
                         dtype=torch.complex64),
    "tdg": torch.tensor([[1, 0], [0, math.cos(math.pi/4) - 1j*math.sin(math.pi/4)]],
                         dtype=torch.complex64),
    "id" : torch.eye(2, dtype=torch.complex64),
    "i"  : torch.eye(2, dtype=torch.complex64),
}

_FIXED_4x4 = {
    "cx"  : _cx_matrix(),
    "cnot": _cx_matrix(),
    "swap": _permute_2q(torch.tensor(
        [[1, 0, 0, 0],
         [0, 0, 1, 0],
         [0, 1, 0, 0],
         [0, 0, 0, 1]], dtype=torch.complex64)),
}


def _apply_1q_gate_batch(
    M2: torch.Tensor,   # (N, 2, 2) — one 2×2 gate per sample
    qubit: int,         # which qubit this gate acts on (Qiskit little-endian)
    n: int,             # total number of qubits
    U: torch.Tensor,    # (N, d, d) running product — MODIFIED IN PLACE
) -> None:
    """
    Apply a batch of single-qubit gates to the running unitary product U,
    WITHOUT building the full (N, d, d) embedded matrix.

    Cost: O(N × d)  instead of  O(N × d²)  — the critical optimisation.

    How it works
    ------------
    A single-qubit gate G on qubit q mixes pairs of computational basis states
    that differ only in the q-th bit. In Qiskit little-endian convention,
    qubit q contributes bit value 2^q to the state index.

    For every "context" (the other n-1 bits held fixed), the gate touches
    exactly two rows of U:
        row_0 = index where bit q = 0   → rows[ctx]
        row_1 = index where bit q = 1   → rows[ctx] | (1 << q)

    We collect all (row_0, row_1) pairs, stack those rows, apply G as a
    batched 2×2 matmul, then write back — touching only d rows total, never
    allocating the d×d embedded matrix.

    Memory: O(N × d) for the two gathered row blocks.
    """
    d    = 1 << n
    step = 1 << qubit          # stride between the |0⟩ and |1⟩ rows for this qubit

    # Build the list of |0⟩-row indices for this qubit
    # These are all indices in [0, d) whose qubit-q bit is 0
    idx0 = torch.arange(0, d, device=U.device)
    idx0 = idx0[idx0 & step == 0]          # keep only those with bit q = 0
    idx1 = idx0 | step                     # corresponding |1⟩ rows

    # Gather the two row blocks: shape (N, d//2, d)
    rows0 = U[:, idx0, :]       # (N, d/2, d)
    rows1 = U[:, idx1, :]       # (N, d/2, d)

    # Stack into (N, d/2, 2, d) so we can apply the 2×2 gate along axis -2
    pair = torch.stack([rows0, rows1], dim=2)    # (N, d/2, 2, d)

    # M2: (N, 2, 2) — broadcast over the d/2 contexts and the d columns
    # We want:  new_pair[n, ctx, :, col] = M2[n] @ pair[n, ctx, :, col]
    # einsum "nij, nкjc -> nкic"  (к = context index, c = column index)
    new_pair = torch.einsum("nij,nkjc->nkic", M2, pair)   # (N, d/2, 2, d)

    # Scatter back — use index_put_ for in-place update
    U[:, idx0, :] = new_pair[:, :, 0, :]
    U[:, idx1, :] = new_pair[:, :, 1, :]


def _apply_2q_gate_batch(
    M4: torch.Tensor,   # (d_loc, d_loc) fixed 4×4 matrix  OR  (N, 4, 4)
    qubits: list[int],  # [control, target] in Qiskit convention
    n: int,
    U: torch.Tensor,    # (N, d, d) — modified in place
) -> None:
    """
    Apply a (possibly batched) two-qubit gate to U without embedding.

    Cost: O(N × d)  — same gain as the single-qubit case.

    For a two-qubit gate on qubits [q0, q1] we group the d basis states
    into blocks of 4 (the 4 combinations of bits q0 and q1), apply the
    4×4 gate to each block of 4 rows, and write back.
    """
    d  = 1 << n
    q0, q1 = qubits[0], qubits[1]
    s0, s1 = 1 << q0, 1 << q1

    # Build the 4 index arrays for (bit_q0, bit_q1) in {00, 01, 10, 11}
    base = torch.arange(0, d, device=U.device)
    base = base[(base & s0 == 0) & (base & s1 == 0)]  # d/4 "context" indices

    idx00 = base
    idx01 = base | s1
    idx10 = base | s0
    idx11 = base | s0 | s1

    # Stack into (N, d/4, 4, d)
    quad = torch.stack([
        U[:, idx00, :],
        U[:, idx01, :],
        U[:, idx10, :],
        U[:, idx11, :],
    ], dim=2)   # (N, d/4, 4, d)

    # Apply gate: M4 is (4,4) fixed — einsum over the 4-element local space
    if M4.dim() == 2:
        new_quad = torch.einsum("ij,nkjc->nkic", M4, quad)   # (N, d/4, 4, d)
    else:
        new_quad = torch.einsum("nij,nkjc->nkic", M4, quad)

    U[:, idx00, :] = new_quad[:, :, 0, :]
    U[:, idx01, :] = new_quad[:, :, 1, :]
    U[:, idx10, :] = new_quad[:, :, 2, :]
    U[:, idx11, :] = new_quad[:, :, 3, :]


# ─────────────────────────────────────────────────────────────────────────────
# Gate descriptor
# ─────────────────────────────────────────────────────────────────────────────

class UnsupportedGateError(NotImplementedError):
    pass

@dataclass
class GateOp:
    kind        : str
    qubits      : list
    param_idx   : Optional[int]
    local_matrix: Optional[torch.Tensor] = field(default=None, repr=False)
    # local_matrix is the 2×2 or 4×4 gate matrix in its own qubit space.
    # We no longer store the full d×d embedded matrix — that would cost O(4^n)
    # memory per gate and is exactly the source of the performance collapse.


# ─────────────────────────────────────────────────────────────────────────────
# Circuit compiler
# ─────────────────────────────────────────────────────────────────────────────

class CircuitCompiler:
    """
    Compiles a Qiskit QuantumCircuit into a list of GateOps.
    Call .sample() or .sample_torch() to produce N unitaries without Qiskit overhead.
    """

    def __init__(self, n_qubits: int, n_params: int, ops: list):
        self.n_qubits = n_qubits
        self.n_params = n_params
        self.ops      = ops
        self.d        = 2 ** n_qubits

    @classmethod
    def from_qiskit(cls, circuit: QuantumCircuit) -> "CircuitCompiler":
        """Parse circuit once. O(gates) time, called only once per circuit."""
        n = circuit.num_qubits
        param_map = {p: i for i, p in enumerate(
            sorted(circuit.parameters, key=lambda p: p.name))}
        ops = []

        for instruction in circuit.data:
            gate   = instruction.operation
            qubits = [circuit.find_bit(q).index for q in instruction.qubits]
            name   = gate.name.lower()

            if name == "barrier":
                continue

            # ── Parametric single-qubit gates ─────────────────────────────
            if name in _PARAMETRIC_FN:
                raw = gate.params[0]
                if isinstance(raw, (Parameter, ParameterExpression)):
                    free = list(raw.parameters)
                    if len(free) != 1:
                        raise UnsupportedGateError(
                            f"Gate {name} depends on {len(free)} parameters simultaneously."
                        )
                    ops.append(GateOp(
                        kind      = name,
                        qubits    = qubits,
                        param_idx = param_map[free[0]],
                    ))
                else:
                    # Constant angle → pre-bake as a fixed 2×2 matrix
                    val = float(raw)
                    M2  = _PARAMETRIC_FN[name](
                              torch.tensor([val], dtype=torch.float32))[0]  # (2,2)
                    ops.append(GateOp(
                        kind         = name + "_const",
                        qubits       = qubits,
                        param_idx    = None,
                        local_matrix = M2.to(torch.complex64),
                    ))

            # ── Fixed single-qubit gates ──────────────────────────────────
            elif name in _FIXED_2x2:
                ops.append(GateOp(
                    kind         = name,
                    qubits       = qubits,
                    param_idx    = None,
                    local_matrix = _FIXED_2x2[name].clone(),
                ))

            # ── Fixed two-qubit gates ─────────────────────────────────────
            elif name in _FIXED_4x4:
                ops.append(GateOp(
                    kind         = name,
                    qubits       = qubits,
                    param_idx    = None,
                    local_matrix = _FIXED_4x4[name].clone(),
                ))

            else:
                raise UnsupportedGateError(
                    f"Gate '{name}' is not supported.\n"
                    f"Supported parametric : {list(_PARAMETRIC_FN.keys())}\n"
                    f"Supported fixed 1q   : {list(_FIXED_2x2.keys())}\n"
                    f"Supported fixed 2q   : {list(_FIXED_4x4.keys())}\n"
                    f"Add it to the dicts at the top of fast_sampler.py."
                )

        return cls(n, len(param_map), ops)

    def __repr__(self) -> str:
        n_fix = sum(1 for op in self.ops if op.param_idx is None)
        n_par = sum(1 for op in self.ops if op.param_idx is not None)
        return (f"CircuitCompiler(n_qubits={self.n_qubits}, n_params={self.n_params}, "
                f"gates={len(self.ops)} [{n_fix} fixed, {n_par} parametric])")

    # ── Core evaluation ───────────────────────────────────────────────────────

    def evaluate_batch(
        self,
        thetas: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Compute U(theta[i]) for all i simultaneously.

        Parameters
        ----------
        thetas : (N, n_params) float32 tensor — one row per sample
        device : inferred from thetas if None

        Returns
        -------
        (N, d, d) complex64 tensor — one unitary per row, no Python loop over N

        Key optimisation vs the original _embed_batch approach
        -------------------------------------------------------
        Instead of building the full (N, d, d) embedded gate matrix and doing
        bmm, we call _apply_1q_gate_batch / _apply_2q_gate_batch which directly
        modify the relevant rows of U using index slicing.

        Cost per gate:
            old: O(N × d²)   — embed_batch allocates a full (N,d,d) tensor
            new: O(N × d)    — only touches 2 (or 4) rows out of d

        This is the fix for the speedup collapse above n=5 qubits.
        """
        if device is None:
            device = thetas.device
        N = thetas.shape[0]

        # Running product — N copies of the identity
        U = torch.eye(self.d, dtype=torch.complex64, device=device) \
                  .unsqueeze(0).expand(N, -1, -1).clone()

        for op in self.ops:
            if op.param_idx is None:
                # ── Fixed gate ──────────────────────────────────────────────
                M = op.local_matrix.to(device=device, dtype=torch.complex64)
                if len(op.qubits) == 1:
                    # Broadcast (2,2) matrix as a constant batch
                    M2_batch = M.unsqueeze(0).expand(N, -1, -1)  # (N,2,2)
                    _apply_1q_gate_batch(M2_batch, op.qubits[0], self.n_qubits, U)
                else:
                    _apply_2q_gate_batch(M, op.qubits, self.n_qubits, U)

            else:
                # ── Parametric single-qubit gate ────────────────────────────
                angles = thetas[:, op.param_idx]              # (N,) float32
                M2     = _PARAMETRIC_FN[op.kind](angles)      # (N, 2, 2) complex64
                _apply_1q_gate_batch(M2, op.qubits[0], self.n_qubits, U)

        return U

    # ── Public sampling API ───────────────────────────────────────────────────

    def sample_torch(
        self,
        n_samples: int,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype   = torch.complex64,
        seed: Optional[int]  = None,
    ) -> torch.Tensor:
        """
        Sample n_samples unitaries. Returns (N, d, d) tensor on `device`.
        Parameters drawn uniformly from [0, 2π].
        """
        gen = torch.Generator(device="cpu")
        if seed is not None:
            gen.manual_seed(seed)
        thetas = torch.rand(n_samples, self.n_params, generator=gen) * (2 * math.pi)
        thetas = thetas.to(device)
        return self.evaluate_batch(thetas, device=device).to(dtype)

    def sample(self, n_samples: int, seed: Optional[int] = None) -> np.ndarray:
        """Convenience wrapper — returns numpy (N, d, d) complex128."""
        Us = self.sample_torch(n_samples, device=torch.device("cpu"), seed=seed)
        return Us.numpy().astype(complex)
