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

def _cx_matrix() -> torch.Tensor:
    # Qiskit CX: control=qubit[0], target=qubit[1]
    # In little-endian: |ct⟩ ordering is |00⟩,|01⟩,|10⟩,|11⟩
    # CX flips target when control=1:
    #   |00⟩→|00⟩, |01⟩→|01⟩, |10⟩→|11⟩, |11⟩→|10⟩
    return torch.tensor(
        [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 0, 1],
         [0, 0, 1, 0]], dtype=torch.complex64)

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
    "swap": torch.tensor(
        [[1, 0, 0, 0],
         [0, 0, 1, 0],
         [0, 1, 0, 0],
         [0, 0, 0, 1]], dtype=torch.complex64),
}


# ─────────────────────────────────────────────────────────────────────────────
# Embed a local gate matrix into the full n-qubit space
# ─────────────────────────────────────────────────────────────────────────────

def _embed(M_local: torch.Tensor, qubits: list[int], n: int) -> torch.Tensor:
    """
    Embed a 2^k × 2^k gate acting on `qubits` into the full 2^n × 2^n space.

    Accounts for Qiskit's little-endian convention: qubit k occupies tensor
    position (n-1-k) from the left, so the Kronecker structure is:

        I_{n-1} ⊗ ... ⊗ I_{k+1} ⊗ M ⊗ I_{k-1} ⊗ ... ⊗ I_0

    For a contiguous block of qubits [q0, q1, ..., qk-1]:
        before = 2^(n - 1 - max(qubits))   (qubits above the block in little-endian)
        after  = 2^min(qubits)              (qubits below the block)
    """
    q_min = min(qubits)
    q_max = max(qubits)
    before = 2 ** (n - 1 - q_max)   # tensor positions LEFT of this gate
    after  = 2 ** q_min              # tensor positions RIGHT of this gate

    I_b = torch.eye(before, dtype=M_local.dtype, device=M_local.device)
    I_a = torch.eye(after,  dtype=M_local.dtype, device=M_local.device)
    return torch.kron(torch.kron(I_b, M_local), I_a)


def _embed_batch(M_batch: torch.Tensor, qubits: list[int], n: int) -> torch.Tensor:
    """
    Embed a batch of (N, 2^k, 2^k) matrices into (N, 2^n, 2^n).
    Uses the same little-endian convention as _embed.

    Implementation: build the full matrix using einsum to compute the
    batched Kronecker product I_before ⊗ M[i] ⊗ I_after for each sample i.

        kron(A, B[i]) where A=(a,a), B=(N,b,b):
            result[i, a0*b + b0, a1*b + b1] = A[a0,a1] * B[i, b0, b1]

    We use einsum to compute this without any Python loop over N.
    """
    N  = M_batch.shape[0]
    mb = M_batch.shape[1]   # 2^k (local dimension)

    q_min  = min(qubits)
    q_max  = max(qubits)
    before = 2 ** (n - 1 - q_max)
    after  = 2 ** q_min

    device = M_batch.device
    dtype  = M_batch.dtype

    I_b = torch.eye(before, dtype=dtype, device=device)  # (before, before)
    I_a = torch.eye(after,  dtype=dtype, device=device)  # (after, after)

    # Step 1: kron(I_b, M_batch[i]) for each i
    # result[i, a0*mb+m0, a1*mb+m1] = I_b[a0,a1] * M_batch[i, m0, m1]
    # einsum: "ac, nij -> naicj" then reshape
    left = torch.einsum("ac,nij->naicj", I_b, M_batch)          # (N, before, mb, before, mb)
    left = left.reshape(N, before * mb, before * mb)             # (N, before*mb, before*mb)

    # Step 2: kron(left[i], I_a) for each i
    # result[i, p0*after+a0, p1*after+a1] = left[i,p0,p1] * I_a[a0,a1]
    # einsum: "npq, ac -> npapc" then reshape — but use "npa" shape
    # Actually: (N, P, P, after, after) — need (N, P*after, P*after)
    # Let's redo cleanly:
    P = before * mb
    full = torch.einsum("nPQ,ac->nPaQc", left, I_a)             # (N, P, after, P, after)
    full = full.reshape(N, P * after, P * after)                 # (N, d, d)

    return full


# ─────────────────────────────────────────────────────────────────────────────
# Gate descriptor
# ─────────────────────────────────────────────────────────────────────────────

class UnsupportedGateError(NotImplementedError):
    pass

@dataclass
class GateOp:
    kind         : str
    qubits       : list
    param_idx    : Optional[int]
    fixed_matrix : Optional[torch.Tensor] = field(default=None, repr=False)


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
        # Sort parameters by name to get a stable index mapping
        # (matches Qiskit's own sorted order for assign_parameters)

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
                            f"Gate {name} depends on {len(free)} parameters simultaneously. "
                            "Only single-parameter gates are supported."
                        )
                    ops.append(GateOp(
                        kind      = name,
                        qubits    = qubits,
                        param_idx = param_map[free[0]],
                    ))
                else:
                    # Constant angle: pre-compute and store as fixed
                    val    = float(raw)
                    M2     = _PARAMETRIC_FN[name](
                                 torch.tensor([val], dtype=torch.float32))[0]
                    M_full = _embed(M2.to(torch.complex64), qubits, n)
                    ops.append(GateOp(
                        kind         = name + "_const",
                        qubits       = qubits,
                        param_idx    = None,
                        fixed_matrix = M_full,
                    ))

            # ── Fixed single-qubit gates ──────────────────────────────────
            elif name in _FIXED_2x2:
                M_full = _embed(_FIXED_2x2[name].clone(), qubits, n)
                ops.append(GateOp(
                    kind         = name,
                    qubits       = qubits,
                    param_idx    = None,
                    fixed_matrix = M_full,
                ))

            # ── Fixed two-qubit gates ─────────────────────────────────────
            elif name in _FIXED_4x4:
                M_full = _embed(_FIXED_4x4[name].clone(), qubits, n)
                ops.append(GateOp(
                    kind         = name,
                    qubits       = qubits,
                    param_idx    = None,
                    fixed_matrix = M_full,
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
        """
        if device is None:
            device = thetas.device
        N = thetas.shape[0]

        # Running product — initialise as N copies of the identity
        U = torch.eye(self.d, dtype=torch.complex64, device=device) \
                  .unsqueeze(0).expand(N, -1, -1).clone()

        for op in self.ops:
            if op.param_idx is None:
                # Fixed gate: one (d,d) matrix broadcast over all N
                M = op.fixed_matrix.to(device=device, dtype=torch.complex64)
                U = torch.einsum("ij,njk->nik", M, U)

            else:
                # Parametric gate: build (N, d, d) from angles, then bmm
                angles = thetas[:, op.param_idx]                   # (N,) float32
                M2     = _PARAMETRIC_FN[op.kind](angles)           # (N, 2, 2) complex64
                M_full = _embed_batch(M2, op.qubits, self.n_qubits) # (N, d, d)
                U = torch.bmm(M_full, U)

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
