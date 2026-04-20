"""
Frame Potential Calculator for Quantum Ansätze
================================================
Given a parameterised quantum circuit (ansatz), this script:
  1. Samples N random parameter configurations
  2. Evaluates the unitary matrix for each configuration via Qiskit
  3. Estimates the t-design frame potential F^(t) = (1/N²) Σᵢⱼ |Tr(Uᵢ† Uⱼ)|^(2t)
  4. Compares the result to the Haar (t-design lower bound) value
  5. Computes the "expressibility" gap  ΔF = F^(t) - F_Haar^(t)

Dependencies
------------
    pip install qiskit qiskit-aer numpy scipy tqdm

Usage examples
--------------
    # Built-in hardware-efficient ansatz:
    python frame_potential.py --ansatz hea --n_qubits 3 --reps 2 --t 1 --n_samples 500

    # Pass your own Qiskit ParameterizedCircuit via --custom (see bottom of file):
    python frame_potential.py --ansatz custom --n_qubits 2 --t 2 --n_samples 300
"""

import argparse
import math
import warnings
from typing import Optional

import numpy as np
from tqdm import tqdm

# ── Qiskit imports ──────────────────────────────────────────────────────────
try:
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import EfficientSU2, TwoLocal, RealAmplitudes
    from qiskit.quantum_info import Operator
    from qiskit.circuit import ParameterVector
except ImportError as e:
    raise ImportError(
        "Qiskit is required. Install with:  pip install qiskit qiskit-aer"
    ) from e

from circuit_generation import build_ansatz  # from local module (see circuit_generation.py)

# ── Frame-potential core ─────────────────────────────────────────────────────

def sample_unitary(circuit: QuantumCircuit, param_values: np.ndarray) -> np.ndarray:
    """
    Bind parameters to a Qiskit circuit and return its unitary matrix (numpy).

    Parameters
    ----------
    circuit      : QuantumCircuit with free parameters
    param_values : 1-D array of length == circuit.num_parameters

    Returns
    -------
    U : complex numpy array of shape (2^n, 2^n)
    """
    params = circuit.parameters  # sorted ParameterView
    bound = circuit.assign_parameters(dict(zip(params, param_values)))
    return np.array(Operator(bound).data)


def sample_unitaries(
    circuit: QuantumCircuit,
    n_samples: int,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = True,
) -> np.ndarray:
    """
    Draw n_samples random parameter vectors (uniform in [0, 2π]) and return
    the corresponding unitary matrices as an array of shape (n_samples, d, d).
    """
    if rng is None:
        rng = np.random.default_rng()

    n_params = circuit.num_parameters
    d = 2 ** circuit.num_qubits

    unitaries = np.empty((n_samples, d, d), dtype=complex)
    iterator = tqdm(range(n_samples), desc="Sampling unitaries", disable=not verbose)

    for i in iterator:
        theta = rng.uniform(0, 2 * np.pi, size=n_params)
        unitaries[i] = sample_unitary(circuit, theta)

    return unitaries


def frame_potential(
    unitariesA: np.ndarray,
    unitariesB: np.ndarray,
    t: int,
    batch_size: int = 256,
) -> float:
    """
    Estimate the t-design frame potential:

        F^(t) = (1/N²) Σᵢ Σⱼ |Tr(Uᵢ† Uⱼ)|^(2t)

    Computed in batches to limit peak memory usage.

    Parameters
    ----------
    unitariesA : array of shape (N, d, d)
    unitariesB : array of shape (N, d, d)
    t          : design order (positive integer)
    batch_size : number of rows processed at once (tune for your RAM)

    Returns
    -------
    F : float — estimated frame potential
    V : float — variance of the estimate
    """
    N = len(unitariesA)
    total = 0
    X_2 = 0.0  # For variance calculation
    for i in range(0, N, batch_size):
        batch_i = unitariesA[i : i + batch_size]   # (B, d, d)
        for j in range(0, N, batch_size):
            batch_j = unitariesB[j : j + batch_size]  # (B', d, d)

            # Tr(Uᵢ† Uⱼ) = Σₖ (Uᵢ†)ₖₖ... = einsum over shared indices
            # Using matmul: (B, d, d) × (B', d, d)ᵀ  →  (B, B', d, d) is too large.
            # Better: Tr(A†B) = (A* ⊙ B).sum()  entry-wise (Hadamard then sum)
            # Vectorised: broadcast (B,1,d,d).conj * (1,B',d,d) → (B,B',d,d) sum last two
            # For large d, use einsum with explicit axes.
            # For d up to 32 (5 qubits) this is fine.
            Ai = batch_i[:, np.newaxis, :, :]       # (B, 1, d, d)
            Bj = batch_j[np.newaxis, :, :, :]       # (1, B', d, d)
            traces = np.einsum("bipq,bjpq->bij", Ai.conj(), Bj)  # (B, B')
            total += np.sum(np.abs(traces) ** (2 * t))
            X_2 += np.sum(np.abs(traces) ** (4 * t))
    # Calculate final frame potential and the potential error due to sampling variance
    F = total / (N * N)
    V = (X_2 / (N * N) - F ** 2)  # Variance of the estimate
    fidelity_error = np.sqrt(V / (N*N))  # Standard error of the mean
    return {"frame_potential": F, 
            "fidelity_error": fidelity_error,
            "variance": V}


# ── Haar reference value ─────────────────────────────────────────────────────

def haar_frame_potential(t: int, d: int) -> float:
    """
    Exact frame potential of the Haar (CUE) measure on U(d):

        F_Haar^(t) = Σ_{σ,τ ∈ S_t} Wg(σ⁻¹τ, d)²

    For t ≤ d this simplifies to:

        F_Haar^(t) = t!  (in the large-d limit, exact for d → ∞)

    The finite-d exact formula involves the Weingarten function, which is
    harder to compute. For practical purposes (d ≥ 2t) the approximation
    F_Haar ≈ t! is accurate to < 1% and is what most QML papers use.

    For small d, the exact value is computed via the known formula:

        F_Haar^(t) = 1 / C(d²+t-1, t)  ... NO — that's for frames.

    The exact CUE result is:

        F_Haar^(t) = ∫|Tr U|^{2t} dU_Haar
                   = Σ_{λ ⊢ t, ℓ(λ) ≤ d} (dim Vλ)² / (d+t-1 choose t) — still non-trivial.

    We provide both the large-d approximation and a Monte Carlo estimate.
    """
    if d >= 2 * t:
        # Large-d approximation (standard in QML literature)
        return float(math.factorial(t))
    else:
        warnings.warn(
            f"d={d} < 2t={2*t}: large-d approximation may be inaccurate. "
            "Use --haar_mc for a Monte Carlo Haar estimate.",
            stacklevel=2,
        )
        return float(math.factorial(t))


def haar_frame_potential_mc(t: int, d: int, n_samples: int = 2000) -> float:
    """
    Monte Carlo estimate of Haar frame potential via random unitary sampling
    using the QR decomposition of a Ginibre matrix (Haar-distributed U(d)).
    """
    from scipy.stats import unitary_group  # requires scipy

    total = 0.0
    for _ in tqdm(range(n_samples), desc="Haar MC", leave=False):
        U = unitary_group.rvs(d)
        V = unitary_group.rvs(d)
        total += abs(np.trace(U.conj().T @ V)) ** (2 * t)
    return total / n_samples


# ── Main entry point ──────────────────────────────────────────────────────────

def compute_frame_potential(
    circuit: QuantumCircuit,
    t: int = 1,
    n_samples: int = 500,
    batch_size: int = 256,
    haar_mc: bool = False,
    haar_mc_samples: int = 2000,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> dict:
    """
    Full pipeline: sample → compute F^(t) → compare to Haar.

    Returns a dict with keys:
        frame_potential   : estimated F^(t) for the ansatz
        haar_value        : reference F_Haar^(t)
        delta             : F - F_Haar  (expressibility gap, lower = better)
        ratio             : F / F_Haar
        n_qubits          : circuit width
        d                 : Hilbert space dimension
        t                 : design order
        n_samples         : number of Monte Carlo samples
    """
    rng = np.random.default_rng(seed)
    d = 2 ** circuit.num_qubits

    if verbose:
        print(f"\n{'─'*55}")
        print(f"  Qubits         : {circuit.num_qubits}")
        print(f"  Hilbert dim d  : {d}")
        print(f"  Parameters     : {circuit.num_parameters}")
        print(f"  Design order t : {t}")
        print(f"  MC samples N   : {n_samples}")
        print(f"{'─'*55}")

    # Sample unitaries
    Us_A = sample_unitaries(circuit, n_samples, rng=rng, verbose=verbose)
    Us_B = sample_unitaries(circuit, n_samples, rng=rng, verbose=verbose)

    # Compute frame potential
    if verbose:
        print("Computing frame potential …")
    result = frame_potential(Us_A, Us_B, t, batch_size=batch_size)
    F = result["frame_potential"]
    V = result["variance"]
    fidelity_error = result["fidelity_error"]

    # Haar reference
    if haar_mc:
        F_haar = haar_frame_potential_mc(t, d, n_samples=haar_mc_samples)
    else:
        F_haar = haar_frame_potential(t, d)

    delta = F - F_haar
    ratio = F / F_haar

    if verbose:
        print(f"\n{'─'*55}")
        print(f"  F^({t}) (ansatz)   : {F:.6f}")
        print(f"  F^({t}) (Haar)     : {F_haar:.6f}")
        print(f"  ΔF (gap)         : {delta:.6f}")
        print(f"  Ratio F/F_Haar   : {ratio:.4f}")
        if ratio < 1.05:
            print(f"  ✓ Near-{t}-design  (ratio ≈ 1)")
        elif ratio < 2.0:
            print(f"  ~ Moderate expressibility")
        else:
            print(f"  ✗ Far from t-design (low expressibility)")
        print(f"{'─'*55}\n")

    return {
        "frame_potential": F,
        "variance" : V,
        "fidelity_error" : fidelity_error,
        "haar_value": F_haar,
        "delta": delta,
        "ratio": ratio,
        "n_qubits": circuit.num_qubits,
        "d": d,
        "t": t,
        "n_samples": n_samples,
        "n_parameters": circuit.num_parameters,
    }


# ── Sweep over t values ───────────────────────────────────────────────────────

def sweep_t(
    circuit: QuantumCircuit,
    t_values: list[int],
    n_samples: int = 500,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> list[dict]:
    """
    Compute frame potential for multiple t values and print a summary table.
    """
    rng = np.random.default_rng(seed)
    d = 2 ** circuit.num_qubits

    # Sample once, reuse
    Us_A = sample_unitaries(circuit, n_samples, rng=rng, verbose=verbose)
    Us_B = sample_unitaries(circuit, n_samples, rng=rng, verbose=verbose)

    results = []
    print(f"\n{'t':>4}  {'F^(t)':>12}  {'F_Haar':>10}  {'ΔF':>12}  {'Ratio':>8}")
    print("─" * 54)

    for t in t_values:
        result = frame_potential(Us_A, Us_B, t)
        F = result["frame_potential"]
        V = result["variance"]
        fidelity_error = result["fidelity_error"]
        F_haar = haar_frame_potential(t, d)
        results.append({
            "t": t,
            "frame_potential": F,
            "fidelity_error" : fidelity_error,
            "Variance": V,
            "fidelity_error": fidelity_error,
            "haar_value": F_haar,
            "delta": F - F_haar,
            "ratio": F / F_haar,
        })
        print(f"{t:>4}  {F:>12.4f}  {F_haar:>10.4f}  {F - F_haar:>12.4f}  {F/F_haar:>8.3f}")

    print("─" * 54)
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Frame potential estimator for quantum ansätze",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ansatz",      default="hea",  choices=["hea","real_amp","two_local_rx","ghz_like","custom"])
    p.add_argument("--n_qubits",    type=int, default=3)
    p.add_argument("--reps",        type=int, default=2,   help="Circuit repetition layers")
    p.add_argument("--t",           type=int, default=1,   help="Design order (or start of sweep)")
    p.add_argument("--t_max",       type=int, default=None, help="If set, sweep t from --t to --t_max")
    p.add_argument("--n_samples",   type=int, default=500, help="Monte Carlo samples")
    p.add_argument("--batch_size",  type=int, default=256, help="Batch size for pairwise computation")
    p.add_argument("--haar_mc",     action="store_true",   help="Use MC Haar estimate instead of analytic")
    p.add_argument("--haar_mc_samples", type=int, default=2000)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--plot",        action="store_true",   help="Plot F^(t) vs t sweep (requires matplotlib)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # ── Build circuit ──────────────────────────────────────────────────────
    if args.ansatz == "custom":
        # ── INJECT YOUR OWN CIRCUIT HERE ───────────────────────────────────
        # Replace the block below with your own QuantumCircuit.
        # The circuit must have free Parameters (qiskit.circuit.Parameter).
        # Example:
        #   from qiskit.circuit import ParameterVector
        #   n = args.n_qubits
        #   params = ParameterVector('θ', 2*n)
        #   qc = QuantumCircuit(n)
        #   for i in range(n): qc.ry(params[i], i)
        #   for i in range(n-1): qc.cx(i, i+1)
        #   for i in range(n): qc.rz(params[n+i], i)
        #   circuit = qc
        # ──────────────────────────────────────────────────────────────────
        raise NotImplementedError(
            "Edit the 'custom' block in __main__ to provide your own circuit."
        )
    else:
        circuit = build_ansatz(args.ansatz, args.n_qubits, args.reps)

    # ── Run ────────────────────────────────────────────────────────────────
    if args.t_max is not None:
        t_values = list(range(args.t, args.t_max + 1))
        results = sweep_t(circuit, t_values, n_samples=args.n_samples, seed=args.seed)

        if args.plot:
            try:
                import matplotlib.pyplot as plt

                ts      = [r["t"] for r in results]
                Fs      = [r["frame_potential"] for r in results]
                F_haars = [r["haar_value"] for r in results]

                fig, ax = plt.subplots(figsize=(7, 4))
                ax.semilogy(ts, Fs,      "o-", label="Ansatz F^(t)", color="#7f77dd")
                ax.semilogy(ts, F_haars, "s--", label="Haar F^(t)",  color="#1d9e75")
                ax.set_xlabel("Design order t")
                ax.set_ylabel("Frame potential (log scale)")
                ax.set_title(f"Frame potential sweep — {args.ansatz} | {args.n_qubits}q | reps={args.reps}")
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig("frame_potential_sweep.png", dpi=150)
                print("Plot saved to frame_potential_sweep.png")
                plt.show()
            except ImportError:
                print("matplotlib not found; skipping plot.")
    else:
        compute_frame_potential(
            circuit,
            t=args.t,
            n_samples=args.n_samples,
            batch_size=args.batch_size,
            haar_mc=args.haar_mc,
            haar_mc_samples=args.haar_mc_samples,
            seed=args.seed,
            verbose=True,
        )
