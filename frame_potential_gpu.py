"""
Frame Potential — GPU accelerated with PyTorch
===============================================

Installation
------------
The code auto-detects the best available backend in this priority order:
    1. CUDA  (NVIDIA GPU)   — torch.cuda.is_available()
    2. XPU   (Intel GPU)    — torch.xpu.is_available()
    3. CPU   (fallback)

Install PyTorch for your hardware:

    # NVIDIA — CUDA 12.1 (most common on modern systems):
    pip install torch --index-url https://download.pytorch.org/whl/cu121

    # NVIDIA — CUDA 11.8:
    pip install torch --index-url https://download.pytorch.org/whl/cu118

    # Intel GPU — XPU (requires intel-extension-for-pytorch):
    pip install torch intel-extension-for-pytorch

    # CPU-only (fallback, no speedup vs original script):
    pip install torch --index-url https://download.pytorch.org/whl/cpu

Verify the install:
    python -c "from frame_potential_gpu import get_device; get_device()"

Other dependencies:
    pip install qiskit qiskit-aer numpy tqdm

Usage
-----
    from frame_potential_gpu import frame_potential_gpu, compute_frame_potential_gpu
    from qiskit.circuit.library import EfficientSU2

    circuit = EfficientSU2(4, reps=2)
    results = compute_frame_potential_gpu(circuit, t=1, n_samples=500)
"""

import math
import warnings
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.circuit import ParameterVector

from save_read_results import save_results


# ── Device setup ──────────────────────────────────────────────────────────────

def get_device(verbose: bool = True) -> torch.device:
    """
    Return the best available device, checked in this order:
        1. CUDA  (NVIDIA GPU)
        2. XPU   (Intel GPU, requires intel-extension-for-pytorch)
        3. CPU   (fallback)
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            props = torch.cuda.get_device_properties(device)
            vram  = props.total_memory / 1024**3
            print(f"GPU found : {props.name}  (CUDA)")
            print(f"VRAM      : {vram:.1f} GB")
            print(f"CUDA      : {torch.version.cuda}")

    elif torch.xpu.is_available():
        device = torch.device("xpu")
        if verbose:
            props = torch.xpu.get_device_properties(device)
            vram  = props.total_memory / 1024**3
            print(f"GPU found : {props.name}  (Intel XPU)")
            print(f"VRAM      : {vram:.1f} GB")

    else:
        device = torch.device("cpu")
        if verbose:
            print("No GPU found — running on CPU (no speedup vs original script)")

    return device


def recommended_batch_size(n: int, d: int, vram_gb: float, dtype: torch.dtype) -> int:
    """
    Compute the largest batch size B such that one (B, B, d, d) complex
    tensor fits in `vram_gb` gigabytes (using 50% as a safety margin).

    The bottleneck tensor is the broadcasted product
        Ai.conj() * Bj  →  shape (B, B, d, d)
    Each element is 8 bytes (complex64) or 16 bytes (complex128).

    Parameters
    ----------
    n       : number of unitary samples (only used for capping at N)
    d       : Hilbert space dimension (2^n_qubits)
    vram_gb : available VRAM in GB
    dtype   : torch.complex64 or torch.complex128

    Returns
    -------
    B : int
    """
    bytes_per_element = 8 if dtype == torch.complex64 else 16
    usable_bytes = vram_gb * 1024**3 * 0.5         # 50% safety margin
    B = int(math.sqrt(usable_bytes / (d * d * bytes_per_element)))
    B = max(1, min(B, n))                           # clamp to [1, N]
    return B


def _get_vram_gb(device: torch.device) -> float:
    """Return total VRAM in GB for a CUDA or XPU device."""
    if device.type == "cuda":
        return torch.cuda.get_device_properties(device).total_memory / 1024**3
    elif device.type == "xpu":
        return torch.xpu.get_device_properties(device).total_memory / 1024**3
    return 0.0


# ── CPU sampling (unchanged from original) ────────────────────────────────────

def sample_unitary(circuit: QuantumCircuit, param_values: np.ndarray) -> np.ndarray:
    """Bind parameters and return the unitary as a numpy array."""
    params = circuit.parameters
    bound  = circuit.assign_parameters(dict(zip(params, param_values)))
    return np.array(Operator(bound).data)


def sample_unitaries_cpu(
    circuit: QuantumCircuit,
    n_samples: int,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = True,
    parameter_composer = None,  # identity by default, can be customized
) -> np.ndarray:
    """
    Sample n_samples random unitaries on the CPU (Qiskit evaluation).
    Returns a numpy array of shape (N, d, d), dtype complex128.

    This step cannot be moved to the GPU because it goes through Qiskit's
    circuit simulation, which is a CPU-only operation.
    """
    if rng is None:
        rng = np.random.default_rng()

    d        = 2 ** circuit.num_qubits
    n_params = circuit.num_parameters
    Us       = np.empty((n_samples, d, d), dtype=complex)

    for i in tqdm(range(n_samples), desc="Sampling on CPU", disable=not verbose):
        theta  = rng.uniform(0, 2 * np.pi, size=n_params)
        if parameter_composer is not None:
            parameter_composer(theta)
        Us[i]  = sample_unitary(circuit, theta)

    return Us


# ── GPU transfer ──────────────────────────────────────────────────────────────

def to_gpu(
    unitaries: np.ndarray,
    device: torch.device,
    dtype: torch.dtype = torch.complex64,
) -> torch.Tensor:
    """
    Transfer the numpy unitary array to the GPU as a PyTorch tensor.

    We do this ONCE before the frame potential loop — not per batch.
    Doing it per batch would saturate the PCIe bus (~16 GB/s) and
    destroy the speedup.

    dtype choice:
        torch.complex64  — 8 bytes/element, 2-4x faster on consumer GPUs
                           (RTX 3090, 4090, etc. have poor float64 throughput)
        torch.complex128 — 16 bytes/element, exact, needed for large t
                           (A100/H100 have full float64 throughput)
    """
    return torch.tensor(unitaries, dtype=dtype, device=device)


# ── GPU frame potential ───────────────────────────────────────────────────────

def frame_potential_gpu(
    unitariesA: torch.Tensor,
    unitariesB: torch.Tensor,
    t: int,
    batch_size: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> float:
    """
    Compute the t-design frame potential on the GPU:

        F^(t) = (1/N²) Σᵢⱼ |Tr(Uᵢ† Uⱼ)|^{2t}

    Parameters
    ----------
    unitariesA : torch.Tensor of shape (N, d, d), complex, already on GPU
    unitariesB : torch.Tensor of shape (N, d, d), complex, already on GPU
    t          : design order
    batch_size : number of unitaries per block. If None, tries to do all N
                 at once (only safe if N is small or VRAM is large).
    device     : torch.device (inferred from unitaries if None)

    Returns
    -------
    F : float
    """
    if device is None:
        device = unitariesA.device

    N = unitariesA.shape[0]

    # Prefer float64 accumulation for numerical stability.
    # Some XPU and older CUDA devices do not expose fp64 — fall back to float32.
    accum_dtype = torch.float64
    try:
        _ = torch.zeros(1, dtype=torch.float64, device=device)
    except RuntimeError:
        accum_dtype = torch.float32
        warnings.warn(
            "Device does not support fp64; accumulating in float32.",
            RuntimeWarning,
        )

    total = torch.tensor(0.0, dtype=accum_dtype, device=device)
    sum_sq = torch.tensor(0.0, dtype=accum_dtype, device=device)  # Σ |tr|^{4t}, for variance

    # If no batch_size given, try the full N×N at once.
    # This is optimal when it fits — zero loop overhead.
    if batch_size is None:
        batch_size = N

    for i in range(0, N, batch_size):
        # shape: (B, d, d)  →  (B, 1, d, d)
        Ai = unitariesA[i : i + batch_size].unsqueeze(1)

        for j in range(0, N, batch_size):
            # shape: (B', d, d)  →  (1, B', d, d)
            Bj = unitariesB[j : j + batch_size].unsqueeze(0)

            # ── Core computation ───────────────────────────────────────────
            #
            # We want Tr(Uᵢ† Uⱼ) for every pair (i, j).
            #
            # Key identity (same as the CPU version):
            #     Tr(A†B) = Σ_{p,q} conj(A_{pq}) · B_{pq}
            #
            # einsum "bipq, bjpq -> bij":
            #   b = batch index of Ai  (kept)
            #   i = batch index of Bj  (kept, note: variable named 'i' in the string
            #       but refers to j-loop index here — the einsum string just labels axes)
            #   p = row of the d×d matrix  (contracted / summed over)
            #   q = col of the d×d matrix  (contracted / summed over)
            #
            # Broadcasting:
            #   Ai: (B, 1, d, d)  ←  broadcasts over j dimension
            #   Bj: (1, B', d, d) ←  broadcasts over i dimension
            #   result: (B, B') — one complex trace per pair
            #
            traces = torch.einsum("bipq,bjpq->bij", Ai.conj(), Bj)

            # |Tr(Uᵢ†Uⱼ)|^{2t} then sum over all pairs in this block
            # .abs() returns real, ** (2*t) raises to power, .sum() accumulates
            block_sum = torch.sum(torch.abs(traces) ** (2 * t)).to(accum_dtype)
            total   += block_sum
            sum_sq  += torch.sum(torch.abs(traces) ** (4 * t)).to(accum_dtype)

    # F^(t) = mean of |Tr(Ui†Uj)|^{2t}
    F = (total / (N * N)).item()
    # Variance of the estimator: Var[X] = E[X²] - E[X]²
    V = (sum_sq / (N * N) - F ** 2).item()
    fidelity_error = math.sqrt(max(V, 0.0) / (N * N))  # standard error of the mean
    return {
        "frame_potential": F,
        "variance": V,
        "fidelity_error": fidelity_error
    }


# ── Haar reference value ──────────────────────────────────────────────────────

def haar_frame_potential(t: int, d: int) -> float:
    """F_Haar^(t) ≈ t!  (large-d approximation, standard in QML literature)."""
    if d <  t:
        warnings.warn(f"d={d} < 2t={2*t}: large-d approximation may be inaccurate.")
    return float(math.factorial(t))


# ── Full pipeline ─────────────────────────────────────────────────────────────

def compute_frame_potential_gpu(
    circuit: QuantumCircuit,
    t: int = 1,
    n_samples: int = 500,
    dtype: torch.dtype = torch.complex64,
    batch_size: Optional[int] = None,
    seed: Optional[int] = None,
    verbose: bool = True,
    save: bool = False,
    circuit_info: Optional[dict] = {},
    parameter_composer: Optional[callable] = None,
) -> dict:
    """
    Full pipeline:
        1. Detect GPU
        2. Sample unitaries on CPU (Qiskit)
        3. Transfer to GPU (once)
        4. Compute frame potential on GPU
        5. Compare to Haar reference

    Parameters
    ----------
    circuit    : Qiskit QuantumCircuit with free parameters
    t          : design order
    n_samples  : number of Monte Carlo samples N
    dtype      : torch.complex64 (fast) or torch.complex128 (exact)
    batch_size : GPU block size. None = auto (tries full N×N).
    seed       : random seed for reproducibility
    verbose    : print progress and results

    Returns
    -------
    dict with keys: frame_potential, haar_value, delta, ratio,
                    n_qubits, d, t, n_samples, device, dtype
    """
    # ── 1. Device ────────────────────────────────────────────────────────────
    device = get_device(verbose=verbose)

    d = 2 ** circuit.num_qubits

    # Auto batch size from VRAM if on a GPU device and no batch_size given
    if batch_size is None and device.type in ("cuda", "xpu"):
        vram_gb    = _get_vram_gb(device)
        batch_size = recommended_batch_size(n_samples, d, vram_gb, dtype)
        if verbose:
            print(f"Auto batch size : {batch_size}  (VRAM={vram_gb:.1f}GB, d={d})")

    if verbose:
        print(f"\nn_qubits={circuit.num_qubits}  d={d}  N={n_samples}  t={t}")
        print(f"dtype={dtype}  batch_size={batch_size or n_samples}")
        print("─" * 50)

    # ── 2. Sample on CPU ─────────────────────────────────────────────────────
    rng = np.random.default_rng(seed)
    Us_cpu_A = sample_unitaries_cpu(circuit, n_samples, rng=rng, verbose=verbose, parameter_composer=parameter_composer)
    Us_cpu_B = sample_unitaries_cpu(circuit, n_samples, rng=rng, verbose=verbose, parameter_composer=parameter_composer)

    # ── 3. Transfer to GPU (once) ────────────────────────────────────────────
    if verbose:
        nbytes = Us_cpu_A.nbytes / 1024**2
        print(f"Transferring {nbytes:.1f} MB to {device} ...")
    Us_gpu_A = to_gpu(Us_cpu_A, device, dtype=dtype)
    Us_gpu_B = to_gpu(Us_cpu_B, device, dtype=dtype)

    # ── 4. Compute on GPU ────────────────────────────────────────────────────
    if verbose:
        print("Computing frame potential on GPU ...")

    result = frame_potential_gpu(Us_gpu_A, Us_gpu_B, t=t, batch_size=batch_size, device=device)
    F = result["frame_potential"]

    # ── 5. Compare to Haar ───────────────────────────────────────────────────
    F_haar = haar_frame_potential(t, d)
    delta  = F - F_haar
    ratio  = F / F_haar

    if verbose:
        print(f"\n{'─'*50}")
        print(f"  F^({t}) (ansatz)  : {F:.6f}")
        print(f"  F^({t}) (Haar)    : {F_haar:.6f}")
        print(f"  ΔF (gap)         : {delta:.6f}")
        print(f"  Ratio F/F_Haar   : {ratio:.4f}")
        if ratio < 1.05:
            print(f"  ✓ Near-{t}-design")
        elif ratio < 2.0:
            print(f"  ~ Moderate expressibility")
        else:
            print(f"  ✗ Far from {t}-design")
        print(f"{'─'*50}")

    result = {
        "frame_potential" : F,
        "variance"        : result["variance"],
        "fidelity_error"  : result["fidelity_error"],
        "haar_value"      : F_haar,
        "delta"           : delta,
        "n_parameters"    : circuit.num_parameters,
        "circuit_depth"   : circuit.depth(),
        "ratio"           : ratio,
        "n_qubits"        : circuit.num_qubits,
        "d"               : d,
        "t"               : t,
        "n_samples"       : n_samples,
        "device"          : str(device),
        "dtype"           : str(dtype),
    }
    if save:
        if verbose:
            print("\nSaving results ...")
        circuit_info["t"] = t
        circuit_info["n_samples"] = n_samples
        save_results(path="./data/results/frame_potential", model_info=circuit_info, result_info=result, verbose=verbose)

    return result