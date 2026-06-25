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
from gates import apply_Control_gate, get_gate_matrix, apply_1q_gate


# ── Device setup ──────────────────────────────────────────────────────────────

def get_device(verbose: bool = False) -> torch.device:
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


def recommended_batch_size( d: int, vram_gb: float, dtype: torch.dtype, n: int = None) -> int:
    """
    Compute the largest batch size B such that one (B, B, d, d) complex
    tensor fits in `vram_gb` gigabytes (using 70% as a safety margin).

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
    usable_bytes = vram_gb * 1024**3 * 0.7         # 50% safety margin
    B = int(math.sqrt(usable_bytes / (d * d * bytes_per_element)))
    if n is not None:
        return max(1, min(B, n))                   # clamp to [1, N]
    return max(1, B)                                # at least 1, even if VRAM



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


# ── GPU sampling ───────────────────────────────────────────────────────

def sample_parameters(batch_size, n_params, device, parameter_composer=None, **kwargs):
    try:
        parameters =  2 * torch.pi * torch.rand(
            batch_size,
            n_params,
            device=device
        )
        if parameter_composer is not None:
            parameter_composer(parameters)
        return parameters
    except RuntimeError:
        # fallback
        cpu_params = 2 * torch.pi * torch.rand(
            batch_size,
            n_params,
            device="cpu"
        )
        return cpu_params.to(device)

def sample_unitaries_gpu(circuit: QuantumCircuit, 
                         batch_size: int, 
                         device,
                         parameter_composer=None,
                         verbose=False) -> torch.Tensor:
    """
    Sample unitaries on the GPU by generating random parameters and evaluating the circuit.

    Directly build full batch unitaries on the GPU
    Extract the operations from the circuit and apply them in a batched manner.

    """
    instructions = []
    n_params = circuit.num_parameters

    for instruction in circuit:
        instructions.append(instruction)

    parameters = sample_parameters(batch_size, n_params+1, device, parameter_composer=parameter_composer)

    unitaries = torch.eye(
        2**circuit.num_qubits,
        dtype=torch.complex64,
        device=device
    ).expand(batch_size, 2**circuit.num_qubits, 2**circuit.num_qubits).clone()



    i = 0
    for instruction in instructions:
        operation = instruction.operation

        gate_parameters = parameters[:,i]
        gate_name = operation.name

        gate_matrix,i = get_gate_matrix(operation, gate_parameters,i)
        if gate_name == "cx" or gate_name == "cp" or gate_name == "cz":
            # kind horrible but i cant find a way to extract the control and target qubits without going through the instruction object, which is not very user-friendly.
            control_qubit = circuit.find_bit(instruction.qubits[0]).index
            target_qubit = circuit.find_bit(instruction.qubits[1]).index
            unitaries = apply_Control_gate(unitaries, control_qubit=control_qubit,
                                           target_qubit=target_qubit, n_qubits=circuit.num_qubits,
                                            gate_matrix = gate_matrix)
        else:
            qubit_index = circuit.find_bit(instruction.qubits[0]).index
            # We need to apply the gate to the correct qubits. This involves reshaping and permuting the gate matrix to fit into the full Hilbert space of the circuit.
            unitaries = apply_1q_gate(unitaries, gate_matrix, qubit_index=qubit_index, n_qubits=circuit.num_qubits)
    return unitaries

def make_parameter_composer(acos_list: list[int]):
    """
    Returns a vectorised parameter_composer that operates in-place
    on a (batch_size, n_params) tensor.

    For each index i in acos_list:
        params[:, i] = arccos(params[:, i] / π - 1)

    This replaces the scalar loop:
        for i in acos_list:
            params[i] = arccos(params[i] / pi - 1)
    """
    if not acos_list:
        return None

    def parameter_composer(params: torch.Tensor) -> None:
        # params: (batch_size, n_params) — modified in-place
        idx = torch.tensor(acos_list, dtype=torch.long, device=params.device)
        # params[:, idx] shape: (batch_size, len(acos_list))
        # Full operation stays on GPU, no Python loop, no scalar ops
        params[:, idx] = torch.arccos(
            (params[:, idx] / torch.pi - 1).clamp(-1.0, 1.0)
        )
        # .clamp is required: arccos is only defined on [-1, 1].
        # Without it, values slightly outside that range (floating point
        # noise) produce NaN silently.

    return parameter_composer

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

    total = torch.tensor(0.0, dtype=accum_dtype, device=device)
    sum_sq = torch.tensor(0.0, dtype=accum_dtype, device=device)  
    
    # Σ |tr|^{4t}, for variance


    A = unitariesA[:].unsqueeze(1)
    B = unitariesB[:].unsqueeze(0)
    traces = torch.einsum("bipq,bjpq->bij", A.conj(), B)
    # |Tr(Uᵢ†Uⱼ)|^{2t} then sum over all pairs in this block
    # .abs() returns real, ** (2*t) raises to power, .sum() accumulates
    block_sum = torch.sum(torch.abs(traces) ** (2 * t)).to(accum_dtype)
    total   = block_sum
    sum_sq  = torch.sum(torch.abs(traces) ** (4 * t)).to(accum_dtype)

    # F^(t) = mean of |Tr(Ui†Uj)|^{2t}
    # Variance of the estimator: Var[X] = E[X²] - E[X]²
    # Ficelity interval (assuming normal distribution of the estimator, which is reasonable for large N) at 95% confidence:
    # mean ± 1.96 * std_error
    # std_error = sqrt(Var[X] / N_samples)

    F = (total / (N * N)).item()
    V = (sum_sq  / (N * N) - F ** 2).item()
    fidelity_error = 1.96 * math.sqrt(max(V, 0.0) / (N * N))

    return {
        "frame_potential": F,
        "variance": V,
        "fidelity_error": fidelity_error,
        "total_pairs": N * N,
        "total": total.item(),
        "sum_sq": sum_sq.item(),
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
    n_samples: int = None,
    converge_before_return: bool = False,  # not implemented yet
    dtype: torch.dtype = torch.complex64,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
    verbose: bool = True,
    save: bool = False,
    circuit_info: Optional[dict] = {},
    parameter_composer: Optional[callable] = None,
    force_save: bool = False,
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
    seed       : random seed for reproducibility
    verbose    : print progress and results
    force_save : whether to force saving of results
    Returns
    -------
    dict with keys: frame_potential, haar_value, delta, ratio,
                    n_qubits, d, t, n_samples, device, dtype
    """

    # 1. Device and variables setup
    if device is None:
        device = get_device(verbose=verbose)
    

    # Auto batch size from VRAM if on a GPU device and no batch_size given
    vram_gb    = _get_vram_gb(device)

    d = 2 ** circuit.num_qubits

    if n_samples is None and device.type in ("cuda", "xpu"):
        n_samples = recommended_batch_size( d, vram_gb, dtype, n_samples)
        if verbose:
            print(f"Auto batch size : {n_samples}  (VRAM={vram_gb:.1f}GB, d={d})")
    n_samples = n_samples or 2939  # default if not provided

    if seed is None:
        seed = np.random.SeedSequence().entropy % 2**32  # random seed for reproducibility, limited to 32 bits for PyTorch
    # Set the seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)


    # functions for verbose printing and saving results, to avoid cluttering the main logic with these details
    def verbose_print(*args, **kwargs):
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
    
    def save_function():
        if verbose:
            print("\nSaving results ...")
        circuit_info["t"] = t
        circuit_info["n_samples"] = n_samples
        save_results(path="./data/results/frame_potential", 
                     model_info=circuit_info, result_info=result, verbose=verbose, force_save=force_save)

    

    
    # ── 1.1. recursion version for convergance case ─────────────────────────────────────────────
    if converge_before_return:
        if verbose: 
            print("Convergence mode: will keep sampling until the fidelity error is below the threshold.")
        cumulative_total = 0.0
        cumulative_sum_sq = 0.0
        cumulative_pairs = 0
        
        max_batch_size = recommended_batch_size( d, vram_gb, dtype, None)

        F_p = compute_frame_potential_gpu(circuit, t=t, 
                                          n_samples=n_samples, 
                                          dtype=dtype,  
                                          seed=seed, verbose=False, 
                                          save=False,
                                          device=device, 
                                          parameter_composer=parameter_composer, )
        error = F_p["fidelity_error"]
        delta = F_p["delta"]
        cumulative_sum_sq += F_p["sum_sq"]
        cumulative_total += F_p["total"]
        cumulative_pairs += F_p["total_pairs"]
        F = F_p["frame_potential"]
        V = F_p["variance"]


        threshold = 0.4 # arbitrary threshold for convergence, can be adjusted

        if verbose:
            print(f"Convergence check: error={error:.6f}, threshold={np.abs(threshold*delta):.6f}, samples={cumulative_pairs}")
            convergence_target = np.abs(threshold * delta)
            progress_value = 0.0 if convergence_target == 0 else max(0.0, min(1.0, 1.0 - error / convergence_target))
            convergence_bar = tqdm(
                total=1.0,
                initial=progress_value,
                desc="Convergence",
                leave=False,
                disable=not verbose,
            )
            convergence_bar.set_postfix(
                error=f"{error:.3e}",
                target=f"{convergence_target:.3e}",
                samples=cumulative_pairs,
            )
        i=0
        while error > np.abs(threshold*delta) and error > 1e-5 and i < 50: # added a max iteration to avoid infinite loop in case of non convergence, can be adjusted or removed
            # desplay a progression bar of the convergence using tqdm

            n_samples *= 2
            if n_samples > max_batch_size:
                n_samples = max_batch_size

            F_p = compute_frame_potential_gpu(circuit, t=t, 
                                              n_samples=n_samples, 
                                              dtype=dtype, 
                                              device=device,
                                              seed=seed+i*773, verbose=False, 
                                              save=False, 
                                              circuit_info=circuit_info, 
                                              parameter_composer=parameter_composer, )
            i+=1
            cumulative_sum_sq += F_p["sum_sq"]
            cumulative_total += F_p["total"]
            cumulative_pairs += F_p["total_pairs"]
            F = cumulative_total / cumulative_pairs
            V = cumulative_sum_sq / cumulative_pairs - (F) ** 2
            error = 1.96 *math.sqrt(max(V, 0.0) / cumulative_pairs)

            delta = F - F_p["haar_value"]
            if verbose:
                convergence_target = np.abs(threshold * delta)
                progress_value = 0.0 if convergence_target == 0 else max(0.0, min(1.0, np.log(error) / np.log(convergence_target)))
                convergence_bar.n = progress_value
                convergence_bar.set_postfix(
                    error=f"{error:.3e}",
                    target=f"{convergence_target:.3e}",
                    samples=cumulative_pairs,
                )
                convergence_bar.refresh()
        

        F_haar = haar_frame_potential(t, d)
        ratio  = F / F_haar

        result = {
            "frame_potential" : F,
            "variance"        : V,
            "fidelity_error"  : error,
            "haar_value"      : F_haar,
            "delta"           : delta,
            "n_parameters"    : circuit.num_parameters,
            "circuit_depth"   : circuit.depth(),
            "ratio"           : ratio,
            "n_qubits"        : circuit.num_qubits,
            "d"               : d,
            "t"               : t,
            "n_samples"       : np.sqrt(cumulative_pairs),
            "device"          : str(device),
            "dtype"           : str(dtype),
            "total_pairs"     : cumulative_pairs,
            "total"           : cumulative_total,
            "sum_sq"          : cumulative_sum_sq,
        }
        if save:
            save_function()
        if verbose:
            verbose_print()
        return result
        


    # ── 2. Sampling ────────────────────────────────────────────────────────────

    if device.type == "cuda":
        # ── 2.1. Sample directly on GPU ────────────────────────────────────────────
        Us_gpu_A = sample_unitaries_gpu(circuit, n_samples//2, device=device, verbose=verbose, parameter_composer=parameter_composer)
        Us_gpu_B = sample_unitaries_gpu(circuit, n_samples//2, device=device, verbose=verbose, parameter_composer=parameter_composer)
    else:
        # warnings.warn("Intel XPU or CPU detected: performance may be poor due to immature PyTorch/XPU support. Consider using CUDA if available.")
        # ── 2.2. Sample on CPU ─────────────────────────────────────────────────────
        rng = np.random.default_rng(seed)
        Us_cpu_A = sample_unitaries_cpu(circuit, n_samples//2, rng=rng, verbose=verbose, parameter_composer=parameter_composer)
        Us_cpu_B = sample_unitaries_cpu(circuit, n_samples//2, rng=rng, verbose=verbose, parameter_composer=parameter_composer)

        # ── 3.2.1 Transfer to GPU (once) ────────────────────────────────────────────
        Us_gpu_A = to_gpu(Us_cpu_A, device, dtype=dtype)
        Us_gpu_B = to_gpu(Us_cpu_B, device, dtype=dtype)

    # ── 3. Compute on GPU ────────────────────────────────────────────────────

    result = frame_potential_gpu(Us_gpu_A, Us_gpu_B, t=t, device=device)
    F = result["frame_potential"]

    # ── 4. Compare to Haar ───────────────────────────────────────────────────
    F_haar = haar_frame_potential(t, d)
    delta  = F - F_haar
    ratio  = F / F_haar

    if verbose:
        verbose_print()

    # ── 5. Return results ───────────────────────────────────────────────────

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
        "total_pairs"     : result["total_pairs"],
        "total"           : result["total"],
        "sum_sq"          : result["sum_sq"],
    }

    if save:
        save_function()
    return result