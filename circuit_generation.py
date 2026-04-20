from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import EfficientSU2, RealAmplitudes, TwoLocal
from torch import ceil, log2

from frame_potential_gpu import frame_potential_gpu


# ── Built-in ansätze ──────────────────────────────────────────────────────────

def build_ansatz(name: str, 
                 n_qubits: int = None, 
                 n_parameters: int = None, 
                 reps: int = 0, 
                 circular: bool = False, 
                 parameter_prefix: str = "θ") -> QuantumCircuit:
    """
    Return a common hardware-efficient ansatz.

    name options:
        'hea'           – EfficientSU2 (Ry + CNOT entangler)
        'real_amp'      – RealAmplitudes (Ry + CNOT)
        'two_local_rx'  – TwoLocal with Rx/Rz + CX
        'ghz_like'      – simple layered Rx/Ry/Rz + CNOT chain
        'custom'        – user must inject circuit externally
    """
    if n_qubits is None and n_parameters is None:
        raise ValueError("n_qubits or n_parameters need to be specified. Choose one or the other.")

    #determine n_qubits based on n_parameters, circuit type and number of repetitions if n_qubits is not provided
    if n_qubits is None:
        n_qubits = 1
        while build_ansatz(name, n_qubits=n_qubits, n_parameters=n_parameters,  reps=reps).num_parameters < n_parameters:
            n_qubits += 1
    
    if name == "real_amp":
        """
        RealApmplitudes with 2 repetitions and 3 qubits:
        ┌──────────┐ ░            ░ ┌──────────┐ ░            
        ┤ Ry(θ[0]) ├─░────────■───░─┤ Ry(θ[3]) ├─░────────■───
        ├──────────┤ ░      ┌─┴─┐ ░ ├──────────┤ ░      ┌─┴─┐ 
        ┤ Ry(θ[1]) ├─░───■──┤ X ├─░─┤ Ry(θ[4]) ├─░───■──┤ X ├─
        ├──────────┤ ░ ┌─┴─┐└───┘ ░ ├──────────┤ ░ ┌─┴─┐└───┘ 
        ┤ Ry(θ[2]) ├─░─┤ X ├──────░─┤ Ry(θ[5]) ├─░─┤ X ├──────
        └──────────┘ ░ └───┘      ░ └──────────┘ ░ └───┘      
        """
        return RealAmplitudes(n_qubits, 
                              reps=reps, 
                              entanglement="linear",
                              skip_final_rotation_layer = False,
                              parameter_prefix=parameter_prefix)

    

    elif name == "two_local_rx":
        """
             ┌──────────┐┌──────────┐ ░           ░ ┌──────────┐ ┌──────────┐
        q_0: ┤ Ry(θ[0]) ├┤ Rz(θ[3]) ├─░──■──■─────░─┤ Ry(θ[6]) ├─┤ Rz(θ[9]) ├
             ├──────────┤├──────────┤ ░  │  │     ░ ├──────────┤┌┴──────────┤
        q_1: ┤ Ry(θ[1]) ├┤ Rz(θ[4]) ├─░──■──┼──■──░─┤ Ry(θ[7]) ├┤ Rz(θ[10]) ├
             ├──────────┤├──────────┤ ░     │  │  ░ ├──────────┤├───────────┤
        q_2: ┤ Ry(θ[2]) ├┤ Rz(θ[5]) ├─░─────■──■──░─┤ Ry(θ[8]) ├┤ Rz(θ[11]) ├
             └──────────┘└──────────┘ ░           ░ └──────────┘└───────────┘
        """
        return TwoLocal(
            n_qubits,
            rotation_blocks=["rx", "rz"],
            entanglement_blocks="cz",
            reps=reps,
            entanglement="full",
            parameter_prefix=parameter_prefix,
        )

    elif name == "ghz_like":
        # Manual alternating Rx/Rz + CNOT chain
        qc = QuantumCircuit(n_qubits)
        params = ParameterVector("θ", length=2 * n_qubits * (reps + 1))
        idx = 0
        for _ in range(reps + 1):
            for q in range(n_qubits):
                qc.rx(params[idx], q)
                idx += 1
                qc.rz(params[idx], q)
                idx += 1
            for q in range(n_qubits - 1):
                qc.cx(q, q + 1)
        return qc
    elif name == "brickwall":
        return brickwall_anzat(n_qubits, reps, parameter_prefix)
    else:
        raise ValueError(
            f"Unknown ansatz '{name}'. Choose from: hea, real_amp, two_local_rx, ghz_like, brickwall"
        )


def brickwall_anzat(n_qubits: int, reps: int, parameter_prefix: str) -> QuantumCircuit:
    """
    Return a brickwall ansatz with alternating Rx/Rz + CNOT layers.
    """
    qc = QuantumCircuit(n_qubits)
    params = ParameterVector(parameter_prefix, length=4 * n_qubits * (reps + 1))
    idx = 0
    for _ in range(reps + 1):
        for q in range((n_qubits//2)*2):
            qc.rx(params[idx], q)
            idx += 1
            qc.rz(params[idx], q)
            idx += 1
        for q in range(0, n_qubits - 1, 2):
            qc.cx(q+1, q)
        for q in range(1,(n_qubits-1)//2*2+1):
            qc.rx(params[idx], q)
            idx += 1
            qc.rz(params[idx], q)
            idx += 1    
        for q in range(1, n_qubits - 1, 2):
            qc.cx(q+1, q)
    return qc


def low_depth_anzat(n_qubits: int, reps: int, parameter_prefix: str) -> QuantumCircuit:
    """
    Return a low-depth ansatz with alternating Rx/Rz + CNOT layers.
    """
    qc = QuantumCircuit(n_qubits)
    params = ParameterVector(parameter_prefix, length=2 * n_qubits * (reps + 1))
    idx = 0

def circuit_depth(circuit: QuantumCircuit) -> int:
    """
    Return the depth of a quantum circuit.
    """
    return circuit.depth()


#implementation of the low depth t-designs with epsilon approximation, as described in https://arxiv.org/pdf/2407.07754


def epsilon_approx(circuit: QuantumCircuit, epsilon: float, i: int, j: int, t: int, name: str = "brickwall") -> QuantumCircuit:
    """
    Add an epsilon-approximation on the given circuit between qubits i and j.
    This is the function used in order to implement the extremely low depth t-designs.
    """
    # Placeholder for better implementation if i find one
    # For now, i call another ansatz and incress it number of repetitions until the frame potential is low enough, which means that the circuit is an epsilon-approximation of a t-design.
    
    n_qubits = j - i + 1
    reps = 1
    epsilon_circuit = build_ansatz(name, n_qubits=n_qubits, reps=reps)
    while frame_potential_gpu(epsilon_circuit, t) > epsilon:
        reps += 1
        epsilon_circuit = build_ansatz(name, n_qubits=n_qubits, reps=reps)
    circuit.compose(epsilon_circuit, qubits=range(i, j+1), inplace=True)

    return circuit

def low_depth_t_design(n_qubits: int, t: int, epsilon: float) -> QuantumCircuit:
    """
    Construct a low-depth t-design ansatz with epsilon-approximation.
    """
    circuit = QuantumCircuit(n_qubits)

    xi = ceil(log2(n_qubits*t**2/epsilon))

    for i in range(0,n_qubits,2*xi):
        circuit = epsilon_approx(circuit, epsilon/n_qubits, i, min(i+xi-1, n_qubits-1), t)
    for i in range(xi, n_qubits, 2*xi):
        circuit = epsilon_approx(circuit, epsilon/n_qubits, i, min(i+xi-1, n_qubits-1), t)

    return circuit

