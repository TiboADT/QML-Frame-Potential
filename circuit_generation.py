from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import EfficientSU2, RealAmplitudes, TwoLocal
from math import log2, ceil

from frame_potential_gpu import compute_frame_potential_gpu


# ── Built-in ansätze ──────────────────────────────────────────────────────────

def build_ansatz(name: str, 
                 n_qubits: int = None, 
                 n_parameters: int = None, 
                 reps: int = 0, 
                 circular: bool = False, 
                 parameter_prefix: str = "θ",
                 **kwargs) -> QuantumCircuit:
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
        while build_ansatz(name, n_qubits=n_qubits, n_parameters=n_parameters,  reps=reps, **kwargs).num_parameters < n_parameters:
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
    elif name == "set":
        if "number" in kwargs:
            return circuit_set(n_qubits, kwargs["number"], reps+1, parameter_prefix)
        else:
            raise ValueError("For 'set' ansatz, 'number' parameter is required.")

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


def epsilon_approx(circuit: QuantumCircuit, 
                   epsilon: float, i: int, j: int, t: int, 
                   name: str = "brickwall", 
                   reps : int = None,
                   max_reps: int = 10,
                   parameter_prefix: str = "ε") -> QuantumCircuit:
    """
    Add an epsilon-approximation on the given circuit between qubits i and j.
    This is the function used in order to implement the extremely low depth t-designs.
    """
    # Placeholder for better implementation if i find one
    # For now, i call another ansatz and incress it number of repetitions until the frame potential is low enough, which means that the circuit is an epsilon-approximation of a t-design.
    
    n_qubits = j - i + 1
    if reps is None:
        print(f"Finding epsilon-approximation for qubits {i} to {j} with epsilon={epsilon} and t={t}...")
        reps = 1
        epsilon_circuit = build_ansatz(name, n_qubits=n_qubits, reps=reps)
        data = compute_frame_potential_gpu(epsilon_circuit, t=t, verbose=False)
        F_p = data["frame_potential"]
        F_Haar = data["haar_value"]
        while abs(F_p - F_Haar) > epsilon and reps < max_reps:
            reps += 1
            epsilon_circuit = build_ansatz(name, n_qubits=n_qubits, reps=reps)
            data = compute_frame_potential_gpu(epsilon_circuit, t=t, verbose=False)
            F_p = data["frame_potential"]
            F_Haar = data["haar_value"]
    epsilon_circuit = build_ansatz(name, n_qubits=n_qubits, reps=reps)
    params = ParameterVector(parameter_prefix, len(epsilon_circuit.parameters))
    epsilon_circuit.assign_parameters(params, inplace=True)    

    circuit.compose(epsilon_circuit, qubits=range(i, j+1), inplace=True)
    return reps

def low_depth_t_design(n_qubits: int, t: int, epsilon: float, xi: int = None) -> QuantumCircuit:
    """
    Construct a low-depth t-design ansatz with epsilon-approximation.
    """
    circuit = QuantumCircuit(n_qubits)

    if xi is None:
        # use the formula from the paper to determine xi based on n_qubits, t and epsilon
        xi = ceil(log2(n_qubits*t**2/epsilon))
    reps = None
    print(f"Constructing low-depth t-design with n_qubits={n_qubits}, t={t}, epsilon={epsilon}, xi={xi}...")
    for i in range(0,n_qubits,2*xi):
        print(f"Adding epsilon-approximation for qubits {i} to {min(i+2*xi-1, n_qubits-1)}...")
        reps = epsilon_approx(circuit, epsilon/n_qubits, i, min(i+2*xi-1, n_qubits-1), t, reps=reps, parameter_prefix = f"ε1_{i}")
    print(f"Finished first layer of epsilon-approximations with {reps} repetitions. Now adding the second layer...")
    for i in range(xi, n_qubits, 2*xi):
        print(f"Adding epsilon-approximation for qubits {i} to {min(i+2*xi-1, n_qubits-1)}...")
        reps = epsilon_approx(circuit, epsilon/n_qubits, i, min(i+2*xi-1, n_qubits-1), t, reps=reps, parameter_prefix = f"ε2_{i}")
    print(f"Finished second layer of epsilon-approximations with {reps} repetitions. Low-depth t-design construction complete.")

    return circuit

def circuit_set(n_qubits: int, number : int, reps : int = 1,  parameter_prefix: str = "θ") -> QuantumCircuit:
    circuit = QuantumCircuit(n_qubits)
    if number == 1:
        params = ParameterVector(parameter_prefix, length=2 * n_qubits * (reps + 1))
        for i in range(reps):
            for q in range(n_qubits):
                circuit.rx(params[i*(n_qubits*2) + 2*q], q)
                circuit.rz(params[i*(n_qubits*2) + 2*q + 1], q)
    elif number == 2:
        params = ParameterVector(parameter_prefix, length=2 * n_qubits * (reps + 1))
        for i in range(reps):
            for q in range(n_qubits):
                circuit.rx(params[i*(n_qubits*2) + 2*q], q)
                circuit.rz(params[i*(n_qubits*2) + 2*q + 1], q)
            for q in range(n_qubits-1):
                circuit.cx(q, q+1)
    elif number == 3:
        param_per_rep = (2 * n_qubits + n_qubits-1)
        params = ParameterVector(parameter_prefix, length=param_per_rep * (reps + 1))
        for i in range(reps):
            for q in range(n_qubits):
                circuit.rx(params[i*param_per_rep + 2*q], q)
                circuit.rz(params[i*param_per_rep + 2*q + 1], q)
            for q in range(n_qubits-1):
                # Parametrized CZ
                circuit.cp(params[i*param_per_rep + n_qubits*2 + q],q, q+1)
    elif number == 4:
        param_per_rep = (2 * n_qubits + n_qubits-1)
        params = ParameterVector(parameter_prefix, length=param_per_rep * (reps + 1))
        for i in range(reps):
            for q in range(n_qubits):
                circuit.rx(params[i*param_per_rep + 2*q], q)
                circuit.rz(params[i*param_per_rep + 2*q + 1], q)
            for q in range(n_qubits-1):
                # Parametrized CX
                circuit.h(q)
                circuit.cp(params[i*param_per_rep + n_qubits*2 + q],q, q+1)
                circuit.h(q)
    elif number == 5:
        param_per_rep = (4 * n_qubits + (n_qubits-1)*n_qubits)
        params = ParameterVector(parameter_prefix, length=param_per_rep * (reps + 1))
        for i in range(reps):
            for q in range(n_qubits):
                circuit.rx(params[i*param_per_rep + 2*q], q)
                circuit.rz(params[i*param_per_rep + 2*q + 1], q)
            # all to all parametrized CZ
            for q in range(n_qubits):
                for q2 in range(n_qubits):
                    if q2 != q:
                        # Parametrized CZ
                        circuit.cp(params[i*param_per_rep + n_qubits*2 + q*n_qubits + q2],q, q2)
            for q in range(n_qubits):
                circuit.rx(params[(i+1)*param_per_rep - 2* n_qubits + 2*q], q)
                circuit.rz(params[(i+1)*param_per_rep - 2* n_qubits + 2*q + 1], q)
    elif number == 6:
        param_per_rep = (4 * n_qubits + (n_qubits-1)*n_qubits)
        params = ParameterVector(parameter_prefix, length=param_per_rep * (reps + 1))
        for i in range(reps):
            for q in range(n_qubits):
                circuit.rx(params[i*param_per_rep + 2*q], q)
                circuit.rz(params[i*param_per_rep + 2*q + 1], q)
            # all to all parametrized CX
            for q in range(n_qubits):
                for q2 in range(n_qubits):
                    if q2 != q:
                        # Parametrized CX
                        circuit.h(q)
                        circuit.cp(params[i*param_per_rep + n_qubits*2 + q*n_qubits + q2],q, q2)
                        circuit.h(q)
            for q in range(n_qubits):
                circuit.rx(params[(i+1)*param_per_rep - 2* n_qubits + 2*q], q)
                circuit.rz(params[(i+1)*param_per_rep - 2* n_qubits + 2*q + 1], q)
    elif number == 7:
        param_per_rep = (4 * n_qubits + n_qubits//2 + (n_qubits-1)//2)
        params = ParameterVector(parameter_prefix, length=param_per_rep * (reps + 1))
        for i in range(reps):
            for q in range(n_qubits):
                circuit.rx(params[i*param_per_rep + 2*q], q)
                circuit.rz(params[i*param_per_rep + 2*q + 1], q)
            for q in range(0,n_qubits-1,2):
                # Parametrized CZ
                circuit.cp(params[i*param_per_rep + n_qubits*2 + q],q, q+1)
            for q in range(n_qubits):
                circuit.rx(params[param_per_rep + n_qubits * 2 + n_qubits//2 + 2*q], q)
                circuit.rz(params[param_per_rep + n_qubits * 2 + n_qubits//2+ 2*q + 1], q)
            for q in range(1,n_qubits-1,2):
                # Parametrized CZ
                circuit.cp(params[i*param_per_rep + n_qubits * 4 + n_qubits//2+ n_qubits*2 + q],q, q+1)
    elif number == 8:
        param_per_rep = (4 * n_qubits + n_qubits//2 + (n_qubits-1)//2)
        params = ParameterVector(parameter_prefix, length=param_per_rep * (reps + 1))
        for i in range(reps):
            for q in range(n_qubits):
                circuit.rx(params[i*param_per_rep + 2*q], q)
                circuit.rz(params[i*param_per_rep + 2*q + 1], q)
            for q in range(0,n_qubits,2):
                # Parametrized CZ
                circuit.h(q)
                circuit.cp(params[i*param_per_rep + n_qubits*2 + q],q, q+1)
                circuit.h(q)
            for q in range(n_qubits):
                circuit.rx(params[param_per_rep + n_qubits * 2 + n_qubits//2 + 2*q], q)
                circuit.rz(params[param_per_rep + n_qubits * 2 + n_qubits//2+ 2*q + 1], q)
            for q in range(1,n_qubits-1,2):
                # Parametrized CX
                circuit.h(q)
                circuit.cp(params[i*param_per_rep + n_qubits * 4 + n_qubits//2+ n_qubits*2 + q],q, q+1)
                circuit.h(q)
    elif number == 9:
        param_per_rep = 4*n_qubits
        params = ParameterVector(parameter_prefix, length=param_per_rep * (reps + 1))
        for i in range(reps):
            for q in range(n_qubits):
                circuit.h(q)
            for q in range(n_qubits-1):
                circuit.cz(q, q+1)
            for q in range(n_qubits):
                circuit.rx(params[i*param_per_rep + q ], q)
    elif number == 10:
        param_per_rep = 4*n_qubits
        params = ParameterVector(parameter_prefix, length=param_per_rep * (reps + 1))
        for i in range(reps):
            for q in range(n_qubits-1):
                circuit.cz(q, q+1)
            circuit.cz(n_qubits-1, 0)
            for q in range(n_qubits):
                circuit.ry(params[i*param_per_rep + q ], q)
    elif number == 11:
        param_per_rep = 4*n_qubits - 4
        params = ParameterVector(parameter_prefix, length=param_per_rep * (reps + 1))
        for i in range(reps):
            for q in range(n_qubits):
                circuit.ry(params[i*param_per_rep + 2*q], q)
                circuit.rz(params[i*param_per_rep + 2*q + 1], q)
            for q in range(0,n_qubits-1,2):
                circuit.cx(q, q+1)
            for q in range(1,n_qubits-1):
                circuit.ry(params[i*param_per_rep +2*n_qubits + 2*q], q)
                circuit.rz(params[i*param_per_rep +2*n_qubits + 2*q + 1], q)
            for q in range(1,n_qubits-1,2):
                # Parametrized CZ
                circuit.cx(q, q+1)
    elif number == 12:

        param_per_rep = 4*n_qubits - 4
        params = ParameterVector(parameter_prefix, length=param_per_rep * (reps + 1))
        for i in range(reps):
            for q in range(n_qubits):
                circuit.ry(params[i*param_per_rep + 2*q], q)
                circuit.rz(params[i*param_per_rep + 2*q + 1], q)
            for q in range(0,n_qubits-1,2):
                circuit.cz(q, q+1)
            for q in range(1,n_qubits-1):
                circuit.ry(params[i*param_per_rep +2*n_qubits + 2*q], q)
                circuit.rz(params[i*param_per_rep +2*n_qubits + 2*q + 1], q)
            for q in range(1,n_qubits-1,2):
                # Parametrized CZ
                circuit.cz(q, q+1)
    elif number == 13:
        param_per_rep = 4*n_qubits
        params = ParameterVector(parameter_prefix, length=param_per_rep * (reps + 1))
        for i in range(reps):
            for q in range(n_qubits):
                circuit.ry(params[i*param_per_rep + q],q)
            for q in range(n_qubits):
                circuit.cp(params[i*param_per_rep + n_qubits + q], q, (q+1)%n_qubits)
            for q in range(n_qubits):
                circuit.ry(params[i*param_per_rep + 2*n_qubits + q],q)
            for q in range(n_qubits):
                circuit.cp(params[i*param_per_rep + 3*n_qubits + q], q, (q-1)%n_qubits)
    elif number == 14:
        param_per_rep = 4*n_qubits
        params = ParameterVector(parameter_prefix, length=param_per_rep * (reps + 1))
        for i in range(reps):
            for q in range(n_qubits):
                circuit.ry(params[i*param_per_rep + q],q)
            for q in range(n_qubits):
                circuit.h(q)
                circuit.cp(params[i*param_per_rep + n_qubits + q], q, (q+1)%n_qubits)
                circuit.h(q)
            for q in range(n_qubits):
                circuit.ry(params[i*param_per_rep + 2*n_qubits + q],q)
            for q in range(n_qubits):
                circuit.h(q)
                circuit.cp(params[i*param_per_rep + 3*n_qubits + q], (q-1)%n_qubits, q)
                circuit.h(q)
    elif number == 15:
        param_per_rep = 4*n_qubits
        params = ParameterVector(parameter_prefix, length=param_per_rep * (reps + 1))
        for i in range(reps):
            for q in range(n_qubits):
                circuit.ry(params[i*param_per_rep + q],q)
            for q in range(n_qubits):
                circuit.cx(q, (q+1)%n_qubits)
            for q in range(n_qubits):
                circuit.ry(params[i*param_per_rep + 2*n_qubits + q],q)
            for q in range(n_qubits):
                circuit.cx((q-1)%n_qubits, q)
    elif number == 16:
        param_per_rep = (2 * n_qubits + n_qubits-1)
        params = ParameterVector(parameter_prefix, length=param_per_rep * (reps + 1))
        for i in range(reps):
            for q in range(n_qubits):
                circuit.rx(params[i*param_per_rep + 2*q], q)
                circuit.rz(params[i*param_per_rep + 2*q + 1], q)
            for q in range(0,n_qubits-1,2):
                # Parametrized CZ
                circuit.cp(params[i*param_per_rep + n_qubits*2 + q],q, q+1)
            for q in range(1,n_qubits-1,2):
                # Parametrized CZ
                circuit.cp(params[i*param_per_rep + n_qubits*2 + q],q, q+1)
    elif number == 17:
        param_per_rep = (2 * n_qubits + n_qubits-1)
        params = ParameterVector(parameter_prefix, length=param_per_rep * (reps + 1))
        for i in range(reps):
            for q in range(n_qubits):
                circuit.rx(params[i*param_per_rep + 2*q], q)
                circuit.rz(params[i*param_per_rep + 2*q + 1], q)
            for q in range(0,n_qubits-1,2):
                # Parametrized CX
                circuit.h(q)
                circuit.cp(params[i*param_per_rep + n_qubits*2 + q],q, q+1)
                circuit.h(q)
            for q in range(1,n_qubits-1,2):
                # Parametrized CX
                circuit.h(q)
                circuit.cp(params[i*param_per_rep + n_qubits*2 + q],q, q+1)
                circuit.h(q)
    elif number == 18:
        param_per_rep = (2 * n_qubits + n_qubits-1)
        params = ParameterVector(parameter_prefix, length=param_per_rep * (reps + 1))
        for i in range(reps):
            for q in range(n_qubits):
                circuit.rx(params[i*param_per_rep + 2*q], q)
                circuit.rz(params[i*param_per_rep + 2*q + 1], q)
            for q in range(n_qubits):
                # Parametrized CZ
                circuit.cp(params[i*param_per_rep + n_qubits*2 + q],q, (q+1)%n_qubits)
    elif number == 19:
        param_per_rep = (2 * n_qubits + n_qubits-1)
        params = ParameterVector(parameter_prefix, length=param_per_rep * (reps + 1))
        for i in range(reps):
            for q in range(n_qubits):
                circuit.rx(params[i*param_per_rep + 2*q], q)
                circuit.rz(params[i*param_per_rep + 2*q + 1], q)
            for q in range(n_qubits):
                # Parametrized CX
                circuit.h(q)
                circuit.cp(params[i*param_per_rep + n_qubits*2 + q],q, (q+1)%n_qubits)
                circuit.h(q)
    return circuit