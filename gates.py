import torch

def get_gate_matrix(gate_name, parameters):
    if gate_name == "rx":
        return rx(parameters)
    elif gate_name == "ry":
        return ry(parameters)
    elif gate_name == "rz":
        return rz(parameters)
    else:
        raise ValueError(f"Gate {gate_name} not recognized")

def apply_1q_gate(U, gate_matrix, qubit_index, n_qubits):
    """
    U: (B, 2^n, 2^n)
    gate_matrix: (B, 2, 2)
    qubit_index: int
    n_qubits: int
    returns: (B, 2^n, 2^n)
    """
    n = n_qubits

    I = torch.eye(2, dtype=torch.complex64, device=U.device).unsqueeze(0).repeat(U.shape[0], 1, 1)

    # We want to apply the gate to the qubit at index qubit_index
    # This means we need to take the tensor product of I with the gate_matrix at the right place
    if qubit_index == 0:
        full_gate = torch.kron(gate_matrix, torch.eye(2**(n-1), dtype=torch.complex64, device=U.device))
    elif qubit_index == n-1:
        full_gate = torch.kron(torch.eye(2**(n-1), dtype=torch.complex64, device=U.device), gate_matrix)
    else:
        full_gate = torch.kron(torch.kron(torch.eye(2**qubit_index, dtype=torch.complex64, device=U.device), gate_matrix), torch.eye(2**(n-qubit_index-1), dtype=torch.complex64, device=U.device))

    return torch.matmul(full_gate, U)    

def apply_cx_gate(U, control_qubit, target_qubit, n_qubits):
    """
    U: (B, 2^n, 2^n)
    control_qubit: int
    target_qubit: int
    n_qubits: int
    returns: (B, 2^n, 2^n)
    """
    n = n_qubits

    # We want to apply the cx gate to the control and target qubits
    # This means we need to take the tensor product of I with the cx gate at the right place
    # We suppose the control and target qubits are adjacent for simplicity
    if control_qubit < target_qubit:
        full_gate = torch.kron(torch.kron(torch.eye(2**control_qubit, dtype=torch.complex64, device=U.device), cx()), torch.eye(2**(n-control_qubit-2), dtype=torch.complex64, device=U.device))
    else:
        full_gate = torch.kron(torch.kron(torch.eye(2**target_qubit, dtype=torch.complex64, device=U.device), cx()), torch.eye(2**(n-target_qubit-2), dtype=torch.complex64, device=U.device))
    return torch.matmul(full_gate, U)

def rx(theta):
    """
    theta: (B,)
    returns: (B,2,2)
    """
    c = torch.cos(theta / 2)
    s = torch.sin(theta / 2)

    mat = torch.zeros(
        theta.shape[0],
        2,
        2,
        dtype=torch.complex64,
        device=theta.device,
    )

    mat[:, 0, 0] = c
    mat[:, 1, 1] = c
    mat[:, 0, 1] = -1j * s
    mat[:, 1, 0] = -1j * s

    return mat

def rz(theta):
    """
    theta: (B,)
    returns: (B,2,2)
    """
    c = torch.cos(theta / 2)
    s = torch.sin(theta / 2)

    mat = torch.zeros(
        theta.shape[0],
        2,
        2,
        dtype=torch.complex64,
        device=theta.device,
    )

    mat[:, 0, 0] = c - 1j * s
    mat[:, 1, 1] = c + 1j * s

    return mat

def ry(theta):
    """
    theta: (B,)
    returns: (B,2,2)
    """
    c = torch.cos(theta / 2)
    s = torch.sin(theta / 2)

    mat = torch.zeros(
        theta.shape[0],
        2,
        2,
        dtype=torch.complex64,
        device=theta.device,
    )

    mat[:, 0, 0] = c
    mat[:, 1, 1] = c
    mat[:, 0, 1] = -s
    mat[:, 1, 0] = s

    return mat

def cx():
    """
    returns: (4,4)
    """
    mat = torch.zeros(
        4,
        4,
        dtype=torch.complex64,
    )

    mat[0, 0] = 1
    mat[1, 1] = 1
    mat[2, 3] = 1
    mat[3, 2] = 1

    return mat