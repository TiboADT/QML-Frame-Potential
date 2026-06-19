import torch

def get_gate_matrix(operation, parameters, i):
    gate_name = operation.name
    i+=1
    if gate_name == "rx":
        return rx(parameters), i
    elif gate_name == "ry":
        return ry(parameters), i
    elif gate_name == "rz":
        return rz(parameters), i
    elif gate_name == "h":
        i-=1
        return hadamard(parameters.device), i
    elif gate_name == "cx":
        i-=1
        return cx(parameters,False), i
    elif gate_name == "cz":
        i-=1
        return cp(parameters,False), i
    elif gate_name == "cp":
        if operation.params == []:
            i-=1
            return cp(parameters,False), i
        else:
            return cp(parameters), i
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

    # We want to apply the gate to the qubit at index qubit_index
    # This means we need to take the tensor product of I with the gate_matrix at the right place
    full_gate = torch.kron(torch.kron(torch.eye(2**qubit_index, dtype=torch.complex64, device=U.device), gate_matrix), torch.eye(2**(n-qubit_index-1), dtype=torch.complex64, device=U.device))

    return torch.matmul(full_gate, U)    

def apply_Control_gate(U, control_qubit, target_qubit, n_qubits, gate_matrix):
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
    device = U.device
    if torch.abs(control_qubit - target_qubit) == 1:
        if control_qubit < target_qubit:
            full_gate = torch.kron(torch.kron(torch.eye(2**control_qubit, dtype=torch.complex64, device=U.device), gate_matrix), torch.eye(2**(n-control_qubit-2), dtype=torch.complex64, device=U.device))
        else:
            full_gate = torch.kron(torch.kron(torch.eye(2**target_qubit, dtype=torch.complex64, device=U.device), gate_matrix), torch.eye(2**(n-target_qubit-2), dtype=torch.complex64, device=U.device))
        return torch.matmul(full_gate, U)
    else:

        size_between = torch.abs(control_qubit - target_qubit) - 1
        A = torch.eye(size_between, dtype=torch.complex64, device=U.device)
        D = A.copy()
        B = A.copy()
        C = A.copy()
        A = torch.kron(A, gate_matrix[0:2, 0:2])
        B = torch.kron(B, gate_matrix[0:2, 2:4])
        C = torch.kron(C, gate_matrix[2:4, 0:2])
        D = torch.kron(D, gate_matrix[2:4, 2:4])
        gate = torch.cat([
            torch.cat([A, B], dim=1),
            torch.cat([C, D], dim=1)
            ], dim=0)
        
        full_gate = torch.kron(torch.kron(torch.eye(2**min(control_qubit, target_qubit), dtype=torch.complex64, device=U.device), gate), torch.eye(2**(n-max(control_qubit, target_qubit)-1), dtype=torch.complex64, device=U.device))
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

def hadamard(device):
    """
    returns: (2,2)
    """
    mat = torch.zeros(
        2,
        2,
        dtype=torch.complex64,
        device=device
    )

    mat[0, 0] = 1 / torch.sqrt(torch.tensor(2.0))
    mat[0, 1] = 1 / torch.sqrt(torch.tensor(2.0))
    mat[1, 0] = 1 / torch.sqrt(torch.tensor(2.0))
    mat[1, 1] = -1 / torch.sqrt(torch.tensor(2.0))

    return mat

def cx(theta, with_parameters = True):
    """
    returns: (4,4)
    """
    mat = torch.zeros(
        theta.shape[0],
        4,
        4,
        dtype=torch.complex64,
        device=theta.device,
    )
    
    mat[:, 0, 0] = 1
    mat[:, 1, 1] = 1
    mat[:, 2, 3] = 1
    mat[:, 3, 2] = 1

    if with_parameters:
        c = torch.cos(theta / 2)
        s = torch.sin(theta / 2)
        mat[:, 2, 2] = c
        mat[:, 3, 3] = c
        mat[:, 2, 3] = -1j * s
        mat[:, 3, 2] = -1j * s


    return mat

def cp(theta, with_parameters = True):
    """
    returns: (4,4)
    """
    mat = torch.zeros(
        theta.shape[0],
        4,
        4,
        dtype=torch.complex64,
        device=theta.device,
    )

    mat[:, 0, 0] = 1
    mat[:, 1, 1] = 1
    mat[:, 2, 2] = 1
    mat[:, 3, 3] = -1

    if with_parameters:
        
        mat[:, 3, 3] = torch.exp(1j * theta)
        

    return mat