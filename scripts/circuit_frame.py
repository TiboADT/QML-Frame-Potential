from circuit_generation import *
from frame_potential_gpu import compute_frame_potential_gpu
from itertools import product


def circuit_frame_evaluation():
    range_n_qubits = [4, 6, 8, 10]  
    range_reps = [1, 2, 3, 4, 5]
    range_circuits = range(1,20)
    range_t = [1, 2, 3]

    # for number in range_circuits:
    #     circuit = build_ansatz(name="set", n_qubits=4, reps=1, number=number)
    #     print(f"Test done for circuit {number}")
    #     print(circuit.draw())
    #     print()
    # return

    for n_qubits,reps,number,t in product(range_n_qubits, range_reps, range_circuits, range_t):
        circuit = build_ansatz(name="set", n_qubits=n_qubits, reps=reps, number=number)
        F_p = compute_frame_potential_gpu(circuit, t=t, n_samples=2**(n_qubits)*t, save=True, circuit_info={"name": f"set_{number}", "n_qubits": n_qubits, "reps": reps}, verbose=False)
        print(f"Test done for circuit {number} with n_qubits={n_qubits}, reps={reps}, t={t}. Frame potential: {F_p['frame_potential']}")