from circuit_generation import *
from frame_potential_gpu import compute_frame_potential_gpu
from itertools import product
from numpy import arccos, cos, sin, pi




def circuit_frame_evaluation(name = None,n_qubits=8,compose_parameters = False):
    range_reps = [1, 2, 3, 4, 5]
    range_t = [1, 2, 3]
    if n_qubits is None:
        range_n_qubits = [4, 6, 8]
    else:
        range_n_qubits = [n_qubits]
    if name == "set":
        range_circuits = range(1,20)

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
    
    if name == "perfectSU4":


        for n_qubits,reps,t in product(range_n_qubits, range_reps, range_t):
            acos_list = []
            circuit = perfectSU4_anzatz(n_qubits=n_qubits,reps=reps, parameter_prefix="θ", acos_list=acos_list)

            def parameter_composer(params):
                for i in acos_list:
                    params[i] = arccos(params[i]/pi -1)
            
            if compose_parameters:
                name = f"perfectSU4_composed"
            else:
                name = f"perfectSU4"
            F_p = compute_frame_potential_gpu(circuit, t=t, n_samples=2**(n_qubits)*t, save=True, parameter_composer= parameter_composer, circuit_info={"name": name, "n_qubits": n_qubits, "reps": reps}, verbose=False)
            print(f"Test done for {name} with n_qubits={n_qubits}, reps={reps}, t={t}. Frame potential: {F_p['frame_potential']}")