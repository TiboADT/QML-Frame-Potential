from qiskit_machine_learning.optimizers import L_BFGS_B
from qiskit.circuit.library import efficient_su2
from qiskit import QuantumCircuit

from scripts.rolling_in_the_depth import *
from scripts.who_let_the_circuit_out import *
from data_loading import Build_artitifical_data_set
from scripts.circuit_frame import *
import torch

from gates import get_gate_matrix
from frame_potential_gpu import get_device, sample_unitaries_gpu, recommended_batch_size, _get_vram_gb, sample_unitaries_cpu, to_gpu

import time

import argparse

if __name__ == "__main__" and False:

    circuit = build_ansatz(name="set", n_qubits=4, reps=1, number=3)

    print(circuit.draw())
    print(circuit.num_parameters)

    F_p = compute_frame_potential_gpu(circuit, t=2, converge_before_return=True, verbose=True)
    print(f"Frame potential: {F_p['frame_potential']:.6f}")

    F_p = compute_frame_potential_gpu(circuit, t=2, converge_before_return=True, verbose=True, device = torch.device("cpu"))
    print(f"Frame potential: {F_p['frame_potential']:.6f}")



if __name__ == "__main__" and False:

    N_QUBITS = 2
    circuit = efficient_su2(N_QUBITS, reps=2, entanglement='linear')
    device = get_device(False)
    v_ram_gb = _get_vram_gb(device)
    print(f"Recommended batch size for GPU sampling: {recommended_batch_size(N_QUBITS, vram_gb=v_ram_gb,dtype=torch.complex64)}")

    
    t0 = time.perf_counter()
    unitaries = sample_unitaries_gpu(circuit, 1, device=device, verbose = False)
    t1 = time.perf_counter()
    print(f"Time taken to sample 1000 unitaries on GPU: {t1-t0:.2f} seconds")
    shape = unitaries.shape
    print(f"Shape of sampled unitaries: {shape}")


    t0 = time.perf_counter()    
    unitaries_cpu = sample_unitaries_cpu(circuit, 1, verbose = False)
    unitaries_cpu = to_gpu(unitaries_cpu, device=device) # just to make sure the unitaries are on the same device for comparison
    t1 = time.perf_counter()
    print(f"Time taken to sample 1000 unitaries on CPU: {t1-t0:.2f} seconds")
    shape = unitaries_cpu.shape
    print(f"Shape of sampled unitaries: {shape}")

    # Compare the unitaries sampled on GPU and CPU
    difference = torch.norm(unitaries - unitaries_cpu)
    print(unitaries)
    print(unitaries_cpu)
    print(f"Difference between GPU and CPU sampled unitaries: {difference:.2e}")

if __name__ == "__main__" and False:
    # No arguments for now, but we can add some later to choose the architecture, the dataset, the target accuracy, etc.

    # parser = argparse.ArgumentParser(description="Entry point to the scripts on frame potential and QNNs")
    
    # parser.add_argument("--n", type=int, default=100, help="number of samples")
    # parser.add_argument("--t", type=int, default=2, help="design order")


    # args = parser.parse_args()

    n_features = 8
    n_classes = 2
    n_dataset = 3

    numbers = range(3,20) # numbers for the set ansatz to test
    for i in range(n_dataset):
        X,y = Build_artitifical_data_set(500, n_features=n_features, n_classes=n_classes, display=False, seed = 32143236+i*54678)
        print(f"Dataset {i} built with seed {32143236+i*54678}")
        who_let_the_circuit_out(embedding_reps=1,
                            n_feature=n_features, 
                            optimizer=COBYLA(maxiter=100,rhobeg=0.4), 
                            target_accuracy=0.9, max_depth=20, 
                            X=X, y=y,numbers = numbers)
        print(f"first optimizer done for dataset {i}")

        who_let_the_circuit_out(embedding_reps=1,
                            n_feature=n_features, 
                            optimizer=L_BFGS_B(maxiter=50), 
                            target_accuracy=0.9, max_depth=20, 
                            X=X, y=y,numbers = numbers)
        print(f"Dataset {i} done")
        print("--------------------------------------------------")




if __name__ == "__main__":
    # We can run the circuit frame evaluation for different ansatzes and different numbers of qubits, 
    # to see how the frame potential evolves with the depth of the circuit, 
    # and how it compares to the Haar value. 

    circuit_frame_evaluation(name = "set",n_qubits=4,converge=True,compose_parameters=False, range_t=[2])
    circuit_frame_evaluation(name = "perfectSU4",n_qubits=4,converge=True,compose_parameters=False, range_t=[2])

    # circuit_frame_evaluation(name = "perfectSU4",n_qubits=4,compose_parameters=True,n_samples=2000)




if False:

    n_features = 8
    n_classes = 2

    X,y = Build_artitifical_data_set(500, n_features=n_features, n_classes=n_classes, display=False, seed = 43256787564)

    rolling_in_the_depth(name_embeding="real_amp", name_anzats="real_amp", 
                         embedding_reps=1,
                         n_feature=n_features, 
                         optimizer=COBYLA(maxiter=100,rhobeg=0.4), 
                         target_accuracy=0.9, max_depth=20, 
                         X=X, y=y)

