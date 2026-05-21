from qiskit_machine_learning.optimizers import L_BFGS_B

from scripts.rolling_in_the_depth import *
from scripts.who_let_the_circuit_out import *
from data_loading import Build_artitifical_data_set
from scripts.circuit_frame import *

import argparse



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
    circuit_frame_evaluation(name = "perfectSU4",n_qubits=8,compose_parameters=True)




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

