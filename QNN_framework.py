import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification, make_moons
from sklearn.model_selection import train_test_split


from qiskit import QuantumCircuit
from qiskit_machine_learning.utils import algorithm_globals
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms import NeuralNetworkClassifier
from qiskit_machine_learning.optimizers import COBYLA
from datetime import datetime


# Set seed for random generators
# algorithm_globals.random_seed = 654213873



from circuit_generation import build_ansatz


from save_read_results import save_results, read_results, init_results_folder, dict_to_id, check_exisiting_results


class QNN_reuploading(EstimatorQNN):
    def __init__(self, n_feature, n_class,
                 embedding_build = build_ansatz, 
                 anzats_build = build_ansatz,  
                 reps=0, 
                 args_embeding= {"name": "real_amp", "circular": True}, 
                 args_anzats= {"name": "two_local_rx", "circular": True},
                 pre_anzats = False,):
        
        # compute the necessary number of qubits for the circuit
        args_embeding["n_parameters"] = n_feature
        embeding_test = embedding_build(**args_embeding)
        n_qubits = embeding_test.num_qubits

        #build the circuit
        circuit = QuantumCircuit(n_qubits)
        weight_params = []

        #add the number of qubits in the arguments of the build functions
        args_embeding["n_qubits"] = n_qubits
        args_anzats["n_qubits"] = n_qubits


        if pre_anzats:
            args_anzats["parameter_prefix"] = "θ_initial"
            anzats_circuit = anzats_build(**args_anzats)
            weight_params.extend(anzats_circuit.parameters)
            circuit.compose(anzats_circuit, inplace=True)
            circuit.barrier()
        
        args_embeding["parameter_prefix"] = "x"
        embeding_circuit = embedding_build(**args_embeding)
        input_params = list(embeding_circuit.parameters)

        for i in range(reps+1):
            circuit.compose(embeding_circuit, inplace=True)

            circuit.barrier()

            args_anzats["parameter_prefix"] = f"θ_{i}"
            anzats_circuit = anzats_build(**args_anzats)
            weight_params.extend(anzats_circuit.parameters)
            circuit.compose(anzats_circuit, inplace=True)

            circuit.barrier()

        self.args_embeding = args_embeding
        self.args_anzats = args_anzats
        self.reps = reps

        super().__init__(
            circuit=circuit,
            input_params=input_params,
            weight_params=weight_params,
            estimator=Estimator(),
        )
    
    def forward(self, input_data, weights):
        """Forward pass with parameters scaled by π for full rotation."""
        scaled_weights = np.array(weights) * np.pi
        padding = 0.5  # Add padding to ensure we cover the full range of angles
        input_data = np.array(input_data) * np.pi * (1 - padding)  # Scale input data as well for better embedding
        return super().forward(input_data, scaled_weights)


class Reuploading_classifier(NeuralNetworkClassifier):
    """
    A neural network classifier using a reuploading quantum neural network (QNN) as the underlying model.
    """

    def __init__(self, n_feature : int, n_class : int, 
                 optimizer = None, iteration : int = None, 
                 **kwargs):
        # pass optional qnn_args to the QNN_reuploading constructor
        self.qnn_args = kwargs.pop("qnn_args", {})
        qnn = QNN_reuploading(n_feature, n_class, **self.qnn_args)

        if optimizer is not None:
            if iteration is not None:
                optimizer.set_options(maxiter=iteration)
        else:
            # default optimizer is COBYLA with maxiter=60 and rhobeg=0.4
            # it is a fast optimizer, but i wont give as good other optimizers like SPSA or ADAM that take more time to converge
            optimizer = COBYLA(maxiter=60,rhobeg=0.4)

        self.objective_func_vals = []
        self.training_weights = []
        self.score_train = None
        self.score_test = None
        

        super().__init__(
            qnn,
            optimizer=optimizer,
            callback=self.callback_method,
            **kwargs,
        )
    
    def callback_method(self, weights, obj_func_eval):
        # I am not quite sure this is the right way to do it, 
        # but I want to save the objective function values and the weights during training to be able to plot/save them later
        # TODO look up how this should be done in the qiskit documentation, maybe there is a better way to do it
        self.objective_func_vals.append(obj_func_eval)
        self.training_weights.append(np.array(weights, copy=True))
    
    
    def fit(self, X, y, verbose: bool = False):
        #check if the training results already exist for this dataset and model configuration, if they do, load them instead of training again
        if self.check_trained_data(dataset_data={"name": "dataset"}):
            if verbose:
                print("Training results already exist for this dataset and model configuration. Loading them instead of training again.")
            self.load_training(dataset_data={"name": "dataset"})
            return
        # Separate the dataset into a training set and a test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=algorithm_globals.random_seed)
        # fit the model to the data
        super().fit(X_train, y_train)
        self.score_train = self.score(X_train, y_train)
        self.score_test = self.score(X_test, y_test)
    
    def init_save(self):
        # initialize the results folder
        init_results_folder()

    def save_training(self, dataset_data: dict = None, verbose: bool = True, force_save: bool = False):
        # save the training results to a file

        model_info = self.model_info(dataset_data)

        results_info = {
            "seed" : algorithm_globals.random_seed,
            "optimizer": str(self.optimizer),
            "objective_func_vals": self.objective_func_vals,
            "training_score": self.score_train,
            "test_score": self.score_test,
            "number_of_samples": len(self._X),
            "python_version": algorithm_globals.python_version,
            "qiskit_version": algorithm_globals.qiskit_version,
        }

        results_data = {
            "objective_func_vals": self.objective_func_vals,
            "training_weights": self.training_weights,
            "dataset_X": self._X,
            "dataset_y": self._y,
            "date" : datetime.now()
        }
        save_results(model_info=model_info, results_info=results_info, results_data=results_data, verbose=verbose, force_save=force_save)

    def model_info(self, dataset_data: dict = None) -> dict:
        args_embeding = self._neural_network.args_embeding
        args_anzats = self._neural_network.args_anzats

        model_info = {
            "embeding": args_embeding["name"],
            "embeding_reps": args_embeding["reps"],
            "anzats": args_anzats["name"],
            "anzats_reps": args_anzats["reps"],
            "layer_reps" : self._neural_network.reps,
            "number_of_qubits": self._neural_network.circuit.num_qubits,
            "dataset_name": dataset_data["name"]
        }
        return model_info

    def result_id(self,dataset_data: dict = None) -> str:
        # save the training results to a file

        model_info = self.model_info(dataset_data)
        
        return dict_to_id(model_info)

    def load_training(self, dataset_data: dict = None) -> tuple[dict, dict]:
        result_id = self.result_id(dataset_data)
        # load the training results from a file
        results_info, results_data = read_results(result_id)
        self.objective_func_vals = results_info["objective_func_vals"]
        self.training_weights = results_data["training_weights"]
        return results_info, results_data
    
    def check_trained_data(self, dataset_data: dict = None) -> bool:
        result_id = self.result_id(dataset_data)
        # check if the training results exist
        return check_exisiting_results(result_id=result_id)




def Build_artitifical_data_set(n_samples: int, n_features: int, n_classes: int, 
                               name: str = "linear",
                               seed : int = None,
                               display: bool = False,
                               *args, **kwargs) -> tuple[np.array, np.array]:
    """
    Build an artifical dataset for classification.
    """
    if seed is not None:
        algorithm_globals.random_seed = seed

    if name == "linear":
        # Generate random input coordinates (X) and binary labels (y)
        X = 2 * algorithm_globals.random.random([n_samples, n_features]) - 1
        #choose random linear coefficients for the linear decision boundary
        coefficients = 2 * algorithm_globals.random.random(n_features) - 1
        y01 = (np.dot(X, coefficients) > 0).astype(int)  # binary labels based on the linear decision boundary
        y = 2 * y01 - 1  # in {-1, +1}, y will be used for EstimatorQNN example
    elif name == "classification":
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_features,
            n_redundant=0,
            n_classes=n_classes,
            random_state=algorithm_globals.random_seed,
            *args, **kwargs
        )
    elif name == "moons":
        if "noise" not in kwargs:
            kwargs["noise"] = 0.1
        X, y = make_moons(n_samples=n_samples, 
                          random_state=algorithm_globals.random_seed,
                          **kwargs)
        y = np.array(y)
        y = 2*y - 1 # convert to {-1, +1}
    
    else:
        raise ValueError(f"Unknown dataset name: {name}")
    
    X = np.array(X)
    min = np.min(X, axis=0)
    max = np.max(X, axis=0)
    X = 2 * (X - min) / (max - min) - 1

    if display:
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm")
        plt.title(f"Dataset: {name}")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.show()
    
    return X, y

