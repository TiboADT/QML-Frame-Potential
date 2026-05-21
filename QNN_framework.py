import numpy as np

from sklearn.model_selection import train_test_split

from qiskit import QuantumCircuit
from qiskit_machine_learning.utils import algorithm_globals
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms import NeuralNetworkClassifier
from qiskit_machine_learning.optimizers import COBYLA
from datetime import datetime
import time


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
                 pre_anzats = False,
                 padding = 0.3,
                 **kwargs):
        
        # compute the necessary number of qubits for the circuit
        args_embeding["n_parameters"] = n_feature
        embeding_test = embedding_build(**args_embeding, **kwargs)
        n_qubits = embeding_test.num_qubits

        #build the circuit
        circuit = QuantumCircuit(n_qubits)
        weight_params = []

        #add the number of qubits in the arguments of the build functions
        args_embeding["n_qubits"] = n_qubits
        args_anzats["n_qubits"] = n_qubits


        if pre_anzats:
            args_anzats["parameter_prefix"] = "θ_initial"
            anzats_circuit = anzats_build(**args_anzats, **kwargs)
            weight_params.extend(anzats_circuit.parameters)
            circuit.compose(anzats_circuit, inplace=True)
            circuit.barrier()
        
        args_embeding["parameter_prefix"] = "x"
        embeding_circuit = embedding_build(**args_embeding, **kwargs)
        input_params = list(embeding_circuit.parameters)

        for i in range(reps+1):
            circuit.compose(embeding_circuit, inplace=True)

            circuit.barrier()

            args_anzats["parameter_prefix"] = f"θ_{i}"
            anzats_circuit = anzats_build(**args_anzats, **kwargs)
            weight_params.extend(anzats_circuit.parameters)
            circuit.compose(anzats_circuit, inplace=True)

            circuit.barrier()

        self.args_embeding = args_embeding
        self.args_anzats = args_anzats
        self.reps = reps
        self.padding = padding

        super().__init__(
            circuit=circuit,
            input_params=input_params,
            weight_params=weight_params,
            estimator=Estimator(),
        )
    
    def forward(self, input_data, weights):
        """Forward pass with parameters scaled by π for full rotation."""
        scaled_weights = np.array(weights) * np.pi
        padding = self.padding  # Add padding to ensure data points are not too close to the boundaries of the embedding space
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
        self.score_train = None
        self.score_test = None
        self.fiting_time = None
        

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
    
    
    def fit(self, X, y, verbose: bool = False):
        self.fiting_time = time.perf_counter()
        #check if the training results already exist for this dataset and model configuration, if they do, load them instead of training again
        if self.check_trained_data(dataset_data={"name": "dataset"}):
            if verbose:
                print("Training results already exist for this dataset and model configuration. Loading them instead of training again.")
            self.load_training(dataset_data={"name": "dataset"})
            return
        
        # check it the embeding circuit has more parameters than the input data 
        # if so padd the input data with zeros to match the number of parameters of the embeding circuit
        n_input_params = len(self.neural_network.input_params)
        n_features = X.shape[1]
        if n_input_params > n_features:
            if verbose:
                print(f"Padding input data with {n_input_params - n_features} zeros to match the number of parameters of the embeding circuit.")
            X = np.hstack((X, np.zeros((X.shape[0], n_input_params - n_features))))
        elif n_input_params < n_features:
            if verbose:
                print(f"Warning: The number of features in the input data ({n_features}) is greater than the number of parameters in the embeding circuit ({n_input_params}). The extra features will be ignored.")
            X = X[:, :n_input_params]

        # Separate the dataset into a training set and a test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=algorithm_globals.random_seed)
        # fit the model to the data
        super().fit(X_train, y_train)
        self.score_train = self.score(X_train, y_train)
        self.score_test = self.score(X_test, y_test)
        self.fiting_time = time.perf_counter() - self.fiting_time
    

    def score(self, X, y):
        n_input_params = len(self.neural_network.input_params)
        n_features = X.shape[1]
        if n_input_params > n_features:
            X = np.hstack((X, np.zeros((X.shape[0], n_input_params - n_features))))
        elif n_input_params < n_features:
            X = X[:, :n_input_params]

        # compute the score of the model on the given data
        return super().score(X,y)
    
    def init_save(self):
        # initialize the results folder
        init_results_folder()

    def save(self, dataset_data: dict = None, 
             path : str = "./data/results/classifier_results",
             verbose: bool = True, force_save: bool = False):
        # save the training results to a file

        model_info = self.model_info(dataset_data)

        result_info = {
            "seed" : algorithm_globals.random_seed,
            "optimizer": str(self.optimizer),
            "objective_func_vals": self.objective_func_vals,
            "training_score": self.score_train,
            "test_score": self.score_test,
            "number_of_samples": dataset_data["n_samples"] if dataset_data is not None else "None",
            "date" : datetime.now(),
            "trainging_time": self.fiting_time
        }

        result_data = {
            "objective_func_vals": self.objective_func_vals,
        }
        save_results(model_info=model_info, result_info=result_info, result_data=result_data, 
                     path=path,
                     verbose=verbose, force_save=force_save)

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
            "number_of_parameters": len(self._neural_network.weight_params),
            "dataset_name": dataset_data["name"] if dataset_data is not None else "None",
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