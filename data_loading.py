
import numpy as np
from sklearn.datasets import make_classification, make_moons
import matplotlib.pyplot as plt
from qiskit import algorithm_globals

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

