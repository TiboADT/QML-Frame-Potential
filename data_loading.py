
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_moons
import matplotlib.pyplot as plt

def Build_artitifical_data_set(n_samples: int, n_features: int, n_classes: int, 
                               name: str = "linear",
                               seed : int = None,
                               display: bool = False,
                               *args, **kwargs) -> tuple[np.array, np.array]:
    """
    Build an artifical dataset for classification.
    """
    if seed is None:
        seed = np.random.randint(0, 10000)
    np.random.default_rng(seed)

    if name == "linear":
        # Generate random input coordinates (X) and binary labels (y)
        X = 2 * np.random.random([n_samples, n_features]) - 1
        #choose random linear coefficients for the linear decision boundary
        coefficients = 2 * np.random.random(n_features) - 1
        y01 = (np.dot(X, coefficients) > 0).astype(int)  # binary labels based on the linear decision boundary
        y = 2 * y01 - 1  # in {-1, +1}, y will be used for EstimatorQNN example
    elif name == "classification":
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_features,
            n_redundant=0,
            n_classes=n_classes,
            random_state=seed,
            *args, **kwargs
        )
    elif name == "moons":
        if "noise" not in kwargs:
            kwargs["noise"] = 0.1
        X, y = make_moons(n_samples=n_samples, 
                          random_state=seed,
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


def load_csv_data(path: str):
    """
    Load data from a csv file.
    """
    data = pd.read_csv(path, header=None)
    return data

def load_cancer_data(path: str):
    """
    Load breast cancer wisconsin diagnostic data from a csv file.
    """
    data = pd.read_csv(path, header=None)
    X = data.iloc[:, 2:].values
    y = data.iloc[:, 1].values
    y = np.where(y == "M", 1, -1) # convert to {-1, +1}
    return X, y