# Necessary imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qiskit_machine_learning.utils import algorithm_globals

# Set seed for random generators
algorithm_globals.random_seed = 38798451

from QNN_framework import Reuploading_classifier
from data_loading import Build_artitifical_data_set,load_csv_data,load_cancer_data

from qiskit_machine_learning.optimizers import COBYLA, L_BFGS_B, AQGD, ADAM


# read data using pandas
def dignose_of_circuit():
    X, y = load_cancer_data("../data/breast+cancer+wisconsin+diagnostic/wdbc.data")
    n_feature = X.shape[1]
    n_class = len(np.unique(y))

    