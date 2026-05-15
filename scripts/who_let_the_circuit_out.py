# The goal of this scipt is to test the required depth of the circuit needed to optain good results on the frame potential task. We will test the depth of the circuit needed to obtain good results on the classification task

from QNN_framework import *
from circuit_generation import build_ansatz
from heapq import heappush, heappop

from frame_potential_gpu import compute_frame_potential_gpu
from scripts.rolling_in_the_depth import rolling_in_the_depth


def who_let_the_circuit_out(X,y, embedding_reps: int, 
                         n_feature: int, optimizer, 
                         max_depth: int, target_accuracy: float = 0.9,
                         anzatz_type : str = "set",
                         **kwargs):
    """ Take an architecture and dataset in input and a target accuracy and a maximun depth to assure that the programme end.
    The model is trained with increasing depth until the target accuracy is reached or the maximum depth is reached.
    Return the depth at which the target accuracy is reached and the accuracy at each depth. """
    # the depth can be incressed by the number of reuploading, or embeding.

    if anzatz_type == "set":
        if "numbers" in kwargs:
            numbers = kwargs["numbers"]
        else:
            numbers = range(1,20)
        
        for i in numbers:
            print(f"Testing circuit with number {i}")
            rolling_in_the_depth(X,y,
                                 name_embeding="set", 
                                 embedding_reps=embedding_reps, 
                                 name_anzats="set", 
                                 n_feature=n_feature, 
                                 optimizer=optimizer, 
                                 max_depth=max_depth, 
                                 target_accuracy=target_accuracy, 
                                 compute_frame_potential=False,
                                 verbose=True,
                                 number = i)
    else:
        # other circuit type not supported yet
        raise NotImplementedError("Only set ansatz is supported for now.")
    


