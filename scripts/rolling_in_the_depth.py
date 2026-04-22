# The goal of this scipt is to test the required depth of the circuit needed to optain good results on the frame potential task. We will test the depth of the circuit needed to obtain good results on the classification task

from QNN_framework import *
from circuit_generation import build_ansatz
from heapq import heappush, heappop

def test_depth(name_anzats, rep, n_parameters):
    """Test the depth of the circuit needed to obtain good results on the frame potential task."""
    args_anzats= {"name": name_anzats, "reps": rep, "n_parameters": n_parameters}
    anzats_circuit = build_ansatz(**args_anzats)
    depth = anzats_circuit.depth()
    return depth

def rolling_in_the_depth(X,y,name_embeding: str, embedding_reps: int, name_anzats: str,
                         n_feature: int, optimizer, 
                         max_depth: int, target_accuracy: float = 0.9):
    """ Take an architecture and dataset in input and a target accuracy and a maximun depth to assure that the programme end.
    The model is trained with increasing depth until the target accuracy is reached or the maximum depth is reached.
    Return the depth at which the target accuracy is reached and the accuracy at each depth. """
    # the depth can be incressed by the number of reuploading, or embeding.

    anzats_reps = 0
    pre_anzats = False
    reps = 0

    embedding_depth = test_depth(name_embeding, embedding_reps, n_feature)
    anzats_depth = test_depth(name_anzats, anzats_reps, n_feature)

    circuit_done = {}
    circuit_to_do = []
    best_accuracy = 0

    depth = (embedding_depth * embedding_reps + anzats_depth) * (reps + 1) + pre_anzats * embedding_depth * embedding_reps
    heappush(circuit_to_do, (depth, (reps, anzats_reps, pre_anzats)))

    while circuit_to_do[0][0] <= max_depth and best_accuracy < target_accuracy:
        depth, (reps, anzats_reps, pre_anzats) = heappop(circuit_to_do)
        args_embeding= {"name": name_embeding, "reps": embedding_reps}
        args_anzats= {"name": name_anzats, "reps": anzats_reps}

        # construct neural network classifier
        estimator_classifier_linear = Reuploading_classifier(
            n_feature=n_feature,
            n_class=2,
            qnn_args=dict(
                reps=reps,
                anzats_build=build_ansatz,
                args_embeding=args_embeding,
                args_anzats=args_anzats,
                pre_anzats=pre_anzats
            ),
            optimizer=optimizer,
        )
        
        estimator_classifier_linear.fit(X, y)

        accuracy = estimator_classifier_linear.score(X, y)
        circuit_done[(reps, anzats_reps, pre_anzats)] = accuracy
        best_accuracy = max(best_accuracy, accuracy)

        # add the next circuits to do in the heap
        if (reps + 1, anzats_reps, pre_anzats) not in circuit_done:
            depth = (embedding_depth * embedding_reps + anzats_depth) * (reps + 1) + pre_anzats * embedding_depth * embedding_reps
            heappush(circuit_to_do, (depth, (reps + 1, anzats_reps, pre_anzats)))
        if (reps, anzats_reps + 1, pre_anzats) not in circuit_done:
            depth = (embedding_depth * embedding_reps + anzats_depth) * (reps + 1) + pre_anzats * embedding_depth * embedding_reps
            heappush(circuit_to_do, (depth, (reps, anzats_reps + 1, pre_anzats)))
        if not pre_anzats and (reps, anzats_reps, True) not in circuit_done:
            depth = (embedding_depth * embedding_reps + anzats_depth) * (reps + 1) + embedding_depth * embedding_reps
            heappush(circuit_to_do, (depth, (reps, anzats_reps, True)))
