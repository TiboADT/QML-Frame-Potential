from scripts.rolling_in_the_depth import *
from data_loading import Build_artitifical_data_set

import argparse



if __name__ == "__main__":
    # No arguments for now, but we can add some later to choose the architecture, the dataset, the target accuracy, etc.

    # parser = argparse.ArgumentParser(description="Entry point to the scripts on frame potential and QNNs")
    
    # parser.add_argument("--n", type=int, default=100, help="number of samples")
    # parser.add_argument("--t", type=int, default=2, help="design order")


    # args = parser.parse_args()
    n_features = 4
    n_classes = 2

    X,y = Build_artitifical_data_set(500, n_features=n_features, n_classes=n_classes, display=False)

    rolling_in_the_depth(name_embeding="real_amp", name_anzats="real_amp", 
                         embedding_reps=1,
                         n_feature=n_features, 
                         optimizer=COBYLA(maxiter=100,rhobeg=0.4), 
                         target_accuracy=0.9, max_depth=100, 
                         X=X, y=y)

