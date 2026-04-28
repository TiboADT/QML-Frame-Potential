
from QNN_framework import *
from data_loading import Build_artitifical_data_set
from itertools import product

Name = ["real_amp", 
        "two_local_rx",
        "ghz_like",
        "brickwall"]

Reps = [1, 2, 3, 4, 5]

n_Features = [2, 4, 6, 8]

Preanzats = [True, False]

Optimizers = [COBYLA(maxiter=100,rhobeg=0.4)]

print("Starting training...")
print(" ------------------------------------------------------")
Variables = ["name_embeding", "name_anzats", "n_feature", "pre_anzats", "optimizer", "embedding_reps", "anzats_reps", "rep"]
tab = " | "
# make sur all the variables are printed with the same space
for var in Variables:
    print(f"{var:14}", end=tab)
print()
print(" ------------------------------------------------------")

for (
    name_embeding,
    name_anzats,
    n_feature,
    pre_anzats,
    optimizer,
    embedding_reps,
    anzats_reps,
    rep,
) in product(Name, Name, n_Features, Preanzats, Optimizers, Reps, Reps, Reps):
    # make sur all the variables are printed with the same space
    print(f" {name_embeding:14} | {name_anzats:14} | {n_feature:14} | {pre_anzats:14} | {optimizer.__class__.__name__:14} | {embedding_reps:14} | {anzats_reps:14} | {rep:14}")
    args_embeding= {"name": name_embeding, "reps": embedding_reps}
    args_anzats= {"name": name_anzats, "reps": anzats_reps}

    X,y = Build_artitifical_data_set(500, n_features=n_feature, n_classes=2, display=False)
    # construct neural network classifier
    estimator_classifier_linear = Reuploading_classifier(
        n_feature=n_feature,
        n_class=2,
        qnn_args=dict(
            reps=rep,
            anzats_build=build_ansatz,
            args_embeding=args_embeding,
            args_anzats=args_anzats,
            pre_anzats=pre_anzats
        ),
        optimizer=COBYLA(maxiter=100,rhobeg=0.4),
    )

    estimator_classifier_linear.fit(X, y)

    data_dict = {
        "name" : "artificial_dataset",
        "n_samples": len(X),
    }
    estimator_classifier_linear.save(dataset_data=data_dict, verbose=False)