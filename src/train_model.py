""" Script to model train and predict"""
import json
import pickle
import warnings
import yaml
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    confusion_matrix,
    recall_score,
)
from sklearn.model_selection import StratifiedShuffleSplit

warnings.filterwarnings("ignore")


# # Functions


def model_result(model, x_train, y_train, x_test, y_test):
    """Function for training model. Return confusion matrix"""
    mod = model.fit(x_train, y_train)
    y_pred = mod.predict(x_test)

    conf_matrix = pd.DataFrame(
        confusion_matrix(y_test, y_pred), index=class_names, columns=class_names
    )
    conf_matrix["Total"] = (
        conf_matrix.negative
        + conf_matrix.comp_hypoth
        + conf_matrix["prim/sec hypothyroid"]
    )

    with open("model_name.pickle", "wb") as test_model:
        pickle.dump(model, test_model)


def model_result_2(y_test, y_pred):
    """Return confusion matrix"""
    conf_matrix = pd.DataFrame(
        confusion_matrix(y_test, y_pred), index=class_names, columns=class_names
    )
    conf_matrix["Total"] = (
        conf_matrix.negative
        + conf_matrix.comp_hypoth
        + conf_matrix["prim/sec hypothyroid"]
    )

    with open("metrics.json", "w") as outfile:
        json.dump(
            {"recall_macro": recall_score(y_test, y_pred, average="macro")}, outfile
        )
    return recall_score(y_test, y_pred, average="macro")


# Params

with open("./params.yaml", "rb") as fd:
    params = yaml.safe_load(fd)

random_state = params["random_state"]
n_splits = params["n_splits"]
test_size = params["test_size"]


class_names = ["comp_hypoth", "negative", "prim/sec hypothyroid"]

train_X_transform = pd.read_csv("./data/train_X_transform.csv", sep=";")
train_y_transform = pd.read_csv("./data/train_y_transform.csv", sep=";")
test_X_transform = pd.read_csv("./data/test_X_transform.csv", sep=";")
test_y_transform = pd.read_csv(
    "./data/test_y_transform.csv",
    sep=";",
)

cv = StratifiedShuffleSplit(
    n_splits=n_splits, test_size=test_size, random_state=random_state
)

log_reg_cv = LogisticRegressionCV(
    penalty="elasticnet",
    scoring="recall_macro",
    cv=cv,
    solver="saga",
    l1_ratios=np.linspace(0, 1, 10),
    random_state=random_state,
)

best_model = log_reg_cv.fit(train_X_transform, train_y_transform.Class)

with open("./best_model.pickle", "wb") as f:
    pickle.dump(best_model, f)

model_result_2(test_y_transform.Class, best_model.predict(test_X_transform))
