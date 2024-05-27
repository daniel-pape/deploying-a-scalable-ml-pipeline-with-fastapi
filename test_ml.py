from pathlib import Path

import pytest
import os
import pandas as pd
import numpy as np
from config import project_path
from ml.model import load_model, inference, compute_model_metrics
from sklearn.model_selection import train_test_split

from train_model import get_census_data, get_train_test_sets


# TODO: add necessary import


# TODO: implement the first test. Change the function name and input as needed
def test_one():
    """
    # add description for the first test
    """
    # Your code here
    pass


# TODO: implement the second test. Change the function name and input as needed
def test_two():
    """
    # add description for the second test
    """
    # Your code here
    pass


def test_compute_model_metrics():
    """
    Test that the loaded model has the expected
    precision, recall and f1 score (as observed during training).
    """
    model_path = os.path.join(project_path, "model", "model.pkl")
    model = load_model(model_path)

    X_test = np.load(os.path.join(project_path, "data", "X_test.npy"))
    y_test = np.load(os.path.join(project_path, "data", "y_test.npy"))

    preds = inference(model, X_test)

    precision, recall, f1 = compute_model_metrics(y_test, preds)

    assert round(precision, 4) == 0.8030
    assert round(recall, 4) == 0.6225
    assert round(f1, 4) == 0.7013
