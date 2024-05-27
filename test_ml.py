import os

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from config import project_path
from ml.model import load_model, inference, compute_model_metrics, train_model


def test_trained_model_type():
    """
    # add description for the first test
    """
    X_test = np.load(os.path.join(project_path, "data", "X_test.npy"))
    y_test = np.load(os.path.join(project_path, "data", "y_test.npy"))
    model = train_model(X_test, y_test)

    assert isinstance(model, GradientBoostingClassifier)
    assert model.random_state == 42


def test_loaded_model_type():
    """
    Test that the loaded model is a GB classifier
    with the expected random_state.
    """
    model_path = os.path.join(project_path, "model", "model.pkl")
    model = load_model(model_path)

    assert isinstance(model, GradientBoostingClassifier)
    assert model.random_state == 42


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
