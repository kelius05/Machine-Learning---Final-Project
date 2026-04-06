import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from shared.preprocessing.preprocess import prepare_data


def test_logreg_prediction_length():
    data = prepare_data("data/raw/ObesityDataSet_raw_and_data_sinthetic.csv")

    model = LogisticRegression(max_iter=3000)
    model.fit(data["X_train"], data["y_train"])

    predictions = model.predict(data["X_test"])

    assert len(predictions) == len(data["y_test"])


def test_logreg_can_train_successfully():
    data = prepare_data("data/raw/ObesityDataSet_raw_and_data_sinthetic.csv")

    model = LogisticRegression(max_iter=3000)
    model.fit(data["X_train"], data["y_train"])

    assert model is not None


def test_logreg_accuracy_is_above_baseline():
    data = prepare_data("data/raw/ObesityDataSet_raw_and_data_sinthetic.csv")

    model = LogisticRegression(max_iter=3000)
    model.fit(data["X_train"], data["y_train"])

    predictions = model.predict(data["X_test"])
    accuracy = accuracy_score(data["y_test"], predictions)

    assert accuracy > 0.70