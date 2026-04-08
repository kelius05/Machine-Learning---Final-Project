import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from sklearn.metrics import accuracy_score
from member1.model.train_logreg import build_logreg_grid


def test_logreg_pipeline_can_train():
    grid_search, X_train, X_test, y_train, y_test, target_encoder = build_logreg_grid(
        "data/raw/ObesityDataSet_raw_and_data_sinthetic.csv"
    )
    grid_search.fit(X_train, y_train)
    assert grid_search.best_estimator_ is not None


def test_logreg_prediction_length():
    grid_search, X_train, X_test, y_train, y_test, target_encoder = build_logreg_grid(
        "data/raw/ObesityDataSet_raw_and_data_sinthetic.csv"
    )
    grid_search.fit(X_train, y_train)
    predictions = grid_search.best_estimator_.predict(X_test)
    assert len(predictions) == len(y_test)


def test_logreg_accuracy_is_above_baseline():
    grid_search, X_train, X_test, y_train, y_test, target_encoder = build_logreg_grid(
        "data/raw/ObesityDataSet_raw_and_data_sinthetic.csv"
    )
    grid_search.fit(X_train, y_train)
    predictions = grid_search.best_estimator_.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    assert accuracy > 0.70
