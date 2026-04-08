import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from sklearn.metrics import accuracy_score
from member1.model.train_logreg import build_logreg_grid


# Test that the Logistic Regression pipeline can be built and trained successfully
# without failing during GridSearchCV, which confirms the workflow is valid end-to-end.
def test_logreg_pipeline_can_train():
    grid_search, X_train, X_test, y_train, y_test, target_encoder = build_logreg_grid(
        "data/raw/ObesityDataSet_raw_and_data_sinthetic.csv"
    )
    grid_search.fit(X_train, y_train)
    assert grid_search.best_estimator_ is not None


# Test that the prediction output has the same number of items as the test labels,
# which confirms the model returns one prediction for every test sample.
def test_logreg_prediction_length():
    grid_search, X_train, X_test, y_train, y_test, target_encoder = build_logreg_grid(
        "data/raw/ObesityDataSet_raw_and_data_sinthetic.csv"
    )
    grid_search.fit(X_train, y_train)
    predictions = grid_search.best_estimator_.predict(X_test)
    assert len(predictions) == len(y_test)


# Test that the Logistic Regression model reaches an accuracy above a simple baseline threshold,
# which helps confirm the trained model is learning useful patterns from the dataset.
def test_logreg_accuracy_is_above_baseline():
    grid_search, X_train, X_test, y_train, y_test, target_encoder = build_logreg_grid(
        "data/raw/ObesityDataSet_raw_and_data_sinthetic.csv"
    )
    grid_search.fit(X_train, y_train)
    predictions = grid_search.best_estimator_.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    assert accuracy > 0.70
