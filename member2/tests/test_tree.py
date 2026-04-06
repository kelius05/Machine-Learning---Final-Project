import sys
import os
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))) # adds repo root so Python can find the shared folder

from shared.preprocessing.preprocess import prepare_data  # shared preprocessing — same as rest of group
from shared.evaluation.metrics import evaluate_model      # shared evaluation

def get_data_and_model():
    data = prepare_data("../../data/raw/ObesityDataSet_raw_and_data_sinthetic.csv")  # load and preprocess using shared function

    # this loop catches any columns that weren't encoded and fixes them
    for col in data["X_train"].columns:
        if not pd.api.types.is_numeric_dtype(data["X_train"][col]):  # if column is not a number, encode it
            le = LabelEncoder()
            data["X_train"][col] = le.fit_transform(data["X_train"][col].astype(str))
            data["X_test"][col]  = le.transform(data["X_test"][col].astype(str))
            data["X"][col]       = le.fit_transform(data["X"][col].astype(str))

    # unpack after the fix so all variables have the corrected encoded values
    X_train        = data["X_train"]
    X_test         = data["X_test"]
    y_train        = data["y_train"]
    y_test         = data["y_test"]
    target_encoder = data["target_encoder"]
    X              = data["X"]

    clf = DecisionTreeClassifier(max_depth=9, random_state=42)  # same model settings as train_tree.py
    clf.fit(X_train, y_train)  # train the model on training data

    return clf, X, X_train, X_test, y_train, y_test, target_encoder


def test_train_accuracy_higher_than_test():
    #Train accuracy should always be at least as high as test accuracy
    clf, X, X_train, X_test, y_train, y_test, target_encoder = get_data_and_model()
    
    train_acc = accuracy_score(clf.predict(X_train), y_train) #score on data the model has already seen
    test_acc = accuracy_score(clf.predict(X_test), y_test) #score on data the model hasn't seen

    #If test is higher than training, something is wrong
    assert train_acc >= test_acc - 0.01, (f"Unexpected: test({test_acc:.4f}) > train ({train_acc:.4f})")#0.01 tolerance handles any borderline cases

def test_model_achieves_minimum_accuracy():
    # Model should beat 70% random guessing on 7 classes is just 14%
    clf, X, X_train, X_test, y_train, y_test, target_encoder = get_data_and_model()
    acc = accuracy_score(clf.predict(X_test), y_test) #fraction of correct predictions on all test samples

    #70% is a good bar to set, this proves that the model actually learned real patterns
    assert acc >= 0.70, f"Accuracy is too low: {acc:.4f} (need at least 70%)"


def test_feature_importances_sumto1():
    clf, X, X_train, X_test, y_train, y_test, target_encoder = get_data_and_model()
    
    total = np.sum(clf.feature_importances_) #adds up all the importance values aross every feature

    assert round(total, 2) == 1.0, f"Importances sum to {round(total, 2)}, expected to be 1" #Round to 2 decimals before comparing


def test_prediction_output_shape():
    #Model must prduce exactly one prediction per test sample
    clf, X, X_train, X_test, y_train, y_test, target_encoder = get_data_and_model()

    predictions = clf.predict(X_test) #ask the model to predict the labels for all test rows

    #len() counts the items in a array - both counts much match
    assert len(predictions) == len(y_test), (
        f"Expected {len(y_test)} predictions, got {len(predictions)}"
    )


def test_predictions_on_new_people():
    #model must return valid obsesity category names for the new people
    clf, X, X_train, X_test, y_train, y_test, target_encoder = get_data_and_model()
    
    valid_categories = set(target_encoder.classes_) #the 7 known obesity category names

    #every row is a new person with different values
    test_people = [
        pd.DataFrame([[0, 20, 1.65, 55,  0, 0, 2, 3, 2, 0, 2, 0, 3, 1, 0, 3]], columns=X.columns),  # young, light, active
        pd.DataFrame([[1, 35, 1.75, 95,  1, 1, 3, 3, 2, 0, 2, 0, 1, 1, 1, 0]], columns=X.columns),  # middle aged, overweight
        pd.DataFrame([[1, 45, 1.70, 130, 1, 1, 3, 3, 3, 0, 1, 0, 0, 2, 2, 0]], columns=X.columns),  # sedentary, heavy
        pd.DataFrame([[0, 28, 1.68, 62,  0, 0, 2, 3, 1, 0, 3, 1, 3, 0, 0, 3]], columns=X.columns),  # very active, normal weight
        pd.DataFrame([[0, 40, 1.60, 110, 1, 1, 2, 3, 2, 0, 1, 0, 0, 1, 2, 1]], columns=X.columns),  # family history, high weight
    ]

    for i, person in enumerate(test_people):
        prediction = clf.predict(person)  #get the predicted encoded number
        category = target_encoder.inverse_transform(prediction) #decode back to obesity category name
        
        #Every prediction must be one of the 7 known categories
        assert category[0] in valid_categories, (
            f"Person {i+1} returned invalid category: {category[0]}"
        )
        
#Run directly without pytest
#this only runs when we run "python test_tree.py" directly
#when using pytest it is ignored - pytest finds and runs all the test functiions automatically
