import sys
import os
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))  # adds repo root to path so Python can find the shared folder


from shared.evaluation.metrics import evaluate_model      # shared evaluation

def get_data_and_model():
    df = pd.read_csv("../../data/raw/ObesityDataSet_raw_and_data_sinthetic.csv")  # load the dataset

    encoder = LabelEncoder()  # LabelEncoder converts text columns to numbers so the model can use them
    categorical_cols = ['Gender', 'family_history_with_overweight', 'FAVC',
                        'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']  # all the text columns that need converting

    for col in categorical_cols:
        df[col] = encoder.fit_transform(df[col])  # replace each text column with numbers e.g. Female=0, Male=1

    target_encoder = LabelEncoder()  # separate encoder for the target so we can decode predictions back to names later
    df['NObeyesdad'] = target_encoder.fit_transform(df['NObeyesdad'])  # encode the 7 obesity categories to numbers 0-6

    X = df.drop(columns=['NObeyesdad'])  # all input features except the one we want to predict
    y = df['NObeyesdad']                  # the target label we want to predict

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y  # 80% training, 20% testing — same split as train_tree.py
    )

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


def test_feature_importances_sum_to_1():
    clf, X, X_train, X_test, y_train, y_test, target_encoder = get_data_and_model()
    
    total = np.sum(clf.feature_importances_) #adds up all the importance values across every feature

    assert round(total, 2) == 1.0, f"Importances sum to {round(total, 2)}, expected to be 1" #Round to 2 decimals before comparing


def test_prediction_output_shape():
    #Model must produce exactly one prediction per test sample
    clf, X, X_train, X_test, y_train, y_test, target_encoder = get_data_and_model()

    predictions = clf.predict(X_test) #ask the model to predict the labels for all test rows

    #len() counts the items in an array - both counts must match
    assert len(predictions) == len(y_test), (
        f"Expected {len(y_test)} predictions, got {len(predictions)}"
    )


def test_predictions_on_new_people():
    #model must return valid obesity category names for the new people
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
#when using pytest it is ignored - pytest finds and runs all the test functions automatically
if __name__ == "__main__":
    tests = [
        test_train_accuracy_higher_than_test,
        test_model_achieves_minimum_accuracy,
        test_feature_importances_sum_to_1,
        test_prediction_output_shape,
        test_predictions_on_new_people,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            print(f"  PASSED: {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAILED: {test.__name__} — {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed")