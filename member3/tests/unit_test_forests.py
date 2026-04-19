import sys
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
#Recently added to check for correct get push X#5
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

def get_data_and_model():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', '..', 'data', 'raw', 'ObesityDataSet_raw_and_data_sinthetic.csv')
    df = pd.read_csv(data_path)

    encoder = LabelEncoder()
    categorical_cols = ['Gender', 'family_history_with_overweight', 'FAVC',
                        'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
    for col in categorical_cols:
        df[col] = encoder.fit_transform(df[col])

    target_encoder = LabelEncoder()
    df['NObeyesdad'] = target_encoder.fit_transform(df['NObeyesdad'])

    X = df.drop(columns=['NObeyesdad'])
    y = df['NObeyesdad']

# same split as randomforest.py
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y  
    )

    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-2) 

    return rf, X, X_train, X_test, y_train, y_test, target_encoder

def test_model_achieves_minimum_acc():
    #random guessing on 7 class is 14%, model must do better
    rf, X, X_train, X_test, y_train, y_test, target_encoder = get_data_and_model()
    acc = accuracy_score(y_test, rf.predict(X_test))
    assert acc >= 0.7, f"Accuracy too low: {acc:.4f} (Need at least 70%)"

def test_train_acc_higher_then_test():
    #train accuracy should always be at last as high as test accuracy
    rf, X, X_train, X_test, y_train, y_test, target_encoder = get_data_and_model()
    train_acc = accuracy_score(y_train, rf.predict(X_train))
    test_acc = accuracy_score(y_test, rf.predict(X_test))
    assert train_acc >= test_acc - 0.01, f"Unexpected: test({test_acc:.4f}) > train({train_acc:.4f})"

def test_feature_importances_sumto1():
    #All importances added up should equal 100 percent
    rf, X, X_train, X_test, y_train, y_test, target_encoder = get_data_and_model()
    total = np.sum(rf.feature_importances_)
    assert round(total, 2) == 1.0, f"Importances sum to {round(total, 2)}, expected 1."

def test_prediction_output_shape():
    #model must produce exactly one prediction per test sample
    rf, X, X_train, X_test, y_train, y_test, target_encoder = get_data_and_model()
    predictions = rf.predict(X_test)
    assert len(predictions) == len(y_test), f"Expected {len(y_test)} predictions, got {len(predictions)} "

def test_predictions_on_new_people():
    # model must return valid obesity category names for new people
    rf, X, X_train, X_test, y_train, y_test, target_encoder = get_data_and_model()

    valid_categories = set(target_encoder.classes_) 

    # every row is a new person with different values
    test_people = [
        pd.DataFrame([[0, 20, 1.65, 55,  0, 0, 2, 3, 2, 0, 2, 0, 3, 1, 0, 3]], columns=X.columns),  
        pd.DataFrame([[1, 35, 1.75, 95,  1, 1, 3, 3, 2, 0, 2, 0, 1, 1, 1, 0]], columns=X.columns),  
        pd.DataFrame([[1, 45, 1.70, 130, 1, 1, 3, 3, 3, 0, 1, 0, 0, 2, 2, 0]], columns=X.columns),  
        pd.DataFrame([[0, 28, 1.68, 62,  0, 0, 2, 3, 1, 0, 3, 1, 3, 0, 0, 3]], columns=X.columns),  
        pd.DataFrame([[0, 40, 1.60, 110, 1, 1, 2, 3, 2, 0, 1, 0, 0, 1, 2, 1]], columns=X.columns),  
    ]

    for i, person in enumerate(test_people):
        prediction = rf.predict(person)  
        category = target_encoder.inverse_transform(prediction)  
        assert category[0] in valid_categories, (
            f"Person {i+1} returned invalid category: {category[0]}"  
        )

if __name__ == "__main__":
    tests = [
        test_train_acc_higher_then_test,
        test_model_achieves_minimum_acc,
        test_feature_importances_sumto1,
        test_prediction_output_shape,
        test_predictions_on_new_people,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            print(f"Passed: {test.__name__}")
            passed +=1
        except AssertionError as e:
            print(f"Failed: {test.__name__} - {e}")
            failed += 1
    
    print(f"\n{passed} passed, {failed} failed")