import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
import numpy as np

df = pd.read_csv("data/raw/ObesityDataSet_raw_and_data_sinthetic.csv")
X = df.drop(columns=["NObeyesdad"])
y = df["NObeyesdad"]

X['Gender'] = X['Gender'].map({"Female": 0, "Male": 1})
X['family_history_with_overweight'] = X['family_history_with_overweight'].map({"no": 0, "yes": 1})
X['FAVC'] = X['FAVC'].map({"no": 0, "yes": 1})
X['CAEC'] = X['CAEC'].map({"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3})
X['SMOKE'] = X['SMOKE'].map({"no": 0, "yes": 1})
X['SCC'] = X['SCC'].map({"no": 0, "yes": 1})
X['CALC'] = X['CALC'].map({"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3})
X['MTRANS'] = X['MTRANS'].map({"Automobile": 0, "Motorbike": 1, "Bike": 2, "Public_Transportation": 3, "Walking": 4})

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

def test_knn_pipeline_shape():
    pipeline_knn = Pipeline([
        ("scaler", StandardScaler()),
        ("selector", SelectKBest(score_func=f_classif)),
        ("classifier", KNeighborsClassifier())
    ])

    param_grid_knn = {
        "selector__k": [4, 8, 12, 16],
        "classifier__n_neighbors": list(range(1, 21)),
        "classifier__weights": ["uniform", "distance"],
        "classifier__metric": ["euclidean", "manhattan"]
    }

    grid_knn = GridSearchCV(
        pipeline_knn,
        param_grid_knn,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    grid_knn.fit(X_train, y_train)
    
    grid_knn.fit(X_train, y_train)
    y_pred = grid_knn.predict(X_test)
    
    assert y_pred.shape == y_test.shape

def test_gradient_boosting_accuracy():
    pipeline_gb = Pipeline([
        ("scaler", StandardScaler()),
        ("selector", SelectKBest(score_func=f_classif)),
        ("classifier", GradientBoostingClassifier())
    ])

    param_grid_gb = {
        "selector__k": [ 16],
        "classifier__n_estimators": [200],
        "classifier__learning_rate": [ 0.1],
        "classifier__max_depth": [ 5]
    }

    grid_gb = GridSearchCV(
        pipeline_gb,
        param_grid_gb,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    grid_gb.fit(X_train, y_train)
    y_pred = grid_gb.predict(X_test)
    
    accuracy = np.mean(y_pred == y_test)
    
    assert accuracy > 0.9

# in terminal, run the tests with:
# pytest test_pipeline_knn.py

