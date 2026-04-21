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

df = pd.read_csv("data/raw/ObesityDataSet_raw_and_data_sinthetic.csv")
X = df.drop(columns=["NObeyesdad"])
y = df["NObeyesdad"]

print(f"\n{df['CAEC'].value_counts().to_string()}")
print(f"\n{df['CALC'].value_counts().to_string()}")
print(f"\n{df['MTRANS'].value_counts().to_string()}")

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

print("Classes:", le.classes_)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

print("Training shape:", X_train.shape)
print("Testing shape: ", X_test.shape)

# gradient boosting pipeline
pipeline_gb = Pipeline([
    ("scaler", StandardScaler()),
    ("selector", SelectKBest(score_func=f_classif)),
    ("classifier", GradientBoostingClassifier())
])

param_grid_gb = {
    "selector__k": [4, 8, 12, 16],
    "classifier__n_estimators": [100, 200],
    "classifier__learning_rate": [0.01, 0.1],
    "classifier__max_depth": [3, 5]
}

grid_gb = GridSearchCV(
    pipeline_gb,    
    param_grid_gb,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)
grid_gb.fit(X_train, y_train)

print("Best Parameters (GB):", grid_gb.best_params_)
print("Best CV Accuracy (GB):", f"{grid_gb.best_score_:.4f}")

best_model_gb = grid_gb.best_estimator_

print(f"Gradient Boosting Training Accuracy: {best_model_gb.score(X_train, y_train):.4f}")
print(f"Gradient Boosting Test Accuracy:     {best_model_gb.score(X_test, y_test):.4f}")

# classification report
best_model_gb = grid_gb.best_estimator_
y_pred_gb = best_model_gb.predict(X_test)
print(classification_report(y_test, y_pred_gb, target_names=le.classes_))

# confusion matrix
y_pred_gb = best_model_gb.predict(X_test)
cm = confusion_matrix(y_test, y_pred_gb)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(xticks_rotation=45)   
plt.title("Confusion Matrix - Gradient Boosting")
plt.show()