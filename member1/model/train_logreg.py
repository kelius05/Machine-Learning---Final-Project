import os
import sys
import time
import tracemalloc

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression

from shared.preprocessing.preprocess import load_data, split_features_target, get_feature_types
from shared.evaluation.metrics import evaluate_model, print_evaluation_results


def build_logreg_grid(file_path: str):
    # Build the full Logistic Regression workflow: raw data -> split -> preprocessing pipeline -> GridSearchCV
    
    df = load_data(file_path)
    X, y = split_features_target(df, target_column="NObeyesdad")

    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)

    numeric_features, categorical_features = get_feature_types(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=3000))
    ])

    param_grid = {
        "model__C": [0.1, 1.0, 10.0],
        "model__solver": ["lbfgs"]
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    return grid_search, X_train, X_test, y_train, y_test, target_encoder


def main():
    file_path = "data/raw/ObesityDataSet_raw_and_data_sinthetic.csv"

    print("Building Logistic Regression pipeline and GridSearchCV...")
    grid_search, X_train, X_test, y_train, y_test, target_encoder = build_logreg_grid(file_path)

    print("Training the pipeline...")
    start_train = time.time()
    grid_search.fit(X_train, y_train)
    train_time = time.time() - start_train

    print("\nBest Parameters:", grid_search.best_params_)
    print("Best Cross-Validation Score:", round(grid_search.best_score_, 4))
    print(f"Training time: {train_time:.4f} seconds")

    best_model = grid_search.best_estimator_

    start_pred = time.time()
    y_pred = best_model.predict(X_test)
    pred_time = time.time() - start_pred
    print(f"Prediction time: {pred_time:.4f} seconds")

    tracemalloc.start()
    mem_grid_search, mem_X_train, mem_X_test, mem_y_train, mem_y_test, _ = build_logreg_grid(file_path)
    mem_grid_search.fit(mem_X_train, mem_y_train)
    _, peak_train_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    mem_best_model = mem_grid_search.best_estimator_
    tracemalloc.start()
    _ = mem_best_model.predict(mem_X_test)
    _, peak_pred_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Peak memory usage during training: {peak_train_mem / 1024:.2f} KB")
    print(f"Peak memory usage during prediction: {peak_pred_mem / 1024:.2f} KB")

    print("\nEvaluating best model on test set...")
    results = evaluate_model(best_model, X_test, y_test)
    results["y_pred"] = y_pred
    print_evaluation_results(results)

    print(f"\nTest Accuracy: {results['accuracy']:.4f}")
    print("First 10 predictions:")
    print(results["y_pred"][:10])

    print("\nThis Logistic Regression pipeline is used as the baseline model for Member 1.")


if __name__ == "__main__":
    main()
