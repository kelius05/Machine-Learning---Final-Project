import os
import sys
import time
import tracemalloc

# Add the project root to Python's import path so this file can use the shared project modules.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression

from shared.preprocessing.preprocess import load_data, split_features_target, get_feature_types
from shared.evaluation.metrics import evaluate_model, print_evaluation_results


# Build the complete Logistic Regression workflow from raw data to a tunable model.
def build_logreg_grid(file_path: str):
    # Load the CSV file into a DataFrame.
    df = load_data(file_path)
    # Separate the DataFrame into input features and the target label column.
    X, y = split_features_target(df, target_column="NObeyesdad")

    # Create a label encoder for the target classes because the model needs numeric targets.
    target_encoder = LabelEncoder()
    # Convert text labels such as obesity categories into integers.
    y_encoded = target_encoder.fit_transform(y)

    # Detect which columns are numeric and which are categorical.
    numeric_features, categorical_features = get_feature_types(X)

    # Split the data into training and test sets while preserving class balance.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    # Create a preprocessing object that applies different transformations by column type.
    preprocessor = ColumnTransformer(
        transformers=[
            # Standardize numeric features so they are centered and scaled.
            ("num", StandardScaler(), numeric_features),
            # One-hot encode categorical features so Logistic Regression can use them.
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    # Build a pipeline so preprocessing and model fitting happen as one connected workflow.
    pipeline = Pipeline([
        # First transform the raw columns into model-ready numeric features.
        ("preprocessor", preprocessor),
        # Then fit the Logistic Regression classifier.
        ("model", LogisticRegression(max_iter=3000))
    ])

    # Define the hyperparameters we want GridSearchCV to test.
    param_grid = {
        # Compare several values of `C`, which controls regularization strength.
        "model__C": [0.1, 1.0, 10.0],
        # Use the `lbfgs` solver, which works well for multinomial Logistic Regression.
        "model__solver": ["lbfgs"]
    }

    # Create a grid search object that will train and compare several pipeline variants.
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    # Return the configured grid search object and the train/test data used later in the script.
    return grid_search, X_train, X_test, y_train, y_test, target_encoder


# Run the full Member 1 Logistic Regression experiment.
def main():
    # Store the dataset path relative to the project root.
    file_path = "data/raw/ObesityDataSet_raw_and_data_sinthetic.csv"

    # Announce that the training pipeline is being prepared.
    print("Building Logistic Regression pipeline and GridSearchCV...")
    # Build the model pipeline and prepare the train/test splits.
    grid_search, X_train, X_test, y_train, y_test, target_encoder = build_logreg_grid(file_path)

    # Announce that model training is about to begin.
    print("Training the pipeline...")
    # Record the time right before training starts.
    start_train = time.time()
    # Fit the grid search object so it trains and tunes the Logistic Regression pipeline.
    grid_search.fit(X_train, y_train)
    # Compute total training runtime.
    train_time = time.time() - start_train

    # Show the hyperparameter combination that achieved the best validation score.
    print("\nBest Parameters:", grid_search.best_params_)
    # Show the best average cross-validation accuracy found by the search.
    print("Best Cross-Validation Score:", round(grid_search.best_score_, 4))
    # Show how long the full training and tuning stage took.
    print(f"Training time: {train_time:.4f} seconds")

    # Extract the best trained pipeline from GridSearchCV.
    best_model = grid_search.best_estimator_

    # Record the time right before inference starts.
    start_pred = time.time()
    # Generate predictions on the held-out test set.
    y_pred = best_model.predict(X_test)
    # Compute the total prediction runtime.
    pred_time = time.time() - start_pred
    # Print the prediction runtime.
    print(f"Prediction time: {pred_time:.4f} seconds")

    # Start tracking memory usage for a clean training pass.
    tracemalloc.start()
    # Rebuild the workflow so the memory measurement covers a fresh training run.
    mem_grid_search, mem_X_train, mem_X_test, mem_y_train, mem_y_test, _ = build_logreg_grid(file_path)
    # Fit the fresh grid search object to capture peak training memory usage.
    mem_grid_search.fit(mem_X_train, mem_y_train)
    # Read the current and peak tracked memory values and keep the peak value only.
    _, peak_train_mem = tracemalloc.get_traced_memory()
    # Stop tracking memory for the training stage.
    tracemalloc.stop()

    # Extract the best model from the memory-measurement training run.
    mem_best_model = mem_grid_search.best_estimator_
    # Start tracking memory usage for the prediction stage.
    tracemalloc.start()
    # Run prediction so we can measure peak memory used during inference.
    _ = mem_best_model.predict(mem_X_test)
    # Read the current and peak tracked memory values and keep the peak value only.
    _, peak_pred_mem = tracemalloc.get_traced_memory()
    # Stop tracking memory for the prediction stage.
    tracemalloc.stop()

    # Print the highest amount of memory used during training.
    print(f"Peak memory usage during training: {peak_train_mem / 1024:.2f} KB")
    # Print the highest amount of memory used during prediction.
    print(f"Peak memory usage during prediction: {peak_pred_mem / 1024:.2f} KB")

    # Announce that final evaluation on the test set is beginning.
    print("\nEvaluating best model on test set...")
    # Compute evaluation metrics for the best model on the held-out test set.
    results = evaluate_model(best_model, X_test, y_test)
    # Reuse the already computed predictions so the printed examples match the timed inference step.
    results["y_pred"] = y_pred
    # Print the shared evaluation summary, including classification report and confusion matrix.
    print_evaluation_results(results)

    # Print the final test accuracy in a compact format.
    print(f"\nTest Accuracy: {results['accuracy']:.4f}")
    # Print a heading before displaying sample predictions.
    print("First 10 predictions:")
    # Show the first ten predicted class labels.
    print(results["y_pred"][:10])

    # Print a short project note describing the purpose of this model.
    print("\nThis Logistic Regression pipeline is used as the baseline model for Member 1.")


# Check whether this file is being executed directly.
if __name__ == "__main__":
    # Run the main experiment only when this script is executed as the entry point.
    main()
