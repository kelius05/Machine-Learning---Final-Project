import pandas as pd
from sklearn.linear_model import LogisticRegression
from shared.preprocessing.preprocess import prepare_data
from shared.evaluation.metrics import evaluate_model, print_evaluation_results

def main():
    #Load data and path
    file_path = "data/raw/ObesityDataSet_raw_and_data_sinthetic.csv"

    print("Loading and preprocessing data...")
    data = prepare_data(file_path)

    print("Training data shape:", data["X_train"].shape)
    print("Testing data shape:", data["X_test".shape])

    print("Training Logistic Regression model...")
    logreg_model = LogisticRegression(max_iter=3000)
    logreg_model.fit(data["X_train"], data["y_train"])

    print("Model Training completed.")

    print("\nEvaluating model")
    result = evaluate_model(logreg_model, data["X_test"], data["y_test"])
    print_evaluation_results(result)

    print(f"\nAccuracy: {result['accuracy']:.4f}")

    summary_df = pd.DataFrame({
        "Metric": ["Accuracy"],
        "Value": [result["accuracy"]]
    })

    print("\nMetrics Summary:")
    print(summary_df)

    print("\nFirst 10 predictions:")
    print(result["y_pred"][:10])

    print("\nLogistic Regression model is used here as a baseline classification model.")

if __name__ == "__main__":
    main()
