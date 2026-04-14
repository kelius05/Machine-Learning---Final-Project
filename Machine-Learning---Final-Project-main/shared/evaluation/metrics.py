from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    return {
        "y_pred": y_pred,
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": matrix
    }


def print_evaluation_results(results):
    print("Accuracy:", results["accuracy"])
    print("\nClassification Report:")
    print(results["classification_report"])
    print("\nConfusion Matrix:")
    print(results["confusion_matrix"])