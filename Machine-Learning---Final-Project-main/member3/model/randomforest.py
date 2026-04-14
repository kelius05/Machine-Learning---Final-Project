import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc
)
from sklearn.model_selection import cross_val_score, StratifiedKFold

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from shared.preprocessing.preprocess import prepare_data

#file path:
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.abspath(os.path.join(script_dir, "../../data/raw/ObesityDataSet_raw_and_data_sinthetic.csv"))

#prep data
data_dict = prepare_data(
    file_path=data_path,
    target_column="NObeyesdad",
    test_size=0.2,
    random_state=42
)

X_train = data_dict["X_train"]
X_test  = data_dict["X_test"]
y_train = data_dict["y_train"]
y_test  = data_dict["y_test"]
target_encoder = data_dict["target_encoder"]
feature_names = data_dict["X"].columns.tolist()

# Fix non numeric column error mainly gender column issues
for col in X_train.columns:
    if not pd.api.types.is_numeric_dtype(X_train[col]):
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        X_test[col]  = le.transform(X_test[col].astype(str))
        print(f"Encoded column: {col}")

#setup random forest(change later)
rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-2)
rf.fit(X_train, y_train)
# train accuracy (for diagnostic purposes)
y_train_pred = rf.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"\nTrain Accuracy: {train_accuracy:.4f}")

y_pred = rf.predict(X_test)
y_pred_proba = rf.predict_proba(X_test)

test_accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')


print("Test set numbers")
print(f"Test Accuracy:  {test_accuracy:.4f}")
print(f"Precision:      {precision:.4f}")
print(f"Recall:         {recall:.4f}")
print(f"F1-score:       {f1:.4f}")

print("\nClassification Report (per class):")
print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_encoder.classes_)
disp.plot(xticks_rotation=45)
plt.title("Confusion Matrix – Test Set")
plt.tight_layout()
plt.show()

#cross validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring='accuracy')

print("Cross validation(train)")

print(f"CV Accuracy scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

#features
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Random Forest Feature Importances")
plt.bar(range(len(importances)), importances[indices], align="center")
plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()

#Roc curve
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score

y_test_bin = label_binarize(y_test, classes=range(len(target_encoder.classes_)))
n_classes = y_test_bin.shape[1]

plt.figure(figsize=(10, 8))
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{target_encoder.classes_[i]} (AUC = {roc_auc:.2f})')
plt.plot([0,1],[0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC Curves (One-vs-Rest)')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

