import sys
import os
import pandas as pd
import numpy as np
import time
import tracemalloc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))  # adds repo root to path so Python can find the shared folder

from shared.preprocessing.preprocess import prepare_data                            # shared preprocessing — same as rest of group
from shared.evaluation.metrics import evaluate_model, print_evaluation_results      # shared evaluation — keeps results consistent


# ── load and prepare data using shared preprocessing ─────────────────────────
file_path = "../../data/raw/ObesityDataSet_raw_and_data_sinthetic.csv"

print("Loading and preprocessing data...")
data = prepare_data(file_path)

# fix any columns not encoded due to pandas 3.0 StringDtype issue
for col in data["X_train"].columns:
    if not pd.api.types.is_numeric_dtype(data["X_train"][col]):
        le = LabelEncoder()
        data["X_train"][col] = le.fit_transform(data["X_train"][col].astype(str))
        data["X_test"][col]  = le.transform(data["X_test"][col].astype(str))
        data["X"][col]       = le.fit_transform(data["X"][col].astype(str))

# unpack AFTER the fix so these variables get the corrected values
X_train        = data["X_train"]
X_test         = data["X_test"]
y_train        = data["y_train"]
y_test         = data["y_test"]
target_encoder = data["target_encoder"]
X              = data["X"]

feature_names = X.columns.tolist()

print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples:     {X_test.shape[0]}")

print("BASELINE: No depth limit")
print("="*45)

baseline = DecisionTreeClassifier(random_state=42)  #no max depth means the tree will grow until every leaf is pure(overfitting)
baseline.fit(X_train, y_train)  #train the model

train_preb_b = baseline.predict(X_train)  #generates the model's guess and predicts against training data
test_preb_b = baseline.predict(X_test)  #predict the test data it's never seen

print(f"\nTrain Accuracy: {accuracy_score(train_preb_b, y_train):.3f}")  #compare predictions to the labels
print(f"Test Accuracy: {accuracy_score(test_preb_b, y_test):.3f}") #should be lower due to overfitting
print(f"Tree depth: {baseline.get_depth()}")  #shows how deep/how many questions were asked

print("\n" + "="*45)
print("TUNING: max_depth sweep")
print("="*45)

depths_to_try  = [2,3,4,5,6,7,8,9,10, None] #trying every level of questions | From 2 to 10, then infinite questions
results = [] #empty list to collect the depth label and the 2 accuracies for each depth, so we are able to hen plot them

for depth in depths_to_try:
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))
    label = str(depth)if depth is not None else "None"
    results.append((label, train_acc, test_acc))
    print(f"max_depth={label:<6} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")# :<basically aligns the data to the left, 6 meaning the value takes up at least 6 characters

depth_labels = [r[0] for r in results] #pull just the depth labels for the x axis to plot
train_accs = [r[1] for r in results] #pull only the training accuracies for plotting
test_accs = [r[2] for r in results] #pull only the test accuracies for plotting

plt.figure(figsize=(9,5))
plt.plot(depth_labels, train_accs, marker='o', label='Train Accuracy', color='steelblue')  #Blue line = how well model fits the training data
plt.plot(depth_labels, test_accs, marker='s', label='Test accuracy', color='darkorange') #orange line is how well the model generalizes the new data
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.title("Decision tree: Train vs Test accuracy by max depth")
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("../../member2/results/depth_tuning.png", dpi=150) #saves the image into the results folder - dpi=150 makes it a higher quality
print("Saved: depth_tuning.png")
plt.show()

best_depth = 9 #best test accuracy from plot
#The baseline test accuracy got a higher tets accuracy, but we're afraid
#of the training accuracy since its 100 percent. it could perform well on this set
#but terrible on another because it memorized the entire thing.
print("\n" + "="*45)
print(f"BEST MODEL: max_depth={best_depth}")
print("="*45)

start_train = time.time()  # time.time() returns the current moment as a decimal number of seconds — record it before training
classifier = DecisionTreeClassifier(max_depth=best_depth, random_state=42)  # final model with the best depth we found
classifier.fit(X_train, y_train)                                             # train on the full training set
train_time = time.time() - start_train  # subtracting start from now gives the number of seconds training took

start_pred = time.time() #record time before prediction starts
test_pred = classifier.predict(X_test) #generate predictions on the test set - this is what we evaluate against y_test
pred_time = time.time() - start_pred #how many seconds it takes to retrieve all the test samples

train_pred = classifier.predict(X_train) #get training predictions so we can compare train vs test accuracy again

print(f"Training time: {train_time:.4f} seconds")
print(f"Prediction time: {pred_time:.4f}")

tracemalloc.start() #tracemalloc is a built in memory tracker in python, start begins recording the amount of memory that is being used
mem_clf = DecisionTreeClassifier(max_depth = best_depth, random_state=42) 
mem_clf.fit(X_train, y_train)
_, peak_train_mem = tracemalloc.get_traced_memory() #returns the current bytes and peak bytes since .start()
tracemalloc.stop()

tracemalloc.start()
_ = mem_clf.predict(X_test) #_ means that we don't need the actual predictions, just want the memory
_, peak_pred_mem = tracemalloc.get_traced_memory()
tracemalloc.stop()

print(f"Peak memory usage during training: {peak_train_mem / 1024:.2f} KB")  #divide it by 1024 so we can convert bytes to kilobytes
print(f"Peak Memory when predicting: {peak_pred_mem / 1024:.2f}")

print("\n" + "="*45)
print("EVALUATION")
print("="*45)

print("Train Accuracy:", round(accuracy_score(y_train, train_pred), 3))
print("Test Accuracy: ", round(accuracy_score(y_test,  test_pred),  3))

results_eval = evaluate_model(classifier, X_test, y_test)  # calls the shared evaluation function
print_evaluation_results(results_eval)                      # prints using the shared print function

print("\n" + "="*45)
print("FEATURE IMPORTANCE")
print("="*45)

for feature, importance in zip(X.columns, classifier.feature_importances_):
    print(f"{feature}: {importance:.4f}") #feature importances values are between 0 and 1 and always sum up to 1

importances = classifier.feature_importances_
indicies = np.argsort(importances)[::-1][:15] #sorts lowest to highest, displaying only the 1 highest
top_features = [feature_names[i] for i in indicies] #uses the sorted indicies to get the feature names in the correct order of importance
top_values = importances[indicies] #users the same indicies to get the importance values in the exact order as before

plt.figure(figsize=(9,6))
plt.barh(top_features[::-1], top_values[::-1], color='steelblue') #most important features appear at the top of the chart
plt.xlabel('Feature importance score')
plt.title('Top Feature importances - Decision Tree')
plt.tight_layout()
plt.savefig("../../member2/results/feature_importance.png", dpi=150)
print("\nSaved: feature_importance.png")
plt.show()

print("\nSaving tree visualizataion" )

plt.figure(figsize=(20,10))
plot_tree(
    classifier,
    feature_names=feature_names, #Labels each split node with the feature name its splitting on
    class_names = target_encoder.classes_,  #labels each leaf witht he obesity category name
    filled=True, #colors each node by its majority class, making the tree easier to read
    rounded=True, #rounded corners on nodes
    max_depth = 3 #only show a depth of 3 since the full depth would be incredible difficult to read
)
plt.title(f"Decision Tree visualization (top 3 levels shown, full depth={best_depth})")
plt.tight_layout()
plt.savefig("../../member2/results/tree_visualization.png", dpi=150)
print("Saved: tree_visualization.png")
plt.show()

print("\nDone! all output is saved to memeber2/results/")