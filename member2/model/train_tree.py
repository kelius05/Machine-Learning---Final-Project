import sys
import os
import pandas as pd
import numpy as np
import time
import tracemalloc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))  # adds repo root to path so Python can find the shared folder

from shared.evaluation.metrics import evaluate_model, print_evaluation_results      # shared evaluation — keeps results consistent


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

feature_names = X.columns.tolist()  # save column names for visualization later

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # 80% training, 20% testing — stratify keeps class proportions equal
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples:     {X_test.shape[0]}")

print("BASELINE: No depth limit")
print("="*45)

baseline = DecisionTreeClassifier(random_state=42)  #no max depth means the tree will grow until every leaf is pure(overfitting)
baseline.fit(X_train, y_train)  #train the model

train_preb_b = baseline.predict(X_train)  # generates the model's guess and predicts against training data
test_preb_b = baseline.predict(X_test)  # predict the test data it's never seen

print(f"\nTrain Accuracy: {accuracy_score(train_preb_b, y_train):.3f}")  #compare predictions to the labels
print(f"Test Accuracy: {accuracy_score(test_preb_b, y_test):.3f}") # should be lower due to overfitting
print(f"Tree depth: {baseline.get_depth()}")  # shows how deep/how many questions were asked

print("\n" + "="*45)
print("TUNING: max_depth sweep")
print("="*45)

depths_to_try  = [2,3,4,5,6,7,8,9,10, None] # trying every level of questions | From 2 to 10, then infinite questions
results = [] # empty list to collect the depth label and the 2 accuracies for each depth, so we are able to plot them later on

for depth in depths_to_try:
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))
    label = str(depth)if depth is not None else "None" 
    results.append((label, train_acc, test_acc)) # store all 3 values as a tuple so we can use them for plotting later
    print(f"max_depth={label:<6} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")# :<basically aligns the data to the left, 6 meaning the value takes up at least 6 characters

depth_labels = [r[0] for r in results] # pull just the depth labels for the x axis to plot
train_accs = [r[1] for r in results] # pull only the training accuracies for plotting
test_accs = [r[2] for r in results] # pull only the test accuracies for plotting

plt.figure(figsize=(9,5))
plt.plot(depth_labels, train_accs, marker='o', label='Train Accuracy', color='steelblue')  # Blue line = how well model fits the training data
plt.plot(depth_labels, test_accs, marker='s', label='Test accuracy', color='darkorange') # orange line is how well the model generalizes the new data
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.title("Decision tree: Train vs Test accuracy by max depth")
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("../../member2/results/depth_tuning.png", dpi=150) #saves the image into the results folder - dpi=150 makes it a higher quality
print("Saved: depth_tuning.png")
plt.show()

best_depth = 9 # best test accuracy from plot
# The baseline accuracy is higher, but we're afraid
# of the training accuracy being so high at 100 percent. it could perform well on this set
# but terrible on another because it memorized the entire thing.
print("\n" + "="*45)
print(f"BEST MODEL: max_depth={best_depth}")
print("="*45)

start_train = time.time()  # time.time() returns the current moment as a decimal number of seconds — record it before training
classifier = DecisionTreeClassifier(max_depth=best_depth, random_state=42)  # final model with the best depth we found
classifier.fit(X_train, y_train)                                             # train on the full training set
train_time = time.time() - start_train  # subtracting start from now gives the number of seconds training took

start_pred = time.time() # record time before prediction starts
test_pred = classifier.predict(X_test) # generate predictions on the test set - this is what we evaluate against y_test
pred_time = time.time() - start_pred # how many seconds it takes to retrieve all the test samples

train_pred = classifier.predict(X_train) #get training predictions so we can compare train vs test accuracy again

print(f"Training time: {train_time:.4f} seconds")
print(f"Prediction time: {pred_time:.4f}")

tracemalloc.start() # tracemalloc is a built in memory tracker in python, start begins recording the amount of memory that is being used
mem_clf = DecisionTreeClassifier(max_depth = best_depth, random_state=42) 
mem_clf.fit(X_train, y_train)
_, peak_train_mem = tracemalloc.get_traced_memory() # _ discards the current memory, we only care about its peak
tracemalloc.stop()# Stops tracking

tracemalloc.start() # starts tracking the memory being used
_ = mem_clf.predict(X_test) # run prediction to measure memory
_, peak_pred_mem = tracemalloc.get_traced_memory() # _ discards the current memory, we only care about its peak
tracemalloc.stop() # Stops tracking

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
    print(f"{feature}: {importance:.4f}") # feature importances values are between 0 and 1 and always sum up to 1

importances = classifier.feature_importances_
indicies = np.argsort(importances)[::-1][:15] # sorts highest importance to lowest, keeping the top 15
top_features = [feature_names[i] for i in indicies] # uses the sorted indicies to get the feature names in the correct order of importance
top_values = importances[indicies] # uses the same indicies to get the importance values in the exact order as before

plt.figure(figsize=(9,6))
plt.barh(top_features[::-1], top_values[::-1], color='steelblue') #most important features appear at the top of the chart
plt.xlabel('Feature importance score')
plt.title('Top Feature importances - Decision Tree')
plt.tight_layout()
plt.savefig("../../member2/results/feature_importance.png", dpi=150)
print("\nSaved: feature_importance.png")
plt.show()

print("\nSaving tree visualization" )

plt.figure(figsize=(20,10))
plot_tree(
    classifier,
    feature_names=feature_names, # Labels each split node with the feature name its splitting on
    class_names = target_encoder.classes_,  # labels each leaf with the obesity category name
    filled=True, # colors each node by its majority class, making the tree easier to read
    rounded=True, # rounded corners on nodes
    max_depth = 3 # only show a depth of 3 since the full depth would be incredible difficult to read
)
plt.title(f"Decision Tree visualization (top 3 levels shown, full depth={best_depth})")
plt.tight_layout()
plt.savefig("../../member2/results/tree_visualization.png", dpi=150)
print("Saved: tree_visualization.png")
plt.show()

print("\n" + "="*45)
print("PIPELINE AND GRID SEARCH TUNING")
print("="*45)

#A pipeline chains steps together into an object so it all runs automatically in order
#Here we wrap the decision tree into a pipeline as a signel step
pipeline = Pipeline([
    ('classifier', DecisionTreeClassifier(random_state=42)) #our model as a step in the pipeline
])

#now we define the parameter combinations we want Gridsearch to test
#classifier__max_depth means the max_depth setting of th eclassifier step
param_grid={
    'classifier__max_depth': [7,8,9,10,None], #some of the best depths we tested manually
    'classifier__criterion': ['gini', 'entropy'] #2 of the ways to measure impurity
}

#GridSearch tries every combination in param_grid
#cv=5 means 5 fold cross validation- splits the data into 5 parts, trains on 4 and tests on 1, repeat it 5 times
#more reliable than a single train/test split, scoring accuracy means we pick whatever combination is the best
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
)



grid_search.fit(X_train, y_train) #try every combination and find the best one

print(f"Best parameters found: {grid_search.best_params_}") #prints whatever combination was the best
print(f"Best cross-val accuracy: {grid_search.best_score_ :.4f}")


# Evaluate the best model on the held out test set
best_pipeline = grid_search.best_estimator_ # the best pipeline from gridsearch
test_acc_pipeline = accuracy_score(best_pipeline.predict(X_test), y_test)  #Test on data its never seen
print(f"Test accuracy: {test_acc_pipeline:.4f} ")

# Manual vs Pipelin Comparison plot
print("\nSaving manual vs pipeline comparison plot")

# get the cross validation results for every combination
cv_results = grid_search.cv_results_

# pull out gini and entropy results separately so we can plot both
gini_mask    = [p['classifier__criterion'] == 'gini'    for p in cv_results['params']]
entropy_mask = [p['classifier__criterion'] == 'entropy' for p in cv_results['params']]

gini_depths = [p['classifier__max_depth'] for p, m in zip(cv_results['params'], gini_mask) if m]
entropy_depths =[p['classifier__max_depth'] for p, m in zip(cv_results['params'], entropy_mask)if m]

gini_scores = [s for s, m in zip(cv_results['mean_test_score'], gini_mask) if m]
entropy_scores = [s for s, m in zip(cv_results['mean_test_score'], entropy_mask) if m]

# convert None to string "None" so it shows cleanly on x axis
gini_labels    = [str(d) if d is not None else "None" for d in gini_depths]
entropy_labels = [str(d) if d is not None else "None" for d in entropy_depths]

# filter manual sweep to only show 7 depths and above, same as the pipeline range
# this is so we can have the same starting point for everythig
manual_start = depth_labels.index("7")  # find where depth 7 starts in our manual results
manual_labels = depth_labels[manual_start:] # depth labels fro 7 and onwards
manual_test = test_accs[manual_start:]

plt.figure(figsize=(9,5))

# blue dashed line is manual test accuracy
# uses one train/test split, a specific 80/20 cut of the data
plt.plot(manual_labels, manual_test, marker='o', linestyle='--', color='steelblue', label='Manual Sweep (Single Split)')

# Orange solud line is grid search's cross validation accuracy using gini criterion
# averages 5 different splits, it's more reliable than a single split (manual)
plt.plot(gini_labels, gini_scores, marker='s', linestyle='-', color='darkorange', label='GridSearchCV - gini (5 fold cv)')

# green solid line is grid search's cross validation accuracy using entropy criterion
# entropy measures impurity differently than gini, sometimes it has better splits
plt.plot(entropy_labels, entropy_scores, marker='^', linestyle='-', color='green', label='GridSearchCV - entropy (5-fold cv)')

plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.title("Manual Sweep vs GridSearchCV - Gini vs Entropy")
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("../../member2/results/pipeline_comparison.png", dpi=150)  # saves to results folder
print("Saved: pipeline_comparison.png")
plt.show()

print("\nDone! all output is saved to member2/results/")