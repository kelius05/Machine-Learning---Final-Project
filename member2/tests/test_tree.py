import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

#load and prepare data
df = pd.read_csv("../../data/raw/ObesityDataSet_raw_and_data_sinthetic.csv")

encoder = LabelEncoder()  # same LabelEncoder used in all your DT class examples
categorical_cols = ['Gender', 'family_history_with_overweight', 'FAVC',
                    'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']

for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col])  # convert each text column to numbers

target_encoder = LabelEncoder()  # separate encoder for target so we can decode predictions back to names
df['NObeyesdad'] = target_encoder.fit_transform(df['NObeyesdad'])  # encode 7 obesity categories to numbers 0-6

X = df.drop(columns=['NObeyesdad']) #all input features but the target feature
y = df['NObeyesdad'] #the label that we want to predict

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) #80/20 split, same as what we used in train_tree.py

classifier = DecisionTreeClassifier(max_depth=9, random_state=42) #same we used before, the max depth is the most ideal depth we found in the model
classifier.fit(X_train, y_train) #train the model



### Test 1 - Check the training vs test accuracy ###
#training accuracy should be higher than test accuracy
#if test accuracy is very close to training, the model is generalizing well
#if train is 1.0 and test is lower, the model is overfitting

train_pred = classifier.predict(X_train) #predictions on training data
test_pred = classifier.predict(X_test) #predictions on the unseen test data

train_acc = accuracy_score(train_pred, y_train) #fraction of the correct predictions on the training set
test_acc = accuracy_score(test_pred, y_test) #fraction of the correct predictions on test set


print("Test 1 - Training accuacy vs Test Accuracy")
print("-" * 45)

print(f"Train Accuracy: {train_acc:.3f}")
print(f"Test Accuracy: {test_acc:.3f}")

if train_acc >= test_acc: #training should always be at least the same as test if not higher
    print("Passed - Train Accuracy is higher than or equal to the test accuracy")
else:
    print("Failed - somethign is wrong, test accuracy is higher than the training")


### Test 2 - Check if model beats a meaningful threshold
#with 7 different target classes, random guessing only has a 14% chance
#so beating about 70 means the model actually learns real patterns

print("\nTest 2 - Minimim accuracy check")
print("-" * 45)
print(f"Test accuracy {test_acc:.3f}")

if test_acc >= 0.70: #this is a meaningful bar, a percentage that should easily be beaten if the model learned properly
    print("Passed - Model accuracy is above 70%")
else:
    print("Failed - Model accuracy is too low")


###Test 3 - check feature importances sum to 1
#This is a fixed mathematical property of Decision trees, just doing it to be safe
print("\nTest 3 - Feature importances")
print("-" * 45)

for feature, importance in zip(X.columns, classifier.feature_importances_): #exact same loop we used befopre
    print(f"{feature}: {importance:.4f}")

total = np.sum(classifier.feature_importances_)  #Adds all the importance values together, should = to 1
print(f"Total importance sum: {total:.4f}")

if round(total, 5) == 1.0: #always adds up to 1 or 100%
    print("Passed - Feature importances sum to 1.0 or 100%")
else:
    print("Failed - importances/feature importance does not add up to 1.0")


### Test 4 - Predict for multiple new people
# test several different profiles to show the model working across different categories

print("\nTest - Predictions on New people")
print("-" * 45)

#each row has a new person assigned using all 16 features
# Gender, Age, Height, Weight, family_history, FAVC, FCVC, NCP,
# CAEC, SMOKE, CH2O, SCC, FAF, TUE, CALC, MTRANS
# wrap each person in a DataFrame with the correct column names
# this matches the format the model was trained on and removes the warning
test_people = [
    ("Person 1 - young, light, active",
     pd.DataFrame([[0, 20, 1.65, 55, 0, 0, 2, 3, 2, 0, 2, 0, 3, 1, 0, 3]], columns=X.columns)),

    ("Person 2 - middle aged, overweight",
     pd.DataFrame([[1, 35, 1.75, 95, 1, 1, 3, 3, 2, 0, 2, 0, 1, 1, 1, 0]], columns=X.columns)),

    ("Person 3 - sedentary, heavy",
     pd.DataFrame([[1, 45, 1.70, 130, 1, 1, 3, 3, 3, 0, 1, 0, 0, 2, 2, 0]], columns=X.columns)),

    ("Person 4 - very active, normal weight",
     pd.DataFrame([[0, 28, 1.68, 62, 0, 0, 2, 3, 1, 0, 3, 1, 3, 0, 0, 3]], columns=X.columns)),

    ("Person 5 - family history, high weight",
     pd.DataFrame([[0, 40, 1.60, 110, 1, 1, 2, 3, 2, 0, 1, 0, 0, 1, 2, 1]], columns=X.columns)),
]

all_passed = True #Track if all the predictions are valid

for description, person in test_people:
    prediction = classifier.predict(person) #get the predicted encoded number
    category = target_encoder.inverse_transform(prediction) #decode number back to obesity category name

    #check if the prediction is one of the 7 known categories
    if category[0] in target_encoder.classes_:
        status = "Passed"
    else:
        status = "Failed"
        all_passed = False
    
    print(f"{description}")
    print(f"Predicted: {category[0]} - {status}")

if all_passed:
    print("\nAll predictions returned valid obesity categories")
else:
    print("\nSome predictions returned invalid categories")


### Summary ###

print("\nSummary")
print("-" * 45)
print("Train Accuracy:", round(train_acc, 3))
print("Test Accuracy: ", round(test_acc,  3))
print("Feature importances sum:", total)
print("New person predictions: completed")