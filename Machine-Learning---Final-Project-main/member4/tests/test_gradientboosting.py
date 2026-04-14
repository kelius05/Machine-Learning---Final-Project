import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv("data/ObesityDataSet_raw_and_data_sinthetic.csv")
X = df.drop(columns=["NObeyesdad"])
y = df["NObeyesdad"]


# preprocess the data
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

# Split the data into training and testing sets, and scale the features
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# gradient boosting classifier
gbc = GradientBoostingClassifier(random_state=42)
gbc.fit(X_train_scaled, y_train)

# load teacher's dataset
df_teacher = pd.read_csv("data/ObesityDataSet_raw_and_data_sinthetic.csv")

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
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f'Accuracy of Gradient Boosted DTs on new test set: {gbc.score(X_scaled, y_encoded):.4f}\n')

