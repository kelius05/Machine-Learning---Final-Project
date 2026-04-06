import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


def split_features_target(df, target_column="NObeyesdad"):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def encode_features(X):
    X = X.copy()
    feature_encoders = {}
    for col in X.columns:
        if X[col].dtype == "object":
            encoder = LabelEncoder()
            X[col] = encoder.fit_transform(X[col])
            feature_encoders[col] = encoder

    return X, feature_encoders


def encode_target(y):
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)
    return y_encoded, target_encoder


def prepare_data(file_path, target_column="NObeyesdad", test_size=0.2, random_state=42):
    df = load_data(file_path)

    X, y = split_features_target(df, target_column)

    X_encoded, feature_encoders = encode_features(X)
    y_encoded, target_encoder = encode_target(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded,
        y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded
    )

    return {
        "dataframe": df,
        "X": X_encoded,
        "y": y_encoded,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_encoders": feature_encoders,
        "target_encoder": target_encoder
    }