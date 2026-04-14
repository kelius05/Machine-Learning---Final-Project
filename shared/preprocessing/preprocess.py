import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    # Load the raw CSV dataset.
    return pd.read_csv(file_path)


def split_features_target(df: pd.DataFrame, target_column: str = "NObeyesdad"):
    # Split the full dataframe into X features and y target.
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def get_feature_types(X: pd.DataFrame):
    # Return numeric and categorical feature name lists.
    categorical_features = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    numeric_features = [col for col in X.columns if col not in categorical_features]
    return numeric_features, categorical_features
