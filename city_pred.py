import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data_file(metadata_file, abundance_file):
    meta_df = pd.read_csv(metadata_file)
    abundance_df = pd.read_csv(abundance_file)
    return meta_df, abundance_df

def import_coordinates(meta_df, abundance_df):
    if 'uuid' not in meta_df.columns or 'uuid' not in abundance_df.columns:
        raise ValueError("UUID column not found in one or both dataframes.")
    meta_df['uuid'] = meta_df['uuid'].astype(str)
    abundance_df['uuid'] = abundance_df['uuid'].astype(str)
    df = pd.merge(abundance_df, meta_df[['uuid', 'city']], on='uuid', how="inner")
    df.dropna(inplace=True)
    if df.empty:
        print("Warning: Merging resulted in an empty DataFrame.")
    else:
        print("Merge successful, data ready for further processing.")
    return df

def fit_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    return accuracy, model.coef_, model.intercept_

def extract_response_variables(df):
    return df["city"]

def parse_city_model(filename):
    best_accuracy = float('-inf')
    best_predictors = None
    try:
        with open(filename, "r") as file:
            accuracy = None
            predictors = None
            for line in file:
                line = line.strip()
                if line.startswith("R²:"):
                    matches = re.findall(r'R²:\s*(\d+\.\d+)', line)
                    if matches:
                        accuracy = float(matches[0])
                if line.startswith("Selected Features:"):
                    cleaned_predictors = line.replace("Selected Features: ", "")
                    predictors = cleaned_predictors.split(", ")
                if accuracy is not None and predictors is not None and accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_predictors = predictors
    except Exception as e:
        print(f"Error reading or processing file: {e}")
    return best_predictors

# Example usage
meta_data, validation_data = load_data_file("./complete_metadata.csv", "./validation_data")
df = import_coordinates(meta_data, validation_data)
predictors = parse_city_model("best_models_city")
response_variable = "city"

# Scale the predictors
scaler = StandardScaler()
df[predictors] = scaler.fit_transform(df[predictors])

# Fit model
if predictors and response_variable in df:
    y = df[response_variable]
    X = df[predictors]
    results = fit_model(X, y)
    print(f"Model Accuracy: {results[0]}, Coefficients: {results[1]}, Intercept: {results[2]}")
else:
    print("Predictors or response variables not properly defined.")