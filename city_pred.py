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
    best_predictors = []
    with open(filename, "r") as file:
        accuracy = None
        predictors = []
        for line in file:
            line = line.strip()
            if line.startswith("Best Accuracy"):
                matches = re.findall(r'[-+]?[0-9]*\.?[0-9]+', line)
                if matches:
                    accuracy = float(matches[0])
                    print(f"Parsed accuracy: {accuracy}")
            else:
                predictor = line.strip()
                if predictor:  # Ensure non-empty lines are considered
                    predictors.append(predictor)

    return predictors

# Example usage
meta_data, validation_data = load_data_file("./complete_metadata.csv", "./testing_data.csv")
df = import_coordinates(meta_data, validation_data)
predictors = parse_city_model("best_predictors.txt")
response_variable = "city"

# Check if all predictors are present in the DataFrame
if all(item in df.columns for item in predictors) and response_variable in df.columns:
    # Scale the predictors
    scaler = StandardScaler()
    df[predictors] = scaler.fit_transform(df[predictors])

    # Fit model
    y = df[response_variable]
    X = df[predictors]
    results = fit_model(X, y)
    print(f"Model Accuracy: {results[0]}")
else:
    print("Predictors or response variable not properly defined or missing from DataFrame.")
