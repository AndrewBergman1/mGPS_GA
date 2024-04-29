import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def load_and_preprocess_data(metadata_file, abundance_file):
    abundance_df = pd.read_csv(abundance_file)
    meta_df = pd.read_csv(metadata_file, usecols=['uuid', 'longitude', 'latitude', 'city'])
    df = pd.merge(abundance_df, meta_df, on='uuid', how='inner')
    
    city_counts = df['city'].value_counts()
    cities_to_remove = city_counts[city_counts < 5].index.tolist()
    df = df[~df['city'].isin(cities_to_remove)]

    predictors = df.drop(['longitude', 'latitude', 'uuid', 'city'], axis=1)
    response_variables = df['city']

    scaler = StandardScaler()
    scaled_predictors = scaler.fit_transform(predictors)
    return scaled_predictors, response_variables, predictors.columns

metadata_file = "complete_metadata.csv"
abundance_file = "metasub_taxa_abundance.csv"

scaled_predictors, response_variables, feature_names = load_and_preprocess_data(metadata_file, abundance_file)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(scaled_predictors, response_variables, test_size=0.2, random_state=42)

# Training the model
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Evaluating the model on the test data
y_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

# Retrieving feature importances
feature_importances = model.feature_importances_

# Writing feature importances to a CSV file
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

feature_importance_df.to_csv('feature_importances.csv', index=False)

# Optionally, print the test accuracy and the path to the file
print(f"Test Accuracy: {test_accuracy}")
print("Feature importances have been saved to 'feature_importances.csv'")
