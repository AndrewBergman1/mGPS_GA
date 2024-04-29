import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt

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

def select_features(X_train, y_train, X_test):
    fs = SelectKBest(score_func=f_classif, k="all")
    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs

metadata_file = "complete_metadata.csv"
abundance_file = "metasub_taxa_abundance.csv"

scaled_predictors, response_variables, feature_names = load_and_preprocess_data(metadata_file, abundance_file)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(scaled_predictors, response_variables, test_size=0.2, random_state=42)

X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)

# Printing and plotting feature scores
for i in range(len(fs.scores_)):
    print("Feature %d: %f" % (i, fs.scores_[i]))

plt.bar([i for i in range(len(fs.scores_))], fs.scores_)
plt.savefig("feature_importances.png")
