import pandas as pd
import random as rd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import sys 
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor

from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np

def load_data_file(abundance_file, metadata_file) :
    abundance_df = pd.read_csv(abundance_file)
    meta_df = pd.read_csv(metadata_file)
    return abundance_df, meta_df

def import_coordinates(abundance_df, meta_df) :
    coordinates = meta_df[['uuid', 'longitude', 'latitude']] #Retrieve all UUIDs and their corresponding coordinates
    abundance_df['uuid'] = abundance_df['uuid'].astype(str)
    coordinates['uuid'] = coordinates['uuid'].astype(str)
    df = pd.merge(abundance_df, coordinates, on='uuid', how="inner")
    return df

def calculate_vif_single_feature(data, feature_index):
    return variance_inflation_factor(data.values, feature_index)

def calculate_vif(predictors_df):
    predictors_df.drop(["uuid", "longitude", "latitude"], axis=1, inplace=True)
    # Convert all columns to numeric, forcing non-convertible values to NaN
    predictors_df = predictors_df.apply(pd.to_numeric, errors='coerce')
    # Drop rows with NaN values to ensure clean VIF calculation
    predictors_df.dropna(inplace=True)
    predictors_df = sm.add_constant(predictors_df)
    features = predictors_df.columns[1:]  # Skip the constant term for feature names
    
    # Prepare data for parallel VIF computation
    data_for_vif = [predictors_df] * len(features)
    
    with ThreadPoolExecutor() as executor:
        # Calculate VIF in parallel
        vifs = list(executor.map(calculate_vif_single_feature, data_for_vif, range(1, len(features) + 1)))
    
    # Combine feature names with their corresponding VIF
    vif_data = pd.DataFrame()
    vif_data["Feature"] = features
    vif_data["VIF"] = vifs
    vif_data.to_csv("vif_df.csv", index=False)

    return vif_data

def generate_corr_matrix(df):
    corr_mtx = df.corr()
    print(corr_mtx)
    return corr_mtx

def drop_columns(corr_matrix) :
    # Create an empty set to keep track of columns to drop
    columns_to_drop = set()

    # Iterate over the correlation matrix
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > 0.6: # The optimal threshold seem to lie between 0.5 and 0.7. 0.7 yields RuntimeWarning: divide by zero encountered in scalar divide vif = 1. / (1. - r_squared_i)
                # Add the column name to our set
                colname = corr_matrix.columns[i]
                columns_to_drop.add(colname)

    print(columns_to_drop)
    df_dropped = df.drop(columns=columns_to_drop)
    return df_dropped


abundance_df, meta_df = load_data_file(metadata_file="../complete_metadata.csv", abundance_file="../training_data.csv")
df = import_coordinates(abundance_df, meta_df)

new_df = df.iloc[:, 1:-2]
new_df = new_df.loc[:, (df != 0).any(axis=0)]

new_df.to_csv("selected_predictors")

#print(new_df)
corr_mtx = generate_corr_matrix(new_df)
df = drop_columns(corr_mtx)

vif_df = calculate_vif(df)

