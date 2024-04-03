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
    coordinates = meta_df[['uuid', 'city_longitude', 'city_latitude']] #Retrieve all UUIDs and their corresponding coordinates
    df = pd.merge(abundance_df, coordinates, on='uuid', how="inner")
    return df

def calculate_vif_single_feature(data, feature_index):
    return variance_inflation_factor(data.values, feature_index)

def calculate_vif(predictors_df):
    predictors_df.drop(["uuid", "city_longitude_y", "city_latitude_y"], axis=1, inplace=True)
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
    vif_df.to_csv("vif_df.csv", index=False)

    return vif_data


abundance_df, meta_df = load_data_file(metadata_file="./complete_metadata.csv", abundance_file="./first_500")
df = import_coordinates(abundance_df, meta_df)
vif_df = calculate_vif(df)

