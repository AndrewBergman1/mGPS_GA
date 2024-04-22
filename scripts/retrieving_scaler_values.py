import pandas as pd
import numpy as np
import sys
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import re
import matplotlib.pyplot as plt

def load_data_file(training_file, metadata_file) :
    validation_df = pd.read_csv(training_file)
    meta_df = pd.read_csv(metadata_file)
    return validation_df, meta_df

def import_coordinates(validation_df, meta_df) :
    coordinates = meta_df[['uuid', 'longitude', 'latitude']].copy()  # Copying the slice to a new DataFrame
    coordinates['uuid'] = coordinates['uuid'].astype(str)    
    validation_df['uuid'] = validation_df['uuid'].astype(str)
    coordinates['uuid'] = coordinates['uuid'].astype(str)
    df = pd.merge(validation_df, coordinates, on='uuid', how="inner")
    df = df.dropna()
    return df

training_data, meta_data = load_data_file("./training_data", "./complete_metadata.csv")

df = import_coordinates(training_data, meta_data)

scaler = StandardScaler()

cols_to_scale = [col for col in df.columns if col not in ["uuid", "Unnamed: 0", "longitude", "latitude"]]

for col in cols_to_scale:
    df[col] = pd.DataFrame(scaler.fit_transform(pd.DataFrame(df[col])),columns=[col])

df = df.fillna(0)
print(df)
