import pandas as pd
import numpy as np
import sys
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import re
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, mean_absolute_error 
from sklearn import preprocessing 
np.set_printoptions(threshold=np.inf)
import folium 

def find_best_lat() : 
    with open("best_models_lat", "r") as file :
        best_gen = ""
        best_r2 = 9999999999
        best_predictors = []
        best_coefs = []
        best_intercept = 0

        predictors = []
        coefs = []
        r_value = 0
        generation = ""
        intercept = 0
        for line in file :
            last_row = False

            if line.startswith("Generation:") :
                generation = line.strip()
            
            elif line.startswith("R²"): 
                r_value = line.strip()
                matches = re.findall(r'R²:\s*\[(\d+\.\d+)\]', r_value)
                r_value = float(matches[0])
            
            elif line.startswith("Selected Features"):
                predictors = line.strip()
                # Remove the "Selected Features: " part
                cleaned_predictors = predictors.replace("Selected Features: ", "")
                #print(cleaned_predictors)
                # Split the string by comma to get a list of features
                predictors = cleaned_predictors.split(", ")
                #print(len(predictors))
            
            elif line.startswith("Coefficients:"):
                coefs = line.strip()
                matches = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', coefs)
                # Convert all found matches to floats
                coefs = [float(match) for match in matches]

            elif line.startswith("Intercept") :
                intercept = line.strip()
                matches = re.findall(r'Intercept:\s*\[(-?\d+\.\d+)\]', intercept)

                intercept = float(matches[0])
                last_row = True
            
            if r_value < best_r2 and last_row:
                    best_gen = generation
                    best_r2 = r_value
                    best_coefs = coefs
                    best_predictors = predictors
                    best_intercept = intercept
        
       
        return [best_gen, best_r2, best_predictors, best_coefs, best_intercept] #best_alpha, best_means, best_vars
def find_best_long() : 
    with open("best_models_long", "r") as file :
        best_gen = ""
        best_r2 = 9999999999
        best_predictors = []
        best_coefs = []
        best_intercept = 0

        predictors = []
        coefs = []
        r_value = 0
        generation = ""
        intercept = 0
        for line in file :
            last_row = False

            if line.startswith("Generation:") :
                generation = line.strip()
            
            elif line.startswith("R²"): 
                r_value = line.strip()
                matches = re.findall(r'R²:\s*\[(\d+\.\d+)\]', r_value)
                r_value = float(matches[0])
            
            elif line.startswith("Selected Features"):
                predictors = line.strip()
                # Remove the "Selected Features: " part
                cleaned_predictors = predictors.replace("Selected Features: ", "")
                #print(cleaned_predictors)
                # Split the string by comma to get a list of features
                predictors = cleaned_predictors.split(", ")
                #print(len(predictors))
            
            elif line.startswith("Coefficients:"):
                coefs = line.strip()
                matches = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', coefs)
                # Convert all found matches to floats
                coefs = [float(match) for match in matches]

            elif line.startswith("Intercept") :
                intercept = line.strip()
                matches = re.findall(r'Intercept:\s*\[(-?\d+\.\d+)\]', intercept)

                intercept = float(matches[0])
                last_row = True
            
            if r_value < best_r2 and last_row:
                    best_gen = generation
                    best_r2 = r_value
                    best_coefs = coefs
                    best_predictors = predictors
                    best_intercept = intercept
        
       
        return [best_gen, best_r2, best_predictors, best_coefs, best_intercept] #best_alpha, best_means, best_vars
    
def load_data_file(validation_file, metadata_file) :
    validation_df = pd.read_csv(validation_file, index_col=0)
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

def make_prediction(best_model, df):
    # Ensure 'df' is your prepared DataFrame with the same feature columns used during model training

    # Select the columns based on your model's feature importance or coefficients

    selected_columns = [col for col in df.columns if col in best_model[2] if col not in ["uuid", "longitude", "latitude", "Unnamed: 0"]]
    validation_data = df[selected_columns]
    print(len(best_model[3]))
    print(len(validation_data.columns))
    #sys.exit()
    # Initialize a new Linear Regression model
    model = LinearRegression()
    model.coef_ = np.array(best_model[3])
    model.intercept_ = best_model[4]
     
 
    # Assuming best_model[3] is the list of coefficients and best_model[4] is the intercept
    # Make sure that the length of selected_columns matches the number of coefficients
       # Print shapes to debug
    #print("Validation data shape:", validation_data.shape)
    #print("Coefficients shape:", model.coef_.shape)
    #sys.exit()
    if len(selected_columns) != len(best_model[3]):
        raise ValueError("The number of selected features does not match the number of coefficients.")
    
    if not list(validation_data.columns) == best_model[2]:
        raise ValueError("Features in validation data must match the features the model was trained on, in the same order.")


    # Making predictions
    predictions = model.predict(validation_data)
    return predictions
 

def extract_coords(validation_data):
    # Reshape data using .values.reshape(-1, 1) if 'latitude' is a single column
    latitude = validation_data[['latitude']]
    longitude = validation_data[['longitude']]
    return longitude, latitude



validation_df, meta_df = load_data_file(metadata_file="./complete_metadata.csv", validation_file="./validation_200.csv")
df = import_coordinates(validation_df, meta_df)

lat_model = find_best_lat()
long_model = find_best_long()

predicted_lat = make_prediction(lat_model, df)
predicted_long = make_prediction(long_model, df)

actual_long, actual_lat = extract_coords(df)

print(df)

actual_latitudes = df["latitude"].tolist()
actual_longitudes = df["longitude"].tolist()

actual_coordinates = list(zip(actual_latitudes, actual_longitudes))
predicted_coordinates = list(zip(predicted_lat, predicted_long))




if actual_coordinates:
    map = folium.Map(location=[actual_coordinates[0][0], actual_coordinates[0][1]], zoom_start=5)

    # Add a circle for each coordinate in the list
    for coord in actual_coordinates:
        folium.CircleMarker(
            location=coord,
            radius=6,  # radius in pixels
            color='blue',
            fill=True,
            fill_color='blue'
        ).add_to(map)
    # Save or display the map

    for coord in predicted_coordinates :
         folium.CircleMarker(
            location=coord,
            radius=6,  # radius in pixels
            color='red',
            fill=True,
            fill_color='red'
        ).add_to(map)
    
    map.save("map.html")



predicted_lat = np.array(predicted_lat).flatten()
actual_lat = np.array(actual_lat).flatten()
plt.scatter(predicted_lat, actual_lat)
plt.xlabel('Predicted Latitude')
plt.ylabel('Actual Latitude')
plt.title('Predicted vs. Actual Latitude')
plt.grid(True)

# Adding an identity line
lims = [min(min(predicted_lat), min(actual_lat)), max(max(predicted_lat), max(actual_lat))]
plt.plot(lims, lims, 'k-', alpha=0.75, zorder=0)

plt.savefig("results/lat_results.png")
            
predicted_long = np.array(predicted_long).flatten()
actual_long = np.array(actual_long).flatten()
plt.scatter(predicted_long, actual_long)
plt.xlabel('Predicted Longitude')
plt.ylabel('Actual Longitude')
plt.title('Predicted vs. Actual Longitude')
plt.grid(True)

# Adding an identity line
lims = [min(min(predicted_long), min(actual_long)), max(max(predicted_long), max(actual_long))]
plt.plot(lims, lims, 'k-', alpha=0.75, zorder=0)

plt.savefig("results/long_results.png")

