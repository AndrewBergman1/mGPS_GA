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
def find_best_model() : 
    with open("best_models.txt", "r") as file :
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
                matches = re.findall(r'R²:\s*(\d+\.?\d*)', r_value)
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
                matches = re.findall(r'[-+]?\d*\.\d+|[-+]?\d+', coefs)
                # Convert all found matches to floats
                coefs = [float(match) for match in matches]

            elif line.startswith("Intercept") :
                intercept = line.strip()
                matches = re.findall(r'Intercept:\s*(-?\d+\.?\d*)', intercept)
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
    validation_df = pd.read_csv(validation_file)
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
    selected_columns = [col for col in df.columns if col in best_model[2]]
    validation_data = df[selected_columns]
    #print(len(best_model[2]))
    #print(len(validation_data.columns))
    #sys.exit()
    # Initialize a new Linear Regression model
    model = LinearRegression()

    # Assuming best_model[3] is the list of coefficients and best_model[4] is the intercept
    # Make sure that the length of selected_columns matches the number of coefficients
    if len(selected_columns) != len(best_model[3]):
        raise ValueError("The number of selected features does not match the number of coefficients.")

 

    model.coef_ = np.array(best_model[3])
    model.intercept_ = best_model[4]

  
    # Print shapes to debug
    print("Validation data shape:", validation_data.shape)
    print("Coefficients shape:", model.coef_.shape)

    # Making predictions
    predictions = model.predict(validation_data)
    return predictions
    '''
    index = 0
    for series_name, series in validation_data.items():  
        new_series = (series - means[index]) / stds[index]
        validation_data[series_name] = new_series
        index = index + 1      
    '''

    # Change NaN to 0 and remove infinite numbers
    # There are positive infinite numbers in the data frame that are replaced with 0.
    validation_data = validation_data.replace([np.nan, -np.inf, np.inf], 0)

    # Importing means and variance from training data


    #print(len(best_model[5]))
    #print(len(best_model[6]))
    # Standardize to the training data
    #validation_data = (validation_data - means / stds)
    

    #print(np.any(np.isnan(validation_data)))
   # print(np.all(np.isfinite(validation_data)))
    #predictions = model.predict(X_scaled)
    
    return predictions

def extract_lat(validation_data):
    scaler = StandardScaler()
    # Reshape data using .values.reshape(-1, 1) if 'latitude' is a single column
    latitude_scaled = scaler.fit_transform(validation_data[['latitude']])
    return latitude_scaled



validation_df, meta_df = load_data_file(metadata_file="./complete_metadata.csv", validation_file="./validation_data")
df = import_coordinates(validation_df, meta_df)

model = find_best_model()

predicted_lat = make_prediction(model, df)
actual_lat = extract_lat(df)
predicted_lat = np.array(predicted_lat).flatten()
actual_lat = np.array(actual_lat).flatten()
plt.scatter(predicted_lat, actual_lat)
plt.xlabel('Predicted Latitude')
plt.ylabel('Actual Latitude')
plt.title('Predicted vs. Actual Latitude, GA optimizes for R2: 0.89')
plt.grid(True)

# Adding an identity line
lims = [min(min(predicted_lat), min(actual_lat)), max(max(predicted_lat), max(actual_lat))]
plt.plot(lims, lims, 'k-', alpha=0.75, zorder=0)

plt.savefig("results/results.png")
            


