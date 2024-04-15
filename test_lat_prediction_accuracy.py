import pandas as pd
import numpy as np
import sys
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import re
import matplotlib.pyplot as plt

def find_best_model() : 
    with open("best_models.txt", "r") as file :
        generation_found = False
        grab_r_squared = False
        grab_predictors = False
        coefs_found = False
        best_gen = ""
        best_r2 = ""
        predictors = ""
        alpha = 1
        for line in file :
            if line.startswith("Generation:") :
                generation_found = True
                generation = line.strip()
            if generation_found == True : 
                grab_r_squared = True 
            if grab_r_squared and generation_found and line.startswith("R²") : 
                r_value = line.strip()
                if r_value > best_r2 :
                    best_gen = generation
                    best_r2 = r_value
            
            if grab_r_squared and generation_found and line.startswith("Selected Features") :
                predictors = line.strip()
                # Remove the "Selected Features: " part
                cleaned_predictors = predictors.replace("Selected Features: ", "")

                # Split the string by comma to get a list of features
                predictors = cleaned_predictors.split(", ")
                grab_predictors = True

            if generation_found and grab_r_squared and grab_predictors and line.startswith("Coefficients:") :
                coefs = line.strip()

                matches = re.findall(r'[-+]?\d*\.\d+|[-+]?\d+', coefs)

                # Convert all found matches to floats
                coefs = [float(match) for match in matches]

                print(coefs)
                coefs_found = True
                generation_found = False
                grab_r_squared = False
                grab_predictors = False
            
            if generation_found and grab_r_squared and grab_predictors and coefs_found and line.startswith("alpha") :
                alpha = line.strip()
                matches = re.findall(r'alpha:\s*(\d+)', alpha)
                alpha = float(matches)
                coefs_found = False
                generation_found = False
                grab_r_squared = False
                grab_predictors = False
                
        return [best_gen, best_r2, predictors, coefs, alpha]
    
def load_data_file(abundance_file, metadata_file) :
    abundance_df = pd.read_csv(abundance_file)
    meta_df = pd.read_csv(metadata_file)
    return abundance_df, meta_df

def import_coordinates(abundance_df, meta_df) :
    coordinates = meta_df[['uuid', 'longitude', 'latitude']].copy()  # Copying the slice to a new DataFrame
    coordinates['uuid'] = coordinates['uuid'].astype(str)    
    abundance_df['uuid'] = abundance_df['uuid'].astype(str)
    coordinates['uuid'] = coordinates['uuid'].astype(str)
    df = pd.merge(abundance_df, coordinates, on='uuid', how="inner")
    df = df.dropna()
    return df

def make_prediction(best_model, df):
    validation_data = df  # Ensure this is the correct path to your CSV file
    
    # Initialize a new Ridge model with predefined intercept and coefficients
    # Note: You will need to replace `predefined_coefficients` with your actual coefficients array
    model = Ridge(alpha=best_model[4]) # Find the alpha value 
    coefs = [coef for index, coef in enumerate(best_model[3])]

    model.intercept_ = best_model[3][0]

    # Create a list of columns to exclude
    selected_columns = best_model[2]
    selected_columns = [col for col in selected_columns if col not in ["uuid", "unnamed: 0", "longitude", "latitude"]]
    
    # Select columns that are not in the exclude list
    selected_columns = [col for col in validation_data.columns if col in selected_columns]

    # Use the selected columns to index the DataFrame
    validation_data = validation_data[selected_columns]

    model.coef_ = np.array(coefs)

    # Standardizing predictors since it's good practice with Ridge regression
    #scaler = StandardScaler()
    X_scaled = validation_data[selected_columns].values
    predictions = model.predict(X_scaled)

    return predictions

def extract_lat(validation_data):
    scaler = StandardScaler()
    # Reshape data using .values.reshape(-1, 1) if 'latitude' is a single column
    latitude_scaled = scaler.fit_transform(validation_data[['latitude']])
    return latitude_scaled



abundance_df, meta_df = load_data_file(metadata_file="./complete_metadata.csv", abundance_file="./validation_data")
df = import_coordinates(abundance_df, meta_df)

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
            


