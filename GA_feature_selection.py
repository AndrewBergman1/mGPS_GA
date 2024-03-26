# Genetic algorithm for feature selection in metasub dataset.

# Create a starting population of 10 individuals with either 1 or 0, 
# representing the presence of microorganisms in the multiple linear regression to follow.

import pandas as pd
import random as rd
import statsmodels.api as sm


# Loads the CSV-file as a data frame
def load_data_file(abundance_file, metadata_file) :
    abundance_df = pd.read_csv(abundance_file)
    meta_df = pd.read_csv(metadata_file)
    return abundance_df, meta_df

def import_coordinates(abundance_df, meta_df) :
    coordinates = meta_df[['uuid', 'city_longitude', 'city_latitude']] #Retrieve all UUIDs and their corresponding coordinates
    df = pd.merge(abundance_df, coordinates, on='uuid', how="inner")
    return df


def extract_predictors(df) :
    columns = [col for col in df.columns if col not in ['city_longitude', 'city_latitude', 'uuid']]
    df = df[columns]
    df = df.apply(pd.to_numeric, errors='coerce')

    return df

def extract_response_variables(df) : 
    columns = [col for col in df.columns if col in ['city_longitude', 'city_latitude']]
    df['city_longitude'] = pd.to_numeric(df['city_longitude'], errors='coerce')
    df['city_latitude'] = pd.to_numeric(df['city_latitude'], errors='coerce')

    return df[columns]

def initialize_population(predictors) :
    td = 0.5

    population = []

    for individual_number in range(10) : 
        individual = [] # This holds thebinary representation oforganisms

        for predictor in predictors : 
            random_number = rd.random()
            if random_number > td :
                individual.append(1)
            else : 
                individual.append(0)
        population.append(individual)
    return population



#This doesnt work!!
def evaluate_fitness(population, predictors, response_variables) :

    # Loop through the individuals in the population
    for individual in population :
        # Include the predictors where the individual's char = 1
        # Fit a model to the predictor variables and the response variables

        # Fit the model
        model = sm.OLS(response_variables.iloc[:, 0], selected_predictors).fit()
        model2 = sm.OLS(response_variables.iloc[:, 1], selected_predictors).fit()

        # To see the coefficients of the model

    print(model.summary())




#def select_parental_chromosomes() :

#def perform_crossover() :

#def mutate_individuals() : 



#abundance_df, meta_df = load_data_file(metadata_file="/home/andrew/Documents/GA_feature_selection/complete_metadata.csv", abundance_file="/home/andrew/Documents/GA_feature_selection/metasub_taxa_abundance.csv")
#df = import_coordinates(abundance_df, meta_df)

#
#df.to_csv('df.csv', index=False)
df = pd.read_csv('/home/andrew/Documents/GA_feature_selection/first_100') # THIS DATAFRAME CONTAINS THE FIRST 100 ROWS

predictors = extract_predictors(df)
response_variables = extract_response_variables(df)
initial_population =initialize_population(predictors)

# This function doesnt work! 
#evaluate_fitness(initial_population, predictors, response_variables)