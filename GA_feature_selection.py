# Genetic algorithm for feature selection in metasub dataset.

# Create a starting population of 10 individuals with either 1 or 0, 
# representing the presence of microorganisms in the multiple linear regression to follow.

import pandas as pd
import random as rd
import statsmodels.api as sm
import matplotlib.pyplot as plt


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

def evaluate_fitness(population, predictors, response_variables) :
    # Loop through the individuals in the population
    models = []
    for index, individual in enumerate(population) :
        # Include the predictors where the individual's char = 1 
        selected_predictors = [index for index, char in enumerate(individual) if char == 1] #Contains indices of the predictors to be included in the model.
        
        # These are the corresponding predictors 
        selected_predictors = [index for index, predictor in enumerate(predictors) if index in selected_predictors]
        #selected_predictors_names = [predictors.columns[index] for index in selected_predictors]
        selected_predictor_data = predictors.iloc[:, selected_predictors]

        # Retrieve the correspodning data to a data frame
        #print(selected_predictors)
        # Fit a model to the predictor variables and the latitude! 
        model = sm.OLS(response_variables.iloc[:, 0], selected_predictor_data).fit()
        #model2 = sm.OLS(response_variables.iloc[:, 1], selected_predictors).fit()

        # Model number, AIC value and the genetic composition of that individual
        model = [index, model.aic, individual]
        models.append(model)
    return models

def rank_population(models) : 
    sorted_list = sorted(models, key=lambda x: x[1]) # Sorts the list based on the 2nd element (AIC value)

    return sorted_list

# Single point crossover
def select_parents(sorted_models) :
    parents = [sorted_models[0], sorted_models[1]]
    return parents 
    
def crossover(parents) :
    p1 = parents[0][2]
    p2 = parents[1][2]

    offspring_population = []

    # Create 50 offspring 
    for offspring_number in range(50) :
        crossover_point = int(rd.random()*len(p1))
        offspring = p1[0:crossover_point] + p2[crossover_point:]
        offspring_population.append(offspring)
    
    return offspring_population

def mutate_offspring(offspring_population) :
    mutation_rate = 0.01

    for offspring in offspring_population :
        for gene in offspring : 
            mutation_coef = rd.random()

            if mutation_coef < mutation_rate :
                if gene == 1 : 
                    gene = 0
                elif gene == 0: 
                    gene = 1 
    return offspring_population

def run_GA(population, predictors, response_variables) :        
    models = evaluate_fitness(population, predictors, response_variables)
    # Sorts based on AIC (2nd element in list)
    sorted_models = rank_population(models)

    #print(sorted_models)
    # Selects the best suited parents (2, can be changed later)
    parents = select_parents(sorted_models)

    offspring_population = crossover(parents)

    # mutates each offspring (p = 0.01 for each gene)
    population = mutate_offspring(offspring_population)

    return population, sorted_models[0]


#abundance_df, meta_df = load_data_file(metadata_file="/home/andrewbergman/courses/mGPS_GA/complete_metadata.csv", abundance_file="/home/andrewbergman/courses/mGPS_GA/metasub_taxa_abundance.csv")
#df = import_coordinates(abundance_df, meta_df)
#df.to_csv('df.csv', index=False)

df = pd.read_csv('/home/andrewbergman/courses/mGPS_GA/first_500_trimmed') # THIS DATAFRAME CONTAINS THE FIRST 500 ROWS and column 25-4000 are sliced away using awk.
predictors = extract_predictors(df)
response_variables = extract_response_variables(df)  
population =initialize_population(predictors)

best_models  = []
for i in range(10):
    population, best_model_info = run_GA(population, predictors, response_variables)
    best_model = best_model_info[1]  

    best_models.append(best_model)

x = range(len(best_models))
plt.plot(x, best_models)
plt.savefig('fitness.png')
