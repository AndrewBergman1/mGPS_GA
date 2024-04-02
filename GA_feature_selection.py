# Genetic algorithm for feature selection in metasub dataset.

# python3 GA_feature_selection.py 0.3 0.9 0.2 50 50 4 50 3
    # min crossover point: 0.3
    # max crossover point: 0.9
    # mutation chance: 20%
    # number of offspring: 50
    # initial population: 50 
    # reproductive units: 4 
    # Generations: 50 
    # Number of crossovers: 3


# Create a starting population of 10 individuals with either 1 or 0, 
# representing the presence of microorganisms in the multiple linear regression to follow.

import pandas as pd
import random as rd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import sys 
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor

from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np

#from concurrent.futures import ProcessPoolExecutor

#Sets the seed
rd.seed(42)

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


def calculate_vif_single_feature(data, feature_index):
    """
    Calculate the VIF for a single feature.
    """
    return variance_inflation_factor(data.values, feature_index)

def calculate_vif(predictors_df):
    predictors_df.drop(["uuid", "city_longitude", "city_latitude"], axis=1, inplace=True)
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
    return vif_data

def extract_response_variables(df) : 
    columns = [col for col in df.columns if col in ['city_longitude', 'city_latitude']]
    df['city_longitude'] = pd.to_numeric(df['city_longitude'], errors='coerce')
    df['city_latitude'] = pd.to_numeric(df['city_latitude'], errors='coerce')

    return df[columns]

def initialize_population(predictors, init_pop_size) :
    td = 0.5

    population = []

    for individual_number in range(init_pop_size) : 
        individual = [] # This holds thebinary representation oforganisms

        for predictor in predictors : 
            random_number = rd.random()
            if random_number > td :
                individual.append(1)
            else : 
                individual.append(0)
        population.append(individual)
    return population

def evaluate_individual_fitness(individual_index, individual, predictors, response_variables):
    selected_predictor_data = predictors.iloc[:, [i for i, bit in enumerate(individual) if bit == 1]]
    model = sm.OLS(response_variables.iloc[:, 0], selected_predictor_data).fit()
    return [individual_index, model.aic, individual]

def evaluate_fitness(population, predictors, response_variables):
    models = []
    
    # Use ProcessPoolExecutor to parallelize the evaluation
    with ProcessPoolExecutor() as executor:
        # Prepare tasks
        tasks = [executor.submit(evaluate_individual_fitness, index, individual, predictors, response_variables) 
                 for index, individual in enumerate(population)]
        
        # Wait for all tasks to complete and collect results
        for future in tasks:
            models.append(future.result())
    
    # Sort models based on their AIC value if needed
    models.sort(key=lambda x: x[1])
    
    return models

def rank_population(models) : 
    sorted_list = sorted(models, key=lambda x: x[1]) # Sorts the list based on the 2nd element (AIC value)

    return sorted_list

# Single point crossover
def select_parents(sorted_models, reproductive_units) :
    #parents = [sorted_models[0], sorted_models[1]]
    parents = [model for index, model in enumerate(sorted_models) if index <= reproductive_units]

    return parents 


# I need to change this to something for reasonable. Currently, you can select the number of parentts that enter a lottery. 
# Option 1: Best parent gets to mate with random other parent.
# Option 2: All parents above the threshold mate randomly to produce the offspring.
# Option 3: The best mates with many, the second best mates with a few, the third best mates with fewer... etc ...
def crossover(parents, crossover_min, crossover_max, no_offspring, reproductive_units, no_crossovers) :
    selected_parents = [parent[2] for index, parent in enumerate(parents) if index <= reproductive_units]

    while True:
        p1_choice, p2_choice = rd.sample(range(len(selected_parents)), 2)
        if p1_choice != p2_choice:
            break

    p1 = selected_parents[p1_choice]
    p2 = selected_parents[p2_choice]


    offspring_population = []

    for i in range(no_offspring):
        # Ensure crossover points are unique and sorted
        crossover_points = sorted(set([int(rd.uniform(crossover_min, crossover_max) * len(p1)) for i in range(no_crossovers)]))
        
        offspring = []
        last_point = 0
        # Alternate between segments from p1 and p2
        for i, point in enumerate(crossover_points):
            if i % 2 == 0:
                offspring += p1[last_point:point]
            else:
                offspring += p2[last_point:point]
            last_point = point
        # Add the remaining segment from the appropriate parent
        if len(crossover_points) % 2 == 0:
            offspring += p2[last_point:]
        else:
            offspring += p1[last_point:]

        offspring_population.append(offspring)


    return offspring_population

def mutate_offspring(offspring_population, mutation_rate):
    for offspring in offspring_population:
        for i in range(len(offspring)):
            mutation_coef = rd.random()
            if mutation_coef < mutation_rate:
                # Directly mutate the gene in the offspring list
                offspring[i] = 0 if offspring[i] == 1 else 1
    return offspring_population  # Ensure this line is present to return the modified population

def run_GA(population, predictors, response_variables) :        
    models = evaluate_fitness(population, predictors, response_variables)

    print("Models", models)

    # Sorts based on AIC (2nd element in list)
    sorted_models = rank_population(models)

    #print("Sorted Models", sorted_models)

    #print(sorted_models)
    # Selects the best suited parents (2, can be changed later)
    parents = select_parents(sorted_models, reproductive_units)

    #print("Parents:", parents)

    offspring_population = crossover(parents, crossover_min, crossover_max, no_offspring, reproductive_units, no_crossovers)

    #print("offspring population", offspring_population)

    # mutates each offspring (p = 0.01 for each gene)
    population = mutate_offspring(offspring_population, mutation_rate)

    #print("mutated offspring population", population)

    return population, sorted_models[0]

def save_png(best_models):
    title = "Min CP: " + str(sys.argv[1]) + ", " + \
            "Max CP: " + str(sys.argv[2]) + ", " + \
            "Mut. Prob. " + str(sys.argv[3]) + ", " + \
            "No. Offspring: " + str(sys.argv[4]) + ", " + \
            "Init pop size: " + str(sys.argv[5]) + ", " + \
            "Reproductive Units: " + str(sys.argv[6]) + ", " + \
            "Generations: " + str(sys.argv[7]) + ", " + \
            "Crossover points: " + str(sys.argv[8])
    plt.figure(figsize=(19.2, 10.2))
    x = range(len(best_models))
    plt.plot(x, best_models, marker='o')  # Plot the points with a marker
    # Annotate each point with its y-value
    for i, value in enumerate(best_models):
        plt.text(x[i], value, f"{value:.2f}", horizontalalignment='left', verticalalignment='bottom', fontsize=9)  
    plt.title(title)
    # Save the figure with the timestamped filename
    plt.savefig(f'{title}.png')
    

abundance_df, meta_df = load_data_file(metadata_file="./complete_metadata.csv", abundance_file="./first_750")
df = import_coordinates(abundance_df, meta_df)
#df.to_csv('df.csv', index=False)

crossover_min = float(sys.argv[1])
crossover_max = float(sys.argv[2])
mutation_rate = float(sys.argv[3])
no_offspring = int(sys.argv[4])
init_pop_size = int(sys.argv[5])
reproductive_units = int(sys.argv[6]) - 1
no_generations = int(sys.argv[7])
no_crossovers = int(sys.argv[8])

#print(df)

vif_df = calculate_vif(df)

print(vif_df)
sys.exit()

#df = pd.read_csv('./first_100') # THIS DATAFRAME CONTAINS THE FIRST 500 ROWS and column 25-4000 are sliced away using awk.
predictors = extract_predictors(df)
response_variables = extract_response_variables(df)  
population =initialize_population(predictors, init_pop_size)
#print(df.columns)


best_models  = []
model_predictors = []

for i in range(no_generations):
    population, best_model_info = run_GA(population, predictors, response_variables)
    best_model = best_model_info[1]  
    model_predictors.append(best_model_info[2])
    best_models.append(best_model)
    print("Generation:", i)
    print(best_models)

save_png(best_models)

for index, model in enumerate(model_predictors) : 
    columns_to_keep = [df.columns[i] for i, keep in enumerate(model) if keep == 1]

    print("Generation:", index, "\n", "Predictors:", columns_to_keep)
