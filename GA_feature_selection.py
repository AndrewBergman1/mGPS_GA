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
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RepeatedKFold
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np

#from concurrent.futures import ProcessPoolExecutor

#Sets the seed
rd.seed(42)

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

def extract_predictors(df) :
    columns = [col for col in df.columns if col not in ['longitude', 'latitude', 'uuid']]
    df = df[columns]
    df = df.apply(pd.to_numeric, errors='coerce')

    return df

def extract_response_variables(df) : 
    columns = [col for col in df.columns if col in ['longitude', 'latitude']]
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')

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
    # Select predictors based on the individual's genes
    selected_predictor_data = predictors.iloc[:, [i for i, bit in enumerate(individual) if bit == 1]]
    
    # Check if there are any predictors selected; if not, return a penalty score
    if selected_predictor_data.empty:
        return [individual_index, -np.inf, individual, None, None]  # Added None for model coefficients and alpha to maintain the return structure

    # Standardizing predictors since it's good practice with Ridge regression
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(selected_predictor_data)
    
    # Set alpha directly for Ridge regression without cross-validation
    alpha = 40
    model = Ridge(alpha=alpha)
    model.fit(X_scaled, response_variables.iloc[:, 0])

    # Input other metric of optimization!!!!!!!!! 
    # Train the model on 90% of the data, test it on 10%. Rank based on difference in predicted latitude.

    # Calculate R² as the performance metric with the model
    r_squared = model.score(X_scaled, response_variables.iloc[:, 0])

    # Retrieve the model's coefficients
    coefficients = model.coef_

    # Return the individual's index, R², individual representation, the set alpha, and model coefficients
    return [individual_index, r_squared, individual, alpha, coefficients]

def evaluate_fitness(population, predictors, response_variables):
    models = []
    
    with ProcessPoolExecutor() as executor:
        # Prepare tasks
        tasks = [executor.submit(evaluate_individual_fitness, index, individual, predictors, response_variables) 
                 for index, individual in enumerate(population)]
        
        # Wait for all tasks to complete and collect results
        for future in as_completed(tasks):  # Use as_completed to gather results as they complete
            models.append(future.result())
    
    # Sort models based on their R² value, higher is better
    models.sort(key=lambda x: x[1], reverse=True)
    
    return models

def rank_population(models) : 
    sorted_list = sorted(models, key=lambda x: x[1], reverse = True) # Sorts the list based on the 2nd element (R-squared)

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
    title = "GA Feature Selection Performance"
    plt.figure(figsize=(19.2, 10.2))
    # Assuming the first element in each sublist is the R² value you want to plot
    r_squared_values = [model[0] for model in best_models]  # Extract R² values

    #print(r_squared_values)
    x = range(len(r_squared_values))
    plt.plot(x, r_squared_values, marker='o')  # Plot the R² values
    # Annotate each point with its R² value
    #for i, value in enumerate(r_squared_values):
        #plt.text(i, value, f"{value:.2f}", horizontalalignment='left', verticalalignment='bottom', fontsize=9)
    plt.title(title)
    plt.xlabel('Generation')
    plt.ylabel('R² Value')
    plt.savefig('GA_Feature_Selection_Performance.png')
    plt.close()
    


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
### Currently im investigating how to get the project to run. The multicollinearity seems to be significant, as the vif is infinite for many predictor variables.
#vif_df = calculate_vif(df)
#sys.exit()

#df = pd.read_csv('./first_100') # THIS DATAFRAME CONTAINS THE FIRST 500 ROWS and column 25-4000 are sliced away using awk.
abundance_df, meta_df = load_data_file(metadata_file="../complete_metadata.csv", abundance_file="../training_data")
df = import_coordinates(abundance_df, meta_df)
predictors = extract_predictors(df)
response_variables = extract_response_variables(df)  
population =initialize_population(predictors, init_pop_size)
#print(df.columns)


best_models  = []
model_predictors = []

for i in range(no_generations):
    population, best_model_info = run_GA(population, predictors, response_variables)
    best_model = [best_model_info[1]]
    best_model.append(best_model_info[2])
    best_model.append(best_model_info[3])
    best_model.append(best_model_info[4])

    #model_predictors.append(best_model_info[2])
    best_models.append(best_model)
    print("Generation:", (i + 1))
    print(best_models)

save_png(best_models)

# Open the file for writing
with open("best_models.txt", "w") as file:
    # Iterate over each generation's best model data
    for gen_index, model_info in enumerate(best_models):
        r_squared, representation, alpha, coefficients = model_info
        # Convert representation to a string of column names (assuming representation is a list of selected feature indices)
        selected_features = ', '.join(df.columns[i] for i, selected in enumerate(representation) if selected)

        coefficients_list = list(coefficients)  # Convert NumPy array to list

        # Prepare the data line
        data_line = (f"Generation: {gen_index + 1}\n"
                     f"R²: {r_squared}\n"
                     f"Selected Features: {selected_features}\n"
                     f"Alpha: {alpha}\n"
                     f"Coefficients: {coefficients_list}\n\n")
        # Write to file
        file.write(data_line)
        # Also print the data line to console
        print(data_line)

# It's important to close the file after writing to ensure data is properly saved
