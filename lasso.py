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


# python3 single_core_test.py 0.3 0.9 0.05 20 20 2 2 3 TAKES 5 MINUTES PER GENERATION
import math
import pandas as pd
import random as rd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import sys 
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RepeatedKFold
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
import time
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, mean_absolute_error 
from sklearn import preprocessing 
import multiprocessing
import logging

np.set_printoptions(threshold=np.inf, suppress=True, linewidth=np.inf)

#from concurrent.futures import ProcessPoolExecutor

#Sets the seed
start_time = time.time()


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
    columns = [col for col in df.columns if col not in ['longitude', 'latitude', 'uuid', 'Unnamed: 0']]
    df = df[columns]
    df = df.apply(pd.to_numeric, errors='coerce')
    return df

def extract_response_variables(df):
    # Convert 'latitude' column to numeric, coercing errors
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    
    # Return the DataFrame containing only the 'latitude' column
    return df[['latitude']]

def initialize_population(predictors, init_pop_size) :
    td = 0.9

    population = []

    for individual_number in range(init_pop_size) : 
        individual = [] # This holds the binary representation oforganisms

        for predictor in predictors : 
            random_number = rd.random()
            if random_number > td :
                individual.append(1)
            else : 
                individual.append(0)
        population.append(individual)
    return population

def evaluate_batch_fitness(batch):
    results = []
    for individual_index, individual, predictors, response_variables in batch:
        selected_predictor_data = predictors.iloc[:, [i for i, bit in enumerate(individual) if bit == 1]]
              # Prepare predictor data by removing unwanted columns
        excluded_columns = {'uuid', 'Unnamed: 0', 'longitude', 'latitude'}
        selected_predictor_data = selected_predictor_data.drop(columns=[col for col in excluded_columns if col in predictors.columns])


        # Predictors not in uuid, unnamed 0, long lat
        
        
        if selected_predictor_data.empty:
            results.append((individual_index, float('inf'), individual, [], [0, 0]))  # No predictors case
        else:
            X_train, X_test, y_train, y_test = train_test_split(selected_predictor_data, response_variables, test_size=0.1)
            model = LinearRegression()
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            mse = mean_squared_error(y_test, predictions, multioutput='raw_values')  # Separate MSE for each response
            results.append((individual_index, mse, individual, model.coef_, model.intercept_))
            #print(f"Individual {individual_index}: MSE={mse}, Intercept={str(model.intercept_)[1:-1]}, Coefficients={model.coef_}")
    return results

def evaluate_fitness_parallel(executor, population, predictors, response_variables):
    args = [(index, individual, predictors, response_variables) for index, individual in enumerate(population)]
    batches = [args[i:i + 20] for i in range(0, len(args), 20)]
    future_to_batch = {executor.submit(evaluate_batch_fitness, batch): batch for batch in batches}
    results = []
    for future in as_completed(future_to_batch):
        batch_results = future.result()
        results.extend(batch_results)
    return results

def rank_population(models) : 
    sorted_list = sorted(models, key=lambda x: abs(x[1]))

    return sorted_list

def tournament_selection(sorted_models, tournament_size=3):
    """Select parents using tournament selection."""
    winners = []
    for _ in range(2):  # Select two parents
        # Randomly select indices for participants
        participant_indices = np.random.randint(len(sorted_models), size=tournament_size)
        # Fetch the actual participants using indices
        participants = [sorted_models[index] for index in participant_indices]
        # Select the best participant based on the fitness value
        winner = min(participants, key=lambda x: x[1])
        winners.append(winner)
    return winners

def adapt_mutation_rate(initial_rate, generation, total_generations):
    """Dynamically adjust the mutation rate."""
    # Decrease mutation rate as the algorithm progresses
    return initial_rate * (1 - generation / total_generations)

def parallel_crossover_mutation(parents, no_offspring, no_crossovers, mutation_rate, executor):
    tasks = [executor.submit(single_crossover_mutation, parents, no_crossovers, mutation_rate) for _ in range(no_offspring)]
    offspring_population = []
    for future in as_completed(tasks):
        offspring_population.append(future.result())
    return offspring_population

def single_crossover_mutation(parents, no_crossovers, mutation_rate):
    p1, p2 = parents[0][2], parents[1][2]
    offspring = []
    crossover_points = sorted(rd.sample(range(1, len(p1)-1), no_crossovers-1))
    crossover_points = [0] + crossover_points + [len(p1)]
    for i in range(len(crossover_points)-1):
        if i % 2 == 0:
            offspring.extend(p1[crossover_points[i]:crossover_points[i+1]])
        else:
            offspring.extend(p2[crossover_points[i]:crossover_points[i+1]])
    # Mutation
    for i in range(len(offspring)):
        if rd.random() < mutation_rate:
            offspring[i] = 0 if offspring[i] == 1 else 1
    return offspring

def run_GA(executor, population, predictors, response_variables, no_generations, mutation_rate, no_offspring, no_crossovers):
    best_models = []
    no_improvement_count = 0
    best_score = np.inf  
    early_stopping_generations = float(no_generations * 0.9)  # Stop if no improvements after 90% of the gens
    
    
    models = evaluate_fitness_parallel(executor, population, predictors, response_variables)
    sorted_models = rank_population(models)
        
    if sorted_models[0][1] < best_score:
        best_score = sorted_models[0][1]
        no_improvement_count = 0
    else:
        no_improvement_count += 1
        
    if no_improvement_count >= early_stopping_generations:
        print("Early stopping...")

    # Elitism: preserve the top N individuals
    elitism_count = 5  # Number of individuals to pass directly to the next generation
    elite_individuals = [model[2] for model in sorted_models[:elitism_count]]

    # Adjust mutation rate dynamically
    #dynamic_mutation_rate = adapt_mutation_rate(mutation_rate, generation, no_generations)
    # Selection and generation of new offspring
    parents = tournament_selection(sorted_models[:elitism_count])  
    offspring_population = parallel_crossover_mutation(parents, no_offspring - elitism_count, no_crossovers, mutation_rate, executor)

    # Merge elite individuals with offspring to form the new population
    population = elite_individuals + offspring_population

    best_models.append(sorted_models[0])  # Track the best model each generation
    #print(f"Generation {generation + 1}: Best score {best_score}")

    return population, best_models[0]


def save_png(best_models):
    plt.figure(figsize=(10, 5))
    mse = [model[1] for model in best_models]  # Assuming the second element is MSE
    plt.plot(mse, marker='o', linestyle='-', color='b')
    plt.title(f"Genetic Algorithm Performance Over Generations (Total time: {time.time() - start_time:.2f}s)")
    plt.xlabel('Generation')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.grid(True)
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

executor = ProcessPoolExecutor(max_workers=24)

abundance_df, meta_df = load_data_file(metadata_file="./complete_metadata.csv", abundance_file="./training_200.csv")
df = import_coordinates(abundance_df, meta_df)
predictors = extract_predictors(df)
response_variables = extract_response_variables(df)  
population =initialize_population(predictors, init_pop_size)

best_models  = []
model_predictors = []

for i in range(no_generations):
    population, best_model_info = run_GA(executor, population, predictors, response_variables, no_generations, mutation_rate, no_offspring, no_crossovers)
    best_model = [best_model_info[0]]
    best_model.append(best_model_info[1])
    best_model.append(best_model_info[2])
    best_model.append(best_model_info[3])
    best_model.append(best_model_info[4])


    best_models.append(best_model)
    print("Generation:", (i + 1))
    print(best_model)

save_png(best_models)


# Open the file for writing
with open("best_models_td0.9_co10.txt", "w") as file:
    # Iterate over each generation's best model data
    for gen_index, (individual_number, test_error, representation, coefficients, intercept) in enumerate(best_models):
            selected_features = ', '.join(df.columns[i] for i, bit in enumerate(representation) if bit == 1)
            coefficients_str = np.array2string(coefficients, separator=', ')
            data_line = f"Generation: {gen_index + 1}\n" \
                        f"R²: {test_error}\n" \
                        f"Selected Features: {selected_features}\n" \
                        f"Coefficients: {coefficients_str}\n" \
                        f"Intercept: {intercept}\n\n"  # Ensure intercept is written once
            file.write(data_line)