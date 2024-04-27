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
from sklearn.linear_model import LogisticRegression
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

def import_coordinates(abundance_df, meta_df):
    # Check if 'uuid' column exists in both DataFrames
    if 'uuid' not in meta_df.columns:
        raise ValueError("UUID column not found in metadata dataframe.")
    if 'uuid' not in abundance_df.columns:
        raise ValueError("UUID column not found in abundance dataframe.")

    # Ensure 'uuid' is treated as a string in both DataFrames
    meta_df['uuid'] = meta_df['uuid'].astype(str)
    abundance_df['uuid'] = abundance_df['uuid'].astype(str)

    # Merge DataFrames on 'uuid'
    df = pd.merge(abundance_df, meta_df[['uuid', 'longitude', 'latitude', 'city']], on='uuid', how="inner")

    # Drop rows with missing values
    df = df.dropna()

    # Optionally, you might want to check if the merge resulted in an empty DataFrame
    if df.empty:
        print("Warning: Merging resulted in an empty DataFrame. Check the 'uuid' values in both DataFrames.")

    return df

def extract_predictors(df) :
    columns = [col for col in df.columns if col not in ['longitude', 'latitude', 'uuid', 'Unnamed: 0', 'city']]
    df = df[columns]
    df = df.apply(pd.to_numeric, errors='coerce')
    return df

def extract_response_variables(df):
    # Factorize the 'city' column to handle it as categorical data
    df['city'], unique_cities = pd.factorize(df['city'])
    return df[['city']], dict(enumerate(unique_cities))

def initialize_population(predictors, init_pop_size) :
    td = 0.975

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
        excluded_columns = {'uuid', 'Unnamed: 0', 'longitude', 'latitude', 'city'}
        selected_predictor_data = selected_predictor_data.drop(columns=[col for col in excluded_columns if col in predictors.columns])
        
        if selected_predictor_data.empty:
            fitness = float('-inf')  # Adjust based on GA fitness direction
            results.append((individual_index, fitness, individual, [], [0]))
        else:
            model = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')
            # Perform 5-fold cross-validation
            accuracies = cross_val_score(model, selected_predictor_data, response_variables.values.ravel(), cv=5, scoring='accuracy', n_jobs=1)
            mean_accuracy = np.mean(accuracies)
            fitness = mean_accuracy  # Adjust based on GA fitness direction
            model.fit(selected_predictor_data, response_variables.values.ravel())  # Fit model on entire dataset to get coefficients
            results.append((individual_index, fitness, individual, model.coef_, model.intercept_))
    return results

def evaluate_fitness_parallel(executor, population, predictors, response_variables):
    args = [(index, individual, predictors, response_variables) for index, individual in enumerate(population)]
    batches = [args[i:i + 4] for i in range(0, len(args), 4)]
    future_to_batch = {executor.submit(evaluate_batch_fitness, batch): batch for batch in batches}
    results = []
    for future in as_completed(future_to_batch):
        batch_results = future.result()
        results.extend(batch_results)
    return results

def rank_population(models): 
    sorted_list = sorted(models, key=lambda x: x[1], reverse=True)  # Sorting by accuracy descending
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
    accuracy = [-model[1] for model in best_models]  # Convert negative accuracy back to positive
    plt.plot(accuracy, marker='o', linestyle='-', color='b')
    plt.title("Genetic Algorithm Performance Over Generations")
    plt.xlabel('Generation')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig('GA_Classification_Performance.png')
    plt.close()
    


#df.to_csv('df.csv', index=False)

if __name__ == "__main__" : 
    crossover_min = float(sys.argv[1])
    crossover_max = float(sys.argv[2])
    mutation_rate = float(sys.argv[3])
    no_offspring = int(sys.argv[4])
    init_pop_size = int(sys.argv[5])
    reproductive_units = int(sys.argv[6]) - 1
    no_generations = int(sys.argv[7])
    no_crossovers = int(sys.argv[8])

    executor = ProcessPoolExecutor(max_workers=48)

    abundance_df, meta_df = load_data_file(metadata_file="./complete_metadata.csv", abundance_file="./training_data.csv")
    df = import_coordinates(abundance_df, meta_df)
    predictors = extract_predictors(df)
    response_variables, city_mapping = extract_response_variables(df)

    # Scale the predictors
    scaler = StandardScaler()
    scaled_predictors = scaler.fit_transform(predictors)
    # Convert scaled array back to DataFrame (to maintain compatibility with existing code)
    scaled_predictors_df = pd.DataFrame(scaled_predictors, columns=predictors.columns)
    # Replace the predictors DataFrame with the scaled one
    predictors = scaled_predictors_df

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


    with open("best_models_city", "w") as file:
            for gen_index, (individual_number, test_error, representation, coefficients, intercepts) in enumerate(best_models):
                selected_features = [predictors.columns[i] for i, bit in enumerate(representation) if bit == 1]
                data_line = f"Generation: {gen_index + 1}\n" \
                            f"R²: {test_error}\n" \
                            f"Selected Features: {', '.join(selected_features)}\n" \
                            f"Coefficients and Cities:\n"

                # Handling coefficients (2D array) and matching each row to a city
                for i, city_coefs in enumerate(coefficients):
                    city_name = city_mapping.get(i, "Unknown City")
                    coefs_by_feature = ', '.join(f"{feat}: {coef:.4f}" for feat, coef in zip(selected_features, city_coefs))
                    data_line += f"  {city_name} - Coefficients: {coefs_by_feature}\n"
                
                # Handling intercepts and matching each to a city
                data_line += "Intercepts and Cities:\n"
                for i, intercept in enumerate(intercepts):
                    city_name = city_mapping.get(i, "Unknown City")
                    data_line += f"  {city_name} - Intercept: {intercept:.4f}\n"
                
                data_line += "\n"
                file.write(data_line)