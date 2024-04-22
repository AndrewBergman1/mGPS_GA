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
#from concurrent.futures import ProcessPoolExecutor

#Sets the seed
rd.seed(42)
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

def extract_response_variables(df) : 
    columns = [col for col in df.columns if col in ['longitude', 'latitude']]
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')

    return df[columns]

def initialize_population(predictors, init_pop_size) :
    td = 0.5

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

def evaluate_individual_fitness(individual_index, individual, predictors, response_variables):
    # Select predictors based on the individual's genes
    selected_predictor_data = predictors.iloc[:, [i for i, bit in enumerate(individual) if bit == 1]]
    # Check if any predictors are selected; if not, return a penalty score
    if selected_predictor_data.empty:
        return [individual_index, -np.inf, individual, None, None]  # With penalty

    # Split data into training and test sets (90% train, 10% test)
    X_train, X_test, y_train, y_test = train_test_split(
        selected_predictor_data, 
        response_variables.iloc[:, 0], 
        test_size=0.1, 
        random_state=42
    )

     #creating a regression model 
    model = LinearRegression() 
  
    # fitting the model 
    model.fit(X_train, y_train) 
  
    # making predictions 
    predictions = model.predict(X_test) 
    # Standardize predictors
    #scaler = StandardScaler()
    #X_train_scaled = scaler.fit_transform(X_train)
    #X_test_scaled = scaler.transform(X_test)

    # RidgeCV is used to find a reasonable alpha value. The longer the span, the longer the run. 
    #alphas = list(range(950, 1050, 10))  # Alpha range
    #model_cv = RidgeCV(alphas=alphas, store_cv_values=True)
    #model_cv.fit(X_train_scaled, y_train)

    # Train the Ridge model on the entire training set using the best alpha found
    #model = Ridge(alpha=model_cv.alpha_)
    #model.fit(X_train_scaled, y_train)

    # Evaluate the model on the test set
    #y_pred = model.predict(X_test_scaled)
    #test_error = mean_squared_error(y_test, y_pred)
    # fitting the model 
    model.fit(X_train, y_train) 
  
# making predictions 
    predictions = model.predict(X_test) 
    # Retrieve the model's coefficients
    coefficients = model.coef_
    intercept = model.intercept_
    test_error = mean_squared_error(y_test, predictions)
    #means = scaler.mean_
    #var = scaler.var_
    # Return the evaluation results including the test error, best alpha, and model coefficients
    return [individual_index, test_error, individual, 0, coefficients, 0, 0, intercept] #model_cv.alpha, means, vars

def evaluate_fitness(population, predictors, response_variables):
    models = []

    # Sequentially evaluate the fitness of each individual
    for index, individual in enumerate(population):
        result = evaluate_individual_fitness(index, individual, predictors, response_variables)
        models.append(result)
    return models

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

def crossover(parents, no_offspring, no_crossovers):
    p1, p2 = parents[0][2], parents[1][2]  # Assuming parents are tuples with the individual at index 2
    offspring_population = []
    for _ in range(no_offspring):
        offspring = []
        crossover_points = sorted(np.random.randint(1, len(p1)-1, no_crossovers-1).tolist())
        crossover_points = [0] + crossover_points + [len(p1)]  # Ensure starting and ending points
        for i in range(len(crossover_points)-1):
            if i % 2 == 0:
                offspring.extend(p1[crossover_points[i]:crossover_points[i+1]])
            else:
                offspring.extend(p2[crossover_points[i]:crossover_points[i+1]])
        offspring_population.append(offspring)
    return offspring_population

def mutate_offspring(offspring_population, mutation_rate): 
    for offspring in offspring_population[2:]: # Skip the two best individuals.
        for i in range(len(offspring)):
            if np.random.rand() < mutation_rate:
                offspring[i] = 0 if offspring[i] == 1 else 1
    return offspring_population

def run_GA(population, predictors, response_variables) : 
    best_models = []
    no_improvement_count = 0
    best_score = np.inf  
    early_stopping_generations = float(no_generations*0.9) # Stop if no improvements after 90% of the gens
    for generation in range(no_generations):
        models = evaluate_fitness(population, predictors, response_variables)
        sorted_models = rank_population(models)
        
        if sorted_models[0][1] < best_score:
            best_score = sorted_models[0][1]
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        if no_improvement_count >= early_stopping_generations:
            print("Early stopping...")
            break
        #print(len(sorted_models[0][6]))
        #print(len(sorted_models[0][5]))

        dynamic_mutation_rate = adapt_mutation_rate(mutation_rate, generation, no_generations)
        parents = tournament_selection(sorted_models)
        offspring_population = crossover(parents, no_offspring, no_crossovers)
        population = mutate_offspring(offspring_population, dynamic_mutation_rate)
        
        # Elitism is employed: the two best parents are passed to the offspring generation
        elitism_count = 2  # Number of individuals to pass directly
        population[:elitism_count] = [model[2] for model in sorted_models[:elitism_count]]

        #print(sorted_models)
        best_models.append(sorted_models[0])  # track the best model each generation


    return population, best_models[0]

def save_png(best_models):
    end_time = time.time()
    tot_time = end_time - start_time
    title = "GA Feature Selection Performance: " + str(tot_time)
    plt.figure(figsize=(19.2, 10.2))
    mse = [model[0] for model in best_models]  # Extract MSE 

    #print(mse)
    x = range(len(mse))
    plt.plot(x, mse, marker='o')  # Plot MSE values
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

abundance_df, meta_df = load_data_file(metadata_file="./complete_metadata.csv", abundance_file="./training_data")
df = import_coordinates(abundance_df, meta_df)
predictors = extract_predictors(df)
response_variables = extract_response_variables(df)  
population =initialize_population(predictors, init_pop_size)

best_models  = []
model_predictors = []

for i in range(no_generations):
    population, best_model_info = run_GA(population, predictors, response_variables)
    best_model = [best_model_info[1]]
    best_model.append(best_model_info[2])
    best_model.append(best_model_info[3])
    best_model.append(best_model_info[4])
    best_model.append(best_model_info[5])
    best_model.append(best_model_info[6])
    best_model.append(best_model_info[7])

    best_models.append(best_model)
    print("Generation:", (i + 1))
    print(best_model)

save_png(best_models)

# Open the file for writing
with open("best_models.txt", "w") as file:
    # Iterate over each generation's best model data
    for gen_index, model_info in enumerate(best_models):
        r_squared, representation, alpha, coefficients, means, vars, intercept  = model_info
        # Convert representation to a string of column names (assuming representation is a list of selected feature indices)
        selected_features = ', '.join(df.columns[i] for i, selected in enumerate(representation) if selected)

        coefficients_list = list(coefficients)  # Convert NumPy array to list
        means_list = list(means)
        vars_list = list(vars)

        # Prepare the data line
        data_line = (f"Generation: {gen_index + 1}\n"
                     f"R²: {r_squared}\n"
                     f"Selected Features: {selected_features}\n"
                     f"Alpha: {alpha}\n"
                     f"Coefficients: {coefficients_list}\n\n"
                     f"Means : {means_list}\n\n"
                     f"Vars: {vars_list}\n\n"
                     f"Intercept: {intercept}")
        # Write to file
        file.write(data_line)
