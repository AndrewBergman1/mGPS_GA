import random
from deap import base, creator, tools, algorithms
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scoop import futures
from functools import partial

    # Load and process data

# Function to load data
def load_data_file(metadata_file, abundance_file):
    meta_df = pd.read_csv(metadata_file)
    abundance_df = pd.read_csv(abundance_file)
    return meta_df, abundance_df

# Function to import and merge coordinates
def import_coordinates(meta_df, abundance_df):
    if 'uuid' not in meta_df.columns or 'uuid' not in abundance_df.columns:
        raise ValueError("UUID column not found in one or both dataframes.")
    meta_df['uuid'] = meta_df['uuid'].astype(str)
    abundance_df['uuid'] = abundance_df['uuid'].astype(str)
    df = pd.merge(abundance_df, meta_df[['uuid', 'city']], on='uuid', how="inner")
    df.dropna(inplace=True)
    if df.empty:
        print("Warning: Merging resulted in an empty DataFrame.")
    else:
        print("Merge successful, data ready for further processing.")
    return df

# Function to extract predictors
def extract_predictors(df):
    columns = [col for col in df.columns if col not in ['longitude', 'latitude', 'uuid', 'Unnamed: 0', 'city']]
    df = df[columns]
    df = df.apply(pd.to_numeric, errors='coerce')
    return df

# Function to extract response variables
def extract_response_variables(df):
    df['city'], unique_cities = pd.factorize(df['city'])
    return df[['city']]  # Return only the DataFrame of factorized values

# Evaluation function
def eval_features(individual, X, y):
    selected_indices = [index for index, value in enumerate(individual) if value == 1]
    if not selected_indices:
        return (0,)  # Ensure it's a tuple
    X_selected = X.iloc[:, selected_indices]
    classifier = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)
    scores = cross_val_score(classifier, X_selected, y.iloc[:, 0], cv=5, scoring='accuracy')
    return (scores.mean(),)

def create_individual(n_features, n_ones=200):
    """Create an individual with exactly n_ones '1's out of n_features."""
    individual = [0] * n_features  # Start with all zeros
    ones_positions = random.sample(range(n_features), n_ones)  # Randomly pick positions for 1s
    for pos in ones_positions:
        individual[pos] = 1
    return individual

training, meta = load_data_file(metadata_file="./complete_metadata.csv", abundance_file="./training_data.csv")
training = import_coordinates(training, meta)


def main():
    training, meta = load_data_file(metadata_file="./complete_metadata.csv", abundance_file="./training_data.csv")
    training = import_coordinates(training, meta)
    X = extract_predictors(training)
    y = extract_response_variables(training)

    n_features = len(X.columns)

    # Setup DEAP
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Attribute generator
    toolbox.register("attr_bool", random.randint, 0, 1)

    # Structure initializers
    toolbox.register("individual", tools.initIterate, creator.Individual,
                     lambda: create_individual(n_features, 200))
    
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Register genetic operators with parameters bound where necessary
    toolbox.register("evaluate", partial(eval_features, X=X, y=y))
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.02)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("map", futures.map)

    # Population setup
    population = toolbox.population(n=5)
    ngen = 2

    # Genetic algorithm flow using parallel processing
    for gen in range(ngen):
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
        fits = list(toolbox.map(toolbox.evaluate, offspring))
        
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        
        population = toolbox.select(offspring, k=len(population))

    # Extract and print the best individual and top 10
    best_individual = tools.selBest(population, k=1)[0]
    best_features_indices = [index for index, value in enumerate(best_individual) if value == 1]
    best_features = X.columns[best_features_indices]
    best_accuracy = best_individual.fitness.values[0]

    print("Best accuracy: {:.2f}%".format(best_accuracy * 100))
    print("Features selected by the best individual:", best_features.tolist())

    # Optionally, print details for the top 10 individuals
    top10 = tools.selBest(population, k=10)
    print("\nTop 10 Individuals:")
    for i, individual in enumerate(top10, 1):
        selected_indices = [index for index, value in enumerate(individual) if value == 1]
        features = X.columns[selected_indices]
        print(f"{i}. Fitness: {individual.fitness.values[0]:.2f}, Features: {features.tolist()}")

if __name__ == '__main__':
    main()
