import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt

def load_and_preprocess_data(metadata_file, abundance_file):
    abundance_df = pd.read_csv(abundance_file)
    meta_df = pd.read_csv(metadata_file, usecols=['uuid', 'longitude', 'latitude', 'city'])
    df = pd.merge(abundance_df, meta_df, on='uuid', how='inner')
    
    city_counts = df['city'].value_counts()
    cities_to_remove = city_counts[city_counts < 5].index.tolist()
    print("Cities with less than 5 observations:", cities_to_remove)
    df = df[~df['city'].isin(cities_to_remove)]

    predictors = df.drop(['longitude', 'latitude', 'uuid', 'city'], axis=1)
    response_variables = df['city']

    scaler = StandardScaler()
    scaled_predictors = scaler.fit_transform(predictors)
    return predictors, scaled_predictors, response_variables

def initialize_population(population_size, num_features):
    num_ones = int(0.1 * num_features)  # 10% ones
    num_zeros = num_features - num_ones
    population = []
    for _ in range(population_size):
        individual_list = [1] * num_ones + [0] * num_zeros
        np.random.shuffle(individual_list)
        population.append(np.array(individual_list))
    return population

def evaluate_fitness_batch(individuals, predictors, response_variables):
    results = []
    for individual in individuals:
        selected_features = predictors[:, individual == 1]
        if selected_features.shape[1] == 0:
            results.append((-np.inf, individual))
        else:
            model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
            accuracies = cross_val_score(model, selected_features, response_variables, cv=2, n_jobs=2)
            mean_accuracy = np.mean(accuracies)
            results.append((mean_accuracy, individual))
    return results

def evaluate_population_fitness(population, predictors, response_variables, executor, batch_size=5):
    results = []
    for i in range(0, len(population), batch_size):
        batch = population[i:i + batch_size]
        future = executor.submit(evaluate_fitness_batch, batch, predictors, response_variables)
        results.append(future)

    final_results = []
    for future in as_completed(results):
        batch_results = future.result()
        final_results.extend(batch_results)
    return final_results

def crossover(parent1, parent2, no_crossovers):
    points = sorted(np.random.choice(range(1, len(parent1)), no_crossovers, replace=False))
    offspring = np.empty_like(parent1)
    parent_flag = True
    for start, end in zip([0] + points, points + [None]):
        offspring[start:end] = parent1[start:end] if parent_flag else parent2[start:end]
        parent_flag = not parent_flag
    return offspring

def mutate(individual, mutation_rate):
    mutation_mask = np.random.rand(len(individual)) < mutation_rate
    individual[mutation_mask] = 1 - individual[mutation_mask]
    return individual

def crossover_and_mutate(parent1, parent2, mutation_rate, no_crossovers):
    child = crossover(parent1, parent2, no_crossovers)
    return mutate(child, mutation_rate)

def generate_offspring(population, mutation_rate, no_crossovers, executor):
    num_pairs = len(population) // 2
    tasks = []
    for _ in range(num_pairs):
        parent_indices = np.random.randint(0, len(population), size=2)
        task = executor.submit(crossover_and_mutate, population[parent_indices[0]], population[parent_indices[1]], mutation_rate, no_crossovers)
        tasks.append(task)

    offspring = []
    for future in as_completed(tasks):
        offspring.append(future.result())

    if len(offspring) < len(population):  # Ensure population size remains constant
        extra_tasks = [executor.submit(crossover_and_mutate, population[np.random.randint(0, len(population))], population[np.random.randint(0, len(population))], mutation_rate, no_crossovers) for _ in range(len(population) - len(offspring))]
        for future in as_completed(extra_tasks):
            offspring.append(future.result())

    return offspring

def run_genetic_algorithm(executor, predictors, response_variables, population_size, num_generations, mutation_rate, no_crossovers):
    num_features = predictors.shape[1]
    population = initialize_population(population_size, num_features)

    best_fitness_history = []
    best_model = None
    best_fitness = -np.inf

    for generation in range(num_generations):
        fitness_and_individuals = evaluate_population_fitness(population, predictors, response_variables, executor)

        if not fitness_and_individuals:
            print(f"No valid fitness results in generation {generation+1}, skipping to next generation.")
            continue

        fitness_scores = np.array([fi[0] for fi in fitness_and_individuals])
        individuals = [fi[1] for fi in fitness_and_individuals]

        sorted_indices = np.argsort(fitness_scores)[::-1]
        current_best_fitness = fitness_scores[sorted_indices[0]]
        best_individual = individuals[sorted_indices[0]]

        print(f"Generation {generation + 1} completed with best fitness: {current_best_fitness}")

        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_model = best_individual

        best_fitness_history.append(current_best_fitness)

        # Elitism: Carry the best individual forward unchanged
        population = [individuals[i] for i in sorted_indices[:max(1, len(fitness_scores) // 20)]]

        # Generate offspring for the next generation
        population.extend(generate_offspring(population, mutation_rate, no_crossovers, executor))

    return population, best_model, best_fitness_history

def save_best_predictors(best_model, predictors, best_fitness, filename="best_predictors.txt"):
    try:
        selected_features = predictors.columns[best_model == 1].tolist()
        with open(filename, "w") as file:
            file.write(f"Best Accuracy: {best_fitness}\n")
            file.write("\n".join(selected_features))
        print(f"Best predictors and accuracy saved to {filename}")
    except Exception as e:
        print(f"Error in saving predictors: {e}")

def plot_fitness_history(best_fitness_history):
    plt.plot(best_fitness_history, marker='o', linestyle='-', color='b')
    plt.title("Best Fitness Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Fitness (Accuracy)")
    plt.grid(True)
    plt.savefig("fitness_over_generations.png")
    plt.show()

if __name__ == "__main__":
    metadata_file = "complete_metadata.csv"
    abundance_file = "training_data.csv"
    predictors_df, predictors, response_variables = load_and_preprocess_data(metadata_file, abundance_file)
    with ProcessPoolExecutor(max_workers=32) as executor:
        final_population, best_model, best_fitness_history = run_genetic_algorithm(executor, predictors, response_variables, 10, 5, 0.1, 5)
        best_fitness = best_fitness_history[-1]  # Get the last (highest) fitness from the history
        save_best_predictors(best_model, predictors_df, best_fitness)
        plot_fitness_history(best_fitness_history)
