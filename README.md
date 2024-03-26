# Feature selection using GA in mGPS 

## 26/3 2024 

I have developed a python script, 'GA_feature_selection.py'. This script runs a genetic algorithm including the following steps: 
- load_data_file(meta_file, abundance_file): load meta-data and abundance data
- import_coordinates(meta_file, abundance_file): import coordinates from the metadata file
- extract_predictors(df): extract predictor variables (all microogranisms present in the abundance file) and the meta data corresponding to the UUIDs present there from the metadata-file. 
- extract_response_variables(df): extract the response-variables (longitude and latitude) 
- initialize_population(predictors): A binary representation of which predictor variables will be present in the multiple linear regression to come is created. Whether a predictor is included or not is dictated by a coin flip. 
- run_GA(population, predictors, response_variables): The function iterates through a number of sub-functions ((evaluate_fitness(population, predictors, response_variables), rank_populations(models), select_parents(sorted_models), crossover(parents) and mutate_offspring(offspring_population)). 
	- evaluate_fitness(population, predictors, response_variables): Loops through each individual, which in turn is composed on the binary representation of genes. The microorganisms corresponding to the genes present in the individual is retrieved and a multiple-linear regression is fitted with respect to the latitude. The model number, the AIC value and the genetic make-up of the individual is returned. 
	- rank_populations(models): Ranks the model with respect to AIC. The top-ranked individual has the lowest AIC, thus, informativeness of the model with respect to the number of predictors is optimized for. 
	- select_parents(sorted_models): Selects the lowest and second-lowest AIC individuals to reproduce.
	- crossover(parents): The genetic make-up of the lowest AIC-carrying individuals is divided at a random crossover point. Offspring are generated from their make-up.
	- mutate_offspring(offspring_population): A mutation threshold is defined at 0.01. Each gene in the offspring is 1% likely to mutate to the other binary (0 --> 1, 1 --> 0). The mutated offspring is returend.


