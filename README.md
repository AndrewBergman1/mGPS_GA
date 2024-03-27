# Feature selection using GA in mGPS 

Run like: GA_feature_selection.py [crossover_min] [crossover_max] [mutation_rate] [number of offspring per generation] [initial population size] [number of reproductive units]


## 26/3 2024 

### Current architecture of GA_feature_selection.py

As of now, GA_feature_selection.py only performs feature selection to optimize the AIC of multiple linear regressions with respect to latitudes of the samples. In the future, longitudinal predictions will be optimized for as well. I am currently visualizing developing two models, one for longitude and another for latitude.
Today I have produced a first version of GA_feature_selection.py for latitude predictions. 

I have developed a python script, 'GA_feature_selection.py'. This script runs a genetic algorithm including the following steps: 
- **load_data_file(meta_file, abundance_file)**: load meta-data and abundance data
- **import_coordinates(meta_file, abundance_file)**: import coordinates from the metadata file
- **extract_predictors(df)**: extract predictor variables (all microogranisms present in the abundance file) and the meta data corresponding to the UUIDs present there from the metadata-file. 
- **extract_response_variables(df)**: extract the response-variables (longitude and latitude) 
- **initialize_population(predictors)**: A binary representation of which predictor variables will be present in the multiple linear regression to come is created. Whether a predictor is included or not is dictated by a coin flip. 
- **run_GA(population, predictors, response_variables)**: The function iterates through a number of sub-functions ((evaluate_fitness(population, predictors, response_variables), rank_populations(models), select_parents(sorted_models), crossover(parents) and mutate_offspring(offspring_population)). 
	- **evaluate_fitness(population, predictors, response_variables)**: Loops through each individual, which in turn is composed on the binary representation of genes. The microorganisms corresponding to the genes present in the individual is retrieved and a multiple-linear regression is fitted with respect to the latitude. The model number, the AIC value and the genetic make-up of the individual is returned. 
	- **rank_populations(models)**: Ranks the model with respect to AIC. The top-ranked individual has the lowest AIC, thus, informativeness of the model with respect to the number of predictors is optimized for. 
	- **select_parents(sorted_models)**: Selects the lowest and second-lowest AIC individuals to reproduce.
	- **crossover(parents)**: The genetic make-up of the lowest AIC-carrying individuals is divided at a random crossover point. Offspring are generated from their make-up.
	- **mutate_offspring(offspring_population)**: A mutation threshold is defined when running the script. Each gene in the offspring is that likely to mutate to the other binary (0 --> 1, 1 --> 0). The mutated offspring is returend.

### Feature Selection using GA_feature_selection.py, optimizing for minimal AIC when predicting latitudes of samples
GA_feature_selection.py was ran with the following conditions (python3 GA_feature_selection.py 0.3 0.9 0.2 100 100 4 50)
	- Min crossover point: 30% into the chromosomes
	- Max crossover point: 90% into the chromosomes
	- Mutation chance: 10% 
	- Offspring count: 100/generation
	- Initial population: 100 
	- Reproductive units: 4 
	- Generations: 50 

The following result was acquired: 
![The figure illustrates the lowest AIC values acquired for each generation, running GA_feature_selection.py.](https://github.com/AndrewBergman1/mGPS_GA/blob/main/fitness.png)


### Immediate actions to take: 
1) The crossover() function doesnt have a good system for mating parents. Currently, you can select how many parents you want to enter a lottery of sorts. Out of these parents, two are randomly selected. I have three options in mind, i can implement one of them: 
	- Option 1: Best parent gets to mate with random other parent.
	- Option 2: All parents above the threshold mate randomly to produce the offspring.
	- Option 3: The best mates with many, the second best mates with a few, the third best mates with fewer... etc ...

2) Figure out what the implications are of negative AIC values. 

3) Read the paper Eran sent to me.

## 27/3 2024
Today's agenda: 
- Read Eran's paper
- Meet with Eran to discuss project
- Think about how to improve the stochastic components of the GA: mutations, crossovers
- Implement a solution for the crossover() function
  
### Eran's Paper Take-aways
- Evaluate the GA by counting: True positives, True negatives, False negatives and False positives.

### Implementing N-point crossover for the crossover() function
There are three conventional ways of crossing over: one-point crossover, N-point crossover and uniform crossover. I will do a N-point crossover where the user can dictate the number of crossover points. This means that crossovers are randomly generated, sorted, then the parental chromosomes are alternately taken from one and the other at those crossover points. 

![Alt text](https://github.com/AndrewBergman1/mGPS_GA/blob/main/GA_feature_selection.py.drawio.png?raw=true)
![Alt text](https://github.com/AndrewBergman1/mGPS_GA/blob/main/mGPS_GA_workflow.drawio.png?raw=true)
