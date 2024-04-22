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
- Read Eran's paper - COMPLETE
- Meet with Eran to discuss project - COMPLETE
- Think about how to improve the stochastic components of the GA: mutations, crossovers - COMPLETE
- Implement a solution for the crossover() function - COMPLETE
- Try to implement CPU paralelization in the program.
  
### Eran's Paper Take-aways
- Evaluate the GA by counting: True positives, True negatives, False negatives and False positives.

### Implementing N-point crossover for the crossover() function
There are three conventional ways of crossing over: one-point crossover, N-point crossover and uniform crossover. I will do a N-point crossover where the user can dictate the number of crossover points. This means that crossovers are randomly generated, sorted, then the parental chromosomes are alternately taken from one and the other at those crossover points. 

Current GA_feature_selection.py workflow
![Alt text](https://github.com/AndrewBergman1/mGPS_GA/blob/main/flowcharts/GA_feature_selection.py.drawio.png)

Current project plan 
![Alt text](https://github.com/AndrewBergman1/mGPS_GA/blob/main/flowcharts/mGPS_GA_workflow.drawio.png)

I ran a 200 generation long GA. The GA is seemingly still optimizing the solution, i will next try the full file for 500 generations:
![Alt text](https://github.com/AndrewBergman1/mGPS_GA/blob/main/200_gen_tests/Min%20CP:%200.3,%20Max%20CP:%200.9,%20Mut.%20Prob.%200.05,%20No.%20Offspring:%20100,%20Init%20pop%20size:%20100,%20Reproductive%20Units:%202,%20Generations:%20200,%20Crossover%20points:%203.png?raw=true)

## 28/3 2024
I ran GA_feature_selection.py for 500 generations with all data. The program has been running for 22 hours on the university server. Observing 'TIME+', it looks like most cores have only been active for a couple of minutes. This means that the majority of computations are still single-threaded. A reasonable fix for the future is further parallellization of the program. Surely, all steps can be further effectivized. 
	- Paralelize the recombination step
	- See if other parts of the code can be paralelised.
All data for 500 generations: 
The data is 3600 column x 4000 rows. On average, each individual has 2000 predictors in the multiple linear regression. This means that there are 8,000,000,000 calculations per individual per generation. As there are 100 individuals for 500 generations, this means 3,200,000,000,000,000 calculations (three quadrillion 200 trillion calculations).


## 29/3 2024
The 500 generation test failed. I got the following error message: posx and posy should be finite values, leading to a failure to plot the graph. 

I'll re-run 50 generations using the full data to see how it stacks up against the 50 generations ran using 100 observations.

The 50 generation run has the same plotting issue. 

I have now ran some short GAs on the full data [FAILED], the first 1000 entries [FAILED], the first 100 entries [SUCCESS]. I will now try the first 500 entries [SUCCESS]. First 750 failed due to 'city longitude' and 'city latitude' not being present. 

Either the data gets fucked further down the dataset or the multiple linear regression might not be able to handle the data im throwing at it. If the issue is the latter, then i think i have to apply some regularization of the data to remove collienarity. The collinearity might lead to strange values on the coefficients, which in turn leads to a failure to retrieve the AIC values. I should also try to reduce the number of possible predictor variables (just using a subset of them).
	- The fact that first_750 fails suggests that there might be data missing from the meta-data file. I should add a filter, so that if the corr. long and lat are unavailable, the observation is removed!!!
	
## 2/4 2024
The situation is as follows: 
	1. When providing 500 observations, the GA runs properly. 
	2. When providing 750 observations, the multiple linear regression is poorly fitted, leading to AIC = nan. 


Today's worK:
	1. I initially tried calculating VIF for each evualuation of fitness, this proved to be computationally intensive.
	2. I then decided to calculate VIF for all predictor variables before running the GA_feature_selection.py. The VIF calculations didnt work out because some predictor variables seem to be fully correlated. This leads to division by 0 (VIF=1/(1-R^2)). 
	3. I then decided to generate a correlation heatmap to visualise the correlation between different microorganisms. 
	4. I will decide on a correlation threshold at which i'll discard predictor variables.
	5. Calculate VIFs and handle VIFs where division by 0 occurs (catch R2=0).
		- I have generated the VIFs and collected them in a .csv file. The R2=0 still needs to be sorted out...

## 4/4 2024
I calculated the VIFs using all predictor variables and either received "inf" or nothing for each predictor variable. VIF calculations follow this equation: 1/(1-R^2). A multiple linear regression model is fitted for each predictor variable using the rest of them. In order to gain interpretable VIF values i generated a correlation matrix (pd.corr()). Using it, i removed every predictor variable that had correlations with other variables exceeding 0.7 (absolute value). I then calculated the VIFs again, now yielding interpretable values. When i was calculating VIFs using few observations, the error persisted, when calculating it with the full dataset, values were good. 

After having calculated the VIFs, i filtered away predictor varaibles with VIF > 5, as they would cause overfitting of the model. 

Followingly, I tried to run the GA. However, the AIC values returned from the multiple linear regression were still NaN. I found online that Ridge Regularisation could be used in cases where there are very many variables with relatively few observations (as in this case). Thus, i excchanged the multiple linear regression for ridge regressin. The ridge regression is based on multiple linear regression, but instead of optimising for least sum squared, it optimises for least sum squared AND (the slope)^2 * lambda (penalisation constant). In order to find the appropriate penalisation constant i employed 10-fold cross validation. I permited values from 0-39 (step=2). 

## 6/4 2024
Since i hogged the penthouse server, DAg canceled the GA. I removed the 10-fold cross validation to improve performance and makje the runs quicker. I improved the output to include R2, model parameters and all predictor variables. 

I also made a new script for making model predictions. It's not complete yet, first i need to retrieve the validation data.


## 15/4 2024
After a long run (50 generations), the predictive ability of latitude is poor (MSE = 1316). I will remove the early stopping function since it seems to halt the GA too much. 
![Alt text](https://github.com/AndrewBergman1/mGPS_GA/blob/main/results/GA_epochs/15042024.png)

Model predictive ability (OBS: R2 IS IN REALITY MSE!!!!!!!): 
![Alt text](https://github.com/AndrewBergman1/mGPS_GA/blob/main/results/model_performance/150424.png)

## 17/4 2024
Ridge regression (MSE ~ 1300) yields poor predictive ability. The predicted data is centered around lat = 0, perhaps suggesting that the coefficients are penalized to hard. Using cross-validation, alpha = 1040 (950 - 1550, step = 10). The reasoning behind using ridge regression was to deal with the significcant multicollinearity exhibited by some taxa, resulting in either VIF = None or VIF = inf. I have constructed a correlation matrix and removed one of the variables, if abs(corr) > 0.6. Then i filtered for VIF < 5. This reduces multicollinearity and should provide grounds to use MLS again.

I also realized lat_pred_mlr.py had logical issues, where the predictor variables imported from the model didn't correspond to the features that were selected for predictions. This has been resolved. 

## 18/4 2024
I have made adjustments to the prediction-script, it should now work properly. Eran spoke to me about trying to use the metasub-subset that Leo McCarthy used when developing mGPS. I have commenced a run using 100 individuals for 50 generations, we'll see how it fares. It could be a good idea to reduce the number of predictor variables in each individuals (instead of Td = 0.5, perhaps Td = 0.9). This would yield faster run times as well as fewer gits in the end. Another point of improvement would be having more crossover points.

## 20/4 2024
559415 : 3 crossover points, Td = 0.5
561511 : 10 CO, Td = 0.9

## 24/4 2024
Good results from the GA. I've ran lasso.py 0.3 0.9 0.05 500 500 2 500 3 and the following results were yielded. MSE = 122. I will try to further reduce it by implementing crossvalidation.
![Alt text](https://github.com/AndrewBergman1/mGPS_GA/blob/main/results/GA_epochs/220424.png)
![Alt text](https://github.com/AndrewBergman1/mGPS_GA/blob/main/results/model_performance/220424.png)

Although the results are a lot better than what's previously been attained, the outcome is still not good enough. 