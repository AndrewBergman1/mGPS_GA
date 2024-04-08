#! bin/bash

for ((i = 6; i <= 20; i++)); do
    mutation_rate=$(echo "$i/100" | bc -l)
    
    python3 GA_feature_selection.py 0.3 0.9 $mutation_rate 50 50 2 30 2
done

for ((i = 5; i <= 9; i++)); do
    max_crossover=$(echo "$i/10" | bc -l)
    
    python3 GA_feature_selection.py 0.3 $max_crossover 0.05 50 50 2 30 2
done

for ((i = 1; i <= 5; i++)); do
    min_crossover=$(echo "$i/10" | bc -l)
    
    python3 GA_feature_selection.py $min_crossover 0.9 0.05 50 50 2 30 2
done