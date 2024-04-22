#!/bin/bash
#SBATCH -t 12:00:00
#SBATCH --mail-user=andrew.jg.bergman@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH -J GA_run_0.3_0.9_0.2_20_20_2_20_3
#SBATCH --mem=32G 

# Load the module for Anaconda/Miniconda if it's not automatically initialized
module load Anaconda3/2024.02-1  # Adjust as per module available on your cluster

# Activate your Conda environment
source activate /home/andrewb/miniconda3/envs/GA_feature_selection

# Run your Python script
<<<<<<< HEAD
python /home/andrewb/GA_feature_selection/long_GA.py 0.3 0.9 0.05 1000 1000 2 1000 5 || echo "Script failed with exit code $?"
=======
python /home/andrewb/GA_feature_selection/long_GA.py 0.3 0.9 0.05 1000 1000 2 500 3 || echo "Script failed with exit code $?"
>>>>>>> main
