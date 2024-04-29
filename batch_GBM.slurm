#!/bin/bash
#SBATCH --job-name=GA-Job            # Job name
#SBATCH --nodes=1                    # Number of nodes to use
#SBATCH --ntasks=1                   # Total number of tasks across all nodes
#SBATCH --cpus-per-task=32            # CPU cores per task
#SBATCH --mem=32G                     # Memory per node
#SBATCH --time=90:00:00              # Time limit hrs:min:sec
#SBATCH --mail-user=andrew.jg.bergman@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# Load the module for Anaconda/Miniconda if it's not automatically initialized
module load Anaconda3/2024.02-1  # Adjust as per module available on your cluster

# Activate your Conda environment
source activate /home/andrewb/miniconda3/envs/GA_feature_selection

python /home/andrewb/GA_feature_selection/GBC.py
