#!/bin/bash
#SBATCH -t 2:00:00
#SBATCH --mail-user=andrew.jg.bergman@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH -J GA_run
#SBATCH --mem=32G
#SBATCH --nodes=2 
#SBATCH --ntasks-per-node=5

# Load the module for Anaconda/Miniconda
module load Anaconda3/2024.02-1

# Initialize Conda and activate the environment
source /home/andrewb/miniconda3/etc/profile.d/conda.sh
conda activate /home/andrewb/miniconda3/envs/GA_feature_selection


# Run your Python script with scoop
/home/andrewb/miniconda3/envs/GA_feature_selection/bin/python3 -m scoop -n 5 /home/andrewb/GA_feature_selection/package_GA_city.py || echo "Script failed with exit code $?"