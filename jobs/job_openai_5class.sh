#!/bin/bash
#SBATCH --job-name=openai_5class
#SBATCH --output=logs/openai_5class.out
#SBATCH --open-mode=truncate
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu_long       
#SBATCH --mem=32000                # Memory in MB

# Load the Anaconda module
module load miniforge3/25.3.0-3/none-none
module load cuda/12.2.2/none-none

# Activate the Conda environment
source activate base_ml

# Run the batch evaluation script for 5-class modality bias
python -m scripts.run_5class --model-family openai
