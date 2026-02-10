#!/bin/bash
#SBATCH --job-name=openai_history
#SBATCH --output=python_job_openai_history.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu_long       
#SBATCH --mem=32000                # Memory in MB

# Load the Anaconda module
module load anaconda3/2024.06/gcc-13.2.0

# Activate the Conda environment
source activate base_ml

# Run the batch evaluation script for history bias
python eval_g_openai_history.py
