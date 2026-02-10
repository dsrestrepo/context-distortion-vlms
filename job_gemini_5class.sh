#!/bin/bash
#SBATCH --job-name=gemini_5class
#SBATCH --output=python_job_gemini_5class.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu_long       
#SBATCH --mem=32000                # Memory in MB

# Load the Anaconda module
module load anaconda3/2024.06/gcc-13.2.0

# Activate the Conda environment
source activate base_ml

# Run the batch evaluation script for 5-class modality bias
python eval_gemini_5class.py
