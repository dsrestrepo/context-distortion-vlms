#!/bin/bash
#SBATCH --job-name=python_job
#SBATCH --output=python_job.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1               # Include if your script requires a GPU
#SBATCH --partition=gpua100       # Use an appropriate partition
#SBATCH --mem=56000                # Memory in MB

# Load the Anaconda module
module load anaconda3/2024.06/gcc-13.2.0
module load cuda/12.2.1/gcc-11.2.0
#module load cuda/11.7.0/gcc-11.2.0 

# Activate the Conda environment
source activate base_ml
#source activate llava-med

# Run your Python script
#python eval_llavamed.py
python eval_general_vlms_5class.py
