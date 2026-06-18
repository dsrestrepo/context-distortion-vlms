#!/bin/bash
#SBATCH --job-name=llavamed_history
#SBATCH --output=logs/llavamed_history.out
#SBATCH --open-mode=truncate
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1               # Include if your script requires a GPU
#SBATCH --partition=gpua100       # Use an appropriate partition
#SBATCH --mem=56000                # Memory in MB

# Load the Anaconda module
module load miniforge3/25.3.0-3/none-none
module load cuda/12.2.2/none-none
# Activate the Conda environment
source activate llava-med

# Run the LLaVA-Med history experiment
python -m scripts.run_history --model-family llavamed
