#!/bin/bash
#SBATCH --job-name=llavamed_brset
#SBATCH --output=logs/llavamed_brset.out
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

# Run your Python script
python -m scripts.run_llavamed_brset
