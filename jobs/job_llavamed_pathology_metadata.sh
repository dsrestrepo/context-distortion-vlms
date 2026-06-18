#!/bin/bash
#SBATCH --job-name=llavamed_path_meta
#SBATCH --output=logs/llavamed_pathology_metadata.out
#SBATCH --open-mode=truncate
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpua100
#SBATCH --mem=56000

module load miniforge3/25.3.0-3/none-none
module load cuda/12.2.2/none-none
source activate llava-med

python -m scripts.run_mimic_targets --model-family llavamed --experiments pathology_metadata
