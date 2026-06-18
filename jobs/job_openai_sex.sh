#!/bin/bash
#SBATCH --job-name=openai_sex
#SBATCH --output=logs/openai_sex.out
#SBATCH --open-mode=truncate
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu_long
#SBATCH --mem=32000

module load miniforge3/25.3.0-3/none-none
module load cuda/12.2.2/none-none
source activate base_ml

python -m scripts.run_mimic_targets --model-family openai --experiments sex
