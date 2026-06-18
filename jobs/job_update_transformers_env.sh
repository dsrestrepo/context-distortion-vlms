#!/bin/bash
#SBATCH --job-name=update_transformers_env
#SBATCH --output=logs/update_transformers_env.out
#SBATCH --open-mode=truncate
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000

set -euo pipefail

module load miniforge3/25.3.0-3/none-none
module load cuda/12.2.2/none-none

bash scripts/update_transformers_env.sh
