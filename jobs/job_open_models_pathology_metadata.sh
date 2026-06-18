#!/bin/bash
#SBATCH --job-name=open_models_path_meta
#SBATCH --output=logs/open_models_pathology_metadata.out
#SBATCH --open-mode=truncate
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpua100
#SBATCH --mem=56000

set -euo pipefail

module load miniforge3/25.3.0-3/none-none
module load cuda/12.2.2/none-none
source activate base_ml

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_HUB_ETAG_TIMEOUT=60
export VLM_LOCAL_FILES_ONLY="${VLM_LOCAL_FILES_ONLY:-0}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
SRUN_GPU_ARGS=(--ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" --gres=gpu:1)
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
srun "${SRUN_GPU_ARGS[@]}" nvidia-smi
srun "${SRUN_GPU_ARGS[@]}" python -m scripts.check_cuda
srun "${SRUN_GPU_ARGS[@]}" python -m scripts.run_mimic_targets --model-family open_models --experiments pathology_metadata
