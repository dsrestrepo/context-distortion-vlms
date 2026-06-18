#!/bin/bash
#SBATCH --job-name=check_env
#SBATCH --output=logs/check_env.out
#SBATCH --open-mode=truncate
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=cpu_long       
#SBATCH --mem=4000

# Load the Anaconda module
module load miniforge3/25.3.0-3/none-none
module load cuda/12.2.2/none-none

# Activate the Conda environment
source activate base_ml

# Check the updated multimodal Transformers interface without downloading models.
python -c "import bitsandbytes, transformers; from transformers import AutoConfig, AutoModelForImageTextToText, AutoModelForMultimodalLM, AutoProcessor, Qwen3VLForConditionalGeneration; AutoConfig.for_model('gemma4_unified'); print(f'transformers={transformers.__version__}'); print(f'bitsandbytes={bitsandbytes.__version__}'); print('Gemma 4 and required multimodal Transformers classes are registered.')"
python -m pip check || true
