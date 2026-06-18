#!/bin/bash

set -euo pipefail

ENV_NAME="${ENV_NAME:-base_ml}"
TRANSFORMERS_SPEC="${TRANSFORMERS_SPEC:-git+https://github.com/huggingface/transformers.git}"
BITSANDBYTES_VERSION="${BITSANDBYTES_VERSION:-0.46.1}"
BACKUP_ROOT="${BACKUP_ROOT:-environment_backups}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP_DIR="${BACKUP_ROOT}/${ENV_NAME}/${TIMESTAMP}"

if ! conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
    echo "Conda environment '${ENV_NAME}' does not exist."
    exit 1
fi

mkdir -p "${BACKUP_DIR}"

echo "Backing up '${ENV_NAME}' to '${BACKUP_DIR}'..."
conda env export --name "${ENV_NAME}" > "${BACKUP_DIR}/environment.yml"
conda list --name "${ENV_NAME}" --explicit > "${BACKUP_DIR}/conda-explicit.txt"
conda run --name "${ENV_NAME}" python -m pip freeze > "${BACKUP_DIR}/requirements.txt"

echo "Installing '${TRANSFORMERS_SPEC}' in '${ENV_NAME}'..."
conda run --name "${ENV_NAME}" python -m pip install --upgrade \
    "${TRANSFORMERS_SPEC}" \
    "bitsandbytes>=${BITSANDBYTES_VERSION}" \
    accelerate \
    safetensors

echo "Removing unused sentence-transformers package..."
conda run --name "${ENV_NAME}" python -m pip uninstall --yes sentence-transformers

echo "Running compatibility smoke test..."
conda run --name "${ENV_NAME}" python -c \
    "import bitsandbytes, transformers; from transformers import AutoConfig, AutoModelForImageTextToText, AutoModelForMultimodalLM, AutoProcessor, Qwen3VLForConditionalGeneration; AutoConfig.for_model('gemma4_unified'); print(f'transformers={transformers.__version__}'); print(f'bitsandbytes={bitsandbytes.__version__}'); print('Gemma 4 and required multimodal Transformers classes are registered.')"

echo "Checking installed package constraints..."
conda run --name "${ENV_NAME}" python -m pip check

echo
echo "Updated '${ENV_NAME}'. Previous package state: '${BACKUP_DIR}'."
