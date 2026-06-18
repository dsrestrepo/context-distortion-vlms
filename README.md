# Context Distorts Decisions in Vision Language Models for Healthcare

This repository contains the code and experiments for the paper **"Context distorts decisions in vision language models for healthcare"**. We investigate how textual context (clinical history) and prompt variations affect the diagnostic performance of Vision-Language Models (VLMs) in medical settings.

---

## Repository Structure

* `config/open_models.json`: Open and general-medical model settings.
* `config/llavamed.json`: LLaVA-Med settings.
* `config/gemini.json`: Gemini settings.
* `config/openai.json`: OpenAI settings.
* `config/experiment_defaults.json`: Shared dataset paths, columns, sampling, and random seed.
* `config/plotting.json` and `config/brset_visualization.json`: Figure-generation settings.
* `scripts/`: Thin executable entry points for experiments, plotting, BRSET
  visualization, and environment maintenance.
* `src/`: Shared datasets, preprocessing, prompts, unified VLM classes, and evaluation logic.
* `jobs/`: SLURM launchers. Each job selects a model family and calls a module in `scripts/`.
* `logs/`: Fixed-name SLURM output files. Rerunning a job overwrites its previous `.out` file.
* `images/`: Generated summary figures. Calibration plots are grouped under `images/<model_name>/`.
* `utils.py`: Attention-analysis helpers retained at the repository root.
* `environment.base_ml.yml` and `environment.llava-med.yml`: Conda environments.

The two main experiment runners are:

* `scripts/run_5class.py`: MIMIC 5-class modality-bias experiments.
* `scripts/run_history.py`: MIMIC fake/multi-history experiments.
* `scripts/run_mimic_targets.py`: Generalized MIMIC target experiments.

Both use the shared preprocessing in `src/experiment_data.py`. Select `open_models`, `gemini`, `openai`, or `llavamed` with `--model-family`. Each model family has its own config file with explicit settings for the `5class` and `history` tasks.

JSON does not support comments. To skip completed models while preserving their
definitions, set a task's optional `enabled_models` list. When omitted, every model in
the family runs.

Qwen, LLaVA, Llama 3.2 Vision, Gemma 4, GLM-V, and MedGemma use true batched generation. Set
`runtime.batch_size` for each task in its model-family config. The open-model family
defaults to `1` with 4-bit quantization for 40 GB GPUs. Kimi-VL, API models, and
LLaVA-Med use the per-item fallback interface.

### Inference Interface

All supported models use the same interface in `src/vlm.py`. A model object owns its
model/client and processor, and accepts text, an image, or both:

```python
from src.vlm import create_vlm

vlm = create_vlm(
    model_name="qwen3_vl_8b",
    model_id="Qwen/Qwen3-VL-8B-Instruct",
    quantization="16b",
    return_logits=True,
)

result = vlm.generate(
    text="Is there a pleural effusion? Answer yes or no.",
    image="example.png",
    tokens=["yes", "no"],
)
print(result.text)
print(result.token_scores)
```

The returned `GenerationResult` exposes `text`, `generated_ids`, `attentions`, `scores`,
and requested `token_scores`. Model objects also expose their `processor` and `tokenizer`
when available. This makes multi-step workflows direct: pass
`result.text` into another model's `generate()` call. The experiment runners retain
their existing CSV columns and filenames through a legacy adapter in `src/inference.py`.

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/dsrestrepo/context-distortion-vlms.git
cd context-distortion-vlms
```

### 2. Set up environments

For **General VLMs (LLaVA, Qwen, MedGemma, etc.)**:
```bash
conda env create -f environment.base_ml.yml
conda activate base_ml
```

For an existing `base_ml` environment, update it before running Qwen3-VL:
```bash
conda env update -f environment.base_ml.yml
```

To back up and update the existing `base_ml` environment for newer model architectures,
submit:

```bash
sbatch jobs/job_update_transformers_env.sh
tail -f logs/update_transformers_env.out
```

Before updating, the job stores a timestamped Conda YAML, explicit Conda package list,
and `pip freeze` under `environment_backups/base_ml/`. It then updates Transformers and
related runtime packages, runs a multimodal import smoke test that verifies Gemma 4 is
registered, and reports dependency conflicts with `pip check`. By default, Transformers
is installed from the latest official Hugging Face GitHub source. Override the
environment or installation specification when submitting:

```bash
sbatch --export=ALL,ENV_NAME=base_ml,TRANSFORMERS_SPEC=git+https://github.com/huggingface/transformers.git,BITSANDBYTES_VERSION=0.46.1 jobs/job_update_transformers_env.sh
```

The backup files preserve the previous package state in case the update needs to be
inspected or reverted.

For **LLaVA-Med**:
```bash
conda env create -f environment.llava-med.yml
conda activate llava-med
```

### 3. API Keys
Create a `.env` file or export environment variables for proprietary models:
```bash
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="AI..."
# For Hugging Face access (Llama 3, etc.)
export HF_TOKEN="hf_..."
```

---

## Running Experiments

You can run experiments locally or submit them via SLURM.

**Example: Running open-source models on 5-class task**
```bash
conda activate base_ml
python -m scripts.run_5class --model-family open_models
```

**Example: Running Gemini on history task**
```bash
conda activate base_ml
python -m scripts.run_history --model-family gemini
```

**SLURM Jobs**
Scripts are provided in `jobs/`. Submit the descriptive job filename directly:
```bash
sbatch jobs/job_open_models_5class.sh
sbatch jobs/job_gemini_5class.sh
sbatch jobs/job_llavamed_history.sh
```

`google/medgemma-27b-it` is included in `config/open_models.json`. It is gated on
Hugging Face and is substantially larger than the 4B variants, so accept its model
terms before running it and use a GPU with enough memory or change the task
quantization to `4b`.

Set a task's optional `debug_samples` value to run the existing SLURM job on a few
deterministic rows without overwriting full results:

```json
"debug_samples": 4
```

Use `null` for a full run. The task's `enabled_models` controls which models are
debugged. Debug outputs use `mimic_5class_debug_base_shifted_multi_versions.csv`.
The `--models` and `--debug-samples` command-line options remain available as overrides.

### Offline Hugging Face Runs

The `open_models` and `llavamed` families can run without internet only when the model
files, processor/tokenizer files, custom model code, Python packages, and datasets are
already present on the new cluster. API families (`openai` and `gemini`) cannot run
offline.

To force open-model jobs to use only local Hugging Face files:

```bash
sbatch --export=ALL,VLM_LOCAL_FILES_ONLY=1,HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1 jobs/job_open_models_5class.sh
```

If a required snapshot is missing, the job will fail immediately instead of trying to
download it.

### Generalized MIMIC Targets

These experiments use the same prompt-version and modality-shift loop as the MIMIC
5-class experiment: `No`, `Image`, `Text`, `Only_text`, and `Only_image`.

* `race`: predicts race from the image, report, and metadata excluding race.
* `sex`: predicts sex from the image, report, and metadata excluding sex.
* `pathology_metadata`: predicts pathology from the image and metadata only. The report
  and pathology label are excluded.

Each target has an independent job. For open models:

```bash
sbatch jobs/job_open_models_race.sh
sbatch jobs/job_open_models_sex.sh
sbatch jobs/job_open_models_pathology_metadata.sh
```

The same jobs are available for the other model families:

```bash
sbatch jobs/job_llavamed_race.sh
sbatch jobs/job_gemini_sex.sh
sbatch jobs/job_openai_pathology_metadata.sh
```

Run experiments locally:

```bash
python -m scripts.run_mimic_targets --model-family open_models --experiments race
python -m scripts.run_mimic_targets --model-family open_models --experiments sex
python -m scripts.run_mimic_targets --model-family open_models --experiments pathology_metadata
```

Results overwrite the same fixed files on each run:

```text
results/<model_name>/mimic_race_base_shifted_multi_versions.csv
results/<model_name>/mimic_sex_base_shifted_multi_versions.csv
results/<model_name>/mimic_pathology_metadata_base_shifted_multi_versions.csv
```

Change `mimic_target_samples_per_class` in `config/experiment_defaults.json` to
control the balanced sample size used for each target class.

The figures used by the two main analysis notebooks can be regenerated with:
```bash
sbatch jobs/job_generate_plots_5class.sh
sbatch jobs/job_generate_plots_multi_history.sh
```

These jobs run `scripts.generate_plots_5class` and `scripts.generate_plots_multi_history`. Both use the shared final plotting style and fixed output filenames in `images/`.

Model-specific calibration plots are stored under:
```text
images/<model_name>/calibration/<dataset_name>/
```

The plotting jobs also create:

* Regex and first-token accuracy figures.
* First-token ECE figures.
* Regex response distributions containing correct, incorrect, and refusal counts.
* First-token response distributions.

First-token distributions do not contain refusals because first-token evaluation always
selects between the available yes/no token probabilities.

---

## Analysis

The retained experiment and analysis notebooks focus on:

*   MIMIC chest X-ray experiments: `eval_mimic*.ipynb`
*   History and fake-history experiments: `eval_mimic-history.ipynb`, `eval_mimic-multi-history.ipynb`, and `multi_prompt_eval_mimic-multi-history*.ipynb`
*   BRSET and mBRSET experiments: `eval_brset.ipynb` and `eval_mbrset.ipynb`
*   Attention experiments: `llava-med_visualization*.ipynb` and general/general-medical model notebooks under `notebooks/`

Main notebooks for analyzing paper results:
*   `multi_prompt_eval_mimic-5class.ipynb`
*   `multi_prompt_eval_mimic-multi-history.ipynb`

These notebooks contain code to reproduce the figures and tables from the paper, including accuracy plots, flip rates, and calibration curves.

---

## Citation

If you use this codebase, please cite our paper:

*Context distorts decisions in vision language models for healthcare*
