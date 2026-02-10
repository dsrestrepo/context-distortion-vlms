# Context Distorts Decisions in Vision Language Models for Healthcare

This repository contains the code and experiments for the paper **"Context distorts decisions in vision language models for healthcare"**. We investigate how textual context (clinical history) and prompt variations affect the diagnostic performance of Vision-Language Models (VLMs) in medical settings.

---

## 📁 Repository Structure

### Core Evaluation Scripts
These scripts run the main experiments described in the paper:

*   **5-Class Classification Task (Modality Bias)**
    *   `eval_general_vlms_5class.py`: Run open-source VLMs (e.g., LLaVA, Qwen2, MedGemma).
    *   `eval_g_openai_5class.py`: Run GPT-based models.
    *   `eval_gemini_5class.py`: Run Gemini models.
    *   `eval_llavamed_5class.py`: Run LLaVA-Med.

*   **History context Task**
    *   `eval_general_vlms_history.py`: Evaluate effect of contradictory/irrelevant history on open-source VLMs.
    *   `eval_g_openai_history.py`: Evaluate history effects on GPT models.
    *   `eval_gemini_history.py`: Evaluate history effects on Gemini models.
    *   `eval_llavamed_history.py`: Evaluate history effects on LLaVA-Med.

### Data Generation
*   `src/generate_histories.py`: Script to generate contradictory and irrelevant prior reports for the history bias experiments. (Requires OpenAI API key).

### Configuration & Utilities
*   `src/`: Contains core implementation logic (datasets, prompts, model wrappers).
*   `utils.py`: Helper functions for attention analysis and plotting.
*   `environment.base_ml.yml`: Conda environment for general VLMs.
*   `environment.llava-med.yml`: Conda environment for LLaVA-Med.
*   `job*.sh`: SLURM submission scripts for HPC environments.

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone <repository_url>
cd <repository_folder>
```

### 2. Set up environments

For **General VLMs (LLaVA, Qwen, MedGemma, etc.)**:
```bash
conda env create -f environment.base_ml.yml
conda activate base_ml
```

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

## 🚀 Running Experiments

You can run experiments locally or submit them via SLURM.

**Example: Running open-source models on 5-class task**
```bash
conda activate base_ml
python eval_general_vlms_5class.py
```

**Example: Running Gemini on history task**
```bash
conda activate base_ml
python eval_gemini_history.py
```

**SLURM Jobs**
Scripts are provided in `job_*.sh`. For example:
```bash
sbatch job_gemini_5class.sh
```

---

## 📊 Analysis

Notebooks for analyzing results:
*   `multi_prompt_eval_mimic-5class.ipynb`
*   `multi_prompt_eval_mimic-multi-history.ipynb`

These notebooks contain code to reproduce the figures and tables from the paper, including accuracy plots, flip rates, and calibration curves.

---

## 📄 Citation

If you use this codebase, please cite our paper:

*Context distorts decisions in vision language models for healthcare*

