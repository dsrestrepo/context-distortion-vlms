from src.datasets import load_mimic
from src.datasets import load_mbrset
from src.datasets import load_medeval
from src.datasets import load_brset

from src.test import generate_predictions_models
from tqdm import tqdm
import pandas as pd

model_dict = {
    "Qwen/Qwen2-VL-7B-Instruct": "qwen2_vl_7b",
    "llava-hf/llava-1.5-7b-hf": "llava_1_5_7b",
    
    
    #"llava-hf/llava-v1.6-mistral-7b-hf": "llava_1_6_mistral",
    #"llava-hf/llava-v1.6-vicuna-7b-hf": "llava_1_6_vicuna",
    #"llava-hf/llama3-llava-next-8b-hf": "llama3_llava_8b",
    #"google/paligemma2-10b-pt-224": 'paligemma2_10b',
    
    
    "deepseek-ai/Janus-Pro-7B": 'janus_pro_7b',
    "meta-llama/Llama-3.2-11B-Vision-Instruct": "llama3_10b",
    "google/medgemma-4b-it": 'medgemma',
    "google/medgemma-1.5-4b-it": 'medgemma_1_5b'
}

print("#"*100)
print("Evaluating MIMIC with General VLMs")
print("#"*100)

N_SAMPLES = 500 # 250  # The desired number of samples for each category

metadata_test = load_mimic(train=False, validation=False, check_images=False)
#metadata_test = metadata_test.iloc[:10000]

# Define the full set of disease label columns
disease_cols = [
    'Atelectasis',
    'Cardiomegaly',
    'Consolidation',
    'Edema',
    'Enlarged Cardiomediastinum',
    'Fracture',
    'Lung Lesion',
    'Lung Opacity',
    'No Finding',
    'Pleural Effusion',
    'Pleural Other',
    'Pneumonia',
    'Pneumothorax'
]

# Define the specific 5 labels of interest
target_labels = [
    'Atelectasis',
    'Cardiomegaly',
    'Consolidation',
    'Edema',
    'Pleural Effusion'
]

# --- Filter for Single-Label Images ---
clean_df = metadata_test.copy()

# Count how many positive labels per row (excluding 'No Finding' for disease count)
disease_only_cols = [col for col in disease_cols if col != 'No Finding']
clean_df["num_disease_labels"] = clean_df[disease_only_cols].sum(axis=1)

# Rows with exactly 1 label (and that label is not 'No Finding')
single_disease_label_df = clean_df[clean_df["num_disease_labels"] == 1].copy()

# --- Create the Targeted Single-Label Disease Dataset ---

# Filter for rows where the single label belongs to the 'target_labels' set (the 5 classes)
target_label_mask = single_disease_label_df[target_labels].sum(axis=1) == 1

# Apply the mask
target_disease_df = single_disease_label_df[target_label_mask]

# Sample n images, ensuring the size doesn't exceed available data
n_disease = min(N_SAMPLES, len(target_disease_df))
final_disease_df = target_disease_df.sample(n=n_disease, random_state=42)
print(f"Final Disease Dataset Size: {len(final_disease_df)}")

# --- Create the Healthy Dataset ('No Finding' only) ---

# A healthy image has exactly one 'No Finding' label AND zero disease labels.
healthy_df = clean_df[
    (clean_df['No Finding'] == 1) &
    (clean_df["num_disease_labels"] == 0)
].copy()

# Sample n healthy images
n_healthy = min(N_SAMPLES, len(healthy_df))
final_healthy_df = healthy_df.sample(n=n_healthy, random_state=42)
print(f"Final Healthy Dataset Size: {len(final_healthy_df)}")

# --- Combine Datasets (Optional) ---
metadata_test = pd.concat([final_disease_df, final_healthy_df])

#versions = None 
versions = ["default", "v1", "v2", "v3"]  
priority_img = False

#generate_predictions_models(model_dict, metadata_test, quantization="16b", use_flash_attention=True, return_attention=False, return_logits=True, dataset="mimic_5class_test", store_columns=["dicom_id", "age", "sex", "race"], label="class_label", text_col="report", image_col="filepath", metadata_cols=["age", "sex", "race", "PerformedProcedureStepDescription", "ViewPosition"], unmatched=True, versions=versions, tokens=['yes', 'no', 'Yes', 'No', 'None', 'none'])
generate_predictions_models(model_dict, metadata_test, quantization="16b", use_flash_attention=True, return_attention=False, return_logits=True, dataset="mimic_5class", store_columns=["dicom_id", "age", "sex", "race"], label="class_label", text_col="report", image_col="filepath", metadata_cols=["age", "sex", "race", "PerformedProcedureStepDescription", "ViewPosition"], unmatched=False, versions=versions, priority_img=priority_img)
