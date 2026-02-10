#from src.datasets import load_mimic
#from src.datasets import load_mbrset
#from src.datasets import load_medeval
#from src.datasets import load_brset

from src.test import generate_predictions_models_base
from tqdm import tqdm
import pandas as pd

model_dict = {
    #"Qwen/Qwen2-VL-7B-Instruct": "qwen2_vl_7b",
    #"llava-hf/llava-1.5-7b-hf": "llava_1_5_7b",
    
    
    #"llava-hf/llava-v1.6-mistral-7b-hf": "llava_1_6_mistral",
    #"llava-hf/llava-v1.6-vicuna-7b-hf": "llava_1_6_vicuna",
    #"llava-hf/llama3-llava-next-8b-hf": "llama3_llava_8b",
    #"google/paligemma2-10b-pt-224": 'paligemma2_10b',
    
    
    #"deepseek-ai/Janus-Pro-7B": 'janus_pro_7b',
    #"meta-llama/Llama-3.2-11B-Vision-Instruct": "llama3_10b",
    #"google/medgemma-4b-it": 'medgemma',
    "google/medgemma-1.5-4b-it": 'medgemma_1_5b'
}

print("#"*100)
print("Evaluating MIMIC with General VLMs")
print("#"*100)


metadata_test = pd.read_csv('multi_history_5_class.csv')#.iloc[:10, :]
#load_mimic(train=False, validation=False,check_images=False)

#versions = None 
versions = ["default", "v1", "v2", "v3"]  
priority_img = True

#generate_predictions_models_base(model_dict, metadata_test, quantization="16b", use_flash_attention=False, return_attention=False, return_logits=True, dataset="cxr_multi_history_test", store_columns=["dicom_id", "age", "sex", "race", "report", 'contradictory_cxr_prior', 'distractor_mri_brain', 'distractor_ct_abd_pelvis', 'distractor_wrist_ultrasound', 'distractor_knee_xray'], label="label", unmatched=True, history_cols=['contradictory_cxr_prior', 'distractor_mri_brain', 'distractor_ct_abd_pelvis', 'distractor_wrist_ultrasound', 'distractor_knee_xray'], versions=versions, tokens=['yes', 'no', 'Yes', 'No', 'None', 'none'])
generate_predictions_models_base(model_dict, metadata_test, quantization="16b", use_flash_attention=True, return_attention=False, return_logits=True, dataset="cxr_multi_history", store_columns=["dicom_id", "age", "sex", "race", "report", 'contradictory_cxr_prior', 'distractor_mri_brain', 'distractor_ct_abd_pelvis', 'distractor_wrist_ultrasound', 'distractor_knee_xray'], label="label", unmatched=False, history_cols=['contradictory_cxr_prior', 'distractor_mri_brain', 'distractor_ct_abd_pelvis', 'distractor_wrist_ultrasound', 'distractor_knee_xray'], versions=versions, priority_img=priority_img)
