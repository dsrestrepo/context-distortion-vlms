#from src.datasets import load_mimic
#from src.datasets import load_mbrset
#from src.datasets import load_medeval
#from src.datasets import load_brset

from src.test import generate_predictions_models_base
from tqdm import tqdm
import pandas as pd

model_dict = {
    "gemini-3-pro-preview":     "gemini_3_pro"
}

print("#"*100)
print("Evaluating MIMIC with General VLMs")
print("#"*100)


metadata_test = pd.read_csv('multi_history_5_class.csv')
#load_mimic(train=False, validation=False,check_images=False)

#versions = None 
versions = ["v2", "v3"] # ["default", "v1", "v2", "v3"]  
priority_img = False

generate_predictions_models_base(model_dict, metadata_test, return_attention=False, return_logits=True, dataset="cxr_multi_history", store_columns=["dicom_id", "age", "sex", "race", "report", 'contradictory_cxr_prior', 'distractor_mri_brain', 'distractor_ct_abd_pelvis', 'distractor_wrist_ultrasound', 'distractor_knee_xray'], label="label", unmatched=False, history_cols=['contradictory_cxr_prior', 'distractor_mri_brain', 'distractor_ct_abd_pelvis', 'distractor_wrist_ultrasound', 'distractor_knee_xray'], versions=versions, priority_img=priority_img)