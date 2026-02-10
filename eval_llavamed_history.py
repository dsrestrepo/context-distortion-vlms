#from src.datasets import load_mimic
#from src.datasets import load_mbrset
#from src.datasets import load_medeval

from src.test import generate_predictions_models_base

from tqdm import tqdm
import pandas as pd

model_dict = {
    "microsoft/llava-med-v1.5-mistral-7b": "llava_med",
}

conv_mode = "mistral_instruct"
#conv_mode = "llava_v0"
#conv_mode = "llava_v1"

print("#"*100)
print("Evaluating MIMIC with Microsoft's LLava-Med model")
print("#"*100)


metadata_test = pd.read_csv('multi_history_5_class.csv')#.iloc[:20, :]

#versions = None 
versions = ["default", "v1", "v2", "v3"]  
priority_img = False

#generate_predictions_models_base(model_dict, metadata_test, quantization="16b", use_flash_attention=True, return_attention=False, return_logits=True, dataset="cxr_multi_history_test", store_columns=["dicom_id", "age", "sex", "race", "report", 'contradictory_cxr_prior', 'distractor_mri_brain', 'distractor_ct_abd_pelvis', 'distractor_wrist_ultrasound', 'distractor_knee_xray'], label="label", unmatched=False, history_cols=['contradictory_cxr_prior', 'distractor_mri_brain', 'distractor_ct_abd_pelvis', 'distractor_wrist_ultrasound', 'distractor_knee_xray'], conv_mode=conv_mode, versions=versions, tokens=['yes', 'no', 'Yes', 'No', 'None', 'none'])
generate_predictions_models_base(model_dict, metadata_test, quantization=None, use_flash_attention=True, return_attention=False, return_logits=True, dataset="cxr_multi_history", store_columns=["dicom_id", "age", "sex", "race", "report", 'contradictory_cxr_prior', 'distractor_mri_brain', 'distractor_ct_abd_pelvis', 'distractor_wrist_ultrasound', 'distractor_knee_xray'], label="label", unmatched=False, history_cols=['contradictory_cxr_prior', 'distractor_mri_brain', 'distractor_ct_abd_pelvis', 'distractor_wrist_ultrasound', 'distractor_knee_xray'], conv_mode=conv_mode, versions=versions, priority_img=priority_img)


#print("#"*100)
#print("Evaluating VLMed with Microsoft's LLava-Med model")
#print("#"*100)

#metadata_train, metadata_val, metadata_test = load_medeval()

#generate_predictions_models(model_dict, metadata_test, quantization="16b", return_attention=False, return_logits=True, dataset="medeval", store_columns=["filename", "age", "sex", "gender", "race", "ethnicity", "language", "maritalstatus"], label="glaucoma", conv_mode=conv_mode, unmatched=True)



#print("#"*100)
#print("Evaluating mBRSET with Microsoft's LLava-Med model")
#print("#"*100)

#metadata_test = load_mbrset(train=False, validation=False,check_images=False)
#metadata_test = metadata_test.iloc[:5000]

#generate_predictions_models(model_dict, metadata_test, quantization="16b", return_attention=False, return_logits=False, dataset="mbrset", store_columns=["filepath", "age", "sex", "insurance", 'educational_level', 'alcohol_consumption', 'smoking', 'obesity'], label="final_icdr", image_col="filepath")
