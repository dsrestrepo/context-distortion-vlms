import pandas as pd
import random

##############################################################
############ Reusable instruction strings ############
##############################################################

# Mismatch/uncertainty token
MISMATCH_TOKEN = "None"

# Binary response instructions (yes/no only)
BINARY_INSTRUCTION_2OPT = 'Your response must be "Yes" if any condition is detected or "No" if no abnormality is detected. Your response should be only "yes" or "no" (without additional commentary or reasoning).'

# Ternary response instructions (yes/no/?)
BINARY_INSTRUCTION_3OPT = f'Your response must be "yes" if any condition is detected, "no" if no abnormality is detected, or "{MISMATCH_TOKEN}" if there is a mismatch between the image and the text or the information is not conclusive. Your response should be only "yes", "no", or "{MISMATCH_TOKEN}" (without additional commentary or reasoning).'

# 5-class specific instructions (yes/no for specified conditions)
FIVECLASS_INSTRUCTION_2OPT = 'Your response must be "Yes" if any of the specified conditions is detected or "No" if none of the specified conditions is detected. Your response should be only "yes" or "no" (without additional commentary or reasoning).'

# 5-class with uncertainty option
FIVECLASS_INSTRUCTION_3OPT = f'Your response must be "yes" if any of the specified conditions is detected, "no" if none of the specified conditions is detected, or "{MISMATCH_TOKEN}" if there is a mismatch between the image and the text or the information is not conclusive. Your response should be only "yes", "no", or "{MISMATCH_TOKEN}" (without additional commentary or reasoning).'

# Image-only binary instructions
IMAGE_ONLY_BINARY_INSTRUCTION_2OPT = 'Your response must be "Yes" if any condition is detected or "No" if no abnormality is detected. Your response should be only "yes" or "no" (without additional commentary or reasoning).'

# Image-only with uncertainty
IMAGE_ONLY_BINARY_INSTRUCTION_3OPT = f'Your response must be "yes" if any condition is detected, "no" if no abnormality is detected, or "{MISMATCH_TOKEN}" if the information is not conclusive. Your response should be only "yes", "no", or "{MISMATCH_TOKEN}" (without additional commentary or reasoning).'

# Image-only 5-class instructions
IMAGE_ONLY_FIVECLASS_INSTRUCTION_2OPT = 'Your response must be "Yes" if any of the specified conditions is detected or "No" if none of the specified conditions is detected. Your response should be only "yes" or "no" (without additional commentary or reasoning).'

# Image-only 5-class with uncertainty
IMAGE_ONLY_FIVECLASS_INSTRUCTION_3OPT = f'Your response must be "yes" if any of the specified conditions is detected, "no" if none of the specified conditions is detected, or "{MISMATCH_TOKEN}" if the information is not conclusive. Your response should be only "yes", "no", or "{MISMATCH_TOKEN}" (without additional commentary or reasoning).'

# V1 style instructions
V1_INSTRUCTION_2OPT = 'Reply with only **"yes"** if any are present, or **"no"** if none are detected. No additional commentary or reasoning.'
V1_INSTRUCTION_3OPT = f'Reply with only **"yes"** if any are present, **"no"** if none are detected, or **"{MISMATCH_TOKEN}"** if there is a mismatch between the image and the text or the information is not conclusive. No additional commentary or reasoning.'
V1_IMAGE_ONLY_INSTRUCTION_3OPT = f'Reply with only **"yes"** if any are present, **"no"** if none are detected, or **"{MISMATCH_TOKEN}"** if the information is not conclusive. No additional commentary or reasoning.'

# V2 style instructions
V2_INSTRUCTION_2OPT = 'Respond with a single word: **"yes"** (if any condition is present) or **"no"** (if none are found). No additional commentary or reasoning.'
V2_INSTRUCTION_3OPT = f'Respond with **"yes"** (if any condition is present), **"no"** (if none are found), or **"{MISMATCH_TOKEN}"** (if there is a mismatch between the image and the text or the information is not conclusive). No additional commentary or reasoning.'
V2_IMAGE_ONLY_INSTRUCTION_3OPT = f'Respond with **"yes"** (if any condition is present), **"no"** (if none are found), or **"{MISMATCH_TOKEN}"** (if the information is not conclusive). No additional commentary or reasoning.'

# V3 style instructions
V3_INSTRUCTION_2OPT = ('Mark **"yes"** if any of these findings are identified on the X-ray (considering the report and patient data).  \n' +
                        'Mark **"no"** if none of them apply.\n\n' +
                        'Final output: only "yes" or "no". No additional reasoning or checklist. Just \'yes\' or \'no\'.')
V3_INSTRUCTION_3OPT = (f'Mark **"yes"** if any of these findings are identified on the X-ray (considering the report and patient data), **"no"** if none of them apply, or **"{MISMATCH_TOKEN}"** if there is a mismatch between the image and the text or the information is not conclusive.\n\n' +
                        f'Final output: only "yes", "no", or "{MISMATCH_TOKEN}". No additional reasoning or checklist. Just "yes", "no", or "{MISMATCH_TOKEN}".')

# V3 image-only instructions
V3_IMAGE_ONLY_INSTRUCTION_2OPT = ('Mark **"yes"** if any of these findings are identified on the X-ray.\n' +
                                   'Mark **"no"** if none of them apply.\n\n' +
                                   'Final output: only "yes" or "no". No additional reasoning or checklist. Just \'yes\' or \'no\'.')
V3_IMAGE_ONLY_INSTRUCTION_3OPT = (f'Mark **"yes"** if any of these findings are identified on the X-ray, **"no"** if none of them apply, or **"{MISMATCH_TOKEN}"** if the information is not conclusive.\n\n' +
                                   f'Final output: only "yes", "no", or "{MISMATCH_TOKEN}". No additional reasoning or checklist. Just "yes", "no", or "{MISMATCH_TOKEN}".')

# V3 text-only instructions
V3_TEXT_ONLY_INSTRUCTION_2OPT = ('Mark **"yes"** if any of these findings are identified (considering the report and patient data).\n' +
                                  'Mark **"no"** if none of them apply.\n\n' +
                                  'Final output: only "yes" or "no". No additional reasoning or checklist. Just \'yes\' or \'no\'.')
V3_TEXT_ONLY_INSTRUCTION_3OPT = (f'Mark **"yes"** if any of these findings are identified (considering the report and patient data), **"no"** if none of them apply, or **"{MISMATCH_TOKEN}"** if the information is not conclusive.\n\n' +
                                  f'Final output: only "yes", "no", or "{MISMATCH_TOKEN}". No additional reasoning or checklist. Just "yes", "no", or "{MISMATCH_TOKEN}".')

# History-based V1 instructions
HISTORY_V1_INSTRUCTION_2OPT = 'Reply with only **"yes"** if any are present, or **"no"** if none are detected. No additional commentary or reasoning.'
HISTORY_V1_INSTRUCTION_3OPT = f'Reply with only **"yes"** if any are present, **"no"** if none are detected, or **"{MISMATCH_TOKEN}"** if there is a mismatch between the image and the text or the information is not conclusive. No additional commentary or reasoning.'

# History-based V2 instructions
HISTORY_V2_INSTRUCTION_2OPT = 'Respond with a single word:\n- **"yes"** if one or more conditions are present\n- **"no"** if none are present. No additional commentary or reasoning.'
HISTORY_V2_INSTRUCTION_3OPT = f'Respond with:\n- **"yes"** if one or more conditions are present\n- **"no"** if none are present\n- **"{MISMATCH_TOKEN}"** if there is a mismatch between the image and the text or the information is not conclusive. No additional commentary or reasoning.'

# History-based V3 instructions
HISTORY_V3_INSTRUCTION_2OPT = 'Final output: return only **"yes"** if any checklist item is present; otherwise return **"no"**.\n\nFinal output: only "yes" or "no". No additional reasoning or checklist. Just \'yes\' or \'no\'.'
HISTORY_V3_INSTRUCTION_3OPT = f'Final output: return only **"yes"** if any checklist item is present, **"no"** if none are present, or **"{MISMATCH_TOKEN}"** if there is a mismatch between the image and the text or the information is not conclusive.\n\nFinal output: only "yes", "no", or "{MISMATCH_TOKEN}". No additional reasoning or checklist. Just "yes", "no", or "{MISMATCH_TOKEN}".'

##############################################################
############ Helper functions for date generation ############
##############################################################

def _synthetic_current_date(year=2024):
    """Generate a random synthetic current date within a given year."""
    month = random.randint(1, 12)
    # ensure valid day for the month
    day = random.randint(1, 28)  # 28 avoids month-end issues (Feb, etc.)
    return pd.Timestamp(year=year, month=month, day=day)

def _synthetic_past_date(current_dt, min_months=6, max_months=12):
    """Return a random date between 6 and 12 months before current_dt."""
    months_back = random.randint(min_months, max_months)
    # choose a small random day offset for variety
    past = current_dt - pd.DateOffset(months=months_back)
    return past - pd.Timedelta(days=random.randint(0, 30))


##############################################################
###### Helper functions for BRSET and mBRSET generation ######
##############################################################

def convert_sex_brset(sex):
    return {1: "male", 2: "female"}.get(sex, "no sex reported")

def convert_eye(eye):
    return {1: "right", 2: "left"}.get(eye, "no eye reported")

def convert_anatomical(val):
    return {1: "normal", 2: "abnormal"}.get(val, "unknown")

def convert_condition(val):
    return "present" if val == 1 else "absent"


def convert_sex_mbrset(sex):
    return {1: "male", 0: "female"}.get(sex, "sex not reported")

def binary_to_text(value, true_text, false_text):
    return true_text if value == 1 else false_text

education_map = {
    1.0: "illiterate",
    2.0: "with incomplete primary education",
    3.0: "with complete primary education",
    4.0: "with incomplete secondary education",
    5.0: "with complete secondary education",
    6.0: "with incomplete tertiary education",
    7.0: "with complete tertiary education"
}


#########################################################################
####################### BRSET and mBRSET prompts ########################
#########################################################################


######################## Image and Text Prompts #########################
def BRSET_TEXT_PROMPT(row):
    # Age
    age_phrase = (
        f"aged {float(str(row['patient_age']).replace('O', '0').replace(',', '.'))} years"
        if not pd.isnull(row['patient_age'])
        else "with age not reported"
    )

    # Diabetes duration
    diabetes_phrase = (
        f"diagnosed with diabetes for {float(str(row['diabetes_time_y']).replace('O', '0').replace(',', '.'))} years"
        if not pd.isnull(row['diabetes_time_y']) and row['diabetes_time_y'] != 'Não'
        else "with no reported diabetes duration"
    )

    # Comorbidities
    comorb_phrase = (
        "with no comorbidities reported"
        if pd.isnull(row['comorbidities'])
        else f"with comorbidities: {row['comorbidities']}"
    )

    # Insulin use
    insulin_phrase = (
        "using insulin" if str(row["insuline"]).strip().lower() == "yes" else "not using insulin"
    )

    # Anatomical description
    anatomy = (
        f"The optic disc appears {convert_anatomical(row['optic_disc'])}, "
        f"the vessels are {convert_anatomical(row['vessels'])}, "
        f"and the macula is {convert_anatomical(row['macula'])}."
    )

    # Disease/condition labels
    condition_fields = [
        "macular_edema", "scar", "nevus", "amd", "vascular_occlusion",
        "hypertensive_retinopathy", "drusens", "hemorrhage", "retinal_detachment",
        "myopic_fundus", "increased_cup_disc", "other"
    ]

    conditions = ", ".join(
        f"{field.replace('_', ' ')}: {convert_condition(row[field])}"
        for field in condition_fields
    )

    # Compose the full description
    description = (
        f"A {convert_sex_brset(row['patient_sex'])} patient {age_phrase}, "
        f"{diabetes_phrase}, {insulin_phrase}, and {comorb_phrase}. "
        f"{anatomy} Conditions include: {conditions}."
    )

    # Compose the final prompt
    return f"""
{description}

Based on the provided patient information and the associated fundus image, does the patient has Diabetic Retinopathy (DR)?

Respond with **yes** if the patient has any level of diabetic retinopathy (ICDR score ≥ 1), or **no** if the score is 0. 
According to the International Clinical Diabetic Retinopathy (ICDR) classification, an eye is considered ICDR 0 when no retinal abnormalities related to diabetic retinopathy are present. ICDR ≥1 indicates the presence of any diabetic retinopathy, defined by the observation of one or more characteristic lesions such as microaneurysms, intraretinal hemorrhages, hard exudates,  venous beading, intraretinal microvascular abnormalities (IRMA), neovascularization, or vitreous/preretinal hemorrhage. Additionally, the presence of panretinal (panphotocoagulation) laser scars is considered evidence of treated proliferative diabetic retinopathy.

Respond only with "yes" or "no" (without additional commentary).
""".strip()


def mBRSET_TEXT_PROMPT(row):
    # Age
    age_phrase = (
        f"aged {row['age']} years" 
        if not pd.isnull(row['age']) 
        else "with age not reported"
    )

    # Diabetes duration
    diabetes_phrase = (
        f"diagnosed with diabetes for {row['dm_time']} years" 
        if not pd.isnull(row['dm_time']) 
        else "with no reported diabetes duration"
    )

    # Educational level
    education = education_map.get(row['educational_level'], "with no educational level reported")

    # Build descriptions
    sex = convert_sex_mbrset(row['sex'])
    insulin = binary_to_text(row['insulin'], "using insulin", "not using insulin")
    oral = binary_to_text(row['oraltreatment_dm'], "on oral treatment for diabetes", "not on oral treatment for diabetes")
    hypertension = binary_to_text(row['systemic_hypertension'], "with systemic hypertension", "without systemic hypertension")
    alcohol = binary_to_text(row['alcohol_consumption'], "consumes alcohol", "does not consume alcohol")
    smoking = binary_to_text(row['smoking'], "smokes", "does not smoke")
    obesity = binary_to_text(row['obesity'], "with obesity", "without obesity")
    vascular = binary_to_text(row['vascular_disease'], "has vascular disease", "does not have vascular disease")
    infarction = binary_to_text(row['acute_myocardial_infarction'], "has a history of acute myocardial infarction", "no history of acute myocardial infarction")
    nephropathy = binary_to_text(row['nephropathy'], "with nephropathy", "without nephropathy")
    neuropathy = binary_to_text(row['neuropathy'], "with neuropathy", "without neuropathy")
    diabetic_foot = binary_to_text(row['diabetic_foot'], "has diabetic foot", "does not have diabetic foot")

    # Compose patient description
    description = (
        f"A {sex} patient {age_phrase}, {diabetes_phrase}, {insulin}, and {oral}. "
        f"The patient is {hypertension}, {alcohol}, {smoking}, {obesity}, and {vascular}. "
        f"Medical history includes: {infarction}, {nephropathy}, {neuropathy}, and {diabetic_foot}. "
        f"The patient is {education}."
    )

    # LLM prompt
    return f"""
{description}

Based on the provided patient information and the associated fundus image, does the patient has Diabetic Retinopathy (DR)?

Respond with **yes** if the patient has any level of diabetic retinopathy (ICDR score ≥ 1), or **no** if the score is 0. 
According to the International Clinical Diabetic Retinopathy (ICDR) classification, an eye is considered ICDR 0 when no retinal abnormalities related to diabetic retinopathy are present. ICDR ≥1 indicates the presence of any diabetic retinopathy, defined by the observation of one or more characteristic lesions such as microaneurysms, intraretinal hemorrhages, hard exudates,  venous beading, intraretinal microvascular abnormalities (IRMA), neovascularization, or vitreous/preretinal hemorrhage. Additionally, the presence of panretinal (panphotocoagulation) laser scars is considered evidence of treated proliferative diabetic retinopathy.

Respond only with "yes" or "no" (without additional commentary).
""".strip()



######################## Only Image Prompts #########################

BRSET_ONLY_IMAGE_TEXT_PROMPT = f"""
Based on the image, does the patient has Diabetic Retinopathy (DR)?

Respond with **yes** if the patient has any level of diabetic retinopathy (ICDR score ≥ 1), or **no** if the score is 0. 
According to the International Clinical Diabetic Retinopathy (ICDR) classification, an eye is considered ICDR 0 when no retinal abnormalities related to diabetic retinopathy are present. ICDR ≥1 indicates the presence of any diabetic retinopathy, defined by the observation of one or more characteristic lesions such as microaneurysms, intraretinal hemorrhages, hard exudates,  venous beading, intraretinal microvascular abnormalities (IRMA), neovascularization, or vitreous/preretinal hemorrhage. Additionally, the presence of panretinal (panphotocoagulation) laser scars is considered evidence of treated proliferative diabetic retinopathy.

Respond only with "yes" or "no" (without additional commentary).
"""

mBRSET_ONLY_IMAGE_TEXT_PROMPT = f"""
Based on the image, does the patient has Diabetic Retinopathy (DR)?

Respond with **yes** if the patient has any level of diabetic retinopathy (ICDR score ≥ 1), or **no** if the score is 0. 
According to the International Clinical Diabetic Retinopathy (ICDR) classification, an eye is considered ICDR 0 when no retinal abnormalities related to diabetic retinopathy are present. ICDR ≥1 indicates the presence of any diabetic retinopathy, defined by the observation of one or more characteristic lesions such as microaneurysms, intraretinal hemorrhages, hard exudates,  venous beading, intraretinal microvascular abnormalities (IRMA), neovascularization, or vitreous/preretinal hemorrhage. Additionally, the presence of panretinal (panphotocoagulation) laser scars is considered evidence of treated proliferative diabetic retinopathy.

Respond only with "yes" or "no" (without additional commentary).
"""


######################## Only Text Prompts #########################

# Define the text prompt for BRSET.
def BRSET_ONLY_TEXT_PROMPT(row):
    # Age
    age_phrase = (
        f"aged {float(str(row['patient_age']).replace('O', '0').replace(',', '.'))} years"
        if not pd.isnull(row['patient_age'])
        else "with age not reported"
    )

    # Diabetes duration
    diabetes_phrase = (
        f"diagnosed with diabetes for {float(str(row['diabetes_time_y']).replace('O', '0').replace(',', '.'))} years"
        if not pd.isnull(row['diabetes_time_y']) and row['diabetes_time_y'] != 'Não'
        else "with no reported diabetes duration"
    )

    # Comorbidities
    comorb_phrase = (
        "with no comorbidities reported"
        if pd.isnull(row['comorbidities'])
        else f"with comorbidities: {row['comorbidities']}"
    )

    # Insulin use
    insulin_phrase = (
        "using insulin" if str(row["insuline"]).strip().lower() == "yes" else "not using insulin"
    )

    # Anatomical description
    anatomy = (
        f"The optic disc appears {convert_anatomical(row['optic_disc'])}, "
        f"the vessels are {convert_anatomical(row['vessels'])}, "
        f"and the macula is {convert_anatomical(row['macula'])}."
    )

    # Disease/condition labels
    condition_fields = [
        "macular_edema", "scar", "nevus", "amd", "vascular_occlusion",
        "hypertensive_retinopathy", "drusens", "hemorrhage", "retinal_detachment",
        "myopic_fundus", "increased_cup_disc", "other"
    ]

    conditions = ", ".join(
        f"{field.replace('_', ' ')}: {convert_condition(row[field])}"
        for field in condition_fields
    )

    # Compose the full description
    description = (
        f"A {convert_sex_brset(row['patient_sex'])} patient {age_phrase}, "
        f"{diabetes_phrase}, {insulin_phrase}, and {comorb_phrase}. "
        f"{anatomy} Conditions include: {conditions}."
    )

    # Compose the final prompt
    return f"""
{description}

Based on the provided patient information, does the patient has Diabetic Retinopathy (DR)?

Respond with **yes** if the patient has any level of diabetic retinopathy (ICDR score ≥ 1), or **no** if the score is 0. 
According to the International Clinical Diabetic Retinopathy (ICDR) classification, an eye is considered ICDR 0 when no retinal abnormalities related to diabetic retinopathy are present. ICDR ≥1 indicates the presence of any diabetic retinopathy, defined by the observation of one or more characteristic lesions such as microaneurysms, intraretinal hemorrhages, hard exudates,  venous beading, intraretinal microvascular abnormalities (IRMA), neovascularization, or vitreous/preretinal hemorrhage. Additionally, the presence of panretinal (panphotocoagulation) laser scars is considered evidence of treated proliferative diabetic retinopathy.

Respond only with "yes" or "no" (without additional commentary).
""".strip()



def mBRSET_ONLY_TEXT_PROMPT(row):
    # Age
    age_phrase = (
        f"aged {row['age']} years" 
        if not pd.isnull(row['age']) 
        else "with age not reported"
    )

    # Diabetes duration
    diabetes_phrase = (
        f"diagnosed with diabetes for {row['dm_time']} years" 
        if not pd.isnull(row['dm_time']) 
        else "with no reported diabetes duration"
    )

    # Educational level
    education = education_map.get(row['educational_level'], "with no educational level reported")

    # Build descriptions
    sex = convert_sex_mbrset(row['sex'])
    insulin = binary_to_text(row['insulin'], "using insulin", "not using insulin")
    oral = binary_to_text(row['oraltreatment_dm'], "on oral treatment for diabetes", "not on oral treatment for diabetes")
    hypertension = binary_to_text(row['systemic_hypertension'], "with systemic hypertension", "without systemic hypertension")
    alcohol = binary_to_text(row['alcohol_consumption'], "consumes alcohol", "does not consume alcohol")
    smoking = binary_to_text(row['smoking'], "smokes", "does not smoke")
    obesity = binary_to_text(row['obesity'], "with obesity", "without obesity")
    vascular = binary_to_text(row['vascular_disease'], "has vascular disease", "does not have vascular disease")
    infarction = binary_to_text(row['acute_myocardial_infarction'], "has a history of acute myocardial infarction", "no history of acute myocardial infarction")
    nephropathy = binary_to_text(row['nephropathy'], "with nephropathy", "without nephropathy")
    neuropathy = binary_to_text(row['neuropathy'], "with neuropathy", "without neuropathy")
    diabetic_foot = binary_to_text(row['diabetic_foot'], "has diabetic foot", "does not have diabetic foot")

    # Compose patient description
    description = (
        f"A {sex} patient {age_phrase}, {diabetes_phrase}, {insulin}, and {oral}. "
        f"The patient is {hypertension}, {alcohol}, {smoking}, {obesity}, and {vascular}. "
        f"Medical history includes: {infarction}, {nephropathy}, {neuropathy}, and {diabetic_foot}. "
        f"The patient is {education}."
    )

    # LLM prompt
    return f"""
{description}

Based on the provided patient information, does the patient have Diabetic Retinopathy (DR)?

Respond with **yes** if the patient has any level of diabetic retinopathy (ICDR score ≥ 1), or **no** if the score is 0. 
According to the International Clinical Diabetic Retinopathy (ICDR) classification, an eye is considered ICDR 0 when no retinal abnormalities related to diabetic retinopathy are present. ICDR ≥1 indicates the presence of any diabetic retinopathy, defined by the observation of one or more characteristic lesions such as microaneurysms, intraretinal hemorrhages, hard exudates,  venous beading, intraretinal microvascular abnormalities (IRMA), neovascularization, or vitreous/preretinal hemorrhage. Additionally, the presence of panretinal (panphotocoagulation) laser scars is considered evidence of treated proliferative diabetic retinopathy.

Respond only with "yes" or "no" (without additional commentary).
""".strip()


#########################################################################
########################### HAM 10000 prompts ###########################
#########################################################################


HAM10000_TEXT_PROMPT_FULL = lambda metadata_row: f"""
Patient Information:
- Age: {metadata_row.get('age', 'Not Available')}
- Sex: {metadata_row.get('sex', 'Not Available')}
- Localization: {metadata_row.get('localization', 'Not Available')}
- Diagnostic Technique: {metadata_row.get('dx_type', 'Not Available')}

Based on the provided patient information and the dermoscopic image, please analyze the image and determine the diagnosis from the following list:
• akiec (Actinic keratoses and intraepithelial carcinoma / Bowen's disease)
• bcc (Basal cell carcinoma)
• bkl (Benign keratosis-like lesions, e.g., solar lentigines, seborrheic keratoses, lichen-planus like keratoses)
• df (Dermatofibroma)
• mel (Melanoma)
• nv (Melanocytic nevi)
• vasc (Vascular lesions, e.g., angiomas, angiokeratomas, pyogenic granulomas, hemorrhage)

Your answer must be exactly one label from the above list: 'nv', 'bkl', 'mel', 'akiec', 'bcc', 'vasc', 'df'. Write it exactly like that and do not include any additional text or commentary.
"""

HAM10000_TEXT_PROMPT_BINARY = lambda metadata_row: f"""
Patient Information:
- Age: {metadata_row.get('age', 'Not Available')}
- Sex: {metadata_row.get('sex', 'Not Available')}
- Localization: {metadata_row.get('localization', 'Not Available')}
- Diagnostic Technique: {metadata_row.get('dx_type', 'Not Available')}

Based on the provided patient information and the dermoscopic image, please analyze the image and determine the diagnosis from the following list:
• mel (Melanoma)
• nv (Melanocytic nevi)

Your answer must be exactly one label from the above list: 'nv' or 'mel'. Write it exactly like that and do not include any additional text or commentary.
"""


HAM10000_ONLY_IMAGE_TEXT_PROMPT = f"""
Based on the image only, please analyze the image and determine the diagnosis from the following list:
• akiec (Actinic keratoses and intraepithelial carcinoma / Bowen's disease)
• bcc (Basal cell carcinoma)
• bkl (Benign keratosis-like lesions, e.g., solar lentigines, seborrheic keratoses, lichen-planus like keratoses)
• df (Dermatofibroma)
• mel (Melanoma)
• nv (Melanocytic nevi)
• vasc (Vascular lesions, e.g., angiomas, angiokeratomas, pyogenic granulomas, hemorrhage)

Your answer must be exactly one label from the above list: 'nv', 'bkl', 'mel', 'akiec', 'bcc', 'vasc', 'df'. Write it exactly like that and do not include any additional text or commentary.
"""

HAM10000_ONLY_TEXT_PROMPT_FULL = lambda metadata_row: f"""
Patient Information:
- Age: {metadata_row.get('age', 'Not Available')}
- Sex: {metadata_row.get('sex', 'Not Available')}
- Localization: {metadata_row.get('localization', 'Not Available')}
- Diagnostic Technique: {metadata_row.get('dx_type', 'Not Available')}

Based on the provided patient information, please analyze the image and determine the diagnosis from the following list:
• akiec (Actinic keratoses and intraepithelial carcinoma / Bowen's disease)
• bcc (Basal cell carcinoma)
• bkl (Benign keratosis-like lesions, e.g., solar lentigines, seborrheic keratoses, lichen-planus like keratoses)
• df (Dermatofibroma)
• mel (Melanoma)
• nv (Melanocytic nevi)
• vasc (Vascular lesions, e.g., angiomas, angiokeratomas, pyogenic granulomas, hemorrhage)

Your answer must be exactly one label from the above list: 'nv', 'bkl', 'mel', 'akiec', 'bcc', 'vasc', 'df'. Write it exactly like that and do not include any additional text or commentary.
"""

HAM10000_ONLY_TEXT_PROMPT_BINARY = lambda metadata_row: f"""
Patient Information:
- Age: {metadata_row.get('age', 'Not Available')}
- Sex: {metadata_row.get('sex', 'Not Available')}
- Localization: {metadata_row.get('localization', 'Not Available')}
- Diagnostic Technique: {metadata_row.get('dx_type', 'Not Available')}

Based on the provided patient information, please analyze the image and determine the diagnosis from the following list:
• mel (Melanoma)
• nv (Melanocytic nevi)

Your answer must be exactly one label from the above list: 'nv' or 'mel'. Write it exactly like that and do not include any additional text or commentary.
"""

#########################################################################
####################### Harvard-FairVLMed prompts #######################
#########################################################################


GLAUCOMA_TEXT_PROMPT = lambda metadata_row: f"""
    Patient Information:
    - Age: {metadata_row['age']}
    - Gender: {metadata_row['gender']}
    - Race: {metadata_row['race']}
    - Language: {metadata_row['language']}
    - Marital Status: {metadata_row['maritalstatus']}
    - Ethnicity: {metadata_row['ethnicity']}
    - Summary: {metadata_row['note']}   
    Based on this image and patient information, does the patient have glaucoma? 
    Respond with "yes" or "no". (without additional commentary)
    """



GLAUCOMA_ONLY_IMAGE_TEXT_PROMPT = f"""
Based on the image only, does the patient have glaucoma?
Respond with "yes" or "no". (without additional commentary)
"""



GLAUCOMA_ONLY_TEXT_PROMPT = lambda metadata_row: f"""
    Patient Information:
    - Age: {metadata_row['age']}
    - Gender: {metadata_row['gender']}
    - Race: {metadata_row['race']}
    - Language: {metadata_row['language']}
    - Marital Status: {metadata_row['maritalstatus']}
    - Ethnicity: {metadata_row['ethnicity']}
    - Summary: {metadata_row['note']}   
    Based on this patient information, does the patient have glaucoma? 
    Respond with "yes" or "no". (without additional commentary)
    """


#########################################################################
####################### MIMIC-CXR prompts #######################
#########################################################################


##################################################################################
########################### TODO: Revise prompts below ###########################
##################################################################################

""" VALSE MIMIC-CXR """
CXR_VALSE_TEXT_PROMPT = lambda row, original=True: f"""
You are an expert chest-radiology assistant.

Radiology report (verbatim):
{row['report'] if original else row['modified_report']}

Question (answer very concisely):
{row['question']}

✦ Respond **only with the answer** required by the question – no extra words.
* In case that the report does not provide enough information to answer the question, respond with "uncertain".
* If the image and report disagree significantly with respect to the question, respond with "contradictory".
"""


CXR_VALSE_BINARY_TEXT_PROMPT_BINARY = lambda row, original=True: f"""
You are an expert chest-radiology assistant.

Radiology report (verbatim):
{row['report'] if original else row['modified_report']}
Question (The questions are always "yes" and "no questions, but you can additionally find "uncertain" and "contradictory" as answers. Nothing else.):
{row['question']}
✦ Respond **only with the answer** required by the question - no extra words.

Possible answers are:
- "yes" : if the answer is affirmative;
- "no" : if the answer is negative;
- "uncertain": if the report does not provide enough information to answer the question;
- "contradictory": if the report and image disagree significantly.
"""


#### History-based prompt ####

CXR_HISTORY_TEXT_PROMPT = lambda row, original=True, history_cols_to_use=None: f"""
You are an expert chest-radiology assistant.

'Prior reports:'
{'--- --- ---'.join( [str(row[col]) for col in history_cols_to_use if row[col]]) if not original else 'No prior report'}


Current chest X-ray report:
{row['report']}

Based on the provided patient information and the associated chest X-ray image, does the patient has any condition (Pleural Effusion, Atelectasis, Cardiomegaly, Consolidation, Edema, Enlarged Cardiomediastinum, Fracture, Lung Lesion, Lung Opacity, Pleural Other, Pneumonia, or Pneumothorax)?

Your response must be "Yes" if any condition is detected or "No" if no abnormality is detected. Your response should be only "yes" or "no" (without additional commentary).
"""


#### Multimodal prompt ####

MIMIC_TEXT_PROMPT = lambda metadata_row, unmatched=False: f"""
Patient Information:
- Age: {metadata_row.get('age', 'Not Available')}
- Sex: {metadata_row.get('sex', 'Not Available')}
- Race: {metadata_row.get('race', 'Not Available')}
- ViewPosition: {metadata_row.get('ViewPosition', 'Not Available')}
- Procedure description: {metadata_row.get('PerformedProcedureStepDescription', 'N/A')}
- Summary: {metadata_row.get('report', 'N/A')}

Based on the provided patient information and the associated chest X-ray image, does the patien has any condition (Pleural Effusion, Atelectasis, Cardiomegaly, Consolidation, Edema, Enlarged Cardiomediastinum, Fracture, Lung Lesion, Lung Opacity, Pleural Other, Pneumonia, or Pneumothorax)?

""" + (BINARY_INSTRUCTION_3OPT if unmatched else BINARY_INSTRUCTION_2OPT)

### Only Image prompts ###

MIMIC_ONLY_IMAGE_TEXT_PROMPT = lambda unmatched=False: f"""
Based on the image only, does the patient has any condition (Pleural Effusion, Atelectasis, Cardiomegaly, Consolidation, Edema, Enlarged Cardiomediastinum, Fracture, Lung Lesion, Lung Opacity, Pleural Other, Pneumonia, or Pneumothorax)?
""" + (IMAGE_ONLY_BINARY_INSTRUCTION_3OPT if unmatched else IMAGE_ONLY_BINARY_INSTRUCTION_2OPT)

### Only Text prompts ###
MIMIC_ONLY_TEXT_PROMPT = lambda metadata_row, unmatched=False: f"""
Patient Information:
- Age: {metadata_row.get('age', 'Not Available')}
- Sex: {metadata_row.get('sex', 'Not Available')}
- Race: {metadata_row.get('race', 'Not Available')}
- ViewPosition: {metadata_row.get('ViewPosition', 'Not Available')}
- Procedure description: {metadata_row.get('PerformedProcedureStepDescription', 'N/A')}
- Summary: {metadata_row.get('report', 'N/A')}

Based on the provided patient information, does the patien has any condition (Pleural Effusion, Atelectasis, Cardiomegaly, Consolidation, Edema, Enlarged Cardiomediastinum, Fracture, Lung Lesion, Lung Opacity, Pleural Other, Pneumonia, or Pneumothorax)?

""" + (BINARY_INSTRUCTION_3OPT if unmatched else BINARY_INSTRUCTION_2OPT)


### =================================================== ###
### 5-class classification prompts for MIMIC-CXR ###
### ================================================== ###

### Multimodal prompt ####
MIMIC_TEXT_PROMPT_5CLASS  = lambda metadata_row, unmatched=False: f"""
Patient Information:
- Age: {metadata_row.get('age', 'Not Available')}
- Sex: {metadata_row.get('sex', 'Not Available')}
- Race: {metadata_row.get('race', 'Not Available')}
- ViewPosition: {metadata_row.get('ViewPosition', 'Not Available')}
- Procedure description: {metadata_row.get('PerformedProcedureStepDescription', 'N/A')}
- Summary: {metadata_row.get('report', 'N/A')}

Based on the provided patient information and the associated chest X-ray image, does the patient have any of the following conditions: **Atelectasis**, **Cardiomegaly**, **Consolidation**, **Edema**, or **Pleural Effusion**?
""" + (FIVECLASS_INSTRUCTION_3OPT if unmatched else FIVECLASS_INSTRUCTION_2OPT)

### Only Image prompts ###
MIMIC_ONLY_IMAGE_TEXT_PROMPT_5CLASS  = lambda unmatched=False: f"""
You are an expert chest-radiology assistant.

Based on the provided chest X-ray image only, does the patient have any of the following conditions: **Atelectasis**, **Cardiomegaly**, **Consolidation**, **Edema**, or **Pleural Effusion**?
""" + (IMAGE_ONLY_FIVECLASS_INSTRUCTION_3OPT if unmatched else IMAGE_ONLY_FIVECLASS_INSTRUCTION_2OPT)

### Only Text prompts ###

MIMIC_ONLY_TEXT_PROMPT_5CLASS  = lambda metadata_row, unmatched=False: f"""
Patient Information:
- Age: {metadata_row.get('age', 'Not Available')}
- Sex: {metadata_row.get('sex', 'Not Available')}
- Race: {metadata_row.get('race', 'Not Available')}
- ViewPosition: {metadata_row.get('ViewPosition', 'Not Available')}
- Procedure description: {metadata_row.get('PerformedProcedureStepDescription', 'N/A')}
- Summary: {metadata_row.get('report', 'N/A')}

Based on the provided patient information, does the patient have any of the following conditions: **Atelectasis**, **Cardiomegaly**, **Consolidation**, **Edema**, or **Pleural Effusion**?
""" + (FIVECLASS_INSTRUCTION_3OPT if unmatched else FIVECLASS_INSTRUCTION_2OPT)


############# VARIATION 1 #############

### Multimodal prompt ####
MIMIC_TEXT_PROMPT_5CLASS_V1 = lambda metadata_row, unmatched=False: f"""
You are assisting in a radiology review task. Follow these steps carefully:

1. Examine the chest X-ray image.
2. Review the following patient details:
   - Age: {metadata_row.get('age', 'Not Available')}
   - Sex: {metadata_row.get('sex', 'Not Available')}
   - Race: {metadata_row.get('race', 'Not Available')}
   - View position: {metadata_row.get('ViewPosition', 'Not Available')}
   - Procedure: {metadata_row.get('PerformedProcedureStepDescription', 'N/A')}
3. Read the radiology summary below:
   "{metadata_row.get('report', 'N/A')}"

Now integrate visual and textual information, giving priority to the image, to decide whether the case shows **any** of the following:
Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion.

""" + (V1_INSTRUCTION_3OPT if unmatched else V1_INSTRUCTION_2OPT)

### Only Image prompt ####
MIMIC_ONLY_IMAGE_TEXT_PROMPT_5CLASS_V1 = lambda unmatched=False: f"""
You are assisting in a radiology review task. Follow these steps carefully:

1. Examine the chest X-ray image.

Now decide whether the case shows **any** of the following:
Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion.

""" + (V1_IMAGE_ONLY_INSTRUCTION_3OPT if unmatched else V1_INSTRUCTION_2OPT)

### Only Text prompt ####

MIMIC_ONLY_TEXT_PROMPT_5CLASS_V1 = lambda metadata_row, unmatched=False: f"""
You are assisting in a radiology review task. Follow these steps carefully:

1. Review the following patient details:
   - Age: {metadata_row.get('age', 'Not Available')}
   - Sex: {metadata_row.get('sex', 'Not Available')}
   - Race: {metadata_row.get('race', 'Not Available')}
   - View position: {metadata_row.get('ViewPosition', 'Not Available')}
   - Procedure: {metadata_row.get('PerformedProcedureStepDescription', 'N/A')}
2. Read the radiology summary below:
   "{metadata_row.get('report', 'N/A')}"

Now decide whether the case shows **any** of the following:
Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion.

""" + (V1_IMAGE_ONLY_INSTRUCTION_3OPT if unmatched else V1_INSTRUCTION_2OPT)

############# VARIATION 2 #############

MIMIC_TEXT_PROMPT_5CLASS_V2 = lambda metadata_row, unmatched=False: f"""
=== RADIOLOGY CHECK REQUEST ===

Case summary:
Patient is a {metadata_row.get('age', 'Unknown age')}-year-old {metadata_row.get('sex', 'Unknown sex')} ({metadata_row.get('race', 'Unknown race')}).

Exam details:
• Projection: {metadata_row.get('ViewPosition', 'Not Available')}
• Procedure: {metadata_row.get('PerformedProcedureStepDescription', 'N/A')}

Clinical note:
"{metadata_row.get('report', 'N/A')}"

Instructions for analysis:
Additionally, review the chest X-ray image and determine if you can confirm the presence of **any** of the following:
→ Atelectasis  
→ Cardiomegaly  
→ Consolidation  
→ Edema  
→ Pleural Effusion  

""" + (V2_INSTRUCTION_3OPT if unmatched else V2_INSTRUCTION_2OPT)

### Only Image prompt ####
MIMIC_ONLY_IMAGE_TEXT_PROMPT_5CLASS_V2 = lambda unmatched=False: f"""
=== RADIOLOGY CHECK REQUEST ===

Instructions for analysis:

Review the chest X-ray image and determine if you can confirm the presence of **any** of the following:

→ Atelectasis
→ Cardiomegaly
→ Consolidation
→ Edema
→ Pleural Effusion

""" + (V2_IMAGE_ONLY_INSTRUCTION_3OPT if unmatched else V2_INSTRUCTION_2OPT)

### Only Text prompt ####
MIMIC_ONLY_TEXT_PROMPT_5CLASS_V2 = lambda metadata_row, unmatched=False: f"""
=== RADIOLOGY CHECK REQUEST ===

Case summary:
Patient is a {metadata_row.get('age', 'Unknown age')}-year-old {metadata_row.get('sex', 'Unknown sex')} ({metadata_row.get('race', 'Unknown race')}).

Exam details:
• Projection: {metadata_row.get('ViewPosition', 'Not Available')}
• Procedure: {metadata_row.get('PerformedProcedureStepDescription', 'N/A')}

Clinical note:
"{metadata_row.get('report', 'N/A')}"

Instructions for analysis:
Determine if you can confirm the presence of **any** of the following:
→ Atelectasis  
→ Cardiomegaly  
→ Consolidation  
→ Edema  
→ Pleural Effusion  

""" + (V2_IMAGE_ONLY_INSTRUCTION_3OPT if unmatched else V2_INSTRUCTION_2OPT)


############# VARIATION 3 #############

MIMIC_TEXT_PROMPT_5CLASS_V3 = lambda metadata_row, unmatched=False: (
    f"""
You are performing a structured diagnostic checklist on a chest X-ray.

Patient file:
────────────────────────────
Age: {metadata_row.get('age', 'Not Available')}
Sex: {metadata_row.get('sex', 'Not Available')}
Race: {metadata_row.get('race', 'Not Available')}
Projection: {metadata_row.get('ViewPosition', 'Not Available')}
Procedure: {metadata_row.get('PerformedProcedureStepDescription', 'N/A')}
────────────────────────────

Radiology Summary:
{metadata_row.get('report', 'N/A')}
────────────────────────────

Diagnostic Checklist. Don't provide the full checklist, just answer the question:
[ ] Atelectasis  
[ ] Cardiomegaly  
[ ] Consolidation  
[ ] Edema  
[ ] Pleural Effusion  

""" + (V3_INSTRUCTION_3OPT if unmatched else V3_INSTRUCTION_2OPT)
)

### Only Image prompt ####
MIMIC_ONLY_IMAGE_TEXT_PROMPT_5CLASS_V3 = lambda unmatched=False: (
    f"""
You are performing a structured diagnostic checklist on a chest X-ray.

Diagnostic Checklist. Don't provide the full checklist, just answer the question:

[ ] Atelectasis  
[ ] Cardiomegaly  
[ ] Consolidation  
[ ] Edema  
[ ] Pleural Effusion

""" + (V3_IMAGE_ONLY_INSTRUCTION_3OPT if unmatched else V3_IMAGE_ONLY_INSTRUCTION_2OPT)
)

### Only Text prompt ####
MIMIC_ONLY_TEXT_PROMPT_5CLASS_V3 = lambda metadata_row, unmatched=False: (
    f"""
You are performing a structured diagnostic checklist on a chest X-ray.

Patient file:
────────────────────────────
Age: {metadata_row.get('age', 'Not Available')}
Sex: {metadata_row.get('sex', 'Not Available')}
Race: {metadata_row.get('race', 'Not Available')}
Projection: {metadata_row.get('ViewPosition', 'Not Available')}
Procedure: {metadata_row.get('PerformedProcedureStepDescription', 'N/A')}
────────────────────────────

Radiology Summary:
{metadata_row.get('report', 'N/A')}
────────────────────────────

Diagnostic Checklist. Don't provide the full checklist, just answer the question:
[ ] Atelectasis  
[ ] Cardiomegaly  
[ ] Consolidation  
[ ] Edema  
[ ] Pleural Effusion  

""" + (V3_TEXT_ONLY_INSTRUCTION_3OPT if unmatched else V3_TEXT_ONLY_INSTRUCTION_2OPT)
)



### ++++++++++++++++++++++ ###
#### History-based prompt ####
### ++++++++++++++++++++++ ###

def CXR_HISTORY_TEXT_PROMPT_5CLASS(row, original=True, history_cols_to_use=None, unmatched=False):
    # pick a random current date within 2024
    current_dt = _synthetic_current_date(year=2024)

    # Build dated prior section
    if not original and history_cols_to_use:
        pieces = []
        for col in history_cols_to_use:
            txt = row.get(col, None)
            if txt and isinstance(txt, str) and txt.strip():
                date = _synthetic_past_date(current_dt)
                pieces.append(f"[Report Date: {date:%Y-%m-%d}]\n{txt.strip()}")
        prior_block = "\n--- --- ---\n".join(pieces) if pieces else "No prior report"
    else:
        prior_block = "No prior report"

    base_prompt = f"""
You are an expert chest-radiology assistant.

Prior reports:
{prior_block}

Current chest X-ray report (Study Date: {current_dt:%Y-%m-%d}):
{row['report']}

Based on the provided patient information and the associated chest X-ray image, does the patient have any of the following conditions: **Atelectasis**, **Cardiomegaly**, **Consolidation**, **Edema**, or **Pleural Effusion**?
"""
    
    base_prompt += FIVECLASS_INSTRUCTION_3OPT if unmatched else FIVECLASS_INSTRUCTION_2OPT
    
    return base_prompt.strip()



############# VARIATION 1 #############

def CXR_HISTORY_TEXT_PROMPT_5CLASS_V1(row, original=True, history_cols_to_use=None, unmatched=False):
    # pick a random current date within 2024
    current_dt = _synthetic_current_date(year=2024)

    # Build dated prior section
    if not original and history_cols_to_use:
        pieces = []
        for col in history_cols_to_use:
            txt = row.get(col, None)
            if txt and isinstance(txt, str) and txt.strip():
                date = _synthetic_past_date(current_dt)
                pieces.append(f"[Report Date: {date:%Y-%m-%d}]\n{txt.strip()}")
        prior_block = "\n--- --- ---\n".join(pieces) if pieces else "No prior report"
    else:
        prior_block = "No prior report"

    base_prompt = f"""
You are assisting in a radiology review task. Follow these steps carefully:

1) Examine the chest X-ray image.
2) Review prior radiology reports if any:
{prior_block}

3) Review the current chest X-ray report (Study Date: {current_dt:%Y-%m-%d}):
{row['report']}

Now integrate the image with the textual context to decide whether the case shows **any** of:
Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion.

"""
    
    base_prompt += HISTORY_V1_INSTRUCTION_3OPT if unmatched else HISTORY_V1_INSTRUCTION_2OPT
    
    return base_prompt.strip()

############# VARIATION 2 #############
def CXR_HISTORY_TEXT_PROMPT_5CLASS_V2(row, original=True, history_cols_to_use=None, unmatched=False):
    # pick a random current date within 2024
    current_dt = _synthetic_current_date(year=2024)

    # Build dated prior section
    if not original and history_cols_to_use:
        pieces = []
        for col in history_cols_to_use:
            txt = row.get(col, None)
            if txt and isinstance(txt, str) and txt.strip():
                date = _synthetic_past_date(current_dt)
                pieces.append(f"[Report Date: {date:%Y-%m-%d}]\n{txt.strip()}")
        prior_block = "\n--- --- ---\n".join(pieces) if pieces else "No prior report"
    else:
        prior_block = "No prior report"

    base_prompt = f"""
=== RADIOLOGY CHECK REQUEST ===

PRIOR REPORTS:
{prior_block}

CURRENT STUDY (Study Date: {current_dt:%Y-%m-%d}):
{row['report']}

Instructions:
Review the chest X-ray image and, using the reports above only as supporting context, confirm whether **any** of the following are present:
→ Atelectasis
→ Cardiomegaly
→ Consolidation
→ Edema
→ Pleural Effusion

"""
    
    base_prompt += HISTORY_V2_INSTRUCTION_3OPT if unmatched else HISTORY_V2_INSTRUCTION_2OPT
    
    return base_prompt.strip()


############# VARIATION 3 #############
def CXR_HISTORY_TEXT_PROMPT_5CLASS_V3(row, original=True, history_cols_to_use=None, unmatched=False):
    # pick a random current date within 2024
    current_dt = _synthetic_current_date(year=2024)

    # Build dated prior section
    if not original and history_cols_to_use:
        pieces = []
        for col in history_cols_to_use:
            txt = row.get(col, None)
            if txt and isinstance(txt, str) and txt.strip():
                date = _synthetic_past_date(current_dt)
                pieces.append(f"[Report Date: {date:%Y-%m-%d}]\n{txt.strip()}")
        prior_block = "\n--- --- ---\n".join(pieces) if pieces else "No prior report"
    else:
        prior_block = "No prior report"

    base_prompt = f"""
Structured Diagnostic Checklist — Chest X-ray

PRIOR REPORTS
────────────────────────
{prior_block}
────────────────────────

CURRENT REPORT (Study Date: {current_dt:%Y-%m-%d})
────────────────────────
{row['report']}
────────────────────────

Checklist (evaluate on the image; use text as context only). Don't provide the full checklist, just answer the question:
[ ] Atelectasis
[ ] Cardiomegaly
[ ] Consolidation
[ ] Edema
[ ] Pleural Effusion

"""
    
    base_prompt += HISTORY_V3_INSTRUCTION_3OPT if unmatched else HISTORY_V3_INSTRUCTION_2OPT
    
    return base_prompt.strip()

