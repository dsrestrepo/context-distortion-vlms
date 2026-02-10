import io
import base64
import numpy as np
from PIL import Image
from transformers.image_utils import load_image
from src.prompts import (
    # --- CXR VALSE ---
    CXR_VALSE_TEXT_PROMPT,
    CXR_VALSE_BINARY_TEXT_PROMPT_BINARY,
    CXR_HISTORY_TEXT_PROMPT,
    CXR_HISTORY_TEXT_PROMPT_5CLASS,
    CXR_HISTORY_TEXT_PROMPT_5CLASS_V1,
    CXR_HISTORY_TEXT_PROMPT_5CLASS_V2,
    CXR_HISTORY_TEXT_PROMPT_5CLASS_V3,

    # --- Multimodality / VLM Bias Benchmark ---
    GLAUCOMA_TEXT_PROMPT,
    MIMIC_TEXT_PROMPT,
    MIMIC_TEXT_PROMPT_5CLASS,
    MIMIC_TEXT_PROMPT_5CLASS_V1,
    MIMIC_TEXT_PROMPT_5CLASS_V2,
    MIMIC_TEXT_PROMPT_5CLASS_V3,
    
    HAM10000_TEXT_PROMPT_FULL,
    HAM10000_TEXT_PROMPT_BINARY,
    BRSET_TEXT_PROMPT,
    mBRSET_TEXT_PROMPT,
    

    # --- Only Image Prompts ---
    GLAUCOMA_ONLY_IMAGE_TEXT_PROMPT,
    MIMIC_ONLY_IMAGE_TEXT_PROMPT,
    MIMIC_ONLY_IMAGE_TEXT_PROMPT_5CLASS,
    MIMIC_ONLY_IMAGE_TEXT_PROMPT_5CLASS_V1,
    MIMIC_ONLY_IMAGE_TEXT_PROMPT_5CLASS_V2,
    MIMIC_ONLY_IMAGE_TEXT_PROMPT_5CLASS_V3,
    
    HAM10000_ONLY_IMAGE_TEXT_PROMPT,
    BRSET_ONLY_IMAGE_TEXT_PROMPT,
    mBRSET_ONLY_IMAGE_TEXT_PROMPT,

    # --- Only Text Prompts ---
    GLAUCOMA_ONLY_TEXT_PROMPT,
    MIMIC_ONLY_TEXT_PROMPT,
    MIMIC_ONLY_TEXT_PROMPT_5CLASS,
    MIMIC_ONLY_TEXT_PROMPT_5CLASS_V1,
    MIMIC_ONLY_TEXT_PROMPT_5CLASS_V2,
    MIMIC_ONLY_TEXT_PROMPT_5CLASS_V3,
    
    HAM10000_ONLY_TEXT_PROMPT_FULL,
    HAM10000_ONLY_TEXT_PROMPT_BINARY,
    BRSET_ONLY_TEXT_PROMPT,
    mBRSET_ONLY_TEXT_PROMPT,
)

def _pil_to_base64_png(img):
    """Return a data URL 'data:image/png;base64,...' from a PIL.Image or raw bytes/np array path."""
    if img is None:
        return None
    if not isinstance(img, Image.Image):
        img = load_image(img)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"

def prepare_gemini_request(prompt, image, model_id="gemini-1.5-flash-002"):
    """
    Creates a request dictionary for the Gemini Batch API.
    """
    if image is None:
        parts = [{"text": prompt}]
    else:
        # Convert image to base64
        if not isinstance(image, Image.Image):
            image = load_image(image)
        
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        img_bytes = buf.getvalue()
        b64_data = base64.b64encode(img_bytes).decode('utf-8')
        
        parts = [
            {"text": prompt},
            {"inline_data": {"mime_type": "image/png", "data": b64_data}}
        ]

    # Construct the request object matching GenerateContentRequest schema
    generation_config = {
        "temperature": 0.0,
        "max_output_tokens": 8192 if "gemini-3" in model_id else 128,
    }

    if "gemini-3" in model_id:
        generation_config["thinking_config"] = {"include_thoughts": False, "thinking_budget": 1}

    request = {
        "contents": [{"parts": parts}],
        "generation_config": generation_config
    }
    return request

def predict_dataset_batch(metadata_row, dataset='mimic', modality=None, 
                          original=False, unmatched=False, history_cols_to_use=None, version='default', model_id="gemini-1.5-flash-002"):
    """
    Prepares the prompt and image, then returns the batch request object.
    """
    
    # ============================================================================
    # MIMIC Dataset
    # ============================================================================
    if dataset == 'mimic':
        image = load_image(metadata_row['filepath'])
        if modality == 'Only_image':
            text_metadata = MIMIC_ONLY_IMAGE_TEXT_PROMPT(unmatched=unmatched)
        elif modality == 'Only_text':
            text_metadata = MIMIC_ONLY_TEXT_PROMPT(metadata_row, unmatched=unmatched)
        else:
            text_metadata = MIMIC_TEXT_PROMPT(metadata_row, unmatched=unmatched)
            
    # ============================================================================
    # MIMIC 5-Class Dataset
    # ============================================================================
    elif dataset == 'mimic_5class' or dataset == 'mimic_5class_test':
        image = load_image(metadata_row['filepath'])
        
        # Select appropriate prompt based on modality and version
        prompt_map = {
            'Only_image': {
                'v1': MIMIC_ONLY_IMAGE_TEXT_PROMPT_5CLASS_V1,
                'v2': MIMIC_ONLY_IMAGE_TEXT_PROMPT_5CLASS_V2,
                'v3': MIMIC_ONLY_IMAGE_TEXT_PROMPT_5CLASS_V3,
                'default': MIMIC_ONLY_IMAGE_TEXT_PROMPT_5CLASS
            },
            'Only_text': {
                'v1': MIMIC_ONLY_TEXT_PROMPT_5CLASS_V1,
                'v2': MIMIC_ONLY_TEXT_PROMPT_5CLASS_V2,
                'v3': MIMIC_ONLY_TEXT_PROMPT_5CLASS_V3,
                'default': MIMIC_ONLY_TEXT_PROMPT_5CLASS
            },
            'both': {
                'v1': MIMIC_TEXT_PROMPT_5CLASS_V1,
                'v2': MIMIC_TEXT_PROMPT_5CLASS_V2,
                'v3': MIMIC_TEXT_PROMPT_5CLASS_V3,
                'default': MIMIC_TEXT_PROMPT_5CLASS
            }
        }
        
        modality_key = modality if modality in ['Only_image', 'Only_text'] else 'both'
        version_key = version if version in ['v1', 'v2', 'v3'] else 'default'
        prompt_func = prompt_map[modality_key][version_key]
        
        # Apply prompt (some are functions, some are constants)
        if modality_key == 'Only_image':
            text_metadata = prompt_func(unmatched=unmatched)
        else:
            text_metadata = prompt_func(metadata_row, unmatched=unmatched)
                
    # ============================================================================
    # History-based Dataset
    # ============================================================================
    elif 'history' in dataset:
        image = load_image(metadata_row['filepath'])
        
        history_prompts = {
            'v1': CXR_HISTORY_TEXT_PROMPT_5CLASS_V1,
            'v2': CXR_HISTORY_TEXT_PROMPT_5CLASS_V2,
            'v3': CXR_HISTORY_TEXT_PROMPT_5CLASS_V3,
            'default': CXR_HISTORY_TEXT_PROMPT_5CLASS
        }
        
        prompt_func = history_prompts.get(version, history_prompts['default'])
        text_metadata = prompt_func(metadata_row, original=original, history_cols_to_use=history_cols_to_use, unmatched=unmatched)
    
    # ============================================================================
    # CXR VALSE Dataset
    # ============================================================================
    elif 'cxr_valse' in dataset:
        image = load_image(metadata_row['filepath'])
        if 'binary' in dataset:
            text_metadata = CXR_VALSE_BINARY_TEXT_PROMPT_BINARY(metadata_row, original=original)
        else:
            text_metadata = CXR_VALSE_TEXT_PROMPT(metadata_row, original=original)
        
    # ============================================================================
    # HAM10000 Dataset
    # ============================================================================
    elif dataset == 'ham10000':
        image = load_image(metadata_row['filepath'])
        if modality == 'Only_image':
            text_metadata = HAM10000_ONLY_IMAGE_TEXT_PROMPT
        elif modality == 'Only_text':
            text_metadata = HAM10000_ONLY_TEXT_PROMPT_BINARY(metadata_row)
        else:
            text_metadata = HAM10000_TEXT_PROMPT_BINARY(metadata_row)
        
    # ============================================================================
    # mBRSET Dataset
    # ============================================================================
    elif dataset == 'mbrset':
        image = load_image(metadata_row['filepath'])
        if modality == 'Only_image':
            text_metadata = mBRSET_ONLY_IMAGE_TEXT_PROMPT
        elif modality == 'Only_text':
            text_metadata = mBRSET_ONLY_TEXT_PROMPT(metadata_row)
        else:
            text_metadata = mBRSET_TEXT_PROMPT(metadata_row)
    
    # ============================================================================
    # BRSET Dataset
    # ============================================================================
    elif dataset == 'brset':
        image = load_image(metadata_row['filepath'])
        if modality == 'Only_image':
            text_metadata = BRSET_ONLY_IMAGE_TEXT_PROMPT
        elif modality == 'Only_text':
            text_metadata = BRSET_ONLY_TEXT_PROMPT(metadata_row)
        else:
            text_metadata = BRSET_TEXT_PROMPT(metadata_row)
        
    # ============================================================================
    # MedEval Dataset
    # ============================================================================
    elif dataset == 'medeval':
        data = np.load(metadata_row['filepath'])
        image = Image.fromarray(data['slo_fundus'])
        if modality == 'Only_image':
            text_metadata = GLAUCOMA_ONLY_IMAGE_TEXT_PROMPT
        elif modality == 'Only_text':
            text_metadata = GLAUCOMA_ONLY_TEXT_PROMPT(metadata_row)
        else:               
            text_metadata = GLAUCOMA_TEXT_PROMPT(metadata_row)
        
    else:
        raise ValueError(f"Dataset not supported: {dataset}, only 'mimic' and 'ham10000' datasets are supported.")
    
    # Handle text-only modality
    if modality == 'Only_text':
        image = None

    return prepare_gemini_request(text_metadata, image, model_id=model_id)
