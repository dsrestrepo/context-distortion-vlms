from src.models import (
    load_qwen2_vl,
    load_llava,
    load_pali_gemma,
    load_janus_pro,
    load_biomedgpt,
    load_llava_med,
    load_maira2,
    load_medgemma,
    load_chexagent,
    load_llama3_2,
    load_openai,
    load_gpt_oss,
    load_gemini,
    load_claude,
    load_grok,
    load_cohere,
)
from src.inference import predict_dataset

import os, re
import pandas as pd
import torch
from tqdm import tqdm
import random
import traceback


def _normalize_versions(versions):
    """
    Returns (versions_list, save_mode) where save_mode in
    {'single_default', 'single_named', 'multi'}
    """
    if versions is None or (isinstance(versions, str) and versions.lower() == "default"):
        return ["default"], "single_default"
    if isinstance(versions, str):
        return [versions], "single_named"
    if isinstance(versions, (list, tuple)):
        vlist = list(versions)
        if len(vlist) == 0 or (len(vlist) == 1 and str(vlist[0]).lower() == "default"):
            return ["default"], "single_default"
        if len(vlist) == 1:
            return vlist, "single_named"
        return vlist, "multi"
    # Fallback: treat anything else as default
    return ["default"], "single_default"

def _safe_version_tag(version):
    """Sanitize version string for use in filenames."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(version))


def _load_model_and_processor(model_name, model_id, quantization, use_flash_attention, 
                               return_attention, return_logits, conv_mode="mistral_instruct"):
    """
    Load model and processor based on model type.
    
    Args:
        model_name: Name/identifier of the model
        model_id: Model ID for loading
        quantization: Quantization settings
        use_flash_attention: Whether to use flash attention
        return_attention: Whether to return attention weights
        return_logits: Whether to return logits
        conv_mode: Conversation mode for certain models
    
    Returns:
        tuple: (model, processor)
    """
    if "qwen2" in model_name:
        return load_qwen2_vl(quantization=quantization, use_flash_attention=use_flash_attention, 
                            model_id=model_id, return_attention=return_attention, return_logits=return_logits)
    
    elif "llava" in model_name and (not ("llava_med" in model_name) and not("llava-med" in model_name)):
        return load_llava(model_id=model_id, quantization=quantization, use_flash_attention=use_flash_attention,
                         return_attention=return_attention, return_logits=return_logits)
    
    elif "paligemma2" in model_name:
        return load_pali_gemma(model_id=model_id, quantization=quantization,
                              return_attention=return_attention, return_logits=return_logits)
    
    elif "janus" in model_name:
        return load_janus_pro(model_id=model_id, quantization=quantization,
                             return_attention=return_attention, return_logits=return_logits)
    
    elif "biomedgpt" in model_name:
        return load_biomedgpt(model_id=model_id, quantization=quantization,
                             return_attention=return_attention, return_logits=return_logits)
    
    elif "llava_med" in model_name or "llava-med" in model_name:
        print("Using llava-med")
        return load_llava_med(model_id=model_id, quantization=quantization,
                             return_attention=return_attention, return_logits=return_logits, conv_mode=conv_mode)
    
    elif "maira-2" in model_name:
        return load_maira2(model_id=model_id, quantization=quantization,
                          return_attention=return_attention, return_logits=return_logits)
    
    elif "medgemma" in model_name:
        return load_medgemma(model_id=model_id, quantization=quantization,
                            return_attention=return_attention, return_logits=return_logits,
                            use_flash_attention=use_flash_attention)
    
    elif "chexagent" in model_name:
        model, processor, _ = load_chexagent(model_id=model_id, quantization=quantization,
                                            return_attention=return_attention, return_logits=return_logits)
        return model, processor
    
    elif "llama3" in model_name:
        return load_llama3_2(model_id=model_id, quantization=quantization,
                            return_attention=return_attention, return_logits=return_logits)
    
    elif "openai" in model_id or "gpt-5" in model_id or "gpt-4" in model_id:
        return load_openai(model_id=model_name, return_logits=return_logits)
    
    elif "gpt-oss" in model_name or "gpt_oss" in model_name:
        return load_gpt_oss(model_id=model_id, quantization=quantization,
                           return_attention=return_attention, return_logits=return_logits)
    
    elif "gemini" in model_name:
        return load_gemini(model_id=model_id, return_logits=return_logits)
    
    elif "claude" in model_name:
        return load_claude(model_id=model_id, return_logits=return_logits)
    
    elif "grok" in model_name:
        return load_grok(model_id=model_id, return_logits=return_logits)
    
    elif "cohere" in model_name or "command" in model_name:
        return load_cohere(model_id=model_id, return_logits=return_logits)
    
    else:
        raise ValueError(f"Model type not supported: {model_name}, only 'qwen2', 'llava', 'paligemma2', "
                        f"'janus', 'biomedgpt', 'llava_med', 'maira-2', 'medgemma', 'chexagent', 'llama3', "
                        f"'openai', 'gpt-oss', 'gemini', 'claude', 'grok', and 'cohere' are supported")


def _create_result_dict(row, store_columns, label, prediction, p_yes, p_no, p_Yes, p_No, version, prob_dict=None, **extra_fields):
    """
    Create a result dictionary with common fields.
    
    Args:
        row: Data row with metadata
        store_columns: List of column names to store from row
        label: Label field name
        prediction: Model prediction
        p_yes, p_no, p_Yes, p_No: Probability values (for backward compatibility)
        version: Version identifier
        prob_dict: Dictionary with all token probabilities
        **extra_fields: Additional fields to include in result dict
    
    Returns:
        dict: Result dictionary
    """
    aux_dict = {key: row.get(key, row[key] if hasattr(row, '__getitem__') else None) for key in store_columns}
    aux_dict["ground_truth"] = row[label]
    aux_dict["prediction"] = prediction
    aux_dict["p_yes"] = p_yes
    aux_dict["p_no"] = p_no
    aux_dict["p_Yes"] = p_Yes
    aux_dict["p_No"] = p_No
    aux_dict["version"] = version
    
    # Add all token probabilities from prob_dict (more generalizable approach)
    if prob_dict:
        for token, prob in prob_dict.items():
            # Store all tokens as p_{token}, even standard ones for consistency
            # Skip standard ones to avoid duplication since we have separate columns
            if token not in ['yes', 'no', 'Yes', 'No']:
                aux_dict[f"p_{token}"] = prob
    
    aux_dict.update(extra_fields)
    return aux_dict


def _save_results(results_df, save_mode, all_results, model_output_dir, base_filename, version, model_name):
    """
    Save results based on save mode.
    
    Args:
        results_df: DataFrame with results for current version
        save_mode: One of 'single_default', 'single_named', or 'multi'
        all_results: List to accumulate results (for multi mode)
        model_output_dir: Directory to save results
        base_filename: Base filename without version tag
        version: Version identifier
        model_name: Model name for logging
    
    Returns:
        list: Updated all_results list (only relevant for multi mode)
    """
    if save_mode == "multi":
        all_results.append(results_df)
        return all_results
    else:
        if save_mode == "single_default":
            output_csv_path = os.path.join(model_output_dir, f"{base_filename}.csv")
        else:  # single_named
            vtag = _safe_version_tag(version)
            output_csv_path = os.path.join(model_output_dir, f"{base_filename}_{vtag}.csv")
        
        results_df.to_csv(output_csv_path, index=False)
        print(f"[{model_name}] v={version} -> {output_csv_path}")
        return all_results


def generate_predictions_models_base(model_dict, metadata_test, quantization=None, return_attention=True, return_logits=True, dataset="medeval", store_columns=["filename", "age", "sex", "gender", "race", "ethnicity", "language", "maritalstatus"], label="glaucoma", conv_mode="mistral_instruct", use_flash_attention=True, p_yes_and_no=True, unmatched=False, 
                                     history_cols=['prior_report'], versions=None, tokens=None, priority_img=False):

    # Results directory
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    versions_list, save_mode = _normalize_versions(versions)
    
    # Loop over models
    for model_id, model_name in tqdm(model_dict.items(), desc="Processing Models"):
        print(f"Loading model: {model_name} ({model_id})")
        
        # Disable flash attention for medgemma on history tasks due to CUDA alignment issues
        model_use_flash = use_flash_attention
        #if "medgemma" in model_name and "history" in dataset:
        #    model_use_flash = False
        #    print(f"Note: Disabling flash attention for {model_name} on history task to avoid CUDA alignment errors")
        
        # Load the model and processor based on model type
        model, processor = _load_model_and_processor(
            model_name, model_id, quantization, model_use_flash,
            return_attention, return_logits, conv_mode
        )

        # Decide output foldername based on the model
        model_output_dir = os.path.join(results_dir, model_name)
        os.makedirs(model_output_dir, exist_ok=True)
        
        # Decide base output filename
        if unmatched:
            base_filename = f"{dataset}_base_unmatched"
        else:
            base_filename = f"{dataset}_base"
        
        # Add priority suffix if priority_img is True
        if priority_img:
            base_filename = base_filename + "_priority"

        # If saving all versions together:
        all_results = [] if save_mode == "multi" else None
        
        # Iterate over versions
        for version in versions_list:
        
            # Predict and save results
            results = []
            for index, row in tqdm(metadata_test.iterrows(), total=len(metadata_test), desc=f"Predicting with {model_name} | version={version}"):

                original_ops = [True, False]
                
                for original in original_ops:
                    if (not original) and ('valse' not in dataset) and ('history' not in dataset):
                        continue
                    
                    # If using history and multiple history columns are provided aterate over them for the prompt
                    if ('history' in dataset) and (not original):
                        if history_cols is None:
                            raise ValueError("history_cols must be provided if using history in the dataset name")
                        
                        history_cols_to_use = []
                        # shuffle the history columns to use a different order each time
                        random.shuffle(history_cols)
                        
                        for i, history_column in enumerate(history_cols):
                            history_cols_to_use.append(history_column)
                            
                            try:
                                # Get prediction using the modified row
                                prediction, _, _, _, p_yes, p_no, p_Yes, p_No, prob_dict = predict_dataset(
                                    row, model=model, processor=processor, quantization=quantization,
                                    return_attention=return_attention, return_logits=return_logits, dataset=dataset,
                                    original=original, p_yes_and_no=p_yes_and_no, unmatched=unmatched,
                                    history_cols_to_use=history_cols_to_use, version=version, tokens=tokens, priority_img=priority_img
                                )

                            except Exception as e:
                                error_msg = str(e)
                                print(f"Error predicting with {model_name} on index {index}: {e}")
                                # traceback.print_exc()
                                torch.cuda.empty_cache()
                                # Skip this history iteration and continue with next one
                                continue
                                
                            if index % 100 == 0:
                                print(f"File: {row['filepath']} | v={version} | Prediction: {prediction}")
                            
                            aux_dict = _create_result_dict(
                                row, store_columns, label, prediction, p_yes, p_no, p_Yes, p_No, version, prob_dict,
                                original=original if original else history_column if i == 0 else ','.join(history_cols_to_use),
                                history_length=i + 1 if not original else 0
                            )
                            results.append(aux_dict)
                            
                    else:
                        # Non-history path
                        try:
                            # Get prediction using the modified row
                            prediction, _, _, _, p_yes, p_no, p_Yes, p_No, prob_dict = predict_dataset(
                                row, model=model, processor=processor, quantization=quantization,
                                return_attention=return_attention, return_logits=return_logits, dataset=dataset,
                                original=original, p_yes_and_no=p_yes_and_no, unmatched=unmatched, version=version, tokens=tokens, priority_img=priority_img
                            )
                            
                        except Exception as e:
                            error_msg = str(e)
                            print(f"Error predicting with {model_name} on index {index}: {e}")
                            # traceback.print_exc()
                            
                            torch.cuda.empty_cache()
                            
                            # Skip this sample and continue with next one
                            continue
                            
                        if index % 100 == 0:
                            print(f"File: {row['filepath']} | v={version} | Prediction: {prediction}")
                        
                        aux_dict = _create_result_dict(
                            row, store_columns, label, prediction, p_yes, p_no, p_Yes, p_No, version, prob_dict,
                            original=original,
                            history_length=0
                        )
                        results.append(aux_dict)


            # Convert results to a DataFrame and save
            results_df = pd.DataFrame(results)
            all_results = _save_results(results_df, save_mode, all_results, model_output_dir, 
                                       base_filename, version, model_name)
                
        # If multi, concatenate and save once
        if save_mode == "multi":
            final_df = pd.concat(all_results, ignore_index=True) if len(all_results) else pd.DataFrame()
            output_csv_path = os.path.join(model_output_dir, f"{base_filename}_multi_versions.csv")
            final_df.to_csv(output_csv_path, index=False)
            print(f"[{model_name}] multi-versions -> {output_csv_path}")
            
        # print(f"Predictions for {model_name} saved to {output_csv_path}")
        
        # Clear model from memory
        del model, processor
        torch.cuda.empty_cache()
        



def get_shifted_image(current_filepath, current_label, metadata, label_field, image_col="filepath"):
    """
    Returns the filepath of an image from `metadata` that has a label different from `current_label`.
    If no candidate is found, returns the original filepath.
    """
    # Filter metadata for rows with a different label than current_label
    candidates = metadata[metadata[label_field] != current_label]
    if candidates.empty:
        print(f"No candidate found for {current_filepath} with label {current_label}")
        return current_filepath
    else:
        # Randomly sample one candidate image and return its filepath
        return candidates.sample(1).iloc[0][image_col]


def get_shifted_text(current_row, current_label, metadata, label_field, text_col=None):
    """
    Returns a shifted text prompt by selecting a random row from `metadata`
    whose label (in the field `label_field`) differs from that of `current_row`.
    """

    # Filter metadata for rows with a different label than current_label
    candidates = metadata[metadata[label_field] != current_label]
    if candidates.empty:
        print(f"No candidate found for {current_row} with label {current_label}")
        return current_row
    else:
        # Randomly sample one candidate row and return its text
        return candidates.sample(1).iloc[0]#[text_col]
    
def get_shifted_metadata(current_row, current_label, metadata, label_field, metadata_cols=["age", "sex", "gender", "race", "ethnicity", "language", "maritalstatus"]):
    
    # Filter metadata for rows with a different label than current_label
    candidates = metadata[metadata[label_field] != current_label]
    if candidates.empty:
        print(f"No candidate found for {current_row} with label {current_label}")
        return current_row
    else:
        # Randomly sample one candidate row and return its text
        return candidates.sample(1).iloc[0][metadata_cols]


def generate_predictions_models(model_dict, metadata_test, quantization=None, return_attention=False, return_logits=False, use_flash_attention=False,
                                  dataset="medeval", store_columns=["filename", "age", "sex", "gender", "race", "ethnicity", "language", "maritalstatus"],
                                  text_col="note", image_col="filename", metadata_cols=["age", "sex", "gender", "race", "ethnicity", "language", "maritalstatus"], label="glaucoma", conv_mode="mistral_instruct",
                                  p_yes_and_no=True, unmatched=False, versions=None, tokens=None, priority_img=False):
    
    
    # Results directory
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Normalize versions into iterable + save mode
    versions_list, save_mode = _normalize_versions(versions)
    
    # Loop over models
    for model_id, model_name in tqdm(model_dict.items(), desc="Processing Models"):
        print(f"Loading model: {model_name} ({model_id})")
        
        # Load the model and processor based on model type
        model, processor = _load_model_and_processor(
            model_name, model_id, quantization, use_flash_attention,
            return_attention, return_logits, conv_mode
        )

        # Base output dir & filename stem (before version tag)
        model_output_dir = os.path.join(results_dir, model_name)
        os.makedirs(model_output_dir, exist_ok=True)
        
        base_filename = f"{dataset}_base_shifted_unmatched" if unmatched else f"{dataset}_base_shifted"
        
        # Add priority suffix if priority_img is True
        if priority_img:
            base_filename = base_filename + "_priority"

        # When saving all versions together:
        all_results = [] if save_mode == "multi" else None
        
        # ----- iterate versions -----
        for version in versions_list:
            
            results = []
            # For each sample in the test set
            for index, row in tqdm(metadata_test.iterrows(), total=len(metadata_test), desc=f"Predicting with {model_name} | version={version}"):
                
                # None (original), Image shift, Metadata shift, and Text shift.
                for shift_type in [None, "Image", "Text", "Only_text", "Only_image"]:
                    # Create a copy of the row so as not to modify the original
                    row_modified = row.copy() if hasattr(row, "copy") else dict(row)
                    
                    # Modify the row depending on the shift
                    if shift_type == "Image":
                        # Replace the image with one from a different class.
                        # Use the helper function to ensure the new image has a different label.
                        row_modified[image_col] = get_shifted_image(row[image_col], row[label], metadata_test, label, image_col=image_col)
                        
                    elif shift_type == "Text":
                        # Replace the text prompt with one from a different class.
                        # Adjust this as necessary based on your dataset's structure.
                        row_modified = get_shifted_text(row, row[label], metadata_test, label, text_col=text_col)
                        
                        # Replace the image with the original image.
                        row_modified[image_col] = row[image_col]

                    # Paligemma2 does not support text-only inputs
                    if "paligemma2" in model_name and shift_type == "Only_text":
                        continue
                    # Janus does not support text-only inputs
                    if "janus" in model_name and shift_type == "Only_text":
                        continue
                    
                    try:
                        # Get prediction using the modified row
                        prediction, _, _, _, p_yes, p_no, p_Yes, p_No, prob_dict = predict_dataset(
                            row_modified, model=model, processor=processor, quantization=quantization, 
                            return_attention=return_attention, return_logits=return_logits, dataset=dataset, 
                            modality=shift_type, p_yes_and_no=p_yes_and_no, unmatched=unmatched, version=version, tokens=tokens, priority_img=priority_img)#, 
                                                                #text_col=text_col, image_col=image_col, metadata_cols=metadata_cols)
                    except Exception as e:
                        error_msg = str(e)
                        print(f"Error predicting with {model_name} on index {index}: {e}")
                        # traceback.print_exc()
                        
                        torch.cuda.empty_cache()
                        
                        # Skip this row if there's an error
                        continue
                    
                    if index % 100 == 0:
                        print(f"File: {row_modified.get(image_col, 'N/A')} | Shift: {shift_type or 'No'} | v={version} | Pred: {prediction}")
                    
                    # Build a dictionary for this evaluation instance
                    aux_dict = _create_result_dict(
                        row, store_columns, label, prediction, p_yes, p_no, p_Yes, p_No, version, prob_dict,
                        shift="No" if shift_type is None else shift_type
                    )
                    results.append(aux_dict)
                    
            # Convert results to a DataFrame and save
            results_df = pd.DataFrame(results)
            all_results = _save_results(results_df, save_mode, all_results, model_output_dir,
                                       base_filename, version, model_name)
        
        if save_mode == "multi":
            final_df = pd.concat(all_results, ignore_index=True) if len(all_results) else pd.DataFrame()
            output_csv_path = os.path.join(model_output_dir, f"{base_filename}_multi_versions.csv")
            final_df.to_csv(output_csv_path, index=False)
            print(f"[{model_name}] multi-versions -> {output_csv_path}")

        # Clear model from memory
        del model, processor
        torch.cuda.empty_cache()
