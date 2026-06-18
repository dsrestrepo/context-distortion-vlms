from src.inference import predict_dataset_batch
from src.vlm import create_vlm

import os, re
import pandas as pd
import torch
from tqdm import tqdm
import random


def _chunks(items, batch_size):
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def _run_request_batch(
    requests,
    model,
    *,
    batch_size,
    dataset,
    version,
    return_attention,
    return_logits,
    p_yes_and_no,
    unmatched,
    tokens,
    priority_img,
):
    outputs = []
    for batch in _chunks(requests, batch_size):
        error = None
        try:
            predictions = predict_dataset_batch(
                [request["input_row"] for request in batch],
                model,
                return_attention=return_attention,
                return_logits=return_logits,
                dataset=dataset,
                modalities=[request.get("modality") for request in batch],
                p_yes_and_no=p_yes_and_no,
                originals=[request.get("original", False) for request in batch],
                unmatched=unmatched,
                history_cols_to_use=[
                    request.get("history_cols_to_use") for request in batch
                ],
                version=version,
                tokens=tokens,
                priority_img=priority_img,
            )
            outputs.extend(zip(batch, predictions))
        except Exception as exc:
            error = exc
        if error is not None:
            print(f"Batch inference failed: {error}")
            torch.cuda.empty_cache()
            if len(batch) > 1:
                print("Retrying failed batch one item at a time.")
                outputs.extend(
                    _run_request_batch(
                        batch,
                        model,
                        batch_size=1,
                        dataset=dataset,
                        version=version,
                        return_attention=return_attention,
                        return_logits=return_logits,
                        p_yes_and_no=p_yes_and_no,
                        unmatched=unmatched,
                        tokens=tokens,
                        priority_img=priority_img,
                    )
                )
            else:
                raise RuntimeError(
                    "Inference failed for an individual request; refusing to save "
                    "an incomplete result file."
                ) from error
    return outputs


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
                                     history_cols=['prior_report'], versions=None, tokens=None, priority_img=False, batch_size=1):

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
        
        model = create_vlm(
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
            requests = []
            for _, row in metadata_test.iterrows():
                requests.append(
                    {"input_row": row, "result_row": row, "original": True, "history_length": 0}
                )
                if "history" in dataset:
                    shuffled_history = list(history_cols or [])
                    random.shuffle(shuffled_history)
                    selected = []
                    for index, history_column in enumerate(shuffled_history):
                        selected.append(history_column)
                        requests.append(
                            {
                                "input_row": row,
                                "result_row": row,
                                "original": False,
                                "history_cols_to_use": list(selected),
                                "original_value": (
                                    history_column if index == 0 else ",".join(selected)
                                ),
                                "history_length": index + 1,
                            }
                        )

            results = []
            batch_outputs = _run_request_batch(
                requests,
                model,
                batch_size=batch_size,
                dataset=dataset,
                version=version,
                return_attention=return_attention,
                return_logits=return_logits,
                p_yes_and_no=p_yes_and_no,
                unmatched=unmatched,
                tokens=tokens,
                priority_img=priority_img,
            )
            for request, prediction_output in tqdm(
                batch_outputs, desc=f"Collecting {model_name} | version={version}"
            ):
                prediction, _, _, _, p_yes, p_no, p_Yes, p_No, prob_dict = prediction_output
                results.append(
                    _create_result_dict(
                        request["result_row"],
                        store_columns,
                        label,
                        prediction,
                        p_yes,
                        p_no,
                        p_Yes,
                        p_No,
                        version,
                        prob_dict,
                        original=request.get("original_value", request["original"]),
                        history_length=request["history_length"],
                    )
                )


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
        del model
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
                                  p_yes_and_no=True, unmatched=False, versions=None, tokens=None, priority_img=False, batch_size=1):
    
    
    # Results directory
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Normalize versions into iterable + save mode
    versions_list, save_mode = _normalize_versions(versions)
    
    # Loop over models
    for model_id, model_name in tqdm(model_dict.items(), desc="Processing Models"):
        print(f"Loading model: {model_name} ({model_id})")
        
        model = create_vlm(
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
            for shift_type in [None, "Image", "Text", "Only_text", "Only_image"]:
                requests = []
                for _, row in metadata_test.iterrows():
                    row_modified = row.copy() if hasattr(row, "copy") else dict(row)
                    if shift_type == "Image":
                        row_modified[image_col] = get_shifted_image(row[image_col], row[label], metadata_test, label, image_col=image_col)
                    elif shift_type == "Text":
                        row_modified = get_shifted_text(row, row[label], metadata_test, label, text_col=text_col)
                        row_modified[image_col] = row[image_col]
                    requests.append(
                        {
                            "input_row": row_modified,
                            "result_row": row,
                            "modality": shift_type,
                        }
                    )
                for request, prediction_output in _run_request_batch(
                    requests,
                    model,
                    batch_size=batch_size,
                    dataset=dataset,
                    version=version,
                    return_attention=return_attention,
                    return_logits=return_logits,
                    p_yes_and_no=p_yes_and_no,
                    unmatched=unmatched,
                    tokens=tokens,
                    priority_img=priority_img,
                ):
                    prediction, _, _, _, p_yes, p_no, p_Yes, p_No, prob_dict = prediction_output
                    results.append(
                        _create_result_dict(
                            request["result_row"], store_columns, label, prediction,
                            p_yes, p_no, p_Yes, p_No, version, prob_dict,
                            shift="No" if shift_type is None else shift_type,
                        )
                    )
                    
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
        del model
        torch.cuda.empty_cache()
