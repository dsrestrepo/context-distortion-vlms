import os
import json
import time
import pandas as pd
import random
from tqdm import tqdm
from google import genai
from google.genai import types
from src.inference_gemini_batch import predict_dataset_batch
from src.run_experiments import _normalize_versions, _create_result_dict, _save_results, get_shifted_image, get_shifted_text

def _wait_for_batch_job(client, job_name):
    """Polls the batch job status until completion."""
    print(f"Polling status for job: {job_name}")
    while True:
        batch_job = client.batches.get(name=job_name)
        if batch_job.state.name in ('JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED', 'JOB_STATE_CANCELLED'):
            break
        print(f"Job not finished. Current state: {batch_job.state.name}. Waiting 30 seconds...")
        time.sleep(30)
    return batch_job

def _process_batch_results(client, batch_job, request_map, results_dir, base_filename, model_name, store_columns, label, p_yes_and_no):
    """Downloads results, matches with requests, and saves CSVs."""
    
    if batch_job.state.name != 'JOB_STATE_SUCCEEDED':
        print(f"Job failed with state: {batch_job.state.name}")
        if batch_job.error:
            print(f"Error: {batch_job.error}")
        return

    result_file_name = batch_job.dest.file_name
    print(f"Results are in file: {result_file_name}")
    
    print("Downloading results...")
    file_content_bytes = client.files.download(file=result_file_name)
    file_content = file_content_bytes.decode('utf-8')
    
    # Parse results
    results_by_version = {} # version -> list of result dicts
    
    for line in file_content.splitlines():
        if not line: continue
        
        try:
            response_obj = json.loads(line)
            # custom_id is the key we sent
            key = response_obj.get('custom_id') 
            if not key:
                # Fallback if key is not in custom_id (it should be)
                # The batch API documentation says "key" in request becomes "custom_id" in response?
                # Let's check the provided notebook.
                # Notebook says: "The 'key' field is required for correlating inputs to outputs."
                # And in the output parsing: "parsed_response = json.loads(line)"
                # It doesn't explicitly show the key in the output example, but usually it's there.
                # Let's assume it's 'custom_id' or 'key'.
                key = response_obj.get('key')
            
            if not key or key not in request_map:
                print(f"Warning: Unknown key in response: {key}")
                continue
                
            req_info = request_map[key]
            row = req_info['row']
            version = req_info['version']
            
            # Extract prediction
            prediction = ""
            try:
                # The response structure depends on success/failure
                if 'response' in response_obj and 'candidates' in response_obj['response']:
                    candidates = response_obj['response']['candidates']
                    if candidates and 'content' in candidates[0] and 'parts' in candidates[0]['content']:
                        parts = candidates[0]['content']['parts']
                        prediction = "".join([p.get('text', '') for p in parts])
                elif 'error' in response_obj:
                    print(f"Error for key {key}: {response_obj['error']}")
                    prediction = "ERROR"
            except Exception as e:
                print(f"Error parsing response for key {key}: {e}")
                prediction = "ERROR"
            
            # We don't get logprobs easily from batch API unless we requested them and parse them.
            # For now, set probs to None.
            p_yes, p_no, p_Yes, p_No = None, None, None, None
            prob_dict = {}
            
            # Create result dict
            aux_dict = _create_result_dict(
                row, store_columns, label, prediction, p_yes, p_no, p_Yes, p_No, version, prob_dict,
                **req_info['extra_fields']
            )
            
            if version not in results_by_version:
                results_by_version[version] = []
            results_by_version[version].append(aux_dict)
            
        except json.JSONDecodeError:
            print("Error decoding JSON line")
            
    # Save results
    model_output_dir = os.path.join(results_dir, f"gemini_batch_{model_name}")
    os.makedirs(model_output_dir, exist_ok=True)
    
    for version, results in results_by_version.items():
        results_df = pd.DataFrame(results)
        # We use a simplified save logic here since we processed everything at once
        output_csv_path = os.path.join(model_output_dir, f"{base_filename}_{version}.csv")
        results_df.to_csv(output_csv_path, index=False)
        print(f"Saved results to {output_csv_path}")


def generate_predictions_models_base_batch(model_dict, metadata_test, dataset="medeval", 
                                           store_columns=["filename", "age", "sex", "gender", "race", "ethnicity", "language", "maritalstatus"], 
                                           label="glaucoma", unmatched=False, history_cols=['prior_report'], versions=None, tokens=None,
                                           p_yes_and_no=True): # Added p_yes_and_no to signature
    
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    versions_list, _ = _normalize_versions(versions)
    results_dir = "results"
    
    for model_id, model_name in model_dict.items():
        if "gemini" not in model_name.lower():
            print(f"Skipping non-Gemini model: {model_name}")
            continue
            
        print(f"Processing {model_name}...")
        
        requests_data = []
        request_map = {} # key -> {row, version, extra_fields}
        
        for version in versions_list:
            for index, row in tqdm(metadata_test.iterrows(), total=len(metadata_test), desc=f"Prep {model_name} v={version}"):
                original_ops = [True, False]
                for original in original_ops:
                    if (not original) and ('valse' not in dataset) and ('history' not in dataset):
                        continue
                    
                    if ('history' in dataset) and (not original):
                        if history_cols is None:
                            raise ValueError("history_cols must be provided")
                        
                        # We can't shuffle randomly if we want deterministic batch generation?
                        # Actually, run_experiments.py shuffles inside the loop.
                        # We should probably keep it deterministic or just shuffle once.
                        # Let's shuffle once per row to match behavior roughly.
                        current_history_cols = list(history_cols)
                        random.shuffle(current_history_cols)
                        
                        history_cols_to_use = []
                        for i, history_column in enumerate(current_history_cols):
                            history_cols_to_use.append(history_column)
                            
                            key = f"idx_{index}_v_{version}_orig_{original}_hist_{i}"
                            
                            try:
                                req_body = predict_dataset_batch(
                                    row, dataset=dataset, original=original, unmatched=unmatched,
                                    history_cols_to_use=history_cols_to_use, version=version, model_id=model_id
                                )
                                
                                requests_data.append({"key": key, "request": req_body})
                                request_map[key] = {
                                    "row": row.to_dict(),
                                    "version": version,
                                    "extra_fields": {
                                        "original": history_column if i == 0 else ','.join(history_cols_to_use),
                                        "history_length": i + 1
                                    }
                                }
                            except Exception as e:
                                print(f"Error preparing request for {key}: {e}")

                    else:
                        # Non-history path
                        key = f"idx_{index}_v_{version}_orig_{original}"
                        try:
                            req_body = predict_dataset_batch(
                                row, dataset=dataset, original=original, unmatched=unmatched, version=version, model_id=model_id
                            )
                            requests_data.append({"key": key, "request": req_body})
                            request_map[key] = {
                                "row": row.to_dict(),
                                "version": version,
                                "extra_fields": {
                                    "original": original,
                                    "history_length": 0
                                }
                            }
                        except Exception as e:
                            print(f"Error preparing request for {key}: {e}")

        # Submit Batch
        if not requests_data:
            print("No requests generated.")
            continue
            
        timestamp = int(time.time())
        safe_model_name = model_name.replace("/", "-")
        jsonl_filename = f"batch_req_{dataset}_{safe_model_name}_{timestamp}.jsonl"
        with open(jsonl_filename, 'w') as f:
            for req in requests_data:
                f.write(json.dumps(req) + '\n')
        
        print(f"Uploading {jsonl_filename}...")
        batch_input_file = client.files.upload(file=jsonl_filename, config={'mime_type': 'application/json'})
        
        print("Creating batch job...")
        batch_job = client.batches.create(
            model=model_id,
            src=batch_input_file.name,
            config={'display_name': f"batch_{dataset}_{safe_model_name}_{timestamp}"}
        )
        print(f"Job created: {batch_job.name}")
        
        # Wait and Process
        batch_job = _wait_for_batch_job(client, batch_job.name)
        
        base_filename = f"{dataset}_base_unmatched" if unmatched else f"{dataset}_base"
        _process_batch_results(client, batch_job, request_map, results_dir, base_filename, model_name, store_columns, label, p_yes_and_no)
        
        # Cleanup
        if os.path.exists(jsonl_filename):
            os.remove(jsonl_filename)


def generate_predictions_models_batch(model_dict, metadata_test, dataset="medeval", 
                                      store_columns=["filename", "age", "sex", "gender", "race", "ethnicity", "language", "maritalstatus"], 
                                      text_col="note", image_col="filename", label="glaucoma", unmatched=False, versions=None, tokens=None,
                                      p_yes_and_no=True): # Added p_yes_and_no
    
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    versions_list, _ = _normalize_versions(versions)
    results_dir = "results"
    
    for model_id, model_name in model_dict.items():
        if "gemini" not in model_name.lower():
            print(f"Skipping non-Gemini model: {model_name}")
            continue
            
        print(f"Processing {model_name}...")
        
        requests_data = []
        request_map = {} 
        
        for version in versions_list:
            for index, row in tqdm(metadata_test.iterrows(), total=len(metadata_test), desc=f"Prep {model_name} v={version}"):
                
                for shift_type in [None, "Image", "Text", "Only_text", "Only_image"]:
                    row_modified = row.copy() if hasattr(row, "copy") else dict(row)
                    
                    if shift_type == "Image":
                        row_modified[image_col] = get_shifted_image(row[image_col], row[label], metadata_test, label, image_col=image_col)
                    elif shift_type == "Text":
                        row_modified = get_shifted_text(row, row[label], metadata_test, label, text_col=text_col)
                        row_modified[image_col] = row[image_col]
                    
                    key = f"idx_{index}_v_{version}_shift_{shift_type}"
                    
                    try:
                        req_body = predict_dataset_batch(
                            row_modified, dataset=dataset, modality=shift_type, unmatched=unmatched, version=version, model_id=model_id
                        )
                        requests_data.append({"key": key, "request": req_body})
                        request_map[key] = {
                            "row": row.to_dict(),
                            "version": version,
                            "extra_fields": {
                                "shift": "No" if shift_type is None else shift_type
                            }
                        }
                    except Exception as e:
                        print(f"Error preparing request for {key}: {e}")

        # Submit Batch
        if not requests_data:
            print("No requests generated.")
            continue
            
        timestamp = int(time.time())
        safe_model_name = model_name.replace("/", "-")
        jsonl_filename = f"batch_req_{dataset}_{safe_model_name}_{timestamp}.jsonl"
        with open(jsonl_filename, 'w') as f:
            for req in requests_data:
                f.write(json.dumps(req) + '\n')
        
        print(f"Uploading {jsonl_filename}...")
        batch_input_file = client.files.upload(file=jsonl_filename, config={'mime_type': 'application/json'})
        
        print("Creating batch job...")
        batch_job = client.batches.create(
            model=model_id,
            src=batch_input_file.name,
            config={'display_name': f"batch_{dataset}_{safe_model_name}_{timestamp}"}
        )
        print(f"Job created: {batch_job.name}")
        
        # Wait and Process
        batch_job = _wait_for_batch_job(client, batch_job.name)
        
        base_filename = f"{dataset}_base_shifted_unmatched" if unmatched else f"{dataset}_base_shifted"
        _process_batch_results(client, batch_job, request_map, results_dir, base_filename, model_name, store_columns, label, p_yes_and_no)
        
        # Cleanup
        if os.path.exists(jsonl_filename):
            os.remove(jsonl_filename)
