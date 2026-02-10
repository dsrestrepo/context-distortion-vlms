# Import experiment-running functions from run_experiments.py
from src.run_experiments import (
    generate_predictions_models_base,
    generate_predictions_models,
    get_shifted_image,
    get_shifted_text,
    get_shifted_metadata,
    _normalize_versions,
    _safe_version_tag
)

import os, re, json, glob
import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, cohen_kappa_score
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import matplotlib as mpl
from collections import defaultdict

PRETTY_NAMES = {
    "qwen2_vl_7b":        "Qwen-2 VL",
    "llava_1_5_7b":       "LLaVA 1.5",
    "paligemma2_10b":     "PaliGemma-2",
    "janus_pro_7b":       "Janus-Pro",
    "biomedgpt":          "BiomedGPT",
    "medgemma":           "MedGemma",
    "llama3_10b":         "Llama-3.2",
    "llava_med_llava_v1": "LLaVA-Med V1",
    "llava_med_mistral_instruct": "LLaVA-Med I",
    "llava_med_llava_v0": "LLaVA-Med V0",
    "llava_med":               "LLaVA-Med",
    "medgemma_1_5b":    "MedGemma 1.5",
}


PRETTY_NAMES.update({
    "openai/gpt-4o-mini":   "GPT‑4o mini",
    "openai/gpt-4o":        "GPT‑4o",
    "openai/gpt-4.1":       "GPT‑4.1",
    "openai/gpt-5":          "GPT‑5",
    "gpt-5":                 "GPT‑5",
    "gpt-5-2025-08-07": "GPT‑5",
    "openai/gpt-5-mini":    "GPT‑5 mini",
    "openai/gpt-oss-20b":   "GPT‑OSS‑20B",         # text‑only
    "openai/gpt-oss-120b":  "GPT‑OSS‑120B",        # text‑only
    "gemini_3_pro":        "Gemini 3 Pro",
    "gemini-3-pro-preview": "Gemini 3 Pro",
    "gemini-2.0-flash":     "Gemini 2.0",
    "gemini-1.5":           "Gemini 1.5",
    "claude-3-7-sonnet":    "Claude 3.7 Sonnet",
    "grok-4":               "Grok‑4",
})

VERSION_PRETTY_NAMES = {
    "default": "v0",
    "v1": "v1",
    "v2": "v2",
    "v3": "v3",
    "v4": "v4"
}


##############################################################################
# EVALUATION FUNCTIONS
##############################################################################

no_patterns = [
    r'\bno evidence\b',
    r'\bno abnormalities\b',
    r'\bno significant change\b',
    r'\bno focal consolidation\b',
    r'\bno acute cardiopulmonary process\b',
    r'\bwithin normal limits\b',
    r'\bclear lungs\b',
    r'\bno pneumothorax\b',
    r'\bno pleural effusion\b',
    r'\bdoes not (have|show|reveal)\b',
    r'\bappears to be normal\b',
    r'\bdoes not appear to have glaucoma\b',
    r'\bno signs of glaucoma\b',
    r'\bnormal intraocular pressure\b',
    r'\bno thinning retinal nerve fiber layer\b',
    r'\bfull visual fields\b'
]

yes_patterns = [
    r'\bglaucoma\b',
    r'\bglaucoma suspect\b',
    r'\bprimary open-angle glaucoma\b',
    r'\bnarrow angle glaucoma\b',
    r'\bborderline glaucoma\b',
    r'\boptic nerve damage\b',
    r'\belevated intraocular pressure\b',
    r'\bincreased cup:disc ratio\b',
    r'\bcupping of the optic nerve\b',
    r'\bvisual field defects\b',
    r'\boptic nerve head damage\b',
    r'\bhistory of glaucoma\b',
    r'\bglaucoma diagnosis\b'
    r'\bpleural effusion\b', 
    r'\bpulmonary edema\b', 
    r'\bpneumothorax\b', 
    r'\bcardiomegaly\b', 
    r'\bconsolidation\b', 
    r'\binfiltrate\b', 
    r'\bopacification\b', 
    r'\binfection\b', 
    r'\bpneumonia\b'
]


def clean_label(label, allow_unmatched=False):
    """
    Clean and normalize prediction/ground truth labels.
    
    Args:
        label: Raw label to clean
        allow_unmatched: If True, recognize 'unmatched' labels instead of marking as unknown
    
    Returns:
        str: Normalized label ('yes', 'no', 'unmatched', or 'unknown, response: ...')
    """
    label = str(label).lower().strip().replace('.', '')
    
    # Check for unmatched responses first (if allowed)
    if allow_unmatched:
        if label in ['unmatched', 'unmatch', 'unmatched response', 'unmatched response:', 'missmatch']:
            return 'unmatched'
    
    # Exact matches
    if label in ['yes', 'y', "**yes**", "**answer:** yes", '**"yes"**']:
        return 'yes'
    elif label in ['no', 'n', '**no**', "**answer:** no", '**"no"**']:
        return 'no'

    elif "**yes**" in label or "**answer:** yes" in label or "answer: yes" in label or 'output: yes' in label or '**"yes"**' in label:
        return 'yes'
    
    elif "**no**" in label or "**answer:** no" in label  or "answer: no" in label or 'output: no' in label or '**"no"**' in label:
        return 'no'
    
    # Pattern-based classification
    #elif any(re.search(pattern, label) for pattern in no_patterns):
    #    return 'no'
    #elif any(re.search(pattern, label) for pattern in yes_patterns):
    #    return 'yes'
        
    elif re.search(r'\bdoes not (have|show|reveal)\b', label):
        return 'no'
    elif re.search(r'\bappears to be normal\b', label):
        return 'no'
        
    #elif ('does not have' in label) or ('does not show' in label):
    #    return 'no'

    elif label.startswith('no ') or label.startswith('no,'):
        return 'no'
    elif label.startswith('yes ') or label.startswith('yes,'):
        return 'yes'
    else:
        return f'unknown, response: {label}'


def clean_label_missmatch(label):
    """Backward compatibility wrapper for clean_label with allow_unmatched=True."""
    return clean_label(label, allow_unmatched=True)


def _apply_softmax_to_probs(df, p_yes=True):
    """
    Apply softmax normalization to probability columns and create prediction column.
    
    Args:
        df: DataFrame with probability columns
        p_yes: If True, use p_yes/p_no columns; else use p_Yes/p_No
    
    Returns:
        DataFrame with normalized probabilities and prediction column added
    """
    import torch
    import torch.nn.functional as F
    
    df = df.copy()
    
    if p_yes:
        df['p_yes'] = df['p_yes'].astype(float)
        df['p_no'] = df['p_no'].astype(float)
        logits = torch.tensor(df[['p_yes', 'p_no']].values)
        probs = F.softmax(logits, dim=1).numpy()
        df['p_yes'], df['p_no'] = probs[:, 0], probs[:, 1]
        df['prediction'] = np.where(df['p_yes'] > df['p_no'], 'yes', 'no')
    else:
        df['p_Yes'] = df['p_Yes'].astype(float)
        df['p_No'] = df['p_No'].astype(float)
        logits = torch.tensor(df[['p_Yes', 'p_No']].values)
        probs = F.softmax(logits, dim=1).numpy()
        df['p_Yes'], df['p_No'] = probs[:, 0], probs[:, 1]
        df['prediction'] = np.where(df['p_Yes'] > df['p_No'], 'yes', 'no')
    
    return df


def calculate_metrics(y, y_pred, show_unknown_responses=False, show_probs=False, silent=False):
    
    # Check 'unknown' responses
    unknown_responses = y_pred[y_pred.str.startswith('unknown')].unique()
    if len(unknown_responses) > 0 and not silent:
        print(f"Total unknown responses: {len(unknown_responses)}")
        if show_unknown_responses:
            print(f"Unknown responses: {unknown_responses}")
    
    
    # Exclude unknown responses
    if not silent:
        print(f"Excluding {len(y[y_pred.str.startswith('unknown')])} unknown responses out of {len(y)}")
    y = y[~y_pred.str.startswith("unknown")]
    y_pred = y_pred[~y_pred.str.startswith("unknown")]

    if show_probs and not silent:
    
        print(f"Precited probability of Condition=Yes|prompt,image: ", y_pred.value_counts(normalize=True))
        print(f"Precited probability of Condition=No|prompt,image: ", 1 - y_pred.value_counts(normalize=True))
        
        print(f"Actual probability of Condition=Yes|prompt,image: ", y.value_counts(normalize=True))
        print(f"Actual probability of Condition=No|prompt,image: ", 1 - y.value_counts(normalize=True))
    
    
    # Calculate metrics (accuracy, precision, recall, F1, sensitivity, specificity)
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, pos_label='yes', zero_division=np.nan)
    recall = recall_score(y, y_pred, pos_label='yes', zero_division=np.nan)
    f1 = f1_score(y, y_pred, pos_label='yes', zero_division=np.nan)
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=['no', 'yes']).ravel()
    
    # AUC score
    # convert to binary labels
    y = (y == 'yes').astype(int)
    y_pred = (y_pred == 'yes').astype(int)
    
    try:
        auc = roc_auc_score(y, y_pred)
    except ValueError:
        auc = np.nan
    
    sensitivity = tp / (tp + fn) if tp + fn > 0 else np.nan
    specificity = tn / (tn + fp) if tn + fp > 0 else np.nan
    
    return accuracy, precision, recall, f1, sensitivity, specificity, auc


def compute_entropy_and_cross_entropy(df, p_yes_col="p_yes", p_no_col="p_no", label_col="ground_truth"):
    """
    Compute mean entropy and cross-entropy from p_yes, p_no, and ground truth labels.

    Parameters:
    - df (pd.DataFrame): DataFrame containing predicted probabilities and true labels
    - p_yes_col (str): name of the column for probability of "yes"
    - p_no_col (str): name of the column for probability of "no"
    - label_col (str): name of the column for ground truth ("yes"/"no")

    Returns:
    - entropy (float)
    - entropy_std (float)
    - cross_entropy (float)
    - ce_std (float)
    """
    with torch.no_grad():
        # Extract and clamp probabilities
        probs = torch.tensor(df[[p_yes_col, p_no_col]].values, dtype=torch.float32)
        probs = torch.clamp(probs, min=1e-6, max=1. - 1e-6)

        # ---------- ENTROPY ----------
        entropies = -torch.sum(probs * torch.log(probs), dim=1)
        entropy = entropies.mean().item()
        entropy_std = entropies.std().item()

        # ---------- CROSS ENTROPY ----------
        y_true = torch.tensor((df[label_col] == "yes").astype(int).values, dtype=torch.long)
        y_one_hot = torch.zeros_like(probs)
        y_one_hot[torch.arange(len(y_true)), y_true] = 1

        cross_entropies = -torch.sum(y_one_hot * torch.log(probs), dim=1)
        cross_entropy = cross_entropies.mean().item()
        ce_std = cross_entropies.std().item()

    return entropy, entropy_std, cross_entropy, ce_std


def plot_calibration(df, prob_col, label_col, title="Calibration", n_bins=10, out_path=None):
    y_true = (df[label_col] == "yes").astype(int)
    y_prob = df[prob_col]

    # Get calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")

    # Bin counts so we can weight ECE properly
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_counts, _ = np.histogram(y_prob, bins=bin_edges)
    non_empty = bin_counts > 0
    counts_kept = bin_counts[non_empty]

    # Expected Calibration Error
    ece = np.sum((counts_kept / len(y_true)) *
                 np.abs(prob_true - prob_pred))

    # Plot reliability diagram
    plt.figure(figsize=(5, 5))
    plt.plot(prob_pred, prob_true, marker='o', label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    plt.title(title)
    plt.xlabel("Predicted Probability")
    plt.ylabel("True Frequency")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    if out_path is not None:
        plt.savefig(out_path, dpi=200)
    plt.show()
    
    return ece


def plot_all_shifts_calibration(
    model_key,
    dataset="medeval",
    base_dir="results",
    pretty_names=None,
    n_bins=10,
    p_yes=False,
    shift_col="shift",
    shifts = {
        "No": "No Shift",
        "Image": "Img Shift",
        "Text": "Txt Shift",
        "Only_text": "Only Text",
        "Only_image": "Only Image"
    },
    path=None
):
    plt.figure(figsize=(6,6))

    if not path:
        path = os.path.join(base_dir, model_key, f"{dataset}_base_shifted.csv")    

    df = pd.read_csv(path)
    
    # Check for probability columns
    req_cols = ['p_yes', 'p_no'] if p_yes else ['p_Yes', 'p_No']
    if not all(col in df.columns for col in req_cols) or df[req_cols].isna().all().all():
        print(f"Skipping calibration plot for {model_key}: Probability columns missing or empty.")
        return None

    # Clean the ground truth
    df["ground_truth"] = df["ground_truth"].apply(clean_label)
    
    # Normalize using softmax
    import torch
    import torch.nn.functional as F
    
    if p_yes:
        logits = torch.tensor(df[['p_yes','p_no']].values)
        probs  = F.softmax(logits, dim=1).numpy()
        df['p_yes'], df['p_no'] = probs[:,0], probs[:,1]
        df['pred_first_token_yes'] = np.where(df['p_yes'] > df['p_no'], 'yes', 'no')
    else:
        logits = torch.tensor(df[['p_Yes','p_No']].values)
        probs  = F.softmax(logits, dim=1).numpy()
        df['p_Yes'], df['p_No'] = probs[:,0], probs[:,1]
        df['pred_first_token_Yes'] = np.where(df['p_Yes'] > df['p_No'], 'yes', 'no')
    
    df_original = df.copy()
    
    
    for shift_code, shift_label in shifts.items():
        
        df = df_original[df_original[shift_col] == shift_code]

        # if you need first_token probs, apply your softmax logic here
        y_true = (df["ground_truth"] == "yes").astype(int)
        y_prob = df["p_yes"] if p_yes else df["p_Yes"]
        
        # Get calibration curve
        try:
            prob_true, prob_pred = calibration_curve(
                y_true, y_prob, n_bins=n_bins, strategy="uniform"
            )
            plt.plot(prob_pred, prob_true, marker='o', label=shift_label)
        except ValueError as e:
            print(f"Error in calibration for {shift_label}: {e}")
            continue
    plt.plot([0,1],[0,1], '--', color='gray')
    name = pretty_names.get(model_key, model_key) if pretty_names else model_key
    plt.title(f"{name} Calibration ({dataset})")
    plt.xlabel("Predicted Probability")
    plt.ylabel("True Frequency")
    plt.legend()
    os.makedirs("images", exist_ok=True)
    out_path = f"images/{model_key}_{dataset}_calibration.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.show()
    plt.close()
    return out_path


def plot_and_save_calibration(df, shift_type, save_calibration_plot, out_path, prob_col, label_col, title="Calibration"):

    try:
        if save_calibration_plot:
            out_path = f'images/calibration_example_{shift_type}.png'
        else:
            out_path = None
        ece = plot_calibration(df, prob_col=prob_col, label_col=label_col, title=title, out_path=out_path)
        print(f"Expected Calibration Error (ECE): {ece:.4f}")
    except:
        ece = np.nan
        print(f"Error plotting calibration for {shift_type} p_yes")
        
    return ece


def calculate_metrics_dataset(results_dir, subgroup_variables=["gender", "race", "language", "maritalstatus"], counterfactual=False, confusion_matrix=False, first_token=False, show_unknown_responses=False, unmatched=False, calibration=False, p_yes=False, save_calibration_plot=False, shift_col="shift", edited_cols=['Image', 'Text'], 
                              shift_names={
                                    "No": "No Shift",
                                    "Image": "Img Shift",
                                    "Text": "Txt Shift",
                                    "Only_text": "Only Text",
                                    "Only_image": "Only Image"
                                }, multi_history=None, history_length_col='history_length', max_subgroup_categories=5, split_q=3, shifts=None, version=None, silent=False):
    
    global_shifts = shifts
    # Load the results
    df_initial = pd.read_csv(results_dir)
    
    # Check if probability columns exist for first_token
    if first_token:
        # Check required columns
        req_cols = ['p_yes', 'p_no'] if p_yes else ['p_Yes', 'p_No']
        
        # Determine availability
        cols_exist = all(c in df_initial.columns for c in req_cols)
        
        # Also check if they are not all-NaN if they exist
        cols_valid = False
        if cols_exist:
            # Check if at least one row has valid non-NaN data in these columns
            if not df_initial[req_cols].dropna().empty:
                cols_valid = True
        
        if not cols_valid:
            if not silent:
                print(f"Metrics (First Token): Probability columns {req_cols} missing or empty. "
                      "Falling back to regex-based extraction (Entropy/Calibration will be NaN, ECE ignored).")
            # Force first_token to False to trigger regex logic and skip entropy/calibration
            first_token = False
    
    if version is not None:
        df_initial = df_initial[df_initial["version"] == version]
    
    if multi_history is not None:
        # get the history lengths:
        lengths = df_initial[history_length_col].unique().tolist()
    else:
        lengths = [0]
        
    # prepare output dict for this dataset
    shift_metrics = {}
        
    # Iterate over all the pussible history lengths
    for length in lengths:
        shifts = global_shifts
        df = df_initial[df_initial[history_length_col] == length].copy() if multi_history is not None else df_initial.copy()
        
        # Clean the labels
        if "mimic" or "medeval" in results_dir:
            df["ground_truth"] = df["ground_truth"].apply(clean_label)
            if not first_token:
                df["prediction"] = df["prediction"].apply(lambda x: clean_label(x, allow_unmatched=unmatched))
        elif "ham10000" in results_dir:
            df["ground_truth"] = df["ground_truth"].apply(lambda x: x.lower())
            if not first_token:
                df["prediction"] = df["prediction"].apply(lambda x: x.lower()) 
                # mel = melanoma = yes, nv = nevus = no
            df["ground_truth"] = df["ground_truth"].apply(lambda x: "yes" if x == "mel" or x == 'melanoma' else "no")
            if not silent:
                print(f'For simplicity we are considering "mel" and "melanoma" as "yes" and "nv" and "nevus" as "no"')
        
        # remove unknown responses
        df_unfiltered = df.copy()
        if not first_token:
            if show_unknown_responses:
                unknowns = df[df["prediction"].str.startswith("unknown")]["prediction"].unique()
                if len(unknowns) > 0 and not silent:
                    print(f"\n[DEBUG] Unknown responses for history length {length}:")
                    for u in unknowns:
                        print(u)

            df = df[~df["prediction"].str.startswith("unknown")]

        
        # Get first token probability
        if first_token:
            df = _apply_softmax_to_probs(df, p_yes=p_yes)            
            

        df_original = df.copy()
        
        if unmatched and not first_token and not silent:
            # Calculate proportion of unmatched responses by shift type
            unmatched_counts = df_original[df_original["prediction"].str.startswith("unmatched")].groupby(shift_col).size()
            print(f"Unmatched responses by shift type:\n{unmatched_counts}")
            
            # Calculate the number of correct unmatched predictions ("Image", "Text") should be unmatched
            correct_unmatched = df_original[(df_original[shift_col].isin(edited_cols)) & (df_original["prediction"].str.startswith("unmatched"))].shape[0]
            print(f"Number of correct unmatched predictions: {correct_unmatched} of {len(df_original[df_original[shift_col].isin(edited_cols)])}")
                    
            # Calculate the number of incorrect unmatched predictions ("Image", "Text") should not be unmatched
            incorrect_unmatched = df_original[(df_original[shift_col].isin(edited_cols)) & (~df_original["prediction"].str.startswith("unmatched"))].shape[0]
            print(f"Number of unmatched not predicted: {incorrect_unmatched} of {len(df_original[df_original[shift_col].isin(edited_cols)])}")
            
            # Number of incorrect unmatched predictions for all shifts except "Image" and "Text"
            incorrect_unmatched_all = df_original[~df_original[shift_col].isin(edited_cols) & df_original["prediction"].str.startswith("unmatched")].shape[0]
            print(f"Number of incorrect unmatched predicted: {incorrect_unmatched_all} of {len(df_original[~df_original[shift_col].isin(edited_cols)])}")
            
            # remove unmatched responses from the dataframe
            print(f'Filtering unmatched responses from the dataframe...')
        
        if unmatched and not first_token:
            df_original = df_original[~df_original["prediction"].str.startswith("unmatched")]

        if shifts is None:
            shifts = pd.unique(df_original[shift_col])
            
            if "No" in shifts:
                shifts = ["No"] + [s for s in shifts if s != "No"]
        
        for counter, shift_type in enumerate(shifts):
            
            if counterfactual:  
                if length > 1:
                    if counter > 0:
                        continue
                    else:
                        if not silent:
                            print(f"History length: {length}")
                        df = df_original
                        df_unfiltered_subset = df_unfiltered
                        
                else:
                    if not silent:
                        print(40 * "=" + f" Metrics for {shift_type} Shift " + 40 * "=")
                    df = df_original[df_original[shift_col] == shift_type].copy()
                    df_unfiltered_subset = df_unfiltered[df_unfiltered[shift_col] == shift_type]
            else:
                df = df_original
                df_unfiltered_subset = df_unfiltered
                
                if counter > 0:
                    continue
            
            # Calculate rejection stats
            total_count = len(df_unfiltered_subset)
            rejection_count = 0
            if not first_token:
                rejection_count = df_unfiltered_subset["prediction"].str.startswith("unknown").sum()
            
            correct_count = (df["ground_truth"] == df["prediction"]).sum()
            incorrect_count = (df["ground_truth"] != df["prediction"]).sum()
                
            
            #print(f"DEBUG: After filtering by length {length} and shift {shift_type}, df shape: {df.shape}")
            #print(df[["ground_truth", "prediction"]])
            
            
            # Print the confusion matrix
            if confusion_matrix and not silent:
                if not first_token:
                    print(40 * "=" + f" Confusion Matrix for {shift_type} Shift " + 40 * "=")
                    print(pd.crosstab(df["ground_truth"], df["prediction"], rownames=["Actual"], colnames=["Predicted"]))
                else:
                    # Use the 'prediction' column created by _apply_softmax_to_probs
                    if p_yes:
                        print(40 * "=" + f" Confusion Matrix for First Token Probability (p_yes/p_no) for {shift_type} Shift " + 40 * "=")
                    else:
                        print(40 * "=" + f" Confusion Matrix for First Token Probability (p_Yes/p_No) for {shift_type} Shift " + 40 * "=")
                    print(pd.crosstab(df["ground_truth"], df["prediction"], rownames=["Actual"], colnames=["Predicted"]))        
            

            if not first_token:
                # Overall metrics:
                accuracy, precision, recall, f1, sensitivity, specificity, auc = calculate_metrics(df["ground_truth"], df["prediction"], show_unknown_responses=show_unknown_responses, silent=silent)
                if not silent:
                    print(40 * "=" + " Overall Metrics " + 40 * "=")
                
                # Set ece and entropy to nan for non-first_token case
                ece = np.nan
                entropy, entropy_std, cross_entropy, ce_std = np.nan, np.nan, np.nan, np.nan
                
            if first_token:
                if p_yes:
                    if not silent:
                        print(40 * "=" + " Overall Metrics for First Token Probability yes " + 40 * "=")
                                  
                    ## Compute the entropy and cross-entropy
                    entropy, entropy_std, cross_entropy, ce_std = compute_entropy_and_cross_entropy(df, p_yes_col="p_yes", p_no_col="p_no", label_col="ground_truth")
                    
                    
                    if calibration:
                        # Always calculate ECE, but only plot if not silent
                        if not silent:
                            ece = plot_and_save_calibration(df, shift_type, save_calibration_plot, out_path=f'images/calibration_example_{shift_type}.png', prob_col="p_yes", label_col="ground_truth", title=f"Calibration: {shift_type} p_yes")
                        else:
                            # Calculate ECE without plotting (for bootstrap)
                            from sklearn.calibration import calibration_curve
                            y_true = (df["ground_truth"] == "yes").astype(int)
                            y_prob = df["p_yes"]
                            n_bins = 10
                            bin_edges = np.linspace(0, 1, n_bins + 1)
                            bin_counts, _ = np.histogram(y_prob, bins=bin_edges)
                            non_empty = bin_counts > 0
                            counts_kept = bin_counts[non_empty]
                            try:
                                prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
                                ece = np.sum((counts_kept / len(y_true)) * np.abs(prob_true - prob_pred))
                            except:
                                ece = np.nan
                    else:
                        ece = np.nan

                else:
                    if not silent:
                        print(40 * "=" + " Overall Metrics for First Token Probability Yes " + 40 * "=")

                    ## Compute the entropy and cross-entropy
                    entropy, entropy_std, cross_entropy, ce_std = compute_entropy_and_cross_entropy(df, p_yes_col="p_Yes", p_no_col="p_No", label_col="ground_truth")
                    
                    if calibration:
                        # Always calculate ECE, but only plot if not silent
                        if not silent:
                            ece = plot_and_save_calibration(df, shift_type, save_calibration_plot, out_path=f'images/calibration_example_{shift_type}.png', prob_col="p_Yes", label_col="ground_truth", title=f"Calibration: {shift_type} p_Yes")
                        else:
                            # Calculate ECE without plotting (for bootstrap)
                            from sklearn.calibration import calibration_curve
                            y_true = (df["ground_truth"] == "yes").astype(int)
                            y_prob = df["p_Yes"]
                            n_bins = 10
                            bin_edges = np.linspace(0, 1, n_bins + 1)
                            bin_counts, _ = np.histogram(y_prob, bins=bin_edges)
                            non_empty = bin_counts > 0
                            counts_kept = bin_counts[non_empty]
                            try:
                                prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
                                ece = np.sum((counts_kept / len(y_true)) * np.abs(prob_true - prob_pred))
                            except:
                                ece = np.nan
                    else:
                        ece = np.nan
                
                # Use the 'prediction' column created by _apply_softmax_to_probs
                accuracy, precision, recall, f1, sensitivity, specificity, auc = calculate_metrics(df["ground_truth"], df["prediction"], show_unknown_responses=False, silent=silent)
                
            
            if not silent:
                print(f"Accuracy: {accuracy:.4f}")
                print(f"Precision: {precision:.4f}")
                print(f"Recall: {recall:.4f}")
                #print(f"F1: {f1:.4f}")
                #print(f"Sensitivity: {sensitivity:.4f}")
                print(f"Specificity: {specificity:.4f}")
            
            shift_name = shift_names[shift_type] if shift_type in shift_names else shift_type

            if multi_history is not None:
                if length not in shift_metrics:
                    shift_metrics[length] = {}
                    
                shift_metrics[length][shift_name] = {
                    "Accuracy": accuracy,
                    "Precision": precision,
                    "Recall": recall,
                    "F1": f1,
                    "Specificity": specificity,
                    "ECE": ece if calibration else None,
                    "Entropy": entropy if first_token else None,
                    "Entropy Std": entropy_std if first_token else None,
                    "Cross Entropy": cross_entropy if first_token else None,
                    "Cross Entropy Std": ce_std if first_token else None,
                    "History Length": length,
                    "Correct": correct_count,
                    "Incorrect": incorrect_count,
                    "Rejection": rejection_count,
                    "Total": total_count
                }
            
            else:
                shift_metrics[shift_name] = {
                    "Accuracy": accuracy,
                    "Precision": precision,
                    "Recall": recall,
                    "F1": f1,
                    "Specificity": specificity,
                    "ECE": ece if calibration else None,
                    "Entropy": entropy if first_token else None,
                    "Entropy Std": entropy_std if first_token else None,
                    "Cross Entropy": cross_entropy if first_token else None,
                    "Cross Entropy Std": ce_std if first_token else None,
                    "History Length": length if multi_history is not None else None,
                    "Correct": correct_count,
                    "Incorrect": incorrect_count,
                    "Rejection": rejection_count,
                    "Total": total_count
                }
            
            if counter < 1:
                # Group by demographic and calculate metrics
                for subgroup in subgroup_variables:
                    
                    # Check if is countinuous, if so split into n categories
                    if df[subgroup].dtype in [int, float] and len(df[subgroup].unique()) > max_subgroup_categories:
                        if not silent:
                            print(f"Splitting {subgroup} into {split_q} categories: Low ({df[subgroup].min()} - {df[subgroup].quantile(0.33)}), Medium ({df[subgroup].quantile(0.33)} - {df[subgroup].quantile(0.66)}), High ({df[subgroup].quantile(0.66)} - {df[subgroup].max()})")
                        df[subgroup] = pd.qcut(df[subgroup], q=split_q, labels=["low", "medium", "high"])
                    
                    if not silent:
                        print(40 * "=" + f" Metrics by {subgroup.capitalize()} " + 40 * "=")
                    
                    
                    # Initialize subgroup dict if it doesn't exist
                    #if multi_history is not None:
                    #    if shift_name not in shift_metrics[length]:
                    #        shift_metrics[length][shift_name] = {}
                    #    if subgroup not in shift_metrics[length][shift_name]:
                    #        shift_metrics[length][shift_name][subgroup] = {}
                    #else:
                    #    if shift_name not in shift_metrics:
                    #        shift_metrics[shift_name] = {}
                    #    if subgroup not in shift_metrics[shift_name]:
                    #        shift_metrics[shift_name][subgroup] = {}
                    
                    # Check if soubgroup already in the dict
                    if subgroup not in shift_metrics:
                        shift_metrics[shift_name][subgroup] = {}
                        
                    for subgroup_value in df[subgroup].unique():
                        subgroup_df = df[df[subgroup] == subgroup_value].copy()
                        if len(subgroup_df) == 0:
                            continue
                        
                        if not silent:
                            print(f"Subgroup: {subgroup}={subgroup_value} (n={len(subgroup_df)})")
                        
                        if not first_token:
                            accuracy, precision, recall, f1, sensitivity, specificity, auc = calculate_metrics(subgroup_df["ground_truth"], subgroup_df["prediction"], show_unknown_responses=show_unknown_responses, silent=silent)
                            
                            ece = np.nan
                            entropy, entropy_std, cross_entropy, ce_std = np.nan, np.nan, np.nan, np.nan
                        else:
                            if p_yes:
                                label_logits = "pred_first_token_yes" 
                                accuracy, precision, recall, f1, sensitivity, specificity, auc = calculate_metrics(subgroup_df["ground_truth"], subgroup_df[label_logits], show_unknown_responses=False, silent=silent)
                                entropy, entropy_std, cross_entropy, ce_std = compute_entropy_and_cross_entropy(subgroup_df, p_yes_col="p_yes", p_no_col="p_no", label_col="ground_truth")
                                if calibration:
                                    ece = plot_and_save_calibration(subgroup_df, shift_type, save_calibration_plot, out_path=f'images/calibration_example_{shift_type}_{subgroup}_{subgroup_value}.png', prob_col="p_yes", label_col="ground_truth", title=f"Calibration: {shift_type} {subgroup}={subgroup_value} p_yes")
                            else:
                                label_logits = "pred_first_token_Yes"
                                accuracy, precision, recall, f1, sensitivity, specificity, auc = calculate_metrics(subgroup_df["ground_truth"], subgroup_df[label_logits], show_unknown_responses=False, silent=silent)
                                entropy, entropy_std, cross_entropy, ce_std = compute_entropy_and_cross_entropy(subgroup_df, p_yes_col="p_Yes", p_no_col="p_No", label_col="ground_truth")
                                if calibration:
                                    ece = plot_and_save_calibration(subgroup_df, shift_type, save_calibration_plot, out_path=f'images/calibration_example_{shift_type}_{subgroup}_{subgroup_value}.png', prob_col="p_Yes", label_col="ground_truth", title=f"Calibration: {shift_type} {subgroup}={subgroup_value} p_Yes")
                        
                        if not silent:
                            print(f"Accuracy: {accuracy:.4f}")
                        
                        if subgroup_value not in shift_metrics[shift_name][subgroup]:
                            shift_metrics[shift_name][subgroup][subgroup_value] = {}
                        
                        shift_metrics[shift_name][subgroup][subgroup_value] = {
                            "Accuracy": accuracy,
                            "Precision": precision,
                            "Recall": recall,
                            "F1": f1,
                            "Specificity": specificity,
                            "ECE": ece if calibration else None,
                            "Entropy": entropy if first_token else None,
                            "Entropy Std": entropy_std if first_token else None,
                            "Cross Entropy": cross_entropy if first_token else None,
                            "Cross Entropy Std": ce_std if first_token else None,
                            "History Length": length if multi_history is not None else None
                        }

                    if not silent:
                        print("\n")
                    

            
            if not counterfactual:
                break
        
    return shift_metrics
        
# Implement def calculate_metrics_mimic
def calculate_metrics_all_models(model_dict, subgroup_variables=["gender", "race", "language", "maritalstatus"], results_dir="results", dataset="medeval", counterfactual=False, confusion_matrix=False, first_token=False, show_unknown_responses=False, unmatched=False, calibration=False, p_yes=False, save_calibration_plot=False, shift_col="shift", edited_cols=['Image', 'Text'], file_name=None, 
                                shift_names={
                                    "No": "No Shift",
                                    "Image": "Img Shift",
                                    "Text": "Txt Shift",
                                    "Only_text": "Only Text",
                                    "Only_image": "Only Image"
                                }, max_subgroup_categories=5, split_q=3, shifts=None, versions=None, bootstrap=False, n_bootstrap=1000, random_state=42, history_length_col='history_length', bootstrap_proportion=1.0):
    """
    Calculate metrics for all models in model_dict.
    
    Parameters:
    -----------
    bootstrap : bool, optional (default=False)
        If True, perform bootstrapping over the dataset to compute confidence intervals.
        When bootstrap=True, logging is suppressed (silent=True).
    n_bootstrap : int, optional (default=1000)
        Number of bootstrap iterations to perform.
    random_state : int, optional (default=42)
        Random seed for reproducibility.
    
    Returns:
    --------
    dict : Dictionary of results, structured as:
        - If bootstrap=False: {model_name: {shift: {metric: value}}}
        - If bootstrap=True: {model_name: {variant: {shift: {metric: value}}}}
          where variant is "boot_000", "boot_001", ..., "boot_N"
    """
    all_results = {}
    
    # Set up random state for bootstrap
    if bootstrap:
        rng = np.random.RandomState(random_state)
    
    for model_id, model_name in model_dict.items():
        if not bootstrap:
            print(90 * "=" )
            print(40 * "=" + f" Metrics for {model_name} " + 40 * "=")
            print(90 * "=" )
        else:
            print(f"Bootstrapping {model_name}...")
            
        if not file_name:
            if counterfactual:
                if unmatched:
                    path = os.path.join(results_dir, model_name, f"{dataset}_base_shifted_unmatched.csv")
                elif versions:
                    # _multi_versions
                    path = os.path.join(results_dir, model_name, f"{dataset}_base_shifted_multi_versions.csv")
                else:
                    path = os.path.join(results_dir, model_name, f"{dataset}_base_shifted.csv")
                    
                # metrics = calculate_metrics_dataset(os.path.join(results_dir, model_name, f"{dataset}_base_shifted.csv"), subgroup_variables=subgroup_variables, counterfactual=counterfactual, confusion_matrix=confusion_matrix, first_token=first_token, show_unknown_responses=show_unknown_responses, calibration=calibration, p_yes=p_yes, save_calibration_plot=save_calibration_plot, shift_col=shift_col, edited_cols=edited_cols)
            else:
                path = os.path.join(results_dir, model_name, f"{dataset}_base.csv")
        else:
            path = os.path.join(results_dir, model_name, file_name)
            
        multi_history = True if "multi_history" in dataset else None
        
        # Handle bootstrap mode
        if bootstrap:
            # Read the CSV once
            df_full = pd.read_csv(path)
            
            # Filter by version if needed
            if versions is not None and isinstance(versions, list):
                # For multiple versions, just use the first one for bootstrap
                # or you could bootstrap each version separately
                if "version" in df_full.columns:
                    print(f"   Multiple versions detected. Bootstrap will only use version '{versions[0]}'.")
                    print(f"   Other versions {versions[1:]} will be ignored during bootstrap.")
                    df_full = df_full[df_full["version"] == versions[0]]
            
            # Create bootstrap variants dictionary
            bootstrap_variants = {}
            
            # Perform bootstrap sampling
            for i in tqdm(range(n_bootstrap), desc=f"Bootstrap {model_name}", disable=False):
                # Sample with replacement
                # Check if multi_history mode
                if multi_history and history_length_col in df_full.columns:
                    # For multi_history, sample by history length groups
                    bootstrap_dfs = []
                    for hist_len in df_full[history_length_col].unique():
                        hist_df = df_full[df_full[history_length_col] == hist_len]
                        
                        # For history length 0 and 1, also stratify by shift
                        if hist_len in [0, 1] and shift_col in df_full.columns:
                            for shift in hist_df[shift_col].unique():
                                shift_hist_df = hist_df[hist_df[shift_col] == shift]
                                bootstrap_sample = shift_hist_df.sample(
                                    n=int(len(shift_hist_df)*bootstrap_proportion), 
                                    replace=True, 
                                    random_state=rng.randint(0, 2**31-1)
                                )
                                bootstrap_dfs.append(bootstrap_sample)
                        else:
                            # For history length >= 2, just sample by history length
                            bootstrap_sample = hist_df.sample(
                                n=len(hist_df), 
                                replace=True, 
                                random_state=rng.randint(0, 2**31-1)
                            )
                            bootstrap_dfs.append(bootstrap_sample)
                    df_bootstrap = pd.concat(bootstrap_dfs, ignore_index=True)
                    
                elif shift_col in df_full.columns:
                    # Non-multi_history: Sample within each shift to maintain shift balance
                    bootstrap_dfs = []
                    for shift in df_full[shift_col].unique():
                        shift_df = df_full[df_full[shift_col] == shift]
                        bootstrap_sample = shift_df.sample(n=int(len(shift_df)*bootstrap_proportion), replace=True, random_state=rng.randint(0, 2**31-1))
                        bootstrap_dfs.append(bootstrap_sample)
                    df_bootstrap = pd.concat(bootstrap_dfs, ignore_index=True)
                else:
                    # No stratification
                    df_bootstrap = df_full.sample(n=int(len(df_full)*bootstrap_proportion), replace=True, random_state=rng.randint(0, 2**31-1))
                
                # Save temporary bootstrap CSV
                temp_path = path.replace(".csv", f"_bootstrap_temp_{i}.csv")
                df_bootstrap.to_csv(temp_path, index=False)
                
                # Calculate metrics with silent=True
                metrics = calculate_metrics_dataset(
                    temp_path, 
                    subgroup_variables=subgroup_variables, 
                    counterfactual=counterfactual, 
                    confusion_matrix=False,  # Disable confusion matrix for bootstrap
                    first_token=first_token, 
                    show_unknown_responses=False, 
                    unmatched=unmatched, 
                    calibration=calibration,  # Keep calibration to calculate ECE (plots will be skipped due to silent=True)
                    p_yes=p_yes, 
                    save_calibration_plot=False, 
                    shift_col=shift_col, 
                    edited_cols=edited_cols, 
                    shift_names=shift_names, 
                    multi_history=multi_history, 
                    history_length_col=history_length_col,
                    max_subgroup_categories=max_subgroup_categories, 
                    split_q=split_q, 
                    shifts=shifts,
                    silent=True  # Suppress all prints during bootstrap
                )
                
                # Clean up temp file
                os.remove(temp_path)
                
                # Store with variant name
                variant_name = f"boot_{i:03d}"
                bootstrap_variants[variant_name] = metrics
            
            all_results[model_name] = bootstrap_variants
            print(f"✓ Completed {n_bootstrap} bootstrap iterations for {model_name}")
        
        # Handle normal mode (no bootstrap)
        elif versions is not None and isinstance(versions, list):
            all_version_metrics = {}
            for version in versions:
                print(90 * "=" )
                print(40 * "=" + f" Metrics for {model_name} Version: {version} " + 40 * "=")
                print(90 * "=" )
                metrics = calculate_metrics_dataset(path, subgroup_variables=subgroup_variables, counterfactual=counterfactual, confusion_matrix=confusion_matrix, first_token=first_token, show_unknown_responses=show_unknown_responses, unmatched=unmatched, calibration=calibration, p_yes=p_yes, save_calibration_plot=save_calibration_plot, shift_col=shift_col, edited_cols=edited_cols, shift_names=shift_names, multi_history=multi_history, history_length_col=history_length_col, max_subgroup_categories=max_subgroup_categories, split_q=split_q, shifts=shifts, version=version, silent=False)
                all_version_metrics[version] = metrics
                print("\n\n")
            all_results[model_name] = all_version_metrics
        
        else:
            metrics = calculate_metrics_dataset(path, subgroup_variables=subgroup_variables, counterfactual=counterfactual, confusion_matrix=confusion_matrix, first_token=first_token, show_unknown_responses=show_unknown_responses, unmatched=unmatched, calibration=calibration, p_yes=p_yes, save_calibration_plot=save_calibration_plot, shift_col=shift_col, edited_cols=edited_cols, shift_names=shift_names, multi_history=multi_history, history_length_col=history_length_col, max_subgroup_categories=max_subgroup_categories, split_q=split_q, shifts=shifts, silent=False)
            print("\n\n")
        
            all_results[model_name] = metrics
        
        if calibration and first_token and not bootstrap:
            img_path = plot_all_shifts_calibration(
                model_key=model_name,
                dataset=dataset,
                base_dir=results_dir,
                pretty_names=PRETTY_NAMES,
                n_bins=10,
                p_yes=p_yes,
                shifts=shift_names,
                shift_col=shift_col,
                path=path
            )
    
    all_results = {PRETTY_NAMES.get(key, key): val for key, val in all_results.items()}
            
    return all_results




##############################################################################
# NFR (Negative Flip Rate)
##############################################################################

def compute_nfr_table(df: pd.DataFrame, id: str = 'filename', first_token: bool = False, p_yes: bool = False) -> pd.DataFrame:
    """
    Given a DataFrame from *_base_shifted.csv, return a DataFrame of NFR values
    for Image, Text, Only_text, Only_image shifts.
    
    NFR (Negative Flip Rate) = proportion of samples that were correct in baseline
    but became incorrect after perturbation.
    """
    df = df.copy()
    df["ground_truth"] = df["ground_truth"].apply(clean_label)
    
    if first_token:
        df = _apply_softmax_to_probs(df, p_yes=p_yes)
    else:
        df["prediction"] = df["prediction"].apply(clean_label)
        df = df[~df["prediction"].str.startswith("unknown")]

    if id not in df.columns or "shift" not in df.columns:
        print(df.columns)
        raise ValueError(f"Expected columns: '{id}' and 'shift'")

    base_df = df[df["shift"] == "No"]
    shifts = ["Image", "Text", "Only_text", "Only_image"]

    records = []
    for shift in shifts:
        pert_df = df[df["shift"] == shift]
        merged = base_df[[id, "ground_truth", "prediction"]].rename(
            columns={"prediction": "base_pred"}
        ).merge(
            pert_df[[id, "prediction"]],
            on=id, how="inner"
        ).rename(columns={"prediction": "pert_pred"})

        base_correct = merged["base_pred"] == merged["ground_truth"]
        pert_wrong = merged["pert_pred"] != merged["ground_truth"]
        n_flips = (base_correct & pert_wrong).sum()
        nfr = n_flips / len(merged) if len(merged) > 0 else np.nan

        records.append({"Shift": shift.replace("_", " ").title(), "NFR": nfr})

    return pd.DataFrame(records)


def plot_nfr(nfr_df: pd.DataFrame, model_name: str, save_path: str = None):
    """
    Plot the Negative Flip Rates for one model from the nfr_df returned by compute_nfr_table().
    """
    shift_colors = {
        "Image": "#E6550D",
        "Text": "#FDAE6B",
        "Only Text": "#31A354",
        "Only Image": "#A1D99B"
    }

    fig, ax = plt.subplots(figsize=(7, 5))
    shifts = nfr_df["Shift"]
    values = nfr_df["NFR"]
    colors = [shift_colors[s] for s in shifts]

    bars = ax.bar(shifts, values, color=colors)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.005,
                f"{height:.2f}", ha='center', va='bottom', fontsize=10)

    ax.set_title(f"Negative Flip Rate for {model_name}", fontsize=14)
    ax.set_ylabel("NFR (↓ better)", fontsize=12)
    ax.set_ylim(0, max(values)*1.15 if len(values) else 1.0)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
    



def compute_nfr_all_models(model_dict: dict,
                           results_dir: str = "results",
                           dataset: str = "mimic",
                           id: str = 'filename',
                           first_token: bool = False, 
                           p_yes: bool =False) -> pd.DataFrame:
    """
    Returns a dataframe whose rows are models and columns are the shifts:
    Image | Text | Only Text | Only Image
    """
    shifts = ["Image", "Text", "Only_text", "Only_image"]
    all_rows = []

    for hf_name, folder_name in model_dict.items():
        file_path = os.path.join(results_dir,
                                 folder_name,
                                 f"{dataset}_base_shifted.csv")
        if not os.path.isfile(file_path):
            print(f"[WARN] Missing results file for {folder_name}: {file_path}")
            continue

        df = pd.read_csv(file_path)
        try:
            nfr_df = compute_nfr_table(df, id=id, first_token=first_token, p_yes=p_yes)                 # → columns: Shift, NFR
        except Exception as e:
            print(f"[ERROR] Could not compute NFR for {folder_name}: {e}")
            continue

        # Pivot so every shift becomes a column
        row = nfr_df.set_index("Shift")["NFR"].reindex(
            [s.replace("_", " ").title() for s in shifts]
        )
        row.name = folder_name
        all_rows.append(row)

    if not all_rows:
        raise RuntimeError("No NFR values could be computed – check file paths.")

    nfr_matrix = pd.concat(all_rows, axis=1).T          # models × shifts
    return nfr_matrix


import matplotlib as mpl

def _pick_text_color(rgba, threshold=0.55):
    """Return 'black' or 'white' depending on perceptual lightness of RGBA."""
    r, g, b, _ = rgba
    # WCAG-like relative luminance (0=dark, 1=light)
    L = 0.2126*r + 0.7152*g + 0.0722*b
    return "black" if L > threshold else "white"


##############################################################################
# Flip Rate and Kappa Score per Prompt Variation
##############################################################################

def _prepare_pivot_data(df, id_col, version_col, shift_col, base_shift, first_token, p_yes, history_length_col, history_mode, include_shifts):
    """Helper to prepare pivoted data for agreement metrics."""
    df = df.copy()
    
    # Check for probability columns if first_token is True
    if first_token:
        import torch
        import torch.nn.functional as F
        
        req_cols = ['p_yes', 'p_no'] if p_yes else ['p_Yes', 'p_No']
        
        # Check if columns exist and are not all-NaN (checking first few rows or dropna for speed/robustness)
        cols_exist = all(c in df.columns for c in req_cols)
        cols_valid = False
        if cols_exist:
             if not df[req_cols].dropna().empty:
                cols_valid = True
        
        if not cols_valid:
             # print(f"Warning: Probability columns {req_cols} missing or empty. Falling back to regex predictions for agreement metrics.")
             first_token = False
    
    # Filter by shift type
    if include_shifts is not None:
        if isinstance(include_shifts, bool):
            # If True, include all shifts (no filtering)
            # If False, use base_shift only
            if not include_shifts and shift_col in df.columns:
                df = df[df[shift_col] == base_shift]
        elif isinstance(include_shifts, list) and shift_col in df.columns:
            # If list, filter to those specific shifts
            df = df[df[shift_col].isin(include_shifts)]
    elif shift_col in df.columns:
        df = df[df[shift_col] == base_shift]
    
    if df.empty:
        return None

    # Clean labels
    df["ground_truth"] = df["ground_truth"].apply(clean_label)
    
    if first_token:
        import torch
        import torch.nn.functional as F
        if p_yes:
            df['p_yes'] = df['p_yes'].astype(float)
            df['p_no'] = df['p_no'].astype(float)
            logits = torch.tensor(df[['p_yes','p_no']].values)
            probs = F.softmax(logits, dim=1).numpy()
            df['p_yes'], df['p_no'] = probs[:,0], probs[:,1]
            df['prediction'] = np.where(df['p_yes'] > df['p_no'], 'yes', 'no')
        else:
            logits = torch.tensor(df[['p_Yes','p_No']].values)
            probs = F.softmax(logits, dim=1).numpy()
            df['p_Yes'], df['p_No'] = probs[:,0], probs[:,1]
            df['prediction'] = np.where(df['p_Yes'] > df['p_No'], 'yes', 'no')
    else:
        df["prediction"] = df["prediction"].apply(clean_label)
        df = df[~df["prediction"].str.startswith("unknown")]

    # Handle history mode
    if history_mode == 'no_vs_length_groups' and history_length_col in df.columns:
        # Compare no history (0) vs each history length as separate groups
        # For each sample, we only keep one representative per history length
        # (by taking the first occurrence of each length)
        
        # Check for duplicates on (id, history_length, version)
        # If duplicates exist (likely due to shifts), include shift_col in composite_id
        subset_cols = [id_col, history_length_col, version_col]
        if df.duplicated(subset=subset_cols).any() and shift_col in df.columns:
             df['_composite_id'] = df[id_col].astype(str) + '_len' + df[history_length_col].astype(str) + '_' + df[shift_col].astype(str)
             # Still need to handle if there are duplicates even after adding shift_col (e.g. true duplicates)
             if df.duplicated(subset=['_composite_id', version_col]).any():
                 df = df.drop_duplicates(subset=['_composite_id', version_col], keep='first')
        else:
             df = df.sort_values(subset_cols)
             df = df.drop_duplicates(subset=subset_cols, keep='first')
             df['_composite_id'] = df[id_col].astype(str) + '_len' + df[history_length_col].astype(str)
             
        pivot_df = df.pivot(index='_composite_id', columns=version_col, values='prediction')

    elif history_mode == 'no_vs_length1_only' and history_length_col in df.columns:
        # Compare no history (0) vs history length=1 only (ignore length>1)
        df = df[df[history_length_col].isin([0, 1])]
        
        subset_cols = [id_col, history_length_col, version_col]
        if df.duplicated(subset=subset_cols).any() and shift_col in df.columns:
             df['_composite_id'] = df[id_col].astype(str) + '_len' + df[history_length_col].astype(str) + '_' + df[shift_col].astype(str)
             if df.duplicated(subset=['_composite_id', version_col]).any():
                 df = df.drop_duplicates(subset=['_composite_id', version_col], keep='first')
        else:
             df = df.sort_values(subset_cols)
             df = df.drop_duplicates(subset=subset_cols, keep='first')
             df['_composite_id'] = df[id_col].astype(str) + '_len' + df[history_length_col].astype(str)
             
        pivot_df = df.pivot(index='_composite_id', columns=version_col, values='prediction')
    else:
        # Default: pivot by id only (no history handling)
        # Check for duplicates which might be due to multiple shifts
        if df.duplicated(subset=[id_col, version_col]).any():
            if shift_col in df.columns:
                # Create composite key: id + shift
                df['_composite_id'] = df[id_col].astype(str) + '_' + df[shift_col].astype(str)
                pivot_df = df.pivot(index='_composite_id', columns=version_col, values='prediction')
            else:
                # Fallback: drop duplicates
                print(f"Warning: Duplicates found for {id_col} and {version_col}, but {shift_col} not found. Dropping duplicates.")
                df = df.drop_duplicates(subset=[id_col, version_col])
                pivot_df = df.pivot(index=id_col, columns=version_col, values='prediction')
        else:
            pivot_df = df.pivot(index=id_col, columns=version_col, values='prediction')
            
    # Rename columns using VERSION_PRETTY_NAMES
    if pivot_df is not None:
        pivot_df.columns = [VERSION_PRETTY_NAMES.get(str(c), str(c)) for c in pivot_df.columns]

    return pivot_df


def fleiss_kappa(table):
    """
    Fleiss' Kappa for inter-rater agreement.
    table: array-like, shape (N_subjects, N_categories)
           Element [i, j] is the number of raters who assigned the i-th subject to the j-th category.
    """
    table = np.asarray(table)
    n_sub, n_cat = table.shape
    n_total = table.sum()
    n_rater = table.sum(axis=1)
    
    # Check if number of raters is constant
    if np.unique(n_rater).size > 1:
        # If not constant, Fleiss kappa is not strictly applicable.
        # We will proceed but results might be approximate or we should filter.
        # For now, we use the mean number of raters for calculation or just proceed with the formula.
        # The formula relies on n (number of raters) being constant.
        # Let's use the mode or mean?
        pass
    
    n = np.mean(n_rater) # Average number of raters
    
    # P_j: proportion of all assignments which were to category j
    p = table.sum(axis=0) / n_total
    
    # P_i: extent to which raters agree for the i-th subject
    P_i = (np.sum(table**2, axis=1) - n_rater) / (n_rater * (n_rater - 1))
    
    P_bar = np.mean(P_i)
    P_e = np.sum(p**2)
    
    if P_e == 1:
        return 1.0
        
    kappa = (P_bar - P_e) / (1 - P_e)
    return kappa


def compute_pairwise_prompt_agreement(
    df: pd.DataFrame,
    id_col: str = 'filename',
    version_col: str = 'version',
    shift_col: str = 'shift',
    base_shift: str = 'No',
    first_token: bool = False,
    p_yes: bool = False,
    history_length_col: str = 'history_length',
    history_mode: str = None,
    include_shifts: list = None
):
    """
    Compute pairwise agreement (Kappa and Flip Rate) between different prompt versions.
    """
    pivot_df = _prepare_pivot_data(df, id_col, version_col, shift_col, base_shift, first_token, p_yes, history_length_col, history_mode, include_shifts)
    
    if pivot_df is None:
        return None, None, None
    
    versions = sorted(pivot_df.columns.tolist())
    n_versions = len(versions)
    
    if n_versions < 2:
        # print("Need at least 2 versions to compute pairwise agreement.")
        return None, None, None

    kappa_matrix = pd.DataFrame(index=versions, columns=versions, dtype=float)
    flip_matrix = pd.DataFrame(index=versions, columns=versions, dtype=float)
    
    kappa_vals = []
    flip_vals = []

    for i in range(n_versions):
        for j in range(n_versions):
            v1 = versions[i]
            v2 = versions[j]
            
            if i == j:
                kappa_matrix.loc[v1, v2] = 1.0
                flip_matrix.loc[v1, v2] = 0.0
                continue
            
            # Get common samples
            pair_data = pivot_df[[v1, v2]].dropna()
            
            if len(pair_data) == 0:
                kappa_matrix.loc[v1, v2] = np.nan
                flip_matrix.loc[v1, v2] = np.nan
                continue
                
            y1 = pair_data[v1]
            y2 = pair_data[v2]
            
            # Kappa
            try:
                k = cohen_kappa_score(y1, y2)
            except:
                k = np.nan
            kappa_matrix.loc[v1, v2] = k
            
            # Flip Rate
            flip = (y1 != y2).mean()
            flip_matrix.loc[v1, v2] = flip
            
            if i < j: # Collect upper triangle for averages
                kappa_vals.append(k)
                flip_vals.append(flip)

    avg_metrics = {
        "avg_kappa": np.nanmean(kappa_vals) if kappa_vals else np.nan,
        "avg_flip_rate": np.nanmean(flip_vals) if flip_vals else np.nan
    }
    
    return kappa_matrix, flip_matrix, avg_metrics


def plot_pairwise_matrix(matrix, title, save_path=None, cmap="viridis", vmin=None, vmax=None):
    if matrix is None: return
    
    plt.figure(figsize=(10, 8))
    plt.imshow(matrix.astype(float), cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar()
    
    ticks = np.arange(len(matrix.columns))
    plt.xticks(ticks, matrix.columns, rotation=45, ha="right")
    plt.yticks(ticks, matrix.index)
    
    # Annotate
    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):
            val = matrix.iloc[i, j]
            if not np.isnan(val):
                plt.text(j, i, f"{val:.2f}", ha="center", va="center", color="red")

    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def compute_flip_rate_per_prompt(df: pd.DataFrame, 
                                  id: str = 'filename',
                                  version_col: str = 'version',
                                  shift_col: str = 'shift',
                                  base_shift: str = 'No',
                                  first_token: bool = False,
                                  p_yes: bool = False,
                                  history_length_col: str = 'history_length',
                                  history_mode: str = None,
                                  include_shifts: list = None) -> pd.DataFrame:
    """
    Compute average flip rate for each prompt version against all other versions.
    (Consistency metric).
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with predictions
    id : str
        Column name for sample identifier
    version_col : str
        Column name for prompt version
    shift_col : str
        Column name for shift type
    base_shift : str
        Value representing baseline (no shift)
    first_token : bool
        Whether to use first token probabilities
    p_yes : bool
        Which probability to use if first_token is True
    history_length_col : str
        Column name for history length
    history_mode : str, optional
        How to handle history:
        - None: Use base_shift filtering only (default)
        - 'no_vs_length_groups': Compare no history (0) vs each history length as a group
          (length=1, length=2, length=3, etc. each treated as one group)
        - 'no_vs_length1_only': Compare no history (0) vs history length=1 only
          (ignores all length>1 data)
    include_shifts : list, optional
        List of shift types to include. If None, uses base_shift only.
    
    Returns:
    --------
    pd.DataFrame with flip rates per version
    """
    kappa_mat, flip_mat, _ = compute_pairwise_prompt_agreement(
        df, id, version_col, shift_col, base_shift, first_token, p_yes,
        history_length_col, history_mode, include_shifts
    )
    
    if flip_mat is None:
        return pd.DataFrame()

    results = []
    versions = flip_mat.index
    for v in versions:
        # Average flip rate with other versions (exclude self which is 0)
        others = [ov for ov in versions if ov != v]
        if not others:
            avg_flip = 0.0
        else:
            avg_flip = flip_mat.loc[v, others].mean()
            
        results.append({
            'Version': v,
            'Flip_Rate': avg_flip
        })
        
    return pd.DataFrame(results)


def compute_kappa_per_prompt(df: pd.DataFrame,
                             version_col: str = 'version',
                             first_token: bool = False,
                             p_yes: bool = False,
                             id_col: str = 'filename',
                             shift_col: str = 'shift',
                             base_shift: str = 'No',
                             history_length_col: str = 'history_length',
                             history_mode: str = None,
                             include_shifts: list = None) -> float:
    """
    Compute overall kappa for inter-rater agreement across all prompt versions.
    Uses Fleiss' Kappa for multiple raters.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with predictions
    version_col : str
        Column name for prompt version
    first_token : bool
        Whether to use first token probabilities
    p_yes : bool
        Which probability to use if first_token is True
    id_col : str
        Column name for sample identifier
    shift_col : str
        Column name for shift type
    base_shift : str
        Value representing baseline (no shift)
    history_length_col : str
        Column name for history length
    history_mode : str, optional
        How to handle history:
        - None: Use base_shift filtering only (default)
        - 'no_vs_length_groups': Compare no history (0) vs each history length as a group
        - 'no_vs_length1_only': Compare no history (0) vs history length=1 only
    include_shifts : list, optional
        List of shift types to include. If None, uses base_shift only.
    
    Returns:
    --------
    float : Fleiss' Kappa score
    """
    pivot_df = _prepare_pivot_data(df, id_col, version_col, shift_col, base_shift, first_token, p_yes, history_length_col, history_mode, include_shifts)
    
    if pivot_df is None or pivot_df.empty:
        return np.nan
        
    # Convert pivot_df to counts table for Fleiss Kappa
    # pivot_df has rows=subjects, cols=versions, values=categories
    
    # Drop rows with any missing values to ensure constant number of raters
    pivot_df = pivot_df.dropna()
    
    if pivot_df.empty:
        return np.nan
        
    # Get all unique categories across the dataframe
    categories = set(np.unique(pivot_df.values))
    cat_to_idx = {cat: i for i, cat in enumerate(sorted(list(categories)))}
    
    n_sub = len(pivot_df)
    n_cat = len(categories)
    
    table = np.zeros((n_sub, n_cat), dtype=int)
    
    for i, (idx, row) in enumerate(pivot_df.iterrows()):
        for val in row:
            if val in cat_to_idx:
                table[i, cat_to_idx[val]] += 1
                
    return fleiss_kappa(table)


def compute_flip_rate_all_models(
    model_dict: dict,
    results_dir: str = "results",
    dataset: str = "mimic_5class",
    id: str = 'filename',
    version_col: str = 'version',
    shift_col: str = 'shift',
    base_shift: str = 'No',
    first_token: bool = False,
    p_yes: bool = False,
    file_name: str = None,
    history_length_col: str = 'history_length',
    history_mode: str = None,
    include_shifts: list = None
) -> dict:
    """
    Compute flip rates across all models and prompt variations.
    
    Parameters:
    -----------
    model_dict : dict
        Dictionary mapping model HuggingFace IDs to model keys
    results_dir : str
        Base directory for results
    dataset : str
        Dataset name
    id : str
        Column name for sample identifier
    version_col : str
        Column name for prompt version
    shift_col : str
        Column name for shift type
    base_shift : str
        Value representing baseline (no shift)
    first_token : bool
        Whether to use first token probabilities
    p_yes : bool
        Which probability to use if first_token is True
    file_name : str
        CSV file name (if None, uses dataset_base_shifted_multi_versions.csv)
    history_length_col : str
        Column name for history length
    history_mode : str, optional
        How to handle history:
        - None: Use base_shift filtering only (default)
        - 'no_vs_length_groups': Compare no history (0) vs each history length as a group
          (length=1, length=2, length=3, etc. each treated as one group)
        - 'no_vs_length1_only': Compare no history (0) vs history length=1 only
          (ignores all length>1 data)
    include_shifts : list, optional
        List of shift types to include. If None, uses base_shift only.
    
    Returns:
    --------
    dict: {model_name: DataFrame with flip rates per version}
    """
    if file_name is None:
        file_name = f"{dataset}_base_shifted_multi_versions.csv"
    
    results = {}
    
    for model_id, model_key in tqdm(model_dict.items(), desc="Computing flip rates"):
        csv_path = os.path.join(results_dir, model_key, file_name)
        
        if not os.path.exists(csv_path):
            print(f"⚠️  File not found: {csv_path}")
            continue
        
        try:
            df = pd.read_csv(csv_path)
            flip_rates = compute_flip_rate_per_prompt(
                df,
                id=id,
                version_col=version_col,
                shift_col=shift_col,
                base_shift=base_shift,
                first_token=first_token,
                p_yes=p_yes,
                history_length_col=history_length_col,
                history_mode=history_mode,
                include_shifts=include_shifts
            )
            
            model_name = PRETTY_NAMES.get(model_key, model_key)
            results[model_name] = flip_rates
            
        except Exception as e:
            print(f"❌ Error processing {model_key}: {e}")
            continue
    
    return results


def compute_kappa_all_models(
    model_dict: dict,
    results_dir: str = "results",
    dataset: str = "mimic_5class",
    version_col: str = 'version',
    first_token: bool = False,
    p_yes: bool = False,
    file_name: str = None,
    id_col: str = 'filename',
    shift_col: str = 'shift',
    base_shift: str = 'No',
    history_length_col: str = 'history_length',
    history_mode: str = None,
    include_shifts: list = None
) -> dict:
    """
    Compute overall kappa for inter-rater agreement across all models.
    Returns a single kappa score per model (average of all pairwise kappas).
    
    Parameters:
    -----------
    model_dict : dict
        Dictionary mapping model HuggingFace IDs to model keys
    results_dir : str
        Base directory for results
    dataset : str
        Dataset name
    version_col : str
        Column name for prompt version
    first_token : bool
        Whether to use first token probabilities
    p_yes : bool
        Which probability to use if first_token is True
    file_name : str
        CSV file name (if None, uses dataset_base_shifted_multi_versions.csv)
    id_col : str
        Column name for sample identifier
    shift_col : str
        Column name for shift type
    base_shift : str
        Value representing baseline (no shift)
    history_length_col : str
        Column name for history length
    history_mode : str, optional
        How to handle history:
        - None: Use base_shift filtering only (default)
        - 'no_vs_length_groups': Compare no history (0) vs each history length as a group
        - 'no_vs_length1_only': Compare no history (0) vs history length=1 only
    include_shifts : list, optional
        List of shift types to include. If None, uses base_shift only.
    
    Returns:
    --------
    dict: {model_name: kappa_score (float)}
        Single kappa value per model representing overall inter-rater agreement
    """
    if file_name is None:
        file_name = f"{dataset}_base_shifted_multi_versions.csv"
    
    results = {}
    
    for model_id, model_key in tqdm(model_dict.items(), desc="Computing kappa scores"):
        csv_path = os.path.join(results_dir, model_key, file_name)
        
        if not os.path.exists(csv_path):
            print(f"⚠️  File not found: {csv_path}")
            continue
        
        try:
            df = pd.read_csv(csv_path)
            kappa_scores = compute_kappa_per_prompt(
                df,
                version_col=version_col,
                first_token=first_token,
                p_yes=p_yes,
                id_col=id_col,
                shift_col=shift_col,
                base_shift=base_shift,
                history_length_col=history_length_col,
                history_mode=history_mode,
                include_shifts=include_shifts
            )
            
            model_name = PRETTY_NAMES.get(model_key, model_key)
            results[model_name] = kappa_scores
            
        except Exception as e:
            print(f"❌ Error processing {model_key}: {e}")
            continue
    
    return results


def plot_flip_rates(
    flip_rate_dict: dict,
    save_path: str = "images/flip_rates_prompt_consistency.png",
    figsize: tuple = (18, 10),
    title: str = "Flip Rates Across Prompt Variations"
) -> None:
    """
    Visualize flip rates across models and prompt variations.
    
    Parameters:
    -----------
    flip_rate_dict : dict
        Output from compute_flip_rate_all_models
    save_path : str
        Path to save the figure
    figsize : tuple
        Figure size (width, height)
    title : str
        Plot title
    """
    n_models = len(flip_rate_dict)
    if n_models == 0:
        print("No data to plot")
        return
    
    # Determine grid layout
    ncols = 3
    nrows = (n_models + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=True)
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, (model_name, flip_df) in enumerate(flip_rate_dict.items()):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        
        # Get unique versions
        versions = sorted(flip_df['Version'].unique())
        
        # Check if 'Shift' column exists (old format) or not (new format)
        if 'Shift' in flip_df.columns:
            shifts = sorted(flip_df['Shift'].unique())
            x = np.arange(len(versions))
            width = 0.8 / len(shifts)
            
            for i, shift in enumerate(shifts):
                shift_data = flip_df[flip_df['Shift'] == shift]
                values = [
                    shift_data[shift_data['Version'] == v]['Flip_Rate'].values[0]
                    if v in shift_data['Version'].values else 0
                    for v in versions
                ]
                ax.bar(x + i * width, values, width, label=shift, alpha=0.8)
            
            ax.set_xticks(x + width * (len(shifts) - 1) / 2)
            ax.legend(fontsize=8, loc='best')
            
        else:
            # New format: Version, Flip_Rate
            values = [
                flip_df[flip_df['Version'] == v]['Flip_Rate'].values[0]
                for v in versions
            ]
            ax.bar(versions, values, alpha=0.8, color='salmon')
            
            # Add value labels
            for i, v in enumerate(values):
                ax.text(i, v, f"{v:.2f}", ha='center', va='bottom', fontsize=8)

        ax.set_xlabel('Prompt Version', fontsize=10)
        ax.set_ylabel('Flip Rate', fontsize=10)
        ax.set_title(f'{model_name}', fontsize=12, fontweight='bold', pad=15)
        ax.set_xticklabels(versions, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1)
    
    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    # Add extra spacing
    plt.subplots_adjust(top=0.90, hspace=0.8, wspace=0.3)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ Saved flip rate plot to: {save_path}")


def plot_kappa_scores(
    kappa_dict: dict,
    save_path: str = "images/kappa_scores_prompt_consistency.png",
    figsize: tuple = (12, 6),
    title: str = "Inter-Rater Agreement (Kappa) Across Models"
) -> None:
    """
    Plot a single bar chart showing kappa for each model.
    
    Parameters:
    -----------
    kappa_dict : dict
        Dictionary from compute_kappa_all_models(): {model_name: kappa_score (float)}
    save_path : str
        Path to save the figure
    figsize : tuple
        Figure size (width, height)
    title : str
        Plot title
    
    Returns:
    --------
    None (saves plot and displays it)
    """
    if not kappa_dict:
        print("No data to plot")
        return
    
    # Sort models by kappa score for better visualization
    sorted_items = sorted(kappa_dict.items(), key=lambda x: x[1] if not np.isnan(x[1]) else -1, reverse=True)
    model_names = [item[0] for item in sorted_items]
    kappa_values = [item[1] for item in sorted_items]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create bars
    bars = ax.bar(range(len(model_names)), kappa_values, alpha=0.8)
    
    # Color bars based on kappa value (Fleiss' interpretation)
    colors = []
    for val in kappa_values:
        if np.isnan(val):
            colors.append('gray')
        elif val < 0.4:
            colors.append('#d62728')  # red - poor
        elif val < 0.6:
            colors.append('#ff7f0e')  # orange - fair
        elif val < 0.75:
            colors.append('#2ca02c')  # green - good
        else:
            colors.append('#1f77b4')  # blue - excellent
    
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # Add reference lines
    ax.axhline(y=0.4, color='red', linestyle='--', alpha=0.4, linewidth=1, label='Poor/Fair threshold')
    ax.axhline(y=0.6, color='orange', linestyle='--', alpha=0.4, linewidth=1, label='Fair/Good threshold')
    ax.axhline(y=0.75, color='green', linestyle='--', alpha=0.4, linewidth=1, label='Good/Excellent threshold')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, kappa_values)):
        if not np.isnan(val):
            height = bar.get_height()
            ax.text(i, height + 0.02, f'{val:.3f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel("Cohen's Kappa (Inter-Rater Agreement)", fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=60, ha='right', fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, min(1.0, max(kappa_values) * 1.15) if kappa_values else 1.0)
    
    # Add legend for interpretation
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#d62728', label='Poor (< 0.4)'),
        Patch(facecolor='#ff7f0e', label='Fair (0.4-0.6)'),
        Patch(facecolor='#2ca02c', label='Good (0.6-0.75)'),
        Patch(facecolor='#1f77b4', label='Excellent (≥ 0.75)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    # Add extra bottom margin for labels
    plt.subplots_adjust(bottom=0.25)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ Saved kappa plot to: {save_path}")


def plot_pairwise_flip_rate_heatmaps(
    model_dict: dict,
    results_dir: str = "results",
    dataset: str = "mimic_5class",
    id: str = 'filename',
    version_col: str = 'version',
    shift_col: str = 'shift',
    base_shift: str = 'No',
    first_token: bool = False,
    p_yes: bool = False,
    file_name: str = None,
    history_length_col: str = 'history_length',
    history_mode: str = None,
    include_shifts: list = None,
    save_path: str = "images/flip_rate_heatmaps.png",
    figsize: tuple = (20, 12),
    show_average_bar: bool = True,
    title: str = "Pairwise Flip Rate Heatmaps",
    hspace: float = 0.5
) -> None:
    """
    Create heatmaps showing pairwise flip rates between all prompt versions for each model.
    
    Parameters:
    -----------
    model_dict : dict
        Dictionary mapping model IDs to model keys
    results_dir : str
        Base directory for results
    dataset : str
        Dataset name
    id : str
        Column name for sample identifier
    version_col : str
        Column name for prompt version
    shift_col : str
        Column name for shift type
    base_shift : str
        Value representing baseline (no shift)
    first_token : bool
        Whether to use first token probabilities
    p_yes : bool
        Which probability to use if first_token is True
    file_name : str
        CSV file name
    history_length_col : str
        Column name for history length
    history_mode : str, optional
        How to handle history:
        - None: Use base_shift filtering only (default)
        - 'no_vs_length_groups': Compare no history (0) vs each history length as a group
          (length=1, length=2, length=3, etc. each treated as one group)
        - 'no_vs_length1_only': Compare no history (0) vs history length=1 only
          (ignores all length>1 data)
    include_shifts : list, optional
        List of shift types to include. If None, uses base_shift only.
    save_path : str
        Path to save the figure
    figsize : tuple
        Figure size (width, height)
    show_average_bar : bool
        Whether to show average flip rate as a bar chart below heatmap
    title : str
        Plot title
    """
    if file_name is None:
        file_name = f"{dataset}_base_shifted_multi_versions.csv"
    
    n_models = len(model_dict)
    if n_models == 0:
        print("No models to plot")
        return
    
    # Determine grid layout
    ncols = 4
    if show_average_bar:
        nrows = 2 * ((n_models + ncols - 1) // ncols)  # Double rows for heatmap + bar
    else:
        nrows = (n_models + ncols - 1) // ncols
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows, ncols, hspace=hspace, wspace=0.3)
    
    model_idx = 0
    for model_id, model_key in tqdm(model_dict.items(), desc="Creating flip rate heatmaps"):
        csv_path = os.path.join(results_dir, model_key, file_name)
        
        if not os.path.exists(csv_path):
            print(f"⚠️  File not found: {csv_path}")

            continue
        
        try:
            df = pd.read_csv(csv_path)
            kappa_mat, flip_mat, _ = compute_pairwise_prompt_agreement(
                df,
                id_col=id,
                version_col=version_col,
                shift_col=shift_col,
                base_shift=base_shift,
                first_token=first_token,
                p_yes=p_yes,
                history_length_col=history_length_col,
                history_mode=history_mode,
                include_shifts=include_shifts
            )
            
            if flip_mat is None:
                print(f"⚠️  No data for {model_key}")
                continue
            
            model_name = PRETTY_NAMES.get(model_key, model_key)
            
            # Calculate position in grid
            row = (model_idx // ncols) * (2 if show_average_bar else 1)
            col = model_idx % ncols
            
            # Create heatmap
            ax_heat = fig.add_subplot(gs[row, col])
            
            # Plot heatmap
            im = ax_heat.imshow(flip_mat.values.astype(float), cmap='RdYlGn_r', 
                               vmin=0, vmax=1, aspect='auto')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
            cbar.set_label('Flip Rate', rotation=270, labelpad=15)
            
            # Set ticks and labels
            versions = flip_mat.columns.tolist()
            ax_heat.set_xticks(np.arange(len(versions)))
            ax_heat.set_yticks(np.arange(len(versions)))
            ax_heat.set_xticklabels(versions, rotation=45, ha='right')
            ax_heat.set_yticklabels(versions)
            ax_heat.set_ylabel('Prompt Version', rotation=90, labelpad=10)
            
            # Add text annotations
            for i in range(len(versions)):
                for j in range(len(versions)):
                    val = flip_mat.iloc[i, j]
                    if not np.isnan(val):
                        text_color = 'white' if val > 0.5 else 'black'
                        ax_heat.text(j, i, f'{val:.2f}', ha='center', va='center',
                                   color=text_color, fontsize=8)
            
            ax_heat.set_title(f'{model_name}', fontsize=12, fontweight='bold', pad=20)
            ax_heat.set_xlabel('Prompt Version')
            ax_heat.set_ylabel('Prompt Version', )
            
            # Add average bar chart if requested
            if show_average_bar:
                ax_bar = fig.add_subplot(gs[row + 1, col])
                
                # Calculate average flip rate per version (excluding self)
                avg_flip = []
                for i, v in enumerate(versions):
                    others_idx = [j for j in range(len(versions)) if j != i]
                    if others_idx:
                        avg_flip.append(flip_mat.iloc[i, others_idx].mean())
                    else:
                        avg_flip.append(0.0)
                
                bars = ax_bar.bar(versions, avg_flip, alpha=0.8, color='salmon')
                ax_bar.set_ylabel('Avg Flip Rate', fontsize=9)
                ax_bar.set_xlabel('Prompt Version', fontsize=9)
                ax_bar.set_xticklabels(versions, rotation=45, ha='right')
                ax_bar.set_ylim(0, 1)
                ax_bar.grid(axis='y', alpha=0.3)
                
                # Add value labels
                for bar, val in zip(bars, avg_flip):
                    if not np.isnan(val):
                        ax_bar.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                                  f'{val:.2f}', ha='center', va='bottom', fontsize=8)
            
            model_idx += 1
            
        except Exception as e:
            print(f"❌ Error processing {model_key}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    # Add extra spacing
    plt.subplots_adjust(top=0.90, hspace=hspace, wspace=0.3)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ Saved flip rate heatmaps to: {save_path}")


def plot_pairwise_kappa_heatmaps(
    model_dict: dict,
    results_dir: str = "results",
    dataset: str = "mimic_5class",
    version_col: str = 'version',
    first_token: bool = False,
    p_yes: bool = False,
    file_name: str = None,
    id_col: str = 'filename',
    shift_col: str = 'shift',
    base_shift: str = 'No',
    history_length_col: str = 'history_length',
    history_mode: str = None,
    include_shifts: list = None,
    save_path: str = "images/kappa_heatmaps.png",
    figsize: tuple = (20, 12),
    show_average_bar: bool = True,
    title: str = "Cohen's Kappa Between Prompt Pairs",
    hspace: float = 0.5
) -> None:
    """
    DEPRECATED: Use compute_kappa_all_models() with plot_kappa_scores() instead.
    
    Create heatmaps showing pairwise Cohen's Kappa between all prompt versions for each model.
    
    Parameters:
    -----------
    model_dict : dict
        Dictionary mapping model IDs to model keys
    results_dir : str
        Base directory for results
    dataset : str
        Dataset name
    version_col : str
        Column name for prompt version
    first_token : bool
        Whether to use first token probabilities
    p_yes : bool
        Which probability to use if first_token is True
    file_name : str
        CSV file name
    id_col : str
        Column name for sample identifier
    shift_col : str
        Column name for shift type
    base_shift : str
        Value representing baseline (no shift)
    history_length_col : str
        Column name for history length
    history_mode : str, optional
        How to handle history:
        - None: Use base_shift filtering only (default)
        - 'no_vs_length_groups': Compare no history (0) vs each history length as a group
        - 'no_vs_length1_only': Compare no history (0) vs history length=1 only
    include_shifts : list, optional
        List of shift types to include. If None, uses base_shift only.
    save_path : str
        Path to save the figure
    figsize : tuple
        Figure size (width, height)
    show_average_bar : bool
        Whether to show average kappa as a bar chart below heatmap
    title : str
        Plot title
    """
    print("WARNING: plot_pairwise_kappa_heatmaps() is deprecated.")
    print("         Use compute_kappa_all_models() with plot_kappa_scores() instead.")
    
    if file_name is None:
        file_name = f"{dataset}_base_shifted_multi_versions.csv"
    
    n_models = len(model_dict)
    if n_models == 0:
        print("No models to plot")
        return
    
    # Determine grid layout
    ncols = 4
    if show_average_bar:
        nrows = 2 * ((n_models + ncols - 1) // ncols)
    else:
        nrows = (n_models + ncols - 1) // ncols
    
    fig = plt.figure(figsize=figsize)
    # Increased hspace from 0.4 to 0.8
    gs = fig.add_gridspec(nrows, ncols, hspace=hspace, wspace=0.3)
    
    model_idx = 0
    for model_id, model_key in tqdm(model_dict.items(), desc="Creating kappa heatmaps"):
        csv_path = os.path.join(results_dir, model_key, file_name)
        
        if not os.path.exists(csv_path):
            print(f"⚠️  File not found: {csv_path}")
            continue
        
        try:
            df = pd.read_csv(csv_path)
            kappa_mat, flip_mat, _ = compute_pairwise_prompt_agreement(
                df,
                id_col=id_col,
                version_col=version_col,
                shift_col=shift_col,
                base_shift=base_shift,
                first_token=first_token,
                p_yes=p_yes,
                history_length_col=history_length_col,
                history_mode=history_mode,
                include_shifts=include_shifts
            )
            
            if kappa_mat is None:
                print(f"⚠️  No data for {model_key}")
                continue
            
            model_name = PRETTY_NAMES.get(model_key, model_key)
            
            # Calculate position in grid
            row = (model_idx // ncols) * (2 if show_average_bar else 1)
            col = model_idx % ncols
            
            # Create heatmap
            ax_heat = fig.add_subplot(gs[row, col])
            
            # Plot heatmap
            im = ax_heat.imshow(kappa_mat.values.astype(float), cmap='RdYlGn', 
                               vmin=0, vmax=1, aspect='auto')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
            cbar.set_label("Cohen's Kappa", rotation=270, labelpad=15)
            
            # Set ticks and labels
            versions = kappa_mat.columns.tolist()
            ax_heat.set_xticks(np.arange(len(versions)))
            ax_heat.set_yticks(np.arange(len(versions)))
            ax_heat.set_xticklabels(versions, rotation=45, ha='right')
            ax_heat.set_yticklabels(versions)
            ax_heat.set_ylabel('Prompt Version', rotation=90, labelpad=10)
            
            # Add text annotations
            for i in range(len(versions)):
                for j in range(len(versions)):
                    val = kappa_mat.iloc[i, j]
                    if not np.isnan(val):
                        text_color = 'white' if val < 0.5 else 'black'
                        ax_heat.text(j, i, f'{val:.2f}', ha='center', va='center',
                                   color=text_color, fontsize=8)

            ax_heat.set_title(f'{model_name}', fontsize=12, fontweight='bold', pad=20)
            ax_heat.set_xlabel('Prompt Version')
            ax_heat.set_ylabel('Prompt Version')
            
            # Add average bar chart if requested
            if show_average_bar:
                ax_bar = fig.add_subplot(gs[row + 1, col])
                
                # Calculate average kappa per version (excluding self)
                avg_kappa = []
                for i, v in enumerate(versions):
                    others_idx = [j for j in range(len(versions)) if j != i]
                    if others_idx:
                        avg_kappa.append(kappa_mat.iloc[i, others_idx].mean())
                    else:
                        avg_kappa.append(1.0)
                
                bars = ax_bar.bar(versions, avg_kappa, alpha=0.8, color='salmon')
                ax_bar.set_ylabel('Avg Kappa', fontsize=9)
                ax_bar.set_xlabel('Prompt Version', fontsize=9)
                ax_bar.set_xticklabels(versions, rotation=45, ha='right')
                ax_bar.set_ylim(0, 1)
                ax_bar.grid(axis='y', alpha=0.3)
                
                # Add value labels
                for bar, val in zip(bars, avg_kappa):
                    if not np.isnan(val):
                        ax_bar.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                                  f'{val:.2f}', ha='center', va='bottom', fontsize=8)
            
            model_idx += 1
            
        except Exception as e:
            print(f"❌ Error processing {model_key}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, hspace=hspace, wspace=0.3)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ Saved kappa heatmaps to: {save_path}")


##############################################################################
# NFR by History Length
##############################################################################

def compute_nfr_by_history_length(df: pd.DataFrame,
                                   id: str = 'filename',
                                   history_length_col: str = 'history_length',
                                   version_col: str = 'version',
                                   first_token: bool = False,
                                   p_yes: bool = False) -> pd.DataFrame:
    """
    Compute NFR comparing no history (length=0) vs any history (length>0).
    
    Similar to modality shift NFR, but compares predictions with and without history.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with predictions
    id : str
        Column name for sample identifier
    history_length_col : str
        Column name for history length
    version_col : str
        Column name for prompt version (optional)
    first_token : bool
        Whether to use first token probabilities
    p_yes : bool
        Which probability to use if first_token is True
    
    Returns:
    --------
    pd.DataFrame with NFR values per history length (and version if applicable)
    """
    df = df.copy()
    df["ground_truth"] = df["ground_truth"].apply(clean_label)
    
    if first_token:
        df = _apply_softmax_to_probs(df, p_yes=p_yes)
    else:
        df["prediction"] = df["prediction"].apply(clean_label)
        df = df[~df["prediction"].str.startswith("unknown")]

    # Check if version column exists and has multiple values
    has_versions = version_col in df.columns and len(df[version_col].unique()) > 1
    
    results = []
    
    if has_versions:
        versions = df[version_col].unique()
        for version in versions:
            df_v = df[df[version_col] == version]
            nfr_dict = _compute_nfr_history_single(df_v, id, history_length_col)
            nfr_dict['Version'] = version
            results.append(nfr_dict)
    else:
        nfr_dict = _compute_nfr_history_single(df, id, history_length_col)
        results.append(nfr_dict)
    
    return pd.DataFrame(results)


def _compute_nfr_history_single(df: pd.DataFrame,
                                id: str,
                                history_length_col: str) -> dict:
    """
    Helper function to compute NFR for a single dataset/version.
    """
    # Get base (no history) predictions
    base_df = df[df[history_length_col] == 0]
    
    # Get unique history lengths (excluding 0)
    history_lengths = sorted([l for l in df[history_length_col].unique() if l > 0])
    
    result = {}
    
    for length in history_lengths:
        hist_df = df[df[history_length_col] == length]
        
        merged = base_df[[id, "ground_truth", "prediction"]].rename(
            columns={"prediction": "base_pred"}
        ).merge(
            hist_df[[id, "prediction"]],
            on=id, how="inner"
        ).rename(columns={"prediction": "hist_pred"})

        # NFR: proportion where base was correct but history prediction is wrong
        base_correct = merged["base_pred"] == merged["ground_truth"]
        hist_wrong = merged["hist_pred"] != merged["ground_truth"]
        n_flips = (base_correct & hist_wrong).sum()
        nfr = n_flips / len(merged) if len(merged) > 0 else np.nan
        
        result[f"NFR_History_Length_{length}"] = nfr
    
    return result


def compute_nfr_all_models_multiprompt(
    model_dict: dict,
    results_dir: str = "results",
    dataset: str = "mimic",
    id: str = 'filename',
    version_col: str = 'version',
    shift_col: str = 'shift',
    base_shift: str = 'No',
    first_token: bool = False,
    p_yes: bool = False,
    return_per_prompt: bool = False
) -> pd.DataFrame:
    """
    Compute NFR across all models with support for multiple prompt variations.
    
    Parameters:
    -----------
    model_dict : dict
        Dictionary mapping model IDs to folder names
    results_dir : str
        Directory containing results
    dataset : str
        Dataset name
    id : str
        Column name for sample identifier
    version_col : str
        Column name for prompt version
    shift_col : str
        Column name for shift type
    base_shift : str
        Value representing baseline (no shift)
    first_token : bool
        Whether to use first token probabilities
    p_yes : bool
        Which probability to use if first_token is True
    return_per_prompt : bool
        If True, returns a nested dict with NFR per prompt variation
        If False, returns averaged NFR across prompts
    
    Returns:
    --------
    pd.DataFrame or dict with NFR values
    """
    shifts = ["Image", "Text", "Only_text", "Only_image"]
    
    if return_per_prompt:
        all_results = {}  # model -> version -> shift -> nfr
    else:
        all_rows = []
    
    for hf_name, folder_name in model_dict.items():
        file_path = os.path.join(results_dir, folder_name, f"{dataset}_base_shifted_multi_versions.csv")
        
        if not os.path.isfile(file_path):
            print(f"[WARN] Missing results file for {folder_name}: {file_path}")
            continue
        
        df = pd.read_csv(file_path)
        
        if return_per_prompt:
            # Compute NFR per version
            model_results = {}
            versions = df[version_col].unique()
            
            for version in versions:
                df_v = df[df[version_col] == version]
                try:
                    nfr_df = compute_nfr_table(df_v, id=id, first_token=first_token, p_yes=p_yes)
                    model_results[version] = dict(zip(nfr_df['Shift'], nfr_df['NFR']))
                except Exception as e:
                    print(f"[ERROR] Could not compute NFR for {folder_name} version {version}: {e}")
            
            all_results[folder_name] = model_results
        else:
            # Compute average NFR across versions
            try:
                nfr_df = compute_nfr_table(df, id=id, first_token=first_token, p_yes=p_yes)
                row = nfr_df.set_index("Shift")["NFR"].reindex(
                    [s.replace("_", " ").title() for s in shifts]
                )
                row.name = folder_name
                all_rows.append(row)
            except Exception as e:
                print(f"[ERROR] Could not compute NFR for {folder_name}: {e}")
    
    if return_per_prompt:
        return all_results
    else:
        if not all_rows:
            raise RuntimeError("No NFR values could be computed – check file paths.")
        nfr_matrix = pd.concat(all_rows, axis=1).T
        return nfr_matrix


def compute_nfr_history_all_models(
    model_dict: dict,
    results_dir: str = "results",
    dataset: str = "cxr_multi_history",
    id: str = 'filename',
    history_length_col: str = 'history_length',
    version_col: str = 'version',
    first_token: bool = False,
    p_yes: bool = False,
    return_per_prompt: bool = False
) -> pd.DataFrame:
    """
    Compute NFR by history length across all models.
    
    Parameters:
    -----------
    model_dict : dict
        Dictionary mapping model IDs to folder names
    results_dir : str
        Directory containing results
    dataset : str
        Dataset name
    id : str
        Column name for sample identifier
    history_length_col : str
        Column name for history length
    version_col : str
        Column name for prompt version
    first_token : bool
        Whether to use first token probabilities
    p_yes : bool
        Which probability to use if first_token is True
    return_per_prompt : bool
        If True, returns a nested dict with NFR per prompt variation
        If False, returns averaged NFR across prompts
    
    Returns:
    --------
    pd.DataFrame or dict with NFR values by history length
    """
    if return_per_prompt:
        all_results = {}  # model -> version -> history_length -> nfr
    else:
        all_rows = []
    
    for hf_name, folder_name in model_dict.items():
        file_path = os.path.join(results_dir, folder_name, f"{dataset}_base_multi_versions.csv")
        
        if not os.path.isfile(file_path):
            print(f"[WARN] Missing results file for {folder_name}: {file_path}")
            continue
        
        df = pd.read_csv(file_path)
        
        try:
            nfr_df = compute_nfr_by_history_length(
                df, id=id, history_length_col=history_length_col,
                version_col=version_col, first_token=first_token, p_yes=p_yes
            )
            
            if return_per_prompt:
                # Organize by version
                model_results = {}
                for _, row in nfr_df.iterrows():
                    version = row.get('Version', 'default')
                    nfr_values = {k: v for k, v in row.items() if k.startswith('NFR_')}
                    model_results[version] = nfr_values
                all_results[folder_name] = model_results
            else:
                # Average across versions
                nfr_cols = [c for c in nfr_df.columns if c.startswith('NFR_')]
                avg_nfr = nfr_df[nfr_cols].mean()
                avg_nfr.name = folder_name
                all_rows.append(avg_nfr)
        except Exception as e:
            print(f"[ERROR] Could not compute history NFR for {folder_name}: {e}")
    
    if return_per_prompt:
        return all_results
    else:
        if not all_rows:
            raise RuntimeError("No NFR values could be computed – check file paths.")
        nfr_matrix = pd.concat(all_rows, axis=1).T
        return nfr_matrix


def plot_nfr_heatmap(
    nfr_matrix: pd.DataFrame,
    dataset_name: str = "MIMIC-CXR",
    model_name_map: dict | None = None,
    save_path: str | None = None,
):
    """
    Parameters
    ----------
    nfr_matrix : DataFrame
        Rows = internal model IDs, columns = shifts ('Image', 'Text', …).
    dataset_name : str
        Used in the figure title.
    model_name_map : dict
        Optional mapping {internal_id: "Pretty Name"}.
    save_path : str | None
        PNG path; if None, just shows the figure.
    """
    # ── 1) Clean names ────────────────────────────────────────────────────────
    mat = nfr_matrix.copy()
    if model_name_map:
        mat.index = [model_name_map.get(i, i) for i in mat.index]

    mat = mat.rename(
        columns={
            "Image": "Image Shift",
            "Text": "Text Shift",
            "Only Text": "Only Text",
            "Only_text": "Only Text",
            "Only Image": "Only Image",
            "Only_image": "Only Image",
        }
    )

    # Control the column order if desired
    shift_order = ["Image Shift", "Text Shift", "Only Text", "Only Image"]
    mat = mat.loc[:, [c for c in shift_order if c in mat.columns]]

    # ── 2) Figure + softer colormap ───────────────────────────────────────────
    cmap = mpl.cm.get_cmap("YlGnBu_r")  # reversed so low values are darker
    vmax = np.nanmax(mat.values) if np.isfinite(mat.values).any() else 1.0

    fig, ax = plt.subplots(
        figsize=(1.4 * mat.shape[1], 0.6 + 0.55 * mat.shape[0])
    )
    im = ax.imshow(mat.values, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")

    # ── 3) Axis tick labels ──────────────────────────────────────────────────
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels(mat.columns, rotation=30, ha="right", fontsize=11)
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels(mat.index, fontsize=11)

    # ── 4) Annotate each cell ────────────────────────────────────────────────
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat.iat[i, j]
            txt = "—" if np.isnan(val) else f"{val:.2f}"
            rgba = cmap(0 if np.isnan(val) else val / vmax)
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                fontsize=10,
                color=_pick_text_color(rgba),
            )

    # ── 5) Title + colour-bar ────────────────────────────────────────────────
    title = f"Negative-Flip Rate on {dataset_name}"
    ax.set_title(title, fontsize=14, pad=14)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("NFR", rotation=-90, va="bottom")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_metrics_with_ci_split(
    data,
    models,
    shifts_row1,  # Distractor shifts
    shifts_row2,  # History length shifts
    metrics=("Accuracy", "Precision", "Recall", "F1"),
    shift_colors=None,
    sota_lines=None,
    title="Comparison of Evaluation Metrics",
    figsize=(20, 5),
    bar_group_width=0.85,
    capsize=3,
    style="bmh",
    annotate=True,
    compute_f1_if_missing=True,
    ylim=None,
    rotation_xticks=30,
    save_path=None,
    metric_fontsize=9
):
    """
    Generates two independent plots:
    1. Distractor conditions (shifts_row1)
    2. History length conditions (shifts_row2)
    
    Saves them as {save_path}_distractors.png and {save_path}_history.png (stripping extension from save_path)
    """
    
    plt.style.use(style)

    # ---------- helpers ----------
    def _safe_f1(p, r):
        if p is None or r is None or np.isnan(p) or np.isnan(r): return np.nan
        s = p + r
        if s == 0: return 0.0
        return 2 * p * r / s

    def _flatten_one_version(version_dict):
        out = {}
        for hist_len, block in version_dict.items():
            if hist_len == 0:
                if isinstance(block, dict) and block:
                    d = next(iter(block.values()))
                    out['No History'] = d
            elif hist_len == 1:
                # Add individual items
                for name, d in block.items():
                    out[name] = d
                
                # Add aggregated "History len(1)"
                if block:
                    # Compute average metrics across all items in block
                    avg_d = {}
                    # Collect all keys from all dicts
                    all_keys = set()
                    for d in block.values():
                        all_keys.update(d.keys())
                    
                    for k in all_keys:
                        vals = [d.get(k) for d in block.values() if d.get(k) is not None and not (isinstance(d.get(k), float) and np.isnan(d.get(k)))]
                        vals = [v for v in vals if isinstance(v, (int, float))]
                        if vals:
                            avg_d[k] = np.mean(vals)
                        else:
                            avg_d[k] = np.nan
                    out['History len(1)'] = avg_d

            else:
                if isinstance(block, dict) and block:
                    d = next(iter(block.values()))
                    out[f'History len({hist_len})'] = d

        if compute_f1_if_missing and ("F1" in metrics):
            for k, d in out.items():
                if 'F1' not in d or d['F1'] is None or (isinstance(d['F1'], float) and np.isnan(d['F1'])):
                    d['F1'] = _safe_f1(d.get('Precision', np.nan), d.get('Recall', np.nan))
        return out

    def _collect_versions_for(model_name):
        model_blob = data.get(model_name, {})
        version_names = list(model_blob.keys())
        return [_flatten_one_version(model_blob.get(vn, {})) for vn in version_names]

    def _aggregate_mean_ci(values):
        arr = np.array([v for v in values if v is not None and not np.isnan(v)], dtype=float)
        n = arr.size
        if n == 0:
            return np.nan, 0.0
        mean = float(np.nanmean(arr))
        if n == 1:
            return float(np.clip(mean, 0.0, 1.0)), 0.0
        sd = float(np.nanstd(arr, ddof=1))
        ci95 = 1.96 * sd / np.sqrt(n)
        return float(np.clip(mean, 0.0, 1.0)), ci95

    # ---------- aggregate across versions for both sets of shifts ----------
    # We need to process all potential shifts
    all_shifts = list(set(shifts_row1 + shifts_row2))
    agg = {m: {s: {met: {'mean': np.nan, 'ci': 0.0} for met in metrics} for s in all_shifts} for m in models}
    
    for m in models:
        vlist = _collect_versions_for(m)
        for s in all_shifts:
            for met in metrics:
                vals = [vf.get(s, {}).get(met, np.nan) for vf in vlist]
                mean, ci = _aggregate_mean_ci(vals)
                agg[m][s][met]['mean'] = mean
                agg[m][s][met]['ci'] = 0.0 if np.isnan(mean) else (0.0 if np.isnan(ci) else ci)

    # ---------- Plotting Function ----------
    def _plot_subset(shifts, suffix, title_suffix):
        num_models = len(models)
        num_shifts = len(shifts)
        
        # Use the provided figsize directly
        fig, axs = plt.subplots(1, len(metrics), figsize=figsize, sharey=True)
        if len(metrics) == 1:
            axs = [axs]
            
        x = np.arange(num_models)
        
        # Match plot_prompt_robustness logic:
        # width = 0.17 (fixed in other func) -> total group width ~0.85
        # Here we calculate width dynamically to fit ~0.85 total width
        total_group_width = bar_group_width
        width = total_group_width / num_shifts

        for idx, metric in enumerate(metrics):
            ax = axs[idx]

            # Filter models to show (hide models with all-NaN values for this metric e.g. ECE)
            models_to_plot = []
            for m in models:
                has_data = False
                for shift in shifts:
                    val = agg[m][shift][metric]['mean']
                    if not np.isnan(val):
                        has_data = True
                        break
                if has_data:
                    models_to_plot.append(m)
            
            x_plot = np.arange(len(models_to_plot))

            for i, shift in enumerate(shifts):
                vals = [agg[m][shift][metric]['mean'] for m in models_to_plot]
                errs = [agg[m][shift][metric]['ci'] for m in models_to_plot]
                
                # Position bars starting from x, shifting right by width
                # This matches plot_prompt_robustness: x + i*width
                positions = x_plot + i * width

                is_missing = [np.isnan(v) for v in vals]
                vals_plot = [0.0 if miss else v for miss, v in zip(is_missing, vals)]
                errs_plot = [0.0 if miss else e for miss, e in zip(is_missing, errs)]

                bars = ax.bar(
                    positions, vals_plot, width,
                    label=shift,
                    color=(shift_colors.get(shift) if shift_colors else None),
                    yerr=errs_plot, capsize=capsize, linewidth=0
                )

                if annotate:
                    for b, v, e, miss in zip(bars, vals_plot, errs_plot, is_missing):
                        if miss:
                            # Only annotate N/A if model is plotted but specific shift is missing
                            # Since we filtered models_to_plot, "all-NaN" models are gone.
                            # But partial NaNs (some shifts missing) might remain.
                            ax.text(b.get_x() + b.get_width()/2, 0.015, "N/A",
                                    ha="center", va="bottom", fontsize=metric_fontsize, rotation=90)
                        else:
                            # Place text above the bar + error
                            y_pos = v + e + 0.02
                            ax.text(b.get_x() + b.get_width()/2, y_pos, f"{v:.2f}",
                                    ha="center", va="bottom", fontsize=metric_fontsize, rotation=90)

            # reference lines
            if isinstance(sota_lines, dict):
                for label, line_map in sota_lines.items():
                    if metric in line_map and line_map[metric] is not None:
                        ax.axhline(y=line_map[metric], linestyle='--', label=label, linewidth=1.5)

            # Center ticks on the group
            # Group spans from x to x + num_shifts*width
            # Center is x + (num_shifts - 1) * width / 2
            if len(models_to_plot) > 0:
                ax.set_xticks(x_plot + (num_shifts - 1) * width / 2)
                ax.set_xticklabels(models_to_plot, fontsize=10, rotation=rotation_xticks)
            else:
                 ax.set_xticks([])
                 ax.set_xticklabels([])
            
            # Ensure ylim is respected and add padding for annotations
            if ylim:
                ax.set_ylim(ylim[0], ylim[1])
            else:
                ax.set_ylim(0, 1.15)
                
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            ax.set_title(metric, fontsize=16, pad=22)
            
            if idx == 0:
                ax.set_ylabel("Score", fontsize=14)

        # Legend
        handles, labels = axs[-1].get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        fig.legend(uniq.values(), uniq.keys(),
                   title="Conditions",
                   loc='upper center', ncol=min(7, len(uniq)),
                   fontsize=13, title_fontsize=11, 
                   bbox_to_anchor=(0.5, 1.12))

        fig.suptitle(f"{title} - {title_suffix}", fontsize=16)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        if save_path:
            base = os.path.splitext(save_path)[0]
            out_file = f"{base}_{suffix}.png"
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            plt.savefig(out_file, bbox_inches="tight", dpi=300)
            print(f"Saved plot to {out_file}")
            
        plt.show()
        return fig

    # Plot 1: Distractors
    fig1 = _plot_subset(shifts_row1, "distractors", "Distractor Conditions")
    
    # Plot 2: History Lengths
    fig2 = _plot_subset(shifts_row2, "history", "History Length Conditions")

    return fig1, fig2


##############################################################################
# Prompt Robustness Plotting
##############################################################################

def plot_prompt_robustness(
    prompt_results, 
    out_png="images/prompt_robustness_summary.png", 
    metrics=["Accuracy", "Precision", "Recall", "F1"],
    shifts=None,
    shift_colors=None,
    sota_zero_shot=None,
    metric_fontsize=9,
    rotation_xticks=30
):
    
    # =========================
    # CONFIGfroim
    # =========================
    models = list(prompt_results.keys())
    
    if shifts is None:
        shifts = ["No Shift", "Img Shift", "Txt Shift", "Only Text", "Only Image"]
    
    if shift_colors is None:
        shift_colors = {
            "No Shift": "#08306B",
            "Img Shift": "#E6550D",
            "Txt Shift": "#FDAE6B",
            "Only Text": "#31A354",
            "Only Image": "#A1D99B"
        }

    #if sota_zero_shot is None:
    #    sota_zero_shot = {"Accuracy": 0.7579, "Precision": 0.8101, "Recall": 0.7938, "F1": 0.8049}

    # =========================
    # Helpers
    # =========================
    def get_variants(prompt_results, model):
        return list(prompt_results.get(model, {}).keys())

    def safe_get(prompt_results, model, variant, shift, metric):
        try:
            val = prompt_results[model][variant][shift][metric]
            if val is None:
                return np.nan
            val = float(val)
            return val if not np.isnan(val) else np.nan
        except Exception:
            return np.nan

    def aggregate_across_variants(prompt_results, metrics=metrics):
        means = defaultdict(lambda: defaultdict(dict))
        stds  = defaultdict(lambda: defaultdict(dict))

        for m in models:
            variants = get_variants(prompt_results, m)
            for s in shifts:
                for metric in metrics:
                    vals = np.array(
                        [safe_get(prompt_results, m, v, s, metric) for v in variants],
                        dtype=float
                    )
                    vals = vals[~np.isnan(vals)]
                    if vals.size == 0:
                        means[m][s][metric] = np.nan
                        stds[m][s][metric]  = np.nan
                    else:
                        means[m][s][metric] = float(np.nanmean(vals))
                        stds[m][s][metric]  = float(np.nanstd(vals, ddof=1)) if vals.size > 1 else 0.0
        return means, stds

    means, stds = aggregate_across_variants(prompt_results, metrics=metrics)

    # Match your previous layout: one figure with subplots
    fig, axs = plt.subplots(1, len(metrics), figsize=(20, 5), sharey=True)
    if len(metrics) == 1:
        axs = [axs]

    x = np.arange(len(models))
    width = 0.17  # same feel as before

    for idx, metric in enumerate(metrics):
        ax = axs[idx]
        
        # Filter models to show (hide models with all-NaN values for this metric e.g. ECE)
        models_to_plot = []
        for m in models:
            has_data = False
            for shift in shifts:
                 # Check if we have valid data for this shift
                 val = means[m][shift].get(metric, np.nan)
                 if not np.isnan(val):
                     has_data = True
                     break
            if has_data:
                models_to_plot.append(m)
                
        x_plot = np.arange(len(models_to_plot))
        
        for i, shift in enumerate(shifts):
            vals = [means[m][shift].get(metric, np.nan) for m in models_to_plot]
            errs = [stds[m][shift].get(metric, np.nan) for m in models_to_plot]

            bars = ax.bar(
                x_plot + i*width, vals, width,
                label=shift,
                color=shift_colors.get(shift, 'gray'),
                yerr=errs, capsize=3, linewidth=0
            )

            for model_index, (bar, v, e) in enumerate(zip(bars, vals, errs)):
                # Note: models has changed to models_to_plot
                model_name = models_to_plot[model_index]

                # --- SPECIAL CASE: Janus-Pro + Only Text ---
                if "Janus-Pro" in model_name and shift == "Only Text":
                    ax.text(
                        bar.get_x() + bar.get_width()/2, 0.02,
                        "not available",
                        ha="center", va="bottom", fontsize=metric_fontsize, rotation=90
                    )
                    continue

                # Normal case
                if not np.isnan(v):
                    # Place text above the bar + error
                    y_pos = v + (e if not np.isnan(e) else 0) + 0.02
                    ax.text(
                        bar.get_x() + bar.get_width()/2, y_pos,
                        f"{v:.2f}",
                        ha="center", va="bottom", fontsize=metric_fontsize, rotation=90
                    )
                else: 
                     # Annotate N/A if it is a partial missing value for a model that is otherwise shown
                     ax.text(
                        bar.get_x() + bar.get_width()/2, 0.02,
                        "N/A",
                        ha="center", va="bottom", fontsize=metric_fontsize, rotation=90
                    )

        # SOTA zero-shot ref line
        try:
            if sota_zero_shot:
                ax.axhline(y=sota_zero_shot[metric], color='red', linestyle='--', label='Zero-Shot SOTA')
        except KeyError:
            pass
        
        if len(models_to_plot) > 0:
            ax.set_xticks(x_plot + (len(shifts) - 1) * width / 2)
            ax.set_xticklabels(models_to_plot, fontsize=10, rotation=rotation_xticks)
        else:
             ax.set_xticks([])
             ax.set_xticklabels([])
             
        ax.set_title(metric, fontsize=16, pad=22)
        ax.set_ylim(0, 1.15)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        if idx == 0:
            ax.set_ylabel("Score", fontsize=14)

    # Single legend (dedupe labels from repeated axhline)
    handles, labels = axs[-1].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(),
               title="Condition & Reference Lines",
               loc='upper center', ncol=2 + len(shifts),
               fontsize=13, title_fontsize=11, bbox_to_anchor=(0.5, 1.12))

    fig.suptitle("Prompt Robustness (mean ± std across prompt variants)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.show()
    print(f"Saved: {out_png}")


def plot_response_distribution(
    data, 
    save_path=None, 
    title="Response Distribution", 
    figsize=(20, 15),
    normalize=True,
    history_only_lengths=True
):
    """
    Plots the distribution of Correct, Incorrect, and Rejection responses.
    Handles data structures with or without Versions, and with or without Multi-History structure.
    
    Parameters:
    -----------
    data : dict
        Output from calculate_metrics_all_models
    save_path : str
        Path to save the figure
    title : str
        Plot title
    figsize : tuple
        Figure size
    normalize : bool
        If True, plot percentages (stacked to 100%). If False, plot counts.
    history_only_lengths : bool
        If True and multi-history data is detected, only show history length conditions (not distractors).
        If False, show all shifts. Defaults to True to avoid overly large plots.
    """
    import math
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    # Flatten data into a DataFrame
    records = []
    
    # helper to process a metrics dict
    def extract_counts(metrics, context):
        if "Correct" in metrics and "Total" in metrics:
            rec = context.copy()
            rec["Correct"] = metrics.get("Correct", 0)
            rec["Incorrect"] = metrics.get("Incorrect", 0)
            rec["Rejection"] = metrics.get("Rejection", 0)
            rec["Total"] = metrics.get("Total", 0)
            records.append(rec)
            return True
        return False

    for model_name, model_data in data.items():
        # Traverse the nested dictionary structure
        def traverse(d, current_context):
            # Base case: we found the metrics dictionary
            if isinstance(d, dict) and "Correct" in d:
                extract_counts(d, current_context)
                return

            if not isinstance(d, dict):
                return
                
            for k, v in d.items():
                new_context = current_context.copy()
                
                if isinstance(k, int) or (isinstance(k, str) and k.isdigit()):
                    # Identify integer keys as History Length
                    new_context["History Length"] = int(k)
                    traverse(v, new_context)
                
                else:
                    # Treat string keys as part of the path (Version or Shift)
                    new_context["Path"] = current_context.get("Path", []) + [k]
                    traverse(v, new_context)
                    
        traverse(model_data, {"Model": model_name})

    if not records:
        print("No valid metrics found with 'Correct'/'Total' counts.")
        return

    df = pd.DataFrame(records)
    
    # Normalize path columns into Version and Shift
    rows = []
    for i, r in df.iterrows():
        path = r.get("Path", [])
        row = r.to_dict()
        if "Path" in row:
            del row["Path"]
        
        # Heuristic to assign Path items to Version/Shift
        # 5-class w/ versions: Path = [Version, Shift]
        # 5-class w/o versions: Path = [Shift]
        # History w/ versions: Path = [Version, Shift] (History Length is separate)
        
        if len(path) >= 2:
            row["Version"] = path[0]
            row["Shift"] = path[-1] 
        elif len(path) == 1:
            row["Version"] = "Default"
            row["Shift"] = path[0]
        else:
             row["Version"] = "Default"
             row["Shift"] = "Unknown"
        
        rows.append(row)

    df = pd.DataFrame(rows)

    # Apply VERSION_PRETTY_NAMES
    df["Version"] = df["Version"].apply(lambda v: VERSION_PRETTY_NAMES.get(v, v))
    
    # Detect if this is multi-history data
    is_multi_history = "History Length" in df.columns and df["History Length"].notna().any()
    
    # Construct a display "Condition" column (Shift + History)
    if is_multi_history:
        def get_condition(row):
            if pd.notna(row.get("History Length")) and row["History Length"] != 0:
                 l = row["History Length"]
                 return f"History len({l})"
            else:
                 return "No History"
                 
        df["Condition"] = df.apply(get_condition, axis=1)
        
        # If history_only_lengths is True, filter to only history length conditions
        if history_only_lengths:
            # Keep "No History" and "History len(X)" conditions, exclude distractors
            df = df[df["Condition"].str.contains("History|No History", regex=True)]
    else:
        df["Condition"] = df["Shift"]
    
    # Aggregate data: Sum counts for same (Model, Version, Condition) tuple.
    # This ensures that if multiple metrics entries map to the same Condition (e.g. multiple shifts map to 'History len(1)'),
    # their counts are summed (aggregated) rather than duplicated or dropped.
    df = df.groupby(["Model", "Version", "Condition"], as_index=False)[["Correct", "Incorrect", "Rejection", "Total"]].sum()

    if df.empty:
        print("No data to plot after filtering.")
        return
        
    models = df["Model"].unique()
    n_models = len(models)
    
    # Calculate grid size
    cols = 2 if n_models > 1 else 1
    rows = math.ceil(n_models / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(figsize[0], figsize[1] * rows / 2))
    if n_models == 1: axes = [axes]
    axes = np.array(axes).flatten()
    
    outcome_colors = {
        "Correct": "#2ca02c",   # Green
        "Incorrect": "#d62728", # Red
        "Rejection": "#7f7f7f"  # Gray
    }
    
    legend_patches = [
        plt.Rectangle((0,0),1,1, color=outcome_colors["Correct"], label='Correct'),
        plt.Rectangle((0,0),1,1, color=outcome_colors["Incorrect"], label='Incorrect'),
        plt.Rectangle((0,0),1,1, color=outcome_colors["Rejection"], label='Rejection')
    ]

    for idx, model in enumerate(models):
        ax = axes[idx]
        m_df = df[df["Model"] == model].copy()
        
        # Get unique conditions and versions
        # Sort conditions to ensure consistent order (e.g. No Shift/No History first if possible)
        conditions = list(m_df["Condition"].unique())
        if "No Shift" in conditions:
            conditions.remove("No Shift")
            conditions = ["No Shift"] + sorted([c for c in conditions if c != "No Shift"])
        elif "No History" in conditions:
            conditions.remove("No History")
            # Sort history lengths numerically
            history_conds = sorted([c for c in conditions if c.startswith("History")], 
                                  key=lambda x: int(x.split('(')[1].rstrip(')')))
            conditions = ["No History"] + history_conds
        else:
            conditions = sorted(conditions)
            
        versions = sorted(m_df["Version"].unique())
        
        # We need numerical X positions
        x = np.arange(len(conditions))
        total_width = 0.8
        bar_width = total_width / len(versions)
        
        for i, version in enumerate(versions):
            # Extract data for this version, aligning to conditions
            
            # Since we aggregated above by (Model, Version, Condition), uniqueness is guaranteed.
            v_df = m_df[m_df["Version"] == version].set_index("Condition").reindex(conditions)
            
            correct = v_df["Correct"].fillna(0).values
            incorrect = v_df["Incorrect"].fillna(0).values
            rejection = v_df["Rejection"].fillna(0).values
            total = v_df["Total"].fillna(0).values
            total[total == 0] = 1 # Avoid div by zero
            
            if normalize:
                correct_pct = correct / total * 100
                incorrect_pct = incorrect / total * 100
                rejection_pct = rejection / total * 100
                ylabel = "Percentage (%)"
            else:
                correct_pct = correct
                incorrect_pct = incorrect
                rejection_pct = rejection
                ylabel = "Count"
                
            # Stacked bar bottoms
            b_incorrect = correct_pct
            b_rejection = correct_pct + incorrect_pct
            
            offset = (i - len(versions)/2) * bar_width + bar_width/2
            
            bar1 = ax.bar(x + offset, correct_pct, bar_width, color=outcome_colors["Correct"], alpha=0.9, edgecolor='white', linewidth=0.5)
            bar2 = ax.bar(x + offset, incorrect_pct, bar_width, bottom=b_incorrect, color=outcome_colors["Incorrect"], alpha=0.9, edgecolor='white', linewidth=0.5)
            bar3 = ax.bar(x + offset, rejection_pct, bar_width, bottom=b_rejection, color=outcome_colors["Rejection"], alpha=0.9, edgecolor='white', linewidth=0.5)

            # Add version label on top of each bar
            if len(versions) > 1:
                bar_heights = b_rejection + rejection_pct
                for j, height in enumerate(bar_heights):
                    if height > 0:  # Only add label if bar has content
                        ax.text(x[j] + offset, height + 1, version, 
                               ha='center', va='bottom', fontsize=8, fontweight='bold')
            
        ax.set_title(model, fontsize=14, fontweight='bold', y=1.05)
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, rotation=45, ha='right', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_ylim(0, (105 if normalize else None))
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Only add legend to first plot to avoid clutter
        if idx == 0:
            ax.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False)

    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved: {save_path}")
    plt.show()


