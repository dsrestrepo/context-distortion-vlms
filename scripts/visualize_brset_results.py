"""
Visualize images, prompts, and model outputs for BrSET and mBrSET datasets.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from src.experiment_config import load_json_config
from src.prompts import BRSET_TEXT_PROMPT, BRSET_ONLY_IMAGE_TEXT_PROMPT, BRSET_ONLY_TEXT_PROMPT
from src.prompts import mBRSET_TEXT_PROMPT, mBRSET_ONLY_IMAGE_TEXT_PROMPT, mBRSET_ONLY_TEXT_PROMPT

VISUALIZATION_CONFIG = load_json_config("brset_visualization.json")
MODEL_DICT = VISUALIZATION_CONFIG["models"]

# Pretty names for display
PRETTY_NAMES = {
    "qwen2_vl_7b": "Qwen-2 VL 7B",
    "llava_1_5_7b": "LLaVA 1.5 7B",
    "llava_med_mistral_instruct": "LLaVA-Med Mistral 7B",
    "medgemma": "MedGemma 4B",
    "llama3_10b": "Llama-3.2 11B Vision"
}

# Dataset configurations
DATASETS = VISUALIZATION_CONFIG["datasets"]
DATASETS["brset"]["prompt_funcs"] = {
    "No": BRSET_TEXT_PROMPT,
    "Image": BRSET_ONLY_TEXT_PROMPT,
    "Text": BRSET_ONLY_IMAGE_TEXT_PROMPT,
    "Only_text": BRSET_ONLY_TEXT_PROMPT,
    "Only_image": BRSET_ONLY_IMAGE_TEXT_PROMPT,
}
DATASETS["mbrset"]["prompt_funcs"] = {
    "No": mBRSET_TEXT_PROMPT,
    "Image": mBRSET_ONLY_TEXT_PROMPT,
    "Text": mBRSET_ONLY_IMAGE_TEXT_PROMPT,
    "Only_text": mBRSET_ONLY_TEXT_PROMPT,
    "Only_image": mBRSET_ONLY_IMAGE_TEXT_PROMPT,
}


def load_image(image_path):
    """Load and return an image."""
    try:
        img = Image.open(image_path)
        return img
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def truncate_text(text, max_length=150):
    """Truncate text to max_length characters."""
    if isinstance(text, str) and len(text) > max_length:
        return text[:max_length] + "..."
    return text


def visualize_sample(ax_img, ax_text, row, model_name, dataset_info, metadata_df=None, show_prediction=True):
    """
    Visualize a single sample with image and text information.
    
    Args:
        ax_img: matplotlib axis for image
        ax_text: matplotlib axis for text
        row: DataFrame row with sample data
        model_name: Name of the model
        dataset_info: Dataset configuration dict
        metadata_df: DataFrame with metadata for prompt generation
        show_prediction: Whether to show prediction output
    """
    # Load and display image using 'filepath' column
    if 'filepath' in row.index:
        image_path = row['filepath']
        img = load_image(image_path)
        
        if img is not None:
            ax_img.imshow(img)
            ax_img.axis('off')
            filename = os.path.basename(image_path)
            ax_img.set_title(f"{filename}", fontsize=10)
        else:
            ax_img.text(0.5, 0.5, 'Image not found', ha='center', va='center')
            ax_img.axis('off')
    else:
        ax_img.text(0.5, 0.5, 'No filepath column found', ha='center', va='center')
        ax_img.axis('off')
    
    # Generate prompt from metadata
    prompt_text = "Prompt not available"
    if metadata_df is not None and 'filepath' in row.index:
        # Extract filename from filepath
        filename = os.path.basename(row['filepath'])
        # Remove extension for matching
        filename_no_ext = os.path.splitext(filename)[0]
        
        # Get the metadata image column name
        meta_img_col = dataset_info.get('metadata_image_col', 'image')
        
        # Try to find matching metadata row
        # First try exact match with extension
        meta_row = metadata_df[metadata_df[meta_img_col] == filename]
        
        # If not found, try without extension
        if meta_row.empty:
            meta_row = metadata_df[metadata_df[meta_img_col] == filename_no_ext]
        
        # If still not found, try contains
        if meta_row.empty:
            meta_row = metadata_df[metadata_df[meta_img_col].astype(str).str.contains(filename_no_ext, na=False)]
        
        if not meta_row.empty:
            meta_row = meta_row.iloc[0]
            shift_type = row.get('shift', 'No')
            prompt_func = dataset_info['prompt_funcs'].get(shift_type, dataset_info['prompt_funcs']['No'])
            
            try:
                if callable(prompt_func):
                    prompt_text = prompt_func(meta_row)
                else:
                    prompt_text = str(prompt_func)
            except Exception as e:
                prompt_text = f"Error generating prompt: {e}"
        else:
            prompt_text = f"Metadata not found for: {filename}"
    
    # Prepare text information
    text_info = []
    text_info.append(f"Model: {PRETTY_NAMES.get(model_name, model_name)}")
    text_info.append(f"Shift Type: {row.get('shift', 'N/A')}")
    text_info.append(f"Ground Truth: {row.get('ground_truth', 'N/A')}")
    text_info.append("")
    text_info.append("Prompt:")
    #text_info.append(truncate_text(prompt_text, 200))
    text_info.append(prompt_text)
    
    
    if show_prediction:
        text_info.append("")
        text_info.append("Prediction:")
        #text_info.append(truncate_text(str(row.get('prediction', 'N/A')), 200))
        text_info.append(str(row.get('prediction', 'N/A')))
    
    # Display text
    ax_text.text(0.05, 0.95, '\n'.join(text_info), 
                verticalalignment='top', fontsize=8,
                family='monospace', wrap=True)
    ax_text.axis('off')


def visualize_dataset_model(dataset_key, model_name, results_dir="results", 
                           num_samples=3, shift_types=None, save_path=None):
    """
    Visualize samples from a dataset for a specific model.
    
    Args:
        dataset_key: Key from DATASETS dict ('brset' or 'mbrset')
        model_name: Model key from MODEL_DICT
        results_dir: Base directory for results
        num_samples: Number of samples to visualize per shift type
        shift_types: List of shift types to visualize (None = all)
        save_path: Path to save the figure (None = show only)
    """
    dataset_info = DATASETS[dataset_key]
    csv_path = os.path.join(results_dir, model_name, dataset_info["file_pattern"])
    
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return
    
    # Load data
    print(f"\nLoading {dataset_info['name']} results for {PRETTY_NAMES.get(model_name, model_name)}...")
    df = pd.read_csv(csv_path)
    
    # Print available columns for debugging
    print(f"Available columns: {df.columns.tolist()}")
    
    # Load metadata
    metadata_df = None
    if 'metadata_path' in dataset_info:
        try:
            metadata_df = pd.read_csv(dataset_info['metadata_path'])
            print(f"Loaded metadata with {len(metadata_df)} rows")
            print(f"Metadata columns: {metadata_df.columns.tolist()}")
            meta_img_col = dataset_info.get('metadata_image_col', 'image')
            print(f"Using metadata image column: '{meta_img_col}'")
            if meta_img_col in metadata_df.columns:
                print(f"Sample metadata image values: {metadata_df[meta_img_col].head().tolist()}")
            else:
                print(f"WARNING: Column '{meta_img_col}' not found in metadata!")
        except Exception as e:
            print(f"Warning: Could not load metadata: {e}")
    
    # Get shift types
    if shift_types is None:
        shift_types = df['shift'].unique().tolist() if 'shift' in df.columns else ['No']
    
    print(f"Shift types found: {shift_types}")
    print(f"Total samples: {len(df)}")
    
    # Create figure
    n_shifts = len(shift_types)
    fig = plt.figure(figsize=(16, 4 * n_shifts * num_samples))
    
    sample_idx = 0
    
    for shift_idx, shift in enumerate(shift_types):
        # Filter by shift type
        if 'shift' in df.columns:
            shift_df = df[df['shift'] == shift].head(num_samples)
        else:
            shift_df = df.head(num_samples)
        
        print(f"\nShift: {shift} - Samples: {len(shift_df)}")
        
        for i, (idx, row) in enumerate(shift_df.iterrows()):
            # Create subplot for image
            ax_img = plt.subplot(n_shifts * num_samples, 2, sample_idx * 2 + 1)
            # Create subplot for text
            ax_text = plt.subplot(n_shifts * num_samples, 2, sample_idx * 2 + 2)
            
            visualize_sample(ax_img, ax_text, row, model_name, 
                           dataset_info, metadata_df, show_prediction=True)
            
            sample_idx += 1
    
    plt.suptitle(f"{dataset_info['name']} - {PRETTY_NAMES.get(model_name, model_name)}", 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {save_path}")
    
    plt.show()


def visualize_comparison_across_models(dataset_key, sample_idx=0, results_dir="results",
                                       shift_type="No", save_path=None):
    """
    Compare the same sample across different models.
    
    Args:
        dataset_key: Key from DATASETS dict ('brset' or 'mbrset')
        sample_idx: Index of the sample to compare
        results_dir: Base directory for results
        shift_type: Shift type to filter by
        save_path: Path to save the figure
    """
    dataset_info = DATASETS[dataset_key]
    
    # Load metadata once
    metadata_df = None
    if 'metadata_path' in dataset_info:
        try:
            metadata_df = pd.read_csv(dataset_info['metadata_path'])
        except Exception as e:
            print(f"Warning: Could not load metadata: {e}")
    
    # Create figure
    n_models = len(MODEL_DICT)
    fig = plt.figure(figsize=(18, 4 * n_models))
    
    for model_idx, (hf_name, model_key) in enumerate(MODEL_DICT.items()):
        csv_path = os.path.join(results_dir, model_key, dataset_info["file_pattern"])
        
        if not os.path.exists(csv_path):
            print(f"Skipping {model_key} - file not found")
            continue
        
        # Load data
        df = pd.read_csv(csv_path)
        
        # Filter by shift type
        if 'shift' in df.columns:
            df_filtered = df[df['shift'] == shift_type]
        else:
            df_filtered = df
        
        if len(df_filtered) <= sample_idx:
            print(f"Sample {sample_idx} not found for {model_key}")
            continue
        
        row = df_filtered.iloc[sample_idx]
        
        # Create subplots
        ax_img = plt.subplot(n_models, 2, model_idx * 2 + 1)
        ax_text = plt.subplot(n_models, 2, model_idx * 2 + 2)
        
        visualize_sample(ax_img, ax_text, row, model_key, 
                       dataset_info, metadata_df, show_prediction=True)
    
    plt.suptitle(f"{dataset_info['name']} - Model Comparison (Shift: {shift_type}, Sample: {sample_idx})", 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {save_path}")
    
    plt.show()


def generate_summary_statistics(dataset_key, results_dir="results"):
    """
    Generate summary statistics for all models on a dataset.
    
    Args:
        dataset_key: Key from DATASETS dict
        results_dir: Base directory for results
    """
    dataset_info = DATASETS[dataset_key]
    
    print(f"\n{'='*80}")
    print(f"Summary Statistics for {dataset_info['name']}")
    print(f"{'='*80}\n")
    
    for hf_name, model_key in MODEL_DICT.items():
        csv_path = os.path.join(results_dir, model_key, dataset_info["file_pattern"])
        
        if not os.path.exists(csv_path):
            print(f"{PRETTY_NAMES.get(model_key, model_key)}: File not found")
            continue
        
        df = pd.read_csv(csv_path)
        
        print(f"\n{PRETTY_NAMES.get(model_key, model_key)}:")
        print(f"  Total samples: {len(df)}")
        
        if 'shift' in df.columns:
            print(f"  Shift types: {df['shift'].unique().tolist()}")
            print(f"  Samples per shift:")
            for shift, count in df['shift'].value_counts().items():
                print(f"    {shift}: {count}")
        
        if 'ground_truth' in df.columns:
            print(f"  Ground truth distribution:")
            for gt, count in df['ground_truth'].value_counts().items():
                print(f"    {gt}: {count}")
        
        if 'prediction' in df.columns:
            # Count predictions that are not empty
            valid_preds = df['prediction'].notna().sum()
            print(f"  Valid predictions: {valid_preds}/{len(df)}")


def main():
    """Main function to run visualizations."""
    
    # Configuration
    RESULTS_DIR = VISUALIZATION_CONFIG["results_dir"]
    OUTPUT_DIR = VISUALIZATION_CONFIG["output_dir"]
    samples_per_shift = VISUALIZATION_CONFIG["samples_per_shift"]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate summary statistics
    for dataset_key in DATASETS.keys():
        generate_summary_statistics(dataset_key, RESULTS_DIR)
    
    print("\n" + "="*80)
    print("Generating visualizations...")
    print("="*80 + "\n")
    
    # Visualize each model for each dataset
    for dataset_key in DATASETS.keys():
        for hf_name, model_key in MODEL_DICT.items():
            save_path = os.path.join(OUTPUT_DIR, f"{dataset_key}_{model_key}_samples.png")
            visualize_dataset_model(
                dataset_key=dataset_key,
                model_name=model_key,
                results_dir=RESULTS_DIR,
                num_samples=samples_per_shift,
                save_path=save_path
            )
    
    # Generate comparison plots (same sample across models)
    for dataset_key in DATASETS.keys():
        for shift_type in ["No", "Image", "Text"]:
            for sample_idx in range(samples_per_shift):
                save_path = os.path.join(OUTPUT_DIR, 
                                        f"{dataset_key}_comparison_shift_{shift_type}_sample_{sample_idx}.png")
                visualize_comparison_across_models(
                    dataset_key=dataset_key,
                    sample_idx=sample_idx,
                    results_dir=RESULTS_DIR,
                    shift_type=shift_type,
                    save_path=save_path
                )
    
    print("\n" + "="*80)
    print("Visualization complete!")
    print(f"Images saved to: {OUTPUT_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()
