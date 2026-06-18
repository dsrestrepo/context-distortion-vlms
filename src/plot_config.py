import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns


def configure_plot_style(style):
    sns.set_theme(
        style="white",
        rc={
            "axes.edgecolor": "black",
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.grid": False,
        },
    )
    plt.rcParams.update(style)


def models_with_results(model_dict, result_filename, results_dir="results"):
    available = {}
    for model_id, model_name in model_dict.items():
        result_path = Path(results_dir) / model_name / result_filename
        if result_path.exists():
            available[model_id] = model_name
        else:
            print(f"Skipping {model_name}: missing {result_path}")
    if not available:
        raise FileNotFoundError(
            f"No model results found for {result_filename} under {results_dir}/"
        )
    return available
