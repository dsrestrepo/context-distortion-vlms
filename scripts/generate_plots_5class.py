import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.experiment_config import load_experiment_defaults, load_json_config
from src.plot_config import configure_plot_style, models_with_results
from src.test import (
    PRETTY_NAMES,
    calculate_metrics_all_models,
    compute_kappa_all_models,
    compute_nfr_all_models_multiprompt,
    plot_kappa_scores,
    plot_prompt_robustness,
    plot_response_distribution,
)


SHIFTS = ["No Shift", "Img Shift", "Txt Shift", "Only Text", "Only Image"]
SHIFT_COLORS = {
    "No Shift": "#08306B",
    "Img Shift": "#E6550D",
    "Txt Shift": "#FDAE6B",
    "Only Text": "#31A354",
    "Only Image": "#A1D99B",
}


def calculate_bootstrap(
    model_dict,
    iterations,
    versions,
    random_seed,
    calibration=False,
    bootstrap_proportion=1.0,
):
    return calculate_metrics_all_models(
        model_dict,
        results_dir="results",
        dataset="mimic_5class",
        subgroup_variables=[],
        counterfactual=True,
        first_token=False,
        confusion_matrix=False,
        show_unknown_responses=False,
        p_yes=False,
        calibration=calibration,
        unmatched=False,
        file_name="mimic_5class_base_shifted_multi_versions.csv",
        versions=versions,
        bootstrap=True,
        n_bootstrap=iterations,
        bootstrap_proportion=bootstrap_proportion,
        random_state=random_seed,
    )


def plot_nfr_heatmap(df_avg_nfr):
    df = df_avg_nfr.rename(
        columns={"Image": "Image Shift", "Text": "Text Shift"}
    ).copy()
    df.index = [PRETTY_NAMES.get(model, model) for model in df.index]

    plt.figure(figsize=(14, 8))
    ax = sns.heatmap(
        df,
        cmap="RdYlGn_r",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".3f",
        annot_kws={"size": 18, "weight": "bold", "color": "black"},
        linewidths=0.6,
        linecolor="white",
        mask=df.isna(),
        cbar_kws={"label": "Negative Flip Rate (down is better)"},
    )
    ax.grid(False)
    ax.set_title("Average Negative Flip Rate Across Perturbations", pad=20)
    ax.set_xlabel("Perturbation", labelpad=15)
    ax.set_ylabel("Model", labelpad=15)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("images/nfr_heatmap_mimic_5class.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    defaults = load_experiment_defaults()
    plot_config = load_json_config("plotting.json")
    model_dict = models_with_results(
        plot_config["models"], "mimic_5class_base_shifted_multi_versions.csv"
    )
    iterations = plot_config["bootstrap_iterations"]
    random_seed = defaults["random_seed"]
    os.makedirs("images", exist_ok=True)
    configure_plot_style(plot_config["style"])

    print("Generating MIMIC 5-class regex metrics and refusal distribution...")
    regex_metrics = calculate_metrics_all_models(
        model_dict,
        results_dir="results",
        dataset="mimic_5class",
        subgroup_variables=[],
        counterfactual=True,
        first_token=False,
        confusion_matrix=False,
        show_unknown_responses=False,
        calibration=False,
        unmatched=False,
        file_name="mimic_5class_base_shifted_multi_versions.csv",
        versions=plot_config["prompt_versions"],
    )
    plot_prompt_robustness(
        regex_metrics,
        out_png="images/metrics_mimic_5class_regex.png",
        metrics=["Accuracy"],
        shifts=SHIFTS,
        shift_colors=SHIFT_COLORS,
        rotation_xticks=60,
        metric_fontsize=18,
        figsize=(20, 8),
    )
    plot_response_distribution(
        regex_metrics,
        save_path="images/response_distribution_mimic_5class_regex.png",
        title="Regex Response Distribution Including Refusals (MIMIC 5-Class)",
        figsize=(20, 15),
        normalize=True,
    )

    print("Generating MIMIC 5-class first-token, calibration, and ECE plots...")
    first_token_metrics = calculate_metrics_all_models(
        model_dict,
        results_dir="results",
        dataset="mimic_5class",
        subgroup_variables=[],
        counterfactual=True,
        first_token=True,
        confusion_matrix=False,
        show_unknown_responses=False,
        calibration=True,
        p_yes=False,
        save_calibration_plot=True,
        unmatched=False,
        file_name="mimic_5class_base_shifted_multi_versions.csv",
        versions=plot_config["prompt_versions"],
    )
    plot_prompt_robustness(
        first_token_metrics,
        out_png="images/metrics_mimic_5class_first_token.png",
        metrics=["Accuracy"],
        shifts=SHIFTS,
        shift_colors=SHIFT_COLORS,
        rotation_xticks=60,
        metric_fontsize=18,
        figsize=(20, 8),
    )
    plot_prompt_robustness(
        first_token_metrics,
        out_png="images/ece_mimic_5class_first_token.png",
        metrics=["ECE"],
        shifts=SHIFTS,
        shift_colors=SHIFT_COLORS,
        rotation_xticks=60,
        metric_fontsize=18,
        figsize=(20, 8),
    )
    plot_response_distribution(
        first_token_metrics,
        save_path="images/response_distribution_mimic_5class_first_token.png",
        title="First-Token Response Distribution (MIMIC 5-Class)",
        figsize=(20, 15),
        normalize=True,
    )

    print("Generating MIMIC 5-class bootstrap plot...")
    data_bootstrap = calculate_bootstrap(
        model_dict, iterations, plot_config["prompt_versions"], random_seed
    )
    plot_prompt_robustness(
        data_bootstrap,
        out_png="images/metrics_5_class_bootstrap_ci_split.png",
        metrics=["Accuracy"],
        shifts=SHIFTS,
        shift_colors=SHIFT_COLORS,
        rotation_xticks=60,
        metric_fontsize=18,
        figsize=(20, 8),
    )

    print("Generating MIMIC 5-class image-priority bootstrap plot...")
    image_priority_bootstrap = calculate_bootstrap(
        model_dict,
        iterations,
        ["v1"],
        random_seed,
        calibration=True,
        bootstrap_proportion=plot_config["bootstrap_proportion"],
    )
    plot_prompt_robustness(
        image_priority_bootstrap,
        out_png="images/prompt_robustness_summary_5_class_image_priority_prompt_bootstrap.png",
        metrics=["Accuracy"],
        shifts=SHIFTS,
        shift_colors=SHIFT_COLORS,
        rotation_xticks=60,
        metric_fontsize=18,
        figsize=(20, 8),
    )

    print("Generating MIMIC 5-class NFR and Kappa plots...")
    nfr = compute_nfr_all_models_multiprompt(
        model_dict,
        results_dir="results",
        dataset="mimic_5class",
        id="dicom_id",
        version_col="version",
        shift_col="shift",
        first_token=False,
        return_per_prompt=False,
    )
    plot_nfr_heatmap(nfr)

    kappa = compute_kappa_all_models(
        model_dict,
        results_dir="results",
        dataset="mimic_5class",
        version_col="version",
        first_token=True,
        p_yes=False,
        file_name="mimic_5class_base_shifted_multi_versions.csv",
        id_col="dicom_id",
        shift_col="shift",
        base_shift="No",
        history_mode=None,
        include_shifts=True,
    )
    plot_kappa_scores(
        kappa,
        save_path="images/fleiss_kappa_scores_mimic_5class.png",
        figsize=(12, 6),
        title="Fleiss' Kappa Across Models (Modality Bias)",
    )


if __name__ == "__main__":
    main()
