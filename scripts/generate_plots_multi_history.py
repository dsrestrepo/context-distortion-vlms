import os

import matplotlib.pyplot as plt
import seaborn as sns

from src.experiment_config import load_experiment_defaults, load_json_config
from src.plot_config import configure_plot_style, models_with_results
from src.test import (
    PRETTY_NAMES,
    calculate_metrics_all_models,
    compute_kappa_all_models,
    compute_nfr_history_all_models,
    plot_kappa_scores,
    plot_metrics_with_ci_split,
    plot_response_distribution,
)


DISTRACTOR_SHIFTS = [
    "No History",
    "distractor_mri_brain",
    "contradictory_cxr_prior",
    "distractor_wrist_ultrasound",
    "distractor_ct_abd_pelvis",
    "distractor_knee_xray",
]
HISTORY_SHIFTS = [
    "No History",
    "History len(1)",
    "History len(2)",
    "History len(3)",
    "History len(4)",
    "History len(5)",
]
SHIFT_COLORS = {
    "distractor_mri_brain": "#756BB1",
    "contradictory_cxr_prior": "#E7298A",
    "distractor_wrist_ultrasound": "#1B9E77",
    "distractor_ct_abd_pelvis": "#E41A1C",
    "distractor_knee_xray": "#FF7F00",
    "No History": "#9E9E9E",
    "History len(1)": "#C6DBEF",
    "History len(2)": "#9ECAE1",
    "History len(3)": "#6BAED6",
    "History len(4)": "#3182BD",
    "History len(5)": "#08519C",
}


def plot_nfr_heatmap(df_avg_nfr_history):
    df = df_avg_nfr_history.copy()
    df.index = [PRETTY_NAMES.get(model, model) for model in df.index]
    df.columns = [
        "History = 1",
        "History = 2",
        "History = 3",
        "History = 4",
        "History = 5",
    ]

    plt.figure(figsize=(14, 8))
    ax = sns.heatmap(
        df,
        cmap="RdYlGn_r",
        vmin=0,
        vmax=0.4,
        annot=True,
        fmt=".3f",
        annot_kws={"size": 18, "weight": "bold", "color": "black"},
        linewidths=0.6,
        linecolor="white",
        mask=df.isna(),
        cbar_kws={"label": "Negative Flip Rate (down is better)"},
    )
    ax.grid(False)
    ax.set_title("Average Negative Flip Rate vs History Length", pad=20)
    ax.set_xlabel("History Length", labelpad=15)
    ax.set_ylabel("Model", labelpad=15)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("images/nfr_heatmap_mimic_history.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    defaults = load_experiment_defaults()
    plot_config = load_json_config("plotting.json")
    model_dict = models_with_results(
        plot_config["models"], "cxr_multi_history_base_multi_versions.csv"
    )
    os.makedirs("images", exist_ok=True)
    configure_plot_style(plot_config["style"])

    common_metrics_args = {
        "results_dir": "results",
        "dataset": "cxr_multi_history",
        "subgroup_variables": [],
        "counterfactual": True,
        "confusion_matrix": False,
        "show_unknown_responses": False,
        "unmatched": False,
        "p_yes": False,
        "shift_col": "original",
        "edited_cols": [False],
        "file_name": "cxr_multi_history_base_multi_versions.csv",
        "shift_names": {True: True, False: False},
        "versions": plot_config["prompt_versions"],
        "history_length_col": "history_length",
    }

    print("Generating multi-history regex metrics and refusal distribution...")
    regex_metrics = calculate_metrics_all_models(
        model_dict,
        first_token=False,
        calibration=False,
        **common_metrics_args,
    )
    plot_metrics_with_ci_split(
        data=regex_metrics,
        models=[PRETTY_NAMES.get(model, model) for model in model_dict.values()],
        shifts_row1=DISTRACTOR_SHIFTS,
        shifts_row2=HISTORY_SHIFTS,
        metrics=["Accuracy"],
        shift_colors=SHIFT_COLORS,
        title="Regex Accuracy - Multi-History",
        save_path="images/metrics_multi_history_regex.png",
        figsize=(20, 8),
        metric_fontsize=18,
    )
    plot_response_distribution(
        regex_metrics,
        save_path="images/response_distribution_multi_history_regex.png",
        title="Regex Response Distribution Including Refusals (Multi-History)",
        figsize=(20, 15),
        normalize=True,
        history_only_lengths=False,
    )

    print("Generating multi-history first-token, calibration, and ECE plots...")
    first_token_metrics = calculate_metrics_all_models(
        model_dict,
        first_token=True,
        calibration=True,
        save_calibration_plot=True,
        **common_metrics_args,
    )
    first_token_models = [
        PRETTY_NAMES.get(model, model) for model in model_dict.values()
    ]
    plot_metrics_with_ci_split(
        data=first_token_metrics,
        models=first_token_models,
        shifts_row1=DISTRACTOR_SHIFTS,
        shifts_row2=HISTORY_SHIFTS,
        metrics=["Accuracy"],
        shift_colors=SHIFT_COLORS,
        title="First-Token Accuracy - Multi-History",
        save_path="images/metrics_multi_history_first_token.png",
        figsize=(20, 8),
        metric_fontsize=18,
    )
    plot_metrics_with_ci_split(
        data=first_token_metrics,
        models=first_token_models,
        shifts_row1=DISTRACTOR_SHIFTS,
        shifts_row2=HISTORY_SHIFTS,
        metrics=["ECE"],
        shift_colors=SHIFT_COLORS,
        title="First-Token ECE - Multi-History",
        save_path="images/ece_multi_history_first_token.png",
        figsize=(20, 8),
        metric_fontsize=18,
    )
    plot_response_distribution(
        first_token_metrics,
        save_path="images/response_distribution_multi_history_first_token.png",
        title="First-Token Response Distribution (Multi-History)",
        figsize=(20, 15),
        normalize=True,
        history_only_lengths=False,
    )

    print("Generating MIMIC multi-history bootstrap plot...")
    data_bootstrap = calculate_metrics_all_models(
        model_dict,
        results_dir="results",
        dataset="cxr_multi_history",
        subgroup_variables=[],
        counterfactual=True,
        first_token=False,
        confusion_matrix=False,
        calibration=True,
        show_unknown_responses=False,
        unmatched=False,
        p_yes=False,
        save_calibration_plot=False,
        shift_col="original",
        edited_cols=[False],
        file_name="cxr_multi_history_base_multi_versions.csv",
        shift_names={True: True, False: False},
        versions=None,
        bootstrap=True,
        n_bootstrap=plot_config["bootstrap_iterations"],
        bootstrap_proportion=plot_config["bootstrap_proportion"],
        random_state=defaults["random_seed"],
        history_length_col="history_length",
    )
    data_bootstrap = {
        PRETTY_NAMES.get(model, model): values
        for model, values in data_bootstrap.items()
    }
    models = [PRETTY_NAMES.get(model, model) for model in model_dict.values()]
    plot_metrics_with_ci_split(
        data=data_bootstrap,
        models=models,
        shifts_row1=DISTRACTOR_SHIFTS,
        shifts_row2=HISTORY_SHIFTS,
        metrics=["Accuracy"],
        shift_colors=SHIFT_COLORS,
        sota_lines=None,
        rotation_xticks=60,
        title="Metrics with Bootstrap CIs (n=100) - History Length",
        save_path="images/metrics_multi_history_bootstrap_ci_split.png",
        figsize=(20, 8),
        metric_fontsize=18,
        style=None,
    )

    print("Generating MIMIC multi-history NFR and Kappa plots...")
    nfr = compute_nfr_history_all_models(
        model_dict,
        results_dir="results",
        dataset="cxr_multi_history",
        id="dicom_id",
        version_col="version",
        history_length_col="history_length",
        first_token=False,
        return_per_prompt=False,
    )
    plot_nfr_heatmap(nfr)

    kappa = compute_kappa_all_models(
        model_dict,
        results_dir="results",
        dataset="cxr_multi_history",
        version_col="version",
        first_token=True,
        p_yes=False,
        file_name="cxr_multi_history_base_multi_versions.csv",
        id_col="dicom_id",
        shift_col="original",
        base_shift=True,
        history_mode="no_vs_length_groups",
        include_shifts=True,
    )
    plot_kappa_scores(
        kappa,
        save_path="images/fleiss_kappa_scores_multi_history.png",
        figsize=(12, 6),
        title="Fleiss' Kappa Across Models (Multi-History)",
    )


if __name__ == "__main__":
    main()
