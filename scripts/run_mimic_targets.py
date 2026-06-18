import argparse

import pandas as pd

from src.experiment_config import load_experiment_defaults, load_model_family
from src.experiment_data import prepare_mimic_5class
from src.run_experiments import generate_predictions_models


EXPERIMENTS = {
    "race": {"dataset": "mimic_race", "label": "race"},
    "sex": {"dataset": "mimic_sex", "label": "sex"},
    "pathology_metadata": {
        "dataset": "mimic_pathology_metadata",
        "label": "class_label",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the standard MIMIC shift workflow with alternative targets."
    )
    parser.add_argument(
        "--model-family",
        required=True,
        help="Model-family config: open_models, llavamed, gemini, or openai",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=EXPERIMENTS,
        default=list(EXPERIMENTS),
        help="Experiments to run. Defaults to all.",
    )
    return parser.parse_args()


def _balanced_target_sample(metadata, label, samples_per_class, random_seed):
    frame = metadata.dropna(subset=[label]).copy()
    frame[label] = frame[label].astype(str).str.strip()
    frame = frame[frame[label] != ""]
    sampled = [
        group.sample(n=min(samples_per_class, len(group)), random_state=random_seed)
        for _, group in frame.groupby(label)
    ]
    return pd.concat(sampled, ignore_index=True)


def main():
    args = parse_args()
    defaults = load_experiment_defaults()
    family = load_model_family(args.model_family, "mimic_targets")
    metadata = prepare_mimic_5class(
        samples_per_group=defaults["mimic_5class_samples_per_group"],
        random_seed=defaults["random_seed"],
    )

    for experiment_name in args.experiments:
        experiment = EXPERIMENTS[experiment_name]
        label = experiment["label"]
        experiment_metadata = _balanced_target_sample(
            metadata,
            label,
            defaults["mimic_target_samples_per_class"],
            defaults["random_seed"],
        )
        labels = sorted(experiment_metadata[label].unique().tolist())
        print("#" * 100)
        print(f"Evaluating {experiment_name} with model family: {args.model_family}")
        print("#" * 100)
        generate_predictions_models(
            family["models"],
            experiment_metadata,
            dataset=experiment["dataset"],
            store_columns=defaults["mimic_store_columns"],
            label=label,
            text_col="report" if experiment_name != "pathology_metadata" else None,
            image_col="filepath",
            metadata_cols=defaults["mimic_metadata_columns"],
            unmatched=False,
            versions=family["versions"],
            tokens=labels,
            priority_img=family["priority_img"],
            **family["runtime"],
        )


if __name__ == "__main__":
    main()
