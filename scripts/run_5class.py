import argparse

from src.experiment_config import load_experiment_defaults, load_model_family
from src.experiment_data import prepare_mimic_5class
from src.test import generate_predictions_models


def parse_args():
    parser = argparse.ArgumentParser(description="Run the MIMIC 5-class experiment.")
    parser.add_argument(
        "--model-family",
        required=True,
        help="Model-family config: open_models, llavamed, gemini, or openai",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Run only these configured model output names, such as gemma4_12b.",
    )
    parser.add_argument(
        "--debug-samples",
        type=int,
        help="Run only this many deterministic rows and use debug filenames.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    defaults = load_experiment_defaults()
    model_family = load_model_family(args.model_family, "5class", args.models)
    debug_samples = (
        args.debug_samples
        if args.debug_samples is not None
        else model_family["debug_samples"]
    )
    metadata = prepare_mimic_5class(
        samples_per_group=defaults["mimic_5class_samples_per_group"],
        random_seed=defaults["random_seed"],
    )
    if debug_samples is not None:
        if debug_samples < 1:
            raise ValueError("--debug-samples must be at least 1.")
        metadata = metadata.sample(
            n=min(debug_samples, len(metadata)),
            random_state=defaults["random_seed"],
        )
        print(f"Debug mode: running {len(metadata)} rows.")

    print("#" * 100)
    print(f"Evaluating MIMIC 5-class with model family: {args.model_family}")
    print("#" * 100)
    generate_predictions_models(
        model_family["models"],
        metadata,
        dataset="mimic_5class_debug" if debug_samples is not None else "mimic_5class",
        store_columns=defaults["mimic_store_columns"],
        label="class_label",
        text_col="report",
        image_col="filepath",
        metadata_cols=defaults["mimic_metadata_columns"],
        unmatched=False,
        versions=model_family["versions"],
        priority_img=model_family["priority_img"],
        **model_family["runtime"],
    )


if __name__ == "__main__":
    main()
