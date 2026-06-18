import argparse

from src.experiment_config import load_experiment_defaults, load_model_family
from src.experiment_data import load_history_experiment
from src.test import generate_predictions_models_base


def parse_args():
    parser = argparse.ArgumentParser(description="Run the MIMIC multi-history experiment.")
    parser.add_argument(
        "--model-family",
        required=True,
        help="Model-family config: open_models, llavamed, gemini, or openai",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    defaults = load_experiment_defaults()
    model_family = load_model_family(args.model_family, "history")
    metadata = load_history_experiment(defaults["history_csv"])

    print("#" * 100)
    print(f"Evaluating MIMIC multi-history with model family: {args.model_family}")
    print("#" * 100)
    generate_predictions_models_base(
        model_family["models"],
        metadata,
        dataset="cxr_multi_history",
        store_columns=defaults["history_store_columns"],
        label="label",
        unmatched=False,
        history_cols=defaults["history_columns"],
        versions=model_family["versions"],
        priority_img=model_family["priority_img"],
        **model_family["runtime"],
    )


if __name__ == "__main__":
    main()
