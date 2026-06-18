from src.experiment_config import load_experiment_defaults, load_model_family
from src.experiment_data import load_brset_experiments
from src.test import generate_predictions_models


def main():
    defaults = load_experiment_defaults()
    model_family = load_model_family("llavamed", "5class")
    mbrset, brset = load_brset_experiments(defaults["brset_samples_per_class"])
    runtime = model_family["runtime"]
    runtime["quantization"] = "16b"

    print("#" * 100)
    print("Evaluating mBRSET with LLaVA-Med")
    print("#" * 100)
    generate_predictions_models(
        model_family["models"],
        mbrset,
        dataset="mbrset",
        store_columns=[
            "filepath",
            "age",
            "sex",
            "insurance",
            "educational_level",
            "alcohol_consumption",
            "smoking",
            "obesity",
        ],
        label="final_icdr",
        image_col="filepath",
        **runtime,
    )

    print("#" * 100)
    print("Evaluating BRSET with LLaVA-Med")
    print("#" * 100)
    generate_predictions_models(
        model_family["models"],
        brset,
        dataset="brset",
        store_columns=[
            "filepath",
            "patient_age",
            "patient_sex",
            "insuline",
            "DR_ICDR",
            "focus",
            "iluminaton",
            "image_field",
            "artifacts",
        ],
        label="DR",
        image_col="filepath",
        **runtime,
    )


if __name__ == "__main__":
    main()
