import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "config"
MODEL_FAMILY_FILES = {
    "open_models": "open_models.json",
    "llavamed": "llavamed.json",
    "gemini": "gemini.json",
    "openai": "openai.json",
}


def load_json_config(filename):
    path = CONFIG_DIR / filename
    with Path(path).open(encoding="utf-8") as config_file:
        return json.load(config_file)


def load_experiment_defaults():
    return load_json_config("experiment_defaults.json")


def select_models(models, selected_names=None):
    if not selected_names:
        return models
    selected_names = set(selected_names)
    unknown = sorted(selected_names - set(models.values()))
    if unknown:
        raise ValueError(f"Unknown model output names: {', '.join(unknown)}")
    return {
        model_id: model_name
        for model_id, model_name in models.items()
        if model_name in selected_names
    }


def load_model_family(model_family, task, selected_models=None):
    try:
        config = load_json_config(MODEL_FAMILY_FILES[model_family])
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_FAMILY_FILES))
        raise ValueError(
            f"Unknown model family '{model_family}'. Available families: {available}"
        ) from exc

    try:
        task_config = config["tasks"][task]
    except KeyError as exc:
        raise ValueError(
            f"Model family '{model_family}' does not define task '{task}'."
        ) from exc

    models = config["models"]
    enabled_models = task_config.get("enabled_models")
    if selected_models:
        models = select_models(models, selected_models)
    elif enabled_models is not None:
        unknown = sorted(set(enabled_models) - set(models))
        if unknown:
            raise ValueError(
                f"Task '{task}' enables unknown model IDs: {', '.join(unknown)}"
            )
        models = {
            model_id: model_name
            for model_id, model_name in models.items()
            if model_id in enabled_models
        }

    return {
        "models": models,
        "versions": task_config["prompt_versions"],
        "priority_img": task_config["priority_img"],
        "runtime": task_config["runtime"],
        "debug_samples": task_config.get("debug_samples"),
    }
