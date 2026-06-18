"""Dataset prompt construction and legacy experiment inference adapter."""

from transformers.image_utils import load_image

from src.prompts import (
    BRSET_ONLY_IMAGE_TEXT_PROMPT,
    BRSET_ONLY_TEXT_PROMPT,
    BRSET_TEXT_PROMPT,
    CXR_HISTORY_TEXT_PROMPT_5CLASS,
    CXR_HISTORY_TEXT_PROMPT_5CLASS_V1,
    CXR_HISTORY_TEXT_PROMPT_5CLASS_V2,
    CXR_HISTORY_TEXT_PROMPT_5CLASS_V3,
    MIMIC_ONLY_IMAGE_TEXT_PROMPT_5CLASS,
    MIMIC_ONLY_IMAGE_TEXT_PROMPT_5CLASS_V1,
    MIMIC_ONLY_IMAGE_TEXT_PROMPT_5CLASS_V2,
    MIMIC_ONLY_IMAGE_TEXT_PROMPT_5CLASS_V3,
    MIMIC_ONLY_TEXT_PROMPT_5CLASS,
    MIMIC_ONLY_TEXT_PROMPT_5CLASS_V1,
    MIMIC_ONLY_TEXT_PROMPT_5CLASS_V2,
    MIMIC_ONLY_TEXT_PROMPT_5CLASS_V3,
    MIMIC_TEXT_PROMPT_5CLASS,
    MIMIC_TEXT_PROMPT_5CLASS_V1,
    MIMIC_TEXT_PROMPT_5CLASS_V2,
    MIMIC_TEXT_PROMPT_5CLASS_V3,
    MIMIC_TARGET_PROMPT,
    mBRSET_ONLY_IMAGE_TEXT_PROMPT,
    mBRSET_ONLY_TEXT_PROMPT,
    mBRSET_TEXT_PROMPT,
)


def generate_text(
    model,
    processor,
    prompt,
    image,
    quantization=None,
    return_attention=False,
    return_logits=False,
    dataset=None,
    p_yes_and_no=False,
    row=None,
    unmatched=False,
    tokens=None,
    priority_img=False,
    modality=None,
):
    """Call the unified model interface and preserve the historical tuple output."""
    del processor, quantization, row
    if priority_img and image is not None and modality != "Only_image":
        prompt += (
            "\n\nIMPORTANT: If there is a mismatch or conflict between the image and "
            "the text information provided, prioritize the information from the image."
        )
    result = model.generate(
        text=prompt,
        image=image,
        tokens=tokens,
        return_attention=return_attention,
        return_logits=return_logits,
        dataset=dataset,
        unmatched=unmatched,
        modality=modality,
    )
    if not return_attention:
        result.attentions = None
    if not return_logits:
        result.scores = None
        result.token_probabilities = {}
    else:
        # Experiments persist only requested first-token scores. Keeping every
        # full-vocabulary generation score here retains GPU tensors for an
        # entire shift and eventually exhausts memory.
        result.scores = None
    return result.as_legacy_tuple(include_probabilities=p_yes_and_no)


def generate_text_batch(
    model,
    prompts,
    images,
    *,
    return_attention=False,
    return_logits=False,
    dataset=None,
    p_yes_and_no=False,
    unmatched=False,
    tokens=None,
    priority_img=False,
    modalities=None,
):
    """Generate prepared prompts through the model batch interface."""
    modalities = modalities or [None] * len(prompts)
    prepared_prompts = []
    for prompt, image, modality in zip(prompts, images, modalities):
        if priority_img and image is not None and modality != "Only_image":
            prompt += (
                "\n\nIMPORTANT: If there is a mismatch or conflict between the image and "
                "the text information provided, prioritize the information from the image."
            )
        prepared_prompts.append(prompt)
    results = model.generate_batch(
        prepared_prompts,
        images,
        tokens=tokens,
        return_attention=return_attention,
        return_logits=return_logits,
        contexts=[
            {"dataset": dataset, "unmatched": unmatched, "modality": modality}
            for modality in modalities
        ],
    )
    outputs = []
    for result in results:
        if not return_attention:
            result.attentions = None
        if not return_logits:
            result.token_probabilities = {}
        # Requested token scores have already been converted to Python floats.
        result.scores = None
        outputs.append(result.as_legacy_tuple(include_probabilities=p_yes_and_no))
    return outputs


def predict_dataset(
    metadata_row,
    model=None,
    processor=None,
    quantization=None,
    return_attention=False,
    return_logits=False,
    dataset="mimic",
    modality=None,
    only_prompt=False,
    p_yes_and_no=False,
    original=False,
    unmatched=False,
    history_cols_to_use=None,
    version="default",
    tokens=None,
    priority_img=False,
):
    """Build a dataset-specific prompt, then optionally run the supplied VLM."""
    if dataset in {"mimic_5class", "mimic_5class_test", "mimic_5class_debug"}:
        image = load_image(metadata_row["filepath"])
        prompt_map = {
            "Only_image": {
                "v1": MIMIC_ONLY_IMAGE_TEXT_PROMPT_5CLASS_V1,
                "v2": MIMIC_ONLY_IMAGE_TEXT_PROMPT_5CLASS_V2,
                "v3": MIMIC_ONLY_IMAGE_TEXT_PROMPT_5CLASS_V3,
                "default": MIMIC_ONLY_IMAGE_TEXT_PROMPT_5CLASS,
            },
            "Only_text": {
                "v1": MIMIC_ONLY_TEXT_PROMPT_5CLASS_V1,
                "v2": MIMIC_ONLY_TEXT_PROMPT_5CLASS_V2,
                "v3": MIMIC_ONLY_TEXT_PROMPT_5CLASS_V3,
                "default": MIMIC_ONLY_TEXT_PROMPT_5CLASS,
            },
            "both": {
                "v1": MIMIC_TEXT_PROMPT_5CLASS_V1,
                "v2": MIMIC_TEXT_PROMPT_5CLASS_V2,
                "v3": MIMIC_TEXT_PROMPT_5CLASS_V3,
                "default": MIMIC_TEXT_PROMPT_5CLASS,
            },
        }
        modality_key = modality if modality in {"Only_image", "Only_text"} else "both"
        version_key = version if version in {"v1", "v2", "v3"} else "default"
        prompt_func = prompt_map[modality_key][version_key]
        text_metadata = (
            prompt_func(unmatched=unmatched)
            if modality_key == "Only_image"
            else prompt_func(metadata_row, unmatched=unmatched)
        )

    elif dataset in {"mimic_race", "mimic_sex", "mimic_pathology_metadata"}:
        image = load_image(metadata_row["filepath"])
        target_config = {
            "mimic_race": ("race", True),
            "mimic_sex": ("sex", True),
            "mimic_pathology_metadata": ("class_label", False),
        }
        target, include_report = target_config[dataset]
        text_metadata = MIMIC_TARGET_PROMPT(
            metadata_row,
            target=target,
            labels=tokens,
            version=version,
            modality=modality,
            include_report=include_report,
        )

    elif "history" in dataset:
        image = load_image(metadata_row["filepath"])
        history_prompts = {
            "v1": CXR_HISTORY_TEXT_PROMPT_5CLASS_V1,
            "v2": CXR_HISTORY_TEXT_PROMPT_5CLASS_V2,
            "v3": CXR_HISTORY_TEXT_PROMPT_5CLASS_V3,
            "default": CXR_HISTORY_TEXT_PROMPT_5CLASS,
        }
        text_metadata = history_prompts.get(version, history_prompts["default"])(
            metadata_row,
            original=original,
            history_cols_to_use=history_cols_to_use,
            unmatched=unmatched,
        )

    elif dataset == "mbrset":
        image = load_image(metadata_row["filepath"])
        if modality == "Only_image":
            text_metadata = mBRSET_ONLY_IMAGE_TEXT_PROMPT
        elif modality == "Only_text":
            text_metadata = mBRSET_ONLY_TEXT_PROMPT(metadata_row)
        else:
            text_metadata = mBRSET_TEXT_PROMPT(metadata_row)

    elif dataset == "brset":
        image = load_image(metadata_row["filepath"])
        if modality == "Only_image":
            text_metadata = BRSET_ONLY_IMAGE_TEXT_PROMPT
        elif modality == "Only_text":
            text_metadata = BRSET_ONLY_TEXT_PROMPT(metadata_row)
        else:
            text_metadata = BRSET_TEXT_PROMPT(metadata_row)

    else:
        raise ValueError(f"Dataset not supported: {dataset}")

    if modality == "Only_text":
        image = None
    if only_prompt:
        return text_metadata, image
    return generate_text(
        model,
        processor,
        text_metadata,
        image,
        quantization,
        return_attention,
        return_logits,
        dataset=dataset,
        p_yes_and_no=p_yes_and_no,
        unmatched=unmatched,
        tokens=tokens,
        priority_img=priority_img,
        modality=modality,
    )


def predict_dataset_batch(
    metadata_rows,
    model,
    *,
    return_attention=False,
    return_logits=False,
    dataset="mimic",
    modalities=None,
    p_yes_and_no=False,
    originals=None,
    unmatched=False,
    history_cols_to_use=None,
    version="default",
    tokens=None,
    priority_img=False,
):
    """Prepare dataset prompts and run them through ``model.generate_batch``."""
    count = len(metadata_rows)
    modalities = modalities or [None] * count
    originals = originals or [False] * count
    history_cols_to_use = history_cols_to_use or [None] * count
    prepared = [
        predict_dataset(
            row,
            dataset=dataset,
            modality=modality,
            only_prompt=True,
            original=original,
            unmatched=unmatched,
            history_cols_to_use=history_columns,
            version=version,
            tokens=tokens,
        )
        for row, modality, original, history_columns in zip(
            metadata_rows, modalities, originals, history_cols_to_use
        )
    ]
    prompts, images = zip(*prepared) if prepared else ([], [])
    return generate_text_batch(
        model,
        list(prompts),
        list(images),
        return_attention=return_attention,
        return_logits=return_logits,
        dataset=dataset,
        p_yes_and_no=p_yes_and_no,
        unmatched=unmatched,
        tokens=tokens,
        priority_img=priority_img,
        modalities=modalities,
    )
