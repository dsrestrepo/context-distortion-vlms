import pandas as pd

from src.datasets import load_brset, load_mbrset, load_mimic


MIMIC_DISEASE_COLUMNS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
]

MIMIC_5CLASS_TARGETS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]


def prepare_mimic_5class(samples_per_group=500, random_seed=42):
    metadata = load_mimic(train=False, validation=False, check_images=False).copy()
    disease_columns = [
        column for column in MIMIC_DISEASE_COLUMNS if column != "No Finding"
    ]
    metadata["num_disease_labels"] = metadata[disease_columns].sum(axis=1)

    single_disease = metadata[metadata["num_disease_labels"] == 1].copy()
    target_disease = single_disease[
        single_disease[MIMIC_5CLASS_TARGETS].sum(axis=1) == 1
    ]
    healthy = metadata[
        (metadata["No Finding"] == 1) & (metadata["num_disease_labels"] == 0)
    ]

    disease_sample = target_disease.sample(
        n=min(samples_per_group, len(target_disease)), random_state=random_seed
    )
    healthy_sample = healthy.sample(
        n=min(samples_per_group, len(healthy)), random_state=random_seed
    )
    print(f"Final Disease Dataset Size: {len(disease_sample)}")
    print(f"Final Healthy Dataset Size: {len(healthy_sample)}")
    return pd.concat([disease_sample, healthy_sample])


def load_history_experiment(path):
    return pd.read_csv(path)


def load_brset_experiments(samples_per_class=4000):
    mbrset = load_mbrset(train=False, validation=False, check_images=False)
    brset = load_brset(train=False, validation=False, check_images=False)
    brset = brset[brset.groupby("DR").cumcount() < samples_per_class]
    return mbrset, brset
