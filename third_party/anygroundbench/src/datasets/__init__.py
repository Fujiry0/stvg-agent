"""Dataset helper shim."""

import os

DATA_ROOT = "./data"

DOMAIN_DATASETS = {
    "animal": ["animal_kingdom", "mouse"],
    "industry": ["meccano", "enigma"],
    "surgery": ["egosurgery", "cholectrack20"],
    "safety": ["uca", "dota"],
    "sports": ["multisports", "american_football"],
}

DATASET_TO_DOMAIN = {
    dataset: domain
    for domain, datasets in DOMAIN_DATASETS.items()
    for dataset in datasets
}

DATASET_DIRS = {
    "american_football": "american_football",
    "mouse": "mouse",
    "enigma": "ENIGMA",
    "animal_kingdom": "animal_kingdom",
    "uca": "uca",
    "multisports": "MultiSports",
    "dota": "DoTA",
    "cholectrack20": "CholecTrack20",
    "meccano": "MECCANO",
    "egosurgery": "egosurgery",
}


def get_datasets_from_domain(domain: str) -> list[str]:
    return DOMAIN_DATASETS[domain]


def get_domain_from_dataset_name(dataset_name: str) -> str:
    return DATASET_TO_DOMAIN[dataset_name]


def get_temporal_info(dataset_name: str) -> dict[str, str]:
    dataset_dir = DATASET_DIRS[dataset_name]
    return {
        "annotation_dir": os.path.join(DATA_ROOT, dataset_dir, "simrank"),
        "video_data_dir": os.path.join(DATA_ROOT, dataset_dir, "videos"),
        "meta_data_dir": os.path.join(DATA_ROOT, dataset_dir, "meta-data"),
    }
