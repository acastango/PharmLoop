"""
Dataset and data loading for PharmLoop training.

Loads drug features and interaction pairs from JSON files.
Creates batches of (drug_a, drug_b, target) triples.
Includes negative sampling and fabricated drug injection.
"""

import json
import random
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from pharmloop.output import (
    SEVERITY_NAMES, SEVERITY_UNKNOWN, NUM_MECHANISMS, NUM_FLAGS,
    MECHANISM_NAMES, FLAG_NAMES,
)


class DrugInteractionDataset(Dataset):
    """
    Dataset of drug interaction pairs for PharmLoop training.

    Each sample is a tuple of:
      (drug_a_id, drug_a_features, drug_b_id, drug_b_features,
       target_severity, target_mechanisms, target_flags, is_unknown)

    Args:
        drugs_path: Path to drugs.json with drug features.
        interactions_path: Path to interactions.json with interaction pairs.
        fabricated_ratio: Fraction of fabricated (unknown) drug pairs to inject per epoch.
        feature_dim: Expected feature vector dimension.
        split_indices: If provided, only use interactions at these indices.
    """

    def __init__(
        self,
        drugs_path: str | Path,
        interactions_path: str | Path,
        fabricated_ratio: float = 0.15,
        feature_dim: int = 64,
        split_indices: list[int] | None = None,
    ) -> None:
        self.feature_dim = feature_dim
        self.fabricated_ratio = fabricated_ratio

        # Load drug data
        with open(drugs_path) as f:
            drugs_data = json.load(f)

        self.drug_names: list[str] = []
        self.drug_ids: dict[str, int] = {}
        self.drug_features: dict[int, list[float]] = {}

        for name, info in drugs_data["drugs"].items():
            drug_id = info["id"]
            self.drug_names.append(name)
            self.drug_ids[name] = drug_id
            self.drug_features[drug_id] = info["features"]

        self.num_drugs = len(self.drug_names)

        # Load interaction data
        with open(interactions_path) as f:
            interactions_data = json.load(f)

        self.mechanism_vocab = interactions_data["metadata"]["mechanism_vocabulary"]
        self.severity_classes = interactions_data["metadata"]["severity_classes"]

        # Build mechanism and flag name-to-index maps
        self.mechanism_to_idx = {name: i for i, name in enumerate(MECHANISM_NAMES)}
        self.flag_to_idx = {name: i for i, name in enumerate(FLAG_NAMES)}

        # Parse interactions into samples (optionally filtered by split indices)
        all_interactions = interactions_data["interactions"]
        if split_indices is not None:
            indexed_interactions = [(i, all_interactions[i]) for i in split_indices]
        else:
            indexed_interactions = list(enumerate(all_interactions))

        self.samples: list[dict] = []
        for _idx, inter in indexed_interactions:
            drug_a = inter["drug_a"]
            drug_b = inter["drug_b"]

            if drug_a not in self.drug_ids or drug_b not in self.drug_ids:
                continue

            severity_idx = SEVERITY_NAMES.index(inter["severity"]) if inter["severity"] in SEVERITY_NAMES else SEVERITY_UNKNOWN

            # Mechanism multi-label vector
            mechanisms = torch.zeros(NUM_MECHANISMS)
            for mech in inter.get("mechanisms", []):
                if mech in self.mechanism_to_idx:
                    mechanisms[self.mechanism_to_idx[mech]] = 1.0

            # Flag multi-label vector
            flags = torch.zeros(NUM_FLAGS)
            for flag in inter.get("flags", []):
                if flag in self.flag_to_idx:
                    flags[self.flag_to_idx[flag]] = 1.0

            self.samples.append({
                "drug_a_id": self.drug_ids[drug_a],
                "drug_b_id": self.drug_ids[drug_b],
                "drug_a_features": self.drug_features[self.drug_ids[drug_a]],
                "drug_b_features": self.drug_features[self.drug_ids[drug_b]],
                "severity": severity_idx,
                "mechanisms": mechanisms,
                "flags": flags,
                "is_unknown": False,
            })

        # Add fabricated drug samples (unknown pairs)
        num_fabricated = max(1, int(len(self.samples) * self.fabricated_ratio))
        # Fabricated drug ID is beyond the known vocabulary
        fabricated_id = self.num_drugs + 50  # well into the padding zone
        for _ in range(num_fabricated):
            # Pair fabricated drug with a random real drug
            real_drug = random.choice(self.drug_names)
            real_id = self.drug_ids[real_drug]

            self.samples.append({
                "drug_a_id": fabricated_id,
                "drug_b_id": real_id,
                "drug_a_features": [random.uniform(0, 0.5) for _ in range(self.feature_dim)],
                "drug_b_features": self.drug_features[real_id],
                "severity": SEVERITY_UNKNOWN,
                "mechanisms": torch.zeros(NUM_MECHANISMS),
                "flags": torch.zeros(NUM_FLAGS),
                "is_unknown": True,
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        sample = self.samples[idx]
        return {
            "drug_a_id": torch.tensor(sample["drug_a_id"], dtype=torch.long),
            "drug_a_features": torch.tensor(sample["drug_a_features"], dtype=torch.float32),
            "drug_b_id": torch.tensor(sample["drug_b_id"], dtype=torch.long),
            "drug_b_features": torch.tensor(sample["drug_b_features"], dtype=torch.float32),
            "target_severity": torch.tensor(sample["severity"], dtype=torch.long),
            "target_mechanisms": sample["mechanisms"] if isinstance(sample["mechanisms"], Tensor) else torch.tensor(sample["mechanisms"], dtype=torch.float32),
            "target_flags": sample["flags"] if isinstance(sample["flags"], Tensor) else torch.tensor(sample["flags"], dtype=torch.float32),
            "is_unknown": torch.tensor(sample["is_unknown"], dtype=torch.bool),
        }


def create_dataloader(
    drugs_path: str | Path,
    interactions_path: str | Path,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    split_indices: list[int] | None = None,
) -> DataLoader:
    """Create a DataLoader for PharmLoop training.

    Args:
        split_indices: If provided, only include interactions at these indices.
    """
    dataset = DrugInteractionDataset(
        drugs_path, interactions_path, split_indices=split_indices,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )
