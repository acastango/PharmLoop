"""
Create deterministic train/val/test split for PharmLoop interactions.

Split rules (from phase2.md):
  - 80/10/10 split by interaction index
  - Stratified by severity (proportional representation in each split)
  - Both drugs in a test pair must appear in at least one training pair
  - fluoxetine+tramadol and metformin+lisinopril forced into test set
  - Fabricated drug tests don't use the dataset

Saves split.json with train/val/test indices and metadata.
"""

import json
import random
from collections import defaultdict
from pathlib import Path


# Pairs that MUST be in the test set (for three-way comparison continuity)
FORCED_TEST_PAIRS = {
    ("fluoxetine", "tramadol"),
    ("metformin", "lisinopril"),
}


def create_split(
    interactions_path: str | Path,
    output_path: str | Path,
    seed: int = 42,
    train_frac: float = 0.80,
    val_frac: float = 0.10,
) -> dict:
    """
    Create a stratified train/val/test split.

    Args:
        interactions_path: Path to interactions.json.
        output_path: Path to write split.json.
        seed: Random seed for reproducibility.
        train_frac: Fraction for training.
        val_frac: Fraction for validation (rest goes to test).

    Returns:
        Split dictionary with train/val/test indices.
    """
    with open(interactions_path) as f:
        data = json.load(f)

    interactions = data["interactions"]

    # Group indices by severity for stratification
    severity_groups: dict[str, list[int]] = defaultdict(list)
    forced_test_indices: set[int] = set()

    for idx, inter in enumerate(interactions):
        pair = (inter["drug_a"], inter["drug_b"])
        pair_rev = (inter["drug_b"], inter["drug_a"])

        # Check if this pair is forced into test
        if pair in FORCED_TEST_PAIRS or pair_rev in FORCED_TEST_PAIRS:
            forced_test_indices.add(idx)
        else:
            severity_groups[inter["severity"]].append(idx)

    # Stratified split within each severity group
    rng = random.Random(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = list(forced_test_indices)

    for severity, indices in sorted(severity_groups.items()):
        rng.shuffle(indices)
        n = len(indices)
        n_train = max(1, int(n * train_frac))
        n_val = max(1, int(n * val_frac))
        # Ensure at least 1 in each split for this severity
        n_test = n - n_train - n_val
        if n_test < 0:
            n_train = n - n_val
            n_test = 0

        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train:n_train + n_val])
        test_indices.extend(indices[n_train + n_val:])

    # Verify: both drugs in every test pair appear in at least one training pair
    train_drugs = set()
    for idx in train_indices:
        inter = interactions[idx]
        train_drugs.add(inter["drug_a"])
        train_drugs.add(inter["drug_b"])

    # Move test pairs to train if their drugs are missing from training
    moved = []
    for idx in list(test_indices):
        if idx in forced_test_indices:
            continue  # never move forced test pairs
        inter = interactions[idx]
        if inter["drug_a"] not in train_drugs or inter["drug_b"] not in train_drugs:
            test_indices.remove(idx)
            train_indices.append(idx)
            train_drugs.add(inter["drug_a"])
            train_drugs.add(inter["drug_b"])
            moved.append(idx)

    # Same for val
    for idx in list(val_indices):
        inter = interactions[idx]
        if inter["drug_a"] not in train_drugs or inter["drug_b"] not in train_drugs:
            val_indices.remove(idx)
            train_indices.append(idx)
            train_drugs.add(inter["drug_a"])
            train_drugs.add(inter["drug_b"])
            moved.append(idx)

    # Sort for determinism
    train_indices.sort()
    val_indices.sort()
    test_indices.sort()

    # Build summary
    split_summary = {
        "train": _count_severities(interactions, train_indices),
        "val": _count_severities(interactions, val_indices),
        "test": _count_severities(interactions, test_indices),
    }

    split = {
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
        "seed": seed,
        "strategy": "stratified_by_severity",
        "forced_test_pairs": [list(p) for p in FORCED_TEST_PAIRS],
        "summary": split_summary,
    }

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(split, f, indent=2)

    return split


def _count_severities(interactions: list[dict], indices: list[int]) -> dict[str, int]:
    """Count severity classes for a set of indices."""
    counts: dict[str, int] = defaultdict(int)
    for idx in indices:
        counts[interactions[idx]["severity"]] += 1
    return dict(counts)


def main() -> None:
    """Generate the split file."""
    data_dir = Path("data/processed")
    split = create_split(
        interactions_path=data_dir / "interactions.json",
        output_path=data_dir / "split.json",
    )

    print(f"Split created:")
    print(f"  Train: {len(split['train_indices'])} samples")
    print(f"  Val:   {len(split['val_indices'])} samples")
    print(f"  Test:  {len(split['test_indices'])} samples")
    print(f"  Summary: {json.dumps(split['summary'], indent=4)}")


if __name__ == "__main__":
    main()
