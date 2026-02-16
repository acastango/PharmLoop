"""
Phase 2 model construction: rebuild Hopfield in learned 512-dim space.

Steps:
  1. Load Phase 1 trained model
  2. Encode all training interaction pairs → 512-dim pair patterns
  3. Deduplicate bidirectional pairs
  4. Severity-amplify dangerous interactions (store extra noisy copies)
  5. Build new 512-dim Hopfield with identity-init projections
  6. Store patterns
  7. Build Phase 2 model and transfer Phase 1 weights
"""

import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from pharmloop.hopfield import PharmHopfield
from pharmloop.model import PharmLoopModel

logger = logging.getLogger("pharmloop.build_phase2")


def load_phase1_model(
    checkpoint_path: str | Path,
    drugs_path: str | Path,
) -> PharmLoopModel:
    """Load the Phase 1 trained model."""
    with open(drugs_path) as f:
        drugs_data = json.load(f)

    num_drugs = len(drugs_data["drugs"])
    drug_features_list = []
    for _name, info in sorted(drugs_data["drugs"].items(), key=lambda x: x[1]["id"]):
        drug_features_list.append(info["features"])
    drug_features = torch.tensor(drug_features_list, dtype=torch.float32)

    # Build Phase 1 Hopfield (phase0 mode) for model construction
    hopfield = PharmHopfield(input_dim=64, hidden_dim=512, phase0=True)
    hopfield.store(drug_features)
    for param in hopfield.parameters():
        param.requires_grad = False

    model = PharmLoopModel(num_drugs=num_drugs, hopfield=hopfield)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    logger.info(f"Loaded Phase 1 model from {checkpoint_path}")

    return model, drugs_data


def compute_all_pair_patterns(
    model: PharmLoopModel,
    drugs_data: dict,
    interactions_path: str | Path,
    train_indices: list[int] | None = None,
) -> tuple[Tensor, list[dict]]:
    """
    Compute pair patterns for all (training) interactions using the Phase 1 encoder.

    Returns:
        patterns: (N_unique, 512) tensor of deduplicated pair patterns.
        metadata: List of dicts with drug_a, drug_b, severity, mechanisms per pattern.
    """
    with open(interactions_path) as f:
        interactions_data = json.load(f)

    interactions = interactions_data["interactions"]
    if train_indices is not None:
        interactions = [interactions[i] for i in train_indices]

    drug_ids = {name: info["id"] for name, info in drugs_data["drugs"].items()}
    drug_features = {
        info["id"]: info["features"]
        for name, info in drugs_data["drugs"].items()
    }

    # Compute pair patterns and deduplicate
    seen_pairs: set[tuple[str, str]] = set()
    patterns_list: list[Tensor] = []
    metadata: list[dict] = []

    model.eval()
    with torch.no_grad():
        for inter in interactions:
            drug_a, drug_b = inter["drug_a"], inter["drug_b"]
            if drug_a not in drug_ids or drug_b not in drug_ids:
                continue

            # Deduplicate: canonical order (alphabetical)
            canonical = tuple(sorted([drug_a, drug_b]))
            if canonical in seen_pairs:
                continue
            seen_pairs.add(canonical)

            a_id = torch.tensor([drug_ids[drug_a]], dtype=torch.long)
            a_feat = torch.tensor([drug_features[drug_ids[drug_a]]], dtype=torch.float32)
            b_id = torch.tensor([drug_ids[drug_b]], dtype=torch.long)
            b_feat = torch.tensor([drug_features[drug_ids[drug_b]]], dtype=torch.float32)

            pair_state = model.compute_pair_state(a_id, a_feat, b_id, b_feat)
            patterns_list.append(pair_state.squeeze(0))

            metadata.append({
                "drug_a": drug_a,
                "drug_b": drug_b,
                "severity": inter["severity"],
                "mechanisms": inter.get("mechanisms", []),
            })

    patterns = torch.stack(patterns_list, dim=0)
    logger.info(f"Computed {patterns.shape[0]} unique pair patterns from {len(interactions)} interactions")
    return patterns, metadata


def severity_amplify(
    patterns: Tensor,
    metadata: list[dict],
    noise_scale: float = 0.01,
    extra_copies: int = 2,
) -> Tensor:
    """
    Amplify severe/contraindicated patterns by storing extra noisy copies.

    This gives dangerous interactions more Hopfield "mass" so they are
    retrieved more readily — DO NO HARM at the retrieval level.

    Returns:
        Augmented patterns tensor including original + amplified copies.
    """
    augmented = [patterns]
    for idx, meta in enumerate(metadata):
        if meta["severity"] in ("severe", "contraindicated"):
            pattern = patterns[idx]
            for _ in range(extra_copies):
                noisy = pattern + torch.randn_like(pattern) * noise_scale
                augmented.append(noisy.unsqueeze(0))

    result = torch.cat(augmented, dim=0)
    n_extra = result.shape[0] - patterns.shape[0]
    logger.info(f"Severity amplification: {n_extra} extra patterns for severe/contraindicated")
    return result


def build_phase2_model(
    phase1_checkpoint_path: str | Path,
    drugs_path: str | Path,
    interactions_path: str | Path,
    split_path: str | Path | None = None,
    device: torch.device = torch.device("cpu"),
) -> PharmLoopModel:
    """
    Build the Phase 2 model:
      1. Load Phase 1 trained encoder + oscillator + output head
      2. Build new 512-dim Hopfield from trained encoder pair patterns
      3. Remove dimension projection bottleneck (automatic)
      4. Initialize Hopfield projections as near-identity
      5. Transfer Phase 1 weights

    Args:
        phase1_checkpoint_path: Path to Phase 1 best_model.pt.
        drugs_path: Path to drugs.json.
        interactions_path: Path to interactions.json.
        split_path: Path to split.json (uses train indices for Hopfield patterns).
        device: Target device.

    Returns:
        Phase 2 PharmLoopModel ready for fine-tuning.
    """
    # Load Phase 1 model
    phase1_model, drugs_data = load_phase1_model(phase1_checkpoint_path, drugs_path)

    # Get train indices from split
    train_indices = None
    if split_path is not None:
        with open(split_path) as f:
            split = json.load(f)
        train_indices = split["train_indices"]

    # Compute pair patterns from Phase 1 encoder
    patterns, metadata = compute_all_pair_patterns(
        phase1_model, drugs_data, interactions_path, train_indices=train_indices,
    )

    # Severity amplification
    augmented_patterns = severity_amplify(patterns, metadata)

    # Build Phase 2 Hopfield in 512-dim learned space
    hopfield_v2 = PharmHopfield(input_dim=512, hidden_dim=512, phase0=False)

    # Identity-init projections (preserve meaningful structure of stored patterns)
    with torch.no_grad():
        nn.init.eye_(hopfield_v2.query_proj.weight)
        nn.init.zeros_(hopfield_v2.query_proj.bias)
        nn.init.eye_(hopfield_v2.key_proj.weight)
        nn.init.zeros_(hopfield_v2.key_proj.bias)

    # Store patterns
    hopfield_v2.store(augmented_patterns.detach())
    logger.info(
        f"Phase 2 Hopfield: {hopfield_v2.count} patterns in 512-dim space "
        f"(from {patterns.shape[0]} unique pairs + severity amplification)"
    )

    # Build Phase 2 model (no dimension projection since hopfield.input_dim == state_dim)
    model = PharmLoopModel(
        num_drugs=phase1_model.num_drugs,
        hopfield=hopfield_v2,
    )

    # Transfer Phase 1 weights
    _transfer_phase1_weights(model, phase1_model)

    model = model.to(device)
    counts = model.count_parameters()
    logger.info(f"Phase 2 model — learned: {counts['learned']:,}, buffers: {counts['buffers']:,}, total: {counts['total']:,}")

    return model


def _transfer_phase1_weights(
    phase2_model: PharmLoopModel,
    phase1_model: PharmLoopModel,
) -> None:
    """
    Transfer all learned weights from Phase 1 to Phase 2.

    Transfers encoder, pair_combine, oscillator dynamics, output head.
    Skips: old Hopfield, old hopfield_query_proj/hopfield_value_proj.
    """
    transferred = []
    skipped = []

    phase1_state = phase1_model.state_dict()
    phase2_state = phase2_model.state_dict()

    for key in phase1_state:
        # Skip Phase 1 Hopfield buffers and parameters
        if "hopfield" in key and "cell.hopfield" in key:
            # These are the Hopfield memory contents — skip (replaced)
            skipped.append(key)
            continue

        # Skip old dimension projection layers (removed in Phase 2)
        if "hopfield_query_proj" in key or "hopfield_value_proj" in key:
            skipped.append(key)
            continue

        if key in phase2_state and phase1_state[key].shape == phase2_state[key].shape:
            phase2_state[key] = phase1_state[key]
            transferred.append(key)
        else:
            skipped.append(key)

    phase2_model.load_state_dict(phase2_state, strict=False)
    logger.info(f"Weight transfer: {len(transferred)} transferred, {len(skipped)} skipped")
    if skipped:
        logger.debug(f"Skipped keys: {skipped}")


def rebuild_hopfield_from_current_encoder(
    model: PharmLoopModel,
    drugs_data: dict,
    interactions_path: str | Path,
    train_indices: list[int] | None = None,
) -> Tensor:
    """
    Re-encode all interaction pairs with the CURRENT encoder and rebuild Hopfield.

    Used during annealing cycles: the encoder has shifted during training, so
    stored patterns are stale. This refreshes them.

    Returns:
        Old patterns tensor (for drift measurement).
    """
    # Save old patterns for drift measurement
    old_count = model.cell.hopfield.count
    old_patterns = model.cell.hopfield.stored_values[:old_count].clone()

    # Compute fresh pair patterns
    patterns, metadata = compute_all_pair_patterns(
        model, drugs_data, interactions_path, train_indices=train_indices,
    )
    augmented = severity_amplify(patterns, metadata)

    # Clear and re-store
    model.cell.hopfield.clear()

    # Re-initialize projections from current state (don't reset to identity)
    model.cell.hopfield.store(augmented.detach())

    logger.info(f"Hopfield rebuilt: {model.cell.hopfield.count} patterns from current encoder")
    return old_patterns


def compute_pattern_drift(old_patterns: Tensor, new_patterns: Tensor) -> float:
    """Measure average L2 distance between old and new patterns (for annealing stopping)."""
    n = min(old_patterns.shape[0], new_patterns.shape[0])
    drift = (old_patterns[:n] - new_patterns[:n]).norm(dim=-1).mean().item()
    return drift


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")

    data_dir = Path("data/processed")
    model = build_phase2_model(
        phase1_checkpoint_path="checkpoints/best_model.pt",
        drugs_path=data_dir / "drugs.json",
        interactions_path=data_dir / "interactions.json",
        split_path=data_dir / "split.json",
    )
    print(f"Phase 2 model built. Hopfield patterns: {model.cell.hopfield.count}")

    # Save initial Phase 2 checkpoint
    Path("checkpoints").mkdir(exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "phase": 2,
        "annealing_cycle": 0,
    }, "checkpoints/phase2_initial.pt")
    print("Saved checkpoints/phase2_initial.pt")
