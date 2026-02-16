"""
Phase 2 training loop: end-to-end fine-tuning with Hopfield in learned 512-dim space.

Key differences from Phase 1:
  - Lower learning rate (1e-4 vs 1e-3)
  - Dual param groups (Hopfield projections learn faster)
  - Higher convergence weight (0.7 vs 0.5)
  - Validation loss tracking
  - Hopfield retrieval quality logging
  - Annealing cycles (rebuild Hopfield from updated encoder)
"""

import json
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from pharmloop.model import PharmLoopModel
from training.build_phase2 import (
    build_phase2_model,
    rebuild_hopfield_from_current_encoder,
    compute_pattern_drift,
)
from training.data_loader import create_dataloader
from training.loss import PharmLoopLoss

# Try to load .env if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger("pharmloop.train_phase2")


def _make_optimizer(
    model: PharmLoopModel,
    hopfield_lr: float = 5e-4,
    other_lr: float = 1e-4,
) -> Adam:
    """Create optimizer with separate LR for Hopfield projections."""
    hopfield_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "hopfield" in name and ("query_proj" in name or "key_proj" in name):
            hopfield_params.append(param)
        else:
            other_params.append(param)

    return Adam([
        {"params": hopfield_params, "lr": hopfield_lr},
        {"params": other_params, "lr": other_lr},
    ])


def _log_retrieval_diagnostics(
    model: PharmLoopModel,
    drugs_data: dict,
    diagnostic_pairs: list[tuple[str, str]],
) -> None:
    """Log Hopfield retrieval quality for a set of diagnostic pairs."""
    model.eval()
    drug_ids = {name: info["id"] for name, info in drugs_data["drugs"].items()}
    drug_features = {
        info["id"]: info["features"]
        for name, info in drugs_data["drugs"].items()
    }

    with torch.no_grad():
        for drug_a, drug_b in diagnostic_pairs:
            if drug_a not in drug_ids or drug_b not in drug_ids:
                continue
            a_id = torch.tensor([drug_ids[drug_a]], dtype=torch.long)
            a_feat = torch.tensor([drug_features[drug_ids[drug_a]]], dtype=torch.float32)
            b_id = torch.tensor([drug_ids[drug_b]], dtype=torch.long)
            b_feat = torch.tensor([drug_features[drug_ids[drug_b]]], dtype=torch.float32)

            query = model.compute_pair_state(a_id, a_feat, b_id, b_feat)
            retrieved = model.cell.hopfield.retrieve(query, beta=1.0)
            cosine = F.cosine_similarity(query, retrieved).item()
            logger.info(f"  Retrieval {drug_a}+{drug_b}: cosine={cosine:.3f}")


def _compute_retrieval_entropy(model: PharmLoopModel) -> float:
    """Compute normalized retrieval entropy to monitor for collapse."""
    hopfield = model.cell.hopfield
    n = hopfield.count
    if n == 0:
        return 0.0

    keys = hopfield.stored_keys[:n]
    # Use stored values as queries (self-retrieval check)
    values = hopfield.stored_values[:n]
    q = hopfield.query_proj(values)
    scores = q @ keys.T
    weights = torch.softmax(scores, dim=-1)
    entropy = -(weights * (weights + 1e-10).log()).sum(dim=-1).mean()
    max_entropy = torch.log(torch.tensor(float(n)))
    return (entropy / max_entropy).item()


def _run_epoch(
    model: PharmLoopModel,
    dataloader,
    criterion: PharmLoopLoss,
    optimizer: Adam | None,
    device: torch.device,
    training: bool = True,
) -> dict[str, float]:
    """Run one training or validation epoch."""
    if training:
        model.train()
    else:
        model.eval()

    epoch_losses = {"total": 0.0, "answer": 0.0, "convergence": 0.0, "smoothness": 0.0, "do_no_harm": 0.0}
    num_batches = 0
    converged_count = 0
    total_count = 0
    total_steps = 0

    for batch in dataloader:
        drug_a_id = batch["drug_a_id"].to(device)
        drug_a_features = batch["drug_a_features"].to(device)
        drug_b_id = batch["drug_b_id"].to(device)
        drug_b_features = batch["drug_b_features"].to(device)
        target_severity = batch["target_severity"].to(device)
        target_mechanisms = batch["target_mechanisms"].to(device)
        target_flags = batch["target_flags"].to(device)
        is_unknown = batch["is_unknown"].to(device)

        if training:
            output = model(drug_a_id, drug_a_features, drug_b_id, drug_b_features)
        else:
            with torch.no_grad():
                output = model(drug_a_id, drug_a_features, drug_b_id, drug_b_features)

        losses = criterion(
            severity_logits=output["severity_logits"],
            mechanism_logits=output["mechanism_logits"],
            flag_logits=output["flag_logits"],
            final_v=output["trajectory"]["velocities"][-1],
            velocities=output["trajectory"]["velocities"],
            target_severity=target_severity,
            target_mechanisms=target_mechanisms,
            target_flags=target_flags,
            is_unknown=is_unknown,
        )

        if training:
            optimizer.zero_grad()
            losses["total"].backward()
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            optimizer.step()

        for key in epoch_losses:
            epoch_losses[key] += losses[key].item()
        num_batches += 1
        converged_count += output["converged"].sum().item()
        total_count += drug_a_id.shape[0]
        total_steps += output["trajectory"]["steps"]

    for key in epoch_losses:
        epoch_losses[key] /= max(num_batches, 1)

    epoch_losses["convergence_rate"] = converged_count / max(total_count, 1)
    epoch_losses["avg_steps"] = total_steps / max(num_batches, 1)

    return epoch_losses


def train_phase2(
    data_dir: str = "./data/processed",
    checkpoint_dir: str = "./checkpoints",
    phase1_checkpoint: str = "./checkpoints/best_model.pt",
    epochs_per_cycle: list[int] | None = None,
    batch_size: int = 32,
    hopfield_lr: float = 5e-4,
    other_lr: float = 1e-4,
    device_str: str = "cpu",
    seed: int | None = 42,
    log_level: str = "INFO",
    drift_threshold: float = 0.01,
) -> PharmLoopModel:
    """
    Run Phase 2 training with annealing cycles.

    Args:
        data_dir: Directory containing drugs.json, interactions.json, split.json.
        checkpoint_dir: Directory for model checkpoints.
        phase1_checkpoint: Path to Phase 1 best model.
        epochs_per_cycle: Epochs for each annealing cycle (default [30, 15, 10]).
        batch_size: Training batch size.
        hopfield_lr: Learning rate for Hopfield projections.
        other_lr: Learning rate for everything else.
        device_str: PyTorch device string.
        seed: Random seed.
        log_level: Logging level.
        drift_threshold: Stop annealing when pattern drift falls below this.

    Returns:
        Trained Phase 2 model.
    """
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    if epochs_per_cycle is None:
        epochs_per_cycle = [30, 15, 10]

    if seed is not None:
        torch.manual_seed(seed)
        logger.info(f"Random seed set to {seed}")

    device = torch.device(device_str)
    logger.info(f"Training on device: {device}")

    # Paths
    data_path = Path(data_dir)
    drugs_path = data_path / "drugs.json"
    interactions_path = data_path / "interactions.json"
    split_path = data_path / "split.json"

    # Load split
    with open(split_path) as f:
        split = json.load(f)

    # Load drugs data (needed for diagnostics)
    with open(drugs_path) as f:
        drugs_data = json.load(f)

    # Dataloaders
    train_loader = create_dataloader(
        drugs_path, interactions_path,
        batch_size=batch_size, shuffle=True,
        split_indices=split["train_indices"],
    )
    val_loader = create_dataloader(
        drugs_path, interactions_path,
        batch_size=batch_size, shuffle=False,
        split_indices=split["val_indices"],
    )
    logger.info(f"Train: {len(train_loader.dataset)} samples, Val: {len(val_loader.dataset)} samples")

    # Diagnostic pairs for retrieval logging
    diagnostic_pairs = [
        ("fluoxetine", "tramadol"),
        ("metformin", "lisinopril"),
        ("warfarin", "aspirin"),
        ("simvastatin", "erythromycin"),
    ]

    # Build Phase 2 model
    model = build_phase2_model(
        phase1_checkpoint, drugs_path, interactions_path, split_path, device,
    )

    # Loss with higher convergence weight
    criterion = PharmLoopLoss(convergence_weight=0.7, smoothness_weight=0.1)

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    global_epoch = 0

    # Annealing cycles
    for cycle, cycle_epochs in enumerate(epochs_per_cycle):
        logger.info(f"\n{'='*60}")
        logger.info(f"Annealing Cycle {cycle + 1}/{len(epochs_per_cycle)} — {cycle_epochs} epochs")
        logger.info(f"{'='*60}")

        if cycle > 0:
            # Rebuild Hopfield from current encoder
            model.eval()
            old_patterns = rebuild_hopfield_from_current_encoder(
                model, drugs_data, interactions_path,
                train_indices=split["train_indices"],
            )
            new_patterns = model.cell.hopfield.stored_values[:model.cell.hopfield.count]
            drift = compute_pattern_drift(old_patterns, new_patterns)
            logger.info(f"Pattern drift: {drift:.4f} (threshold: {drift_threshold})")

            if drift < drift_threshold:
                logger.info("Pattern drift below threshold — stopping annealing early")
                break

        # Fresh optimizer for each cycle
        optimizer = _make_optimizer(model, hopfield_lr=hopfield_lr, other_lr=other_lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=cycle_epochs)

        for epoch in range(1, cycle_epochs + 1):
            global_epoch += 1

            # Train
            train_metrics = _run_epoch(model, train_loader, criterion, optimizer, device, training=True)
            scheduler.step()

            # Validate
            val_metrics = _run_epoch(model, val_loader, criterion, None, device, training=False)

            logger.info(
                f"Epoch {global_epoch} (cycle {cycle+1}, ep {epoch}/{cycle_epochs}) — "
                f"train_loss: {train_metrics['total']:.4f} "
                f"(ans: {train_metrics['answer']:.4f}, "
                f"conv: {train_metrics['convergence']:.4f}, "
                f"harm: {train_metrics['do_no_harm']:.4f}) — "
                f"val_loss: {val_metrics['total']:.4f} — "
                f"train_conv: {train_metrics['convergence_rate']:.1%} "
                f"val_conv: {val_metrics['convergence_rate']:.1%}"
            )

            # Check for NaN
            if any(torch.isnan(torch.tensor(v)) for v in train_metrics.values()):
                logger.error(f"NaN detected at epoch {global_epoch}! Stopping.")
                break

            # Save best model (by validation loss)
            if val_metrics["total"] < best_val_loss:
                best_val_loss = val_metrics["total"]
                torch.save({
                    "epoch": global_epoch,
                    "cycle": cycle + 1,
                    "model_state_dict": model.state_dict(),
                    "val_loss": best_val_loss,
                    "phase": 2,
                }, Path(checkpoint_dir) / "best_model_phase2.pt")
                logger.info(f"  Saved best Phase 2 model (val_loss={best_val_loss:.4f})")

        # End-of-cycle diagnostics
        logger.info(f"\nCycle {cycle + 1} diagnostics:")
        _log_retrieval_diagnostics(model, drugs_data, diagnostic_pairs)
        entropy = _compute_retrieval_entropy(model)
        logger.info(f"  Retrieval entropy (normalized): {entropy:.3f}")

    # Save final model
    torch.save({
        "epoch": global_epoch,
        "model_state_dict": model.state_dict(),
        "val_loss": best_val_loss,
        "phase": 2,
        "annealing_cycles_completed": min(cycle + 1, len(epochs_per_cycle)),
    }, Path(checkpoint_dir) / "final_model_phase2.pt")
    logger.info(f"\nPhase 2 training complete. Total epochs: {global_epoch}")

    return model


def main() -> None:
    """Entry point using environment variables for configuration."""
    train_phase2(
        data_dir=os.getenv("PHARMLOOP_DATA_DIR", "./data/processed"),
        checkpoint_dir=os.getenv("PHARMLOOP_CHECKPOINT_DIR", "./checkpoints"),
        phase1_checkpoint=os.getenv("PHARMLOOP_PHASE1_CHECKPOINT", "./checkpoints/best_model.pt"),
        batch_size=int(os.getenv("PHARMLOOP_BATCH_SIZE", "32")),
        hopfield_lr=float(os.getenv("PHARMLOOP_HOPFIELD_LR", "5e-4")),
        other_lr=float(os.getenv("PHARMLOOP_LEARNING_RATE", "1e-4")),
        device_str=os.getenv("PHARMLOOP_DEVICE", "cpu"),
        seed=int(os.getenv("PHARMLOOP_SEED", "42")) if os.getenv("PHARMLOOP_SEED") else 42,
        log_level=os.getenv("PHARMLOOP_LOG_LEVEL", "INFO"),
    )


if __name__ == "__main__":
    main()
