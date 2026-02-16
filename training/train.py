"""
Phase 1 training loop for PharmLoop.

1. Initialize Hopfield from raw 64-dim features (Phase 0 bootstrap).
2. Freeze Hopfield.
3. Train encoder + oscillator + output head.
4. Log loss components, convergence rates, gray zone trajectories.
5. Save checkpoints.
"""

import json
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from pharmloop.hopfield import PharmHopfield
from pharmloop.model import PharmLoopModel
from pharmloop.output import SEVERITY_NAMES
from training.data_loader import create_dataloader
from training.loss import PharmLoopLoss

# Try to load .env if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger("pharmloop.train")


def load_drug_features(drugs_path: str | Path) -> torch.Tensor:
    """Load all 64-dim drug feature vectors as a tensor for Hopfield initialization."""
    with open(drugs_path) as f:
        data = json.load(f)

    features_list = []
    for _name, info in sorted(data["drugs"].items(), key=lambda x: x[1]["id"]):
        features_list.append(info["features"])

    return torch.tensor(features_list, dtype=torch.float32)


def build_model(
    num_drugs: int,
    drug_features: torch.Tensor,
    device: torch.device,
) -> PharmLoopModel:
    """
    Build PharmLoopModel with Phase 0 Hopfield initialization.

    Phase 0: Hopfield bank is built from raw 64-dim feature vectors,
    then frozen. The oscillator learns against stable retrieval targets.
    """
    # Phase 0: create Hopfield in raw 64-dim feature space (no learned projections)
    hopfield = PharmHopfield(input_dim=64, hidden_dim=512, phase0=True)
    hopfield.store(drug_features)
    logger.info(f"Hopfield initialized with {hopfield.count} patterns in 64-dim space (phase0 mode)")

    # Freeze Hopfield — it's a stable reference, not a learnable component in Phase 1
    for param in hopfield.parameters():
        param.requires_grad = False

    # Build full model
    model = PharmLoopModel(
        num_drugs=num_drugs,
        feature_dim=64,
        hopfield=hopfield,
    )
    model = model.to(device)

    # Log parameter counts
    counts = model.count_parameters()
    logger.info(f"Model parameters — learned: {counts['learned']:,}, buffers: {counts['buffers']:,}, total: {counts['total']:,}")

    return model


def train(
    data_dir: str = "./data/processed",
    checkpoint_dir: str = "./checkpoints",
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    device_str: str = "cpu",
    seed: int | None = 42,
    log_level: str = "INFO",
) -> PharmLoopModel:
    """
    Run Phase 1 training.

    Args:
        data_dir: Directory containing drugs.json and interactions.json.
        checkpoint_dir: Directory for saving model checkpoints.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        lr: Learning rate.
        device_str: PyTorch device string.
        seed: Random seed (None for no fixed seed).
        log_level: Logging level string.

    Returns:
        Trained PharmLoopModel.
    """
    logging.basicConfig(level=getattr(logging, log_level), format="%(asctime)s %(name)s %(levelname)s: %(message)s")

    if seed is not None:
        torch.manual_seed(seed)
        logger.info(f"Random seed set to {seed}")

    device = torch.device(device_str)
    logger.info(f"Training on device: {device}")

    # Paths
    drugs_path = Path(data_dir) / "drugs.json"
    interactions_path = Path(data_dir) / "interactions.json"

    # Load data
    drug_features = load_drug_features(drugs_path)
    num_drugs = drug_features.shape[0]
    logger.info(f"Loaded {num_drugs} drugs with {drug_features.shape[1]}-dim features")

    dataloader = create_dataloader(
        drugs_path, interactions_path,
        batch_size=batch_size, shuffle=True,
    )
    logger.info(f"Dataset: {len(dataloader.dataset)} samples, {len(dataloader)} batches per epoch")

    # Build model
    model = build_model(num_drugs, drug_features, device)

    # Optimizer (only train non-frozen params)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(trainable_params, lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # Loss
    criterion = PharmLoopLoss()

    # Checkpoint directory
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Training loop
    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = {"total": 0.0, "answer": 0.0, "convergence": 0.0, "smoothness": 0.0, "do_no_harm": 0.0}
        num_batches = 0
        converged_count = 0
        total_count = 0

        for batch in dataloader:
            # Move to device
            drug_a_id = batch["drug_a_id"].to(device)
            drug_a_features = batch["drug_a_features"].to(device)
            drug_b_id = batch["drug_b_id"].to(device)
            drug_b_features = batch["drug_b_features"].to(device)
            target_severity = batch["target_severity"].to(device)
            target_mechanisms = batch["target_mechanisms"].to(device)
            target_flags = batch["target_flags"].to(device)
            is_unknown = batch["is_unknown"].to(device)

            # Forward pass
            output = model(drug_a_id, drug_a_features, drug_b_id, drug_b_features)

            # Loss — pass final_v and velocities tensors (carry gradients for convergence)
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

            # Backward
            optimizer.zero_grad()
            losses["total"].backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)

            optimizer.step()

            # Accumulate metrics
            for key in epoch_losses:
                epoch_losses[key] += losses[key].item()
            num_batches += 1
            converged_count += output["converged"].sum().item()
            total_count += drug_a_id.shape[0]

        scheduler.step()

        # Epoch averages
        for key in epoch_losses:
            epoch_losses[key] /= max(num_batches, 1)

        convergence_rate = converged_count / max(total_count, 1)

        logger.info(
            f"Epoch {epoch}/{epochs} — "
            f"loss: {epoch_losses['total']:.4f} "
            f"(answer: {epoch_losses['answer']:.4f}, "
            f"conv: {epoch_losses['convergence']:.4f}, "
            f"smooth: {epoch_losses['smoothness']:.4f}, "
            f"harm: {epoch_losses['do_no_harm']:.4f}) — "
            f"convergence rate: {convergence_rate:.2%}"
        )

        # Check for NaN
        if any(torch.isnan(torch.tensor(v)) for v in epoch_losses.values()):
            logger.error(f"NaN detected at epoch {epoch}! Stopping training.")
            break

        # Save best checkpoint
        if epoch_losses["total"] < best_loss:
            best_loss = epoch_losses["total"]
            checkpoint_path = Path(checkpoint_dir) / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
            }, checkpoint_path)
            logger.info(f"  Saved best model (loss={best_loss:.4f})")

        # Periodic checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint_path = Path(checkpoint_dir) / f"checkpoint_epoch_{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "loss": epoch_losses["total"],
            }, checkpoint_path)

    logger.info("Training complete.")
    return model


def main() -> None:
    """Entry point using environment variables for configuration."""
    train(
        data_dir=os.getenv("PHARMLOOP_DATA_DIR", "./data/processed"),
        checkpoint_dir=os.getenv("PHARMLOOP_CHECKPOINT_DIR", "./checkpoints"),
        epochs=int(os.getenv("PHARMLOOP_EPOCHS", "50")),
        batch_size=int(os.getenv("PHARMLOOP_BATCH_SIZE", "32")),
        lr=float(os.getenv("PHARMLOOP_LEARNING_RATE", "1e-3")),
        device_str=os.getenv("PHARMLOOP_DEVICE", "cpu"),
        seed=int(os.getenv("PHARMLOOP_SEED", "42")) if os.getenv("PHARMLOOP_SEED") else 42,
        log_level=os.getenv("PHARMLOOP_LOG_LEVEL", "INFO"),
    )


if __name__ == "__main__":
    main()
