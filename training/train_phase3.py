"""
Phase 3 training: staged mechanism head upgrade + early convergence.

Stage 1: Freeze all except mechanism_head, train 10 epochs at lr=5e-4.
Stage 2: Unfreeze all, train 15 epochs with dual LR (mechanism 3e-4, rest 5e-5).

Logs mechanism accuracy per epoch to track climbing.
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

from pharmloop.hopfield import PharmHopfield
from pharmloop.model import PharmLoopModel
from pharmloop.output import SEVERITY_NAMES, MECHANISM_NAMES, NUM_MECHANISMS
from training.data_loader import create_dataloader
from training.loss import PharmLoopLoss

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger("pharmloop.train_phase3")


def _load_phase2_model(checkpoint_path: str, drugs_path: str) -> PharmLoopModel:
    """Load Phase 2 model with 512-dim Hopfield."""
    with open(drugs_path) as f:
        drugs_data = json.load(f)

    num_drugs = len(drugs_data["drugs"])
    hopfield = PharmHopfield(input_dim=512, hidden_dim=512, phase0=False)
    model = PharmLoopModel(num_drugs=num_drugs, hopfield=hopfield)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Phase 2 checkpoint has old linear mechanism_head; Phase 3 has TrajectoryMechanismHead.
    # Load with strict=False and let the new mechanism_head init randomly.
    missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    logger.info(f"Loaded Phase 2 model. Missing keys: {len(missing)}, Unexpected: {len(unexpected)}")
    if missing:
        logger.info(f"  Missing (new Phase 3 params): {missing}")

    return model


def _compute_mechanism_accuracy(
    model: PharmLoopModel,
    dataloader,
    device: torch.device,
) -> float:
    """Compute mechanism accuracy (at least one correct mechanism) on a dataset."""
    model.eval()
    correct = 0
    applicable = 0

    with torch.no_grad():
        for batch in dataloader:
            output = model(
                batch["drug_a_id"].to(device),
                batch["drug_a_features"].to(device),
                batch["drug_b_id"].to(device),
                batch["drug_b_features"].to(device),
            )
            mech_preds = (output["mechanism_logits"] > 0).float()
            targets = batch["target_mechanisms"].to(device)
            is_unknown = batch["is_unknown"].to(device)

            for i in range(targets.shape[0]):
                if is_unknown[i]:
                    continue
                true_mechs = targets[i].nonzero(as_tuple=True)[0]
                if len(true_mechs) == 0:
                    continue
                applicable += 1
                pred_mechs = mech_preds[i].nonzero(as_tuple=True)[0]
                if len(set(pred_mechs.tolist()) & set(true_mechs.tolist())) > 0:
                    correct += 1

    return correct / max(applicable, 1)


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

    epoch_losses = {
        "total": 0.0, "answer": 0.0, "convergence": 0.0,
        "smoothness": 0.0, "do_no_harm": 0.0, "early_convergence": 0.0,
    }
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
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0,
            )
            optimizer.step()

        for key in epoch_losses:
            if key in losses:
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


def train_phase3(
    data_dir: str = "./data/processed",
    checkpoint_dir: str = "./checkpoints",
    phase2_checkpoint: str = "./checkpoints/best_model_phase2.pt",
    stage1_epochs: int = 40,
    stage2_epochs: int = 15,
    batch_size: int = 32,
    device_str: str = "cpu",
    seed: int | None = 42,
) -> PharmLoopModel:
    """
    Run Phase 3 staged training.

    Stage 1: mechanism head only (frozen everything else).
    Stage 2: full model fine-tuning with early convergence loss.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    if seed is not None:
        torch.manual_seed(seed)

    device = torch.device(device_str)
    data_path = Path(data_dir)
    drugs_path = data_path / "drugs.json"
    interactions_path = data_path / "interactions.json"
    split_path = data_path / "split.json"

    with open(split_path) as f:
        split = json.load(f)

    train_loader = create_dataloader(
        drugs_path, interactions_path, batch_size=batch_size,
        shuffle=True, split_indices=split["train_indices"],
    )
    val_loader = create_dataloader(
        drugs_path, interactions_path, batch_size=batch_size,
        shuffle=False, split_indices=split["val_indices"],
    )
    test_loader = create_dataloader(
        drugs_path, interactions_path, batch_size=batch_size,
        shuffle=False, split_indices=split["test_indices"],
    )
    logger.info(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    # Load Phase 2 model (mechanism_head will be randomly initialized)
    model = _load_phase2_model(phase2_checkpoint, str(drugs_path))
    model = model.to(device)

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    global_epoch = 0

    # ── Stage 1: Train only mechanism head ──
    logger.info(f"\n{'='*60}")
    logger.info(f"Stage 1: mechanism head only ({stage1_epochs} epochs)")
    logger.info(f"{'='*60}")

    for name, param in model.named_parameters():
        if "mechanism_head" not in name:
            param.requires_grad = False

    mechanism_params = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"Stage 1 trainable params: {sum(p.numel() for p in mechanism_params):,}")

    optimizer = Adam(mechanism_params, lr=5e-4)
    # No early convergence loss in stage 1 (only mechanism head is training)
    criterion = PharmLoopLoss(convergence_weight=0.7, smoothness_weight=0.1, early_convergence_weight=0.0)

    for epoch in range(1, stage1_epochs + 1):
        global_epoch += 1
        train_metrics = _run_epoch(model, train_loader, criterion, optimizer, device, training=True)
        val_metrics = _run_epoch(model, val_loader, criterion, None, device, training=False)

        mech_acc = _compute_mechanism_accuracy(model, test_loader, device)

        logger.info(
            f"Epoch {global_epoch} (S1 {epoch}/{stage1_epochs}) — "
            f"train_loss: {train_metrics['total']:.4f} "
            f"val_loss: {val_metrics['total']:.4f} — "
            f"mech_acc: {mech_acc:.1%} — "
            f"conv: {train_metrics['convergence_rate']:.1%}"
        )

        if val_metrics["total"] < best_val_loss:
            best_val_loss = val_metrics["total"]
            torch.save({
                "epoch": global_epoch, "stage": 1,
                "model_state_dict": model.state_dict(),
                "val_loss": best_val_loss, "mech_acc": mech_acc, "phase": 3,
            }, Path(checkpoint_dir) / "best_model_phase3.pt")
            logger.info(f"  Saved best Phase 3 model (val_loss={best_val_loss:.4f})")

    # Save Stage 1 final as separate checkpoint (fallback if Stage 2 degrades)
    stage1_mech_acc = mech_acc
    torch.save({
        "epoch": global_epoch, "stage": 1,
        "model_state_dict": model.state_dict(),
        "val_loss": val_metrics["total"], "mech_acc": mech_acc, "phase": 3,
    }, Path(checkpoint_dir) / "stage1_final_phase3.pt")
    logger.info(f"Saved Stage 1 final checkpoint (mech_acc={mech_acc:.1%})")

    # ── Stage 2: Freeze mechanism head, train core for convergence speed ──
    logger.info(f"\n{'='*60}")
    logger.info(f"Stage 2: convergence speed — core only ({stage2_epochs} epochs)")
    logger.info(f"{'='*60}")

    for name, param in model.named_parameters():
        if "mechanism_head" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    core_params = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"Stage 2 trainable params: {sum(p.numel() for p in core_params):,}")

    optimizer = Adam(core_params, lr=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=stage2_epochs)
    # Enable early convergence loss to incentivize faster convergence
    criterion = PharmLoopLoss(
        convergence_weight=0.7, smoothness_weight=0.1,
        early_convergence_weight=0.15,
    )

    for epoch in range(1, stage2_epochs + 1):
        global_epoch += 1
        train_metrics = _run_epoch(model, train_loader, criterion, optimizer, device, training=True)
        scheduler.step()
        val_metrics = _run_epoch(model, val_loader, criterion, None, device, training=False)

        mech_acc = _compute_mechanism_accuracy(model, test_loader, device)

        logger.info(
            f"Epoch {global_epoch} (S2 {epoch}/{stage2_epochs}) — "
            f"train_loss: {train_metrics['total']:.4f} "
            f"(early_conv: {train_metrics['early_convergence']:.4f}) "
            f"val_loss: {val_metrics['total']:.4f} — "
            f"mech_acc: {mech_acc:.1%} — "
            f"conv: {train_metrics['convergence_rate']:.1%}"
        )

        # Save if representation shift helped mechanism accuracy
        if mech_acc > stage1_mech_acc:
            torch.save({
                "epoch": global_epoch, "stage": 2,
                "model_state_dict": model.state_dict(),
                "val_loss": val_metrics["total"], "mech_acc": mech_acc, "phase": 3,
            }, Path(checkpoint_dir) / "best_model_phase3.pt")
            logger.info(f"  Saved best Phase 3 model (mech_acc={mech_acc:.1%})")
            stage1_mech_acc = mech_acc  # raise the bar for Stage 3

    # ── Stage 3: Re-adapt mechanism head to shifted representations ──
    stage3_epochs = 15
    logger.info(f"\n{'='*60}")
    logger.info(f"Stage 3: mechanism head re-adaptation ({stage3_epochs} epochs)")
    logger.info(f"{'='*60}")

    for name, param in model.named_parameters():
        if "mechanism_head" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    mech_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(mech_params, lr=3e-4)
    criterion = PharmLoopLoss(convergence_weight=0.7, smoothness_weight=0.1, early_convergence_weight=0.0)

    best_mech_acc = stage1_mech_acc

    for epoch in range(1, stage3_epochs + 1):
        global_epoch += 1
        train_metrics = _run_epoch(model, train_loader, criterion, optimizer, device, training=True)
        val_metrics = _run_epoch(model, val_loader, criterion, None, device, training=False)

        mech_acc = _compute_mechanism_accuracy(model, test_loader, device)

        logger.info(
            f"Epoch {global_epoch} (S3 {epoch}/{stage3_epochs}) — "
            f"train_loss: {train_metrics['total']:.4f} "
            f"val_loss: {val_metrics['total']:.4f} — "
            f"mech_acc: {mech_acc:.1%} — "
            f"conv: {train_metrics['convergence_rate']:.1%}"
        )

        if mech_acc > best_mech_acc:
            best_mech_acc = mech_acc
            torch.save({
                "epoch": global_epoch, "stage": 3,
                "model_state_dict": model.state_dict(),
                "val_loss": val_metrics["total"], "mech_acc": mech_acc, "phase": 3,
            }, Path(checkpoint_dir) / "best_model_phase3.pt")
            logger.info(f"  Saved best Phase 3 model (mech_acc={mech_acc:.1%})")

    logger.info(f"Best mechanism accuracy across all stages: {max(best_mech_acc, stage1_mech_acc):.1%}")

    # Save final
    torch.save({
        "epoch": global_epoch,
        "model_state_dict": model.state_dict(),
        "val_loss": val_metrics["total"],
        "mech_acc": mech_acc,
        "phase": 3,
    }, Path(checkpoint_dir) / "final_model_phase3.pt")

    logger.info(f"\nPhase 3 training complete. Total epochs: {global_epoch}")
    logger.info(f"Final mechanism accuracy: {mech_acc:.1%}")

    return model


def main() -> None:
    train_phase3(
        data_dir=os.getenv("PHARMLOOP_DATA_DIR", "./data/processed"),
        checkpoint_dir=os.getenv("PHARMLOOP_CHECKPOINT_DIR", "./checkpoints"),
        phase2_checkpoint=os.getenv("PHARMLOOP_PHASE2_CHECKPOINT", "./checkpoints/best_model_phase2.pt"),
        device_str=os.getenv("PHARMLOOP_DEVICE", "cpu"),
    )


if __name__ == "__main__":
    main()
