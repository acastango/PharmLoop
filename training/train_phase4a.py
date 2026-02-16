"""
Phase 4a training: scaled to ~300 drugs / ~3000 interactions.

Key differences from Phase 2/3 training:
  1. Weight transfer from Phase 3 (original 50 drugs keep embeddings)
  2. Hierarchical Hopfield (class-specific + global banks)
  3. Curriculum learning (high-confidence first, then all)
  4. Feature-dominant encoding (identity suppressed for rare drugs)
  5. Data split on v2 dataset

Usage:
    python -m training.train_phase4a
"""

import json
import logging
import os
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from pharmloop.hierarchical_hopfield import HierarchicalHopfield, DRUG_CLASSES
from pharmloop.hopfield import PharmHopfield
from pharmloop.model import PharmLoopModel
from pharmloop.output import SEVERITY_NAMES, MECHANISM_NAMES, NUM_MECHANISMS
from training.data_loader import DrugInteractionDataset, create_dataloader
from training.loss import PharmLoopLoss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger("pharmloop.train_phase4a")


def _load_v2_data(data_dir: str) -> tuple[dict, list[dict]]:
    """Load v2 drug and interaction data."""
    data_path = Path(data_dir)

    with open(data_path / "drugs_v2.json") as f:
        drugs_data = json.load(f)

    with open(data_path / "interactions_v2.json") as f:
        interactions_data = json.load(f)

    return drugs_data, interactions_data


def _build_drug_class_map(drugs_data: dict) -> dict[int, str]:
    """Build drug_id → class mapping for hierarchical Hopfield routing."""
    class_map: dict[int, str] = {}
    for name, info in drugs_data["drugs"].items():
        drug_class = info.get("class", "other")
        # Map drug classes to DRUG_CLASSES taxonomy
        if drug_class in DRUG_CLASSES:
            class_map[info["id"]] = drug_class
        else:
            class_map[info["id"]] = "other"
    return class_map


def _split_v2_data(
    interactions: list[dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """
    Split v2 interactions into train/val/test indices.

    Stratified by severity to maintain class balance across splits.
    """
    rng = random.Random(seed)

    # Group by severity
    severity_groups: dict[str, list[int]] = {}
    for i, ix in enumerate(interactions):
        sev = ix["severity"]
        severity_groups.setdefault(sev, []).append(i)

    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []

    for sev, indices in severity_groups.items():
        rng.shuffle(indices)
        n = len(indices)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_idx.extend(indices[:n_train])
        val_idx.extend(indices[n_train:n_train + n_val])
        test_idx.extend(indices[n_train + n_val:])

    return train_idx, val_idx, test_idx


def _transfer_phase3_weights(
    phase3_path: str,
    model: PharmLoopModel,
    original_drugs: dict,
    v2_drugs: dict,
) -> None:
    """
    Transfer Phase 3 weights to Phase 4a model.

    - feature_proj and fusion are copied directly (architecture unchanged)
    - Identity embeddings for original 50 drugs are placed at their v2 IDs
    - Oscillator, output head weights are copied
    - New drug embeddings stay random-initialized (feature-dominant scaling handles them)
    """
    phase3_ckpt = torch.load(phase3_path, map_location="cpu", weights_only=True)
    phase3_state = phase3_ckpt["model_state_dict"]

    # Map old drug names to their new v2 IDs
    old_to_new_id: dict[int, int] = {}
    for name, old_info in original_drugs.items():
        if name in v2_drugs:
            old_to_new_id[old_info["id"]] = v2_drugs[name]["id"]

    # Copy identity embeddings for original drugs
    old_embed = phase3_state.get("encoder.identity_embedding.weight")
    if old_embed is not None:
        with torch.no_grad():
            for old_id, new_id in old_to_new_id.items():
                if old_id < old_embed.shape[0] and new_id < model.encoder.total_vocab:
                    model.encoder.identity_embedding.weight[new_id] = old_embed[old_id]
        logger.info(f"Transferred {len(old_to_new_id)} identity embeddings")

    # Copy all other compatible weights
    model_state = model.state_dict()
    transferred = 0
    skipped = []

    for key, value in phase3_state.items():
        if key == "encoder.identity_embedding.weight":
            continue  # already handled
        if key in model_state and model_state[key].shape == value.shape:
            model_state[key] = value
            transferred += 1
        else:
            skipped.append(key)

    model.load_state_dict(model_state, strict=False)
    logger.info(f"Transferred {transferred} parameter tensors, skipped {len(skipped)}")
    if skipped:
        logger.info(f"  Skipped keys: {skipped[:10]}{'...' if len(skipped) > 10 else ''}")


def _build_hierarchical_hopfield(
    model: PharmLoopModel,
    interactions: list[dict],
    drugs_data: dict,
    drug_class_map: dict[int, str],
) -> HierarchicalHopfield:
    """
    Build hierarchical Hopfield from trained encoder.

    Computes pair states for all interactions and stores them in
    both global and class-specific banks with severity amplification.
    """
    hopfield = HierarchicalHopfield(input_dim=512, class_names=DRUG_CLASSES)

    # Identity-init projections on all banks
    all_banks = [hopfield.global_bank] + list(hopfield.class_banks.values())
    for bank in all_banks:
        if isinstance(bank.query_proj, nn.Linear):
            with torch.no_grad():
                nn.init.eye_(bank.query_proj.weight)
                nn.init.zeros_(bank.query_proj.bias)
                nn.init.eye_(bank.key_proj.weight)
                nn.init.zeros_(bank.key_proj.bias)

    model.eval()
    patterns: list[Tensor] = []
    classes: list[tuple[str, str]] = []
    drug_map = drugs_data["drugs"]

    with torch.no_grad():
        for ix in interactions:
            da, db = ix["drug_a"], ix["drug_b"]
            if da not in drug_map or db not in drug_map:
                continue

            info_a, info_b = drug_map[da], drug_map[db]
            a_id = torch.tensor([info_a["id"]])
            a_feat = torch.tensor([info_a["features"]])
            b_id = torch.tensor([info_b["id"]])
            b_feat = torch.tensor([info_b["features"]])

            pair_state = model.compute_pair_state(a_id, a_feat, b_id, b_feat)
            patterns.append(pair_state.squeeze(0))

            cls_a = drug_class_map.get(info_a["id"], "other")
            cls_b = drug_class_map.get(info_b["id"], "other")
            classes.append((cls_a, cls_b))

            # Severity amplification for dangerous interactions
            if ix["severity"] in ("severe", "contraindicated"):
                for _ in range(2):
                    noisy = pair_state.squeeze(0) + torch.randn(512) * 0.01
                    patterns.append(noisy)
                    classes.append((cls_a, cls_b))

    if not patterns:
        logger.warning("No patterns to store in Hopfield!")
        return hopfield

    # Deduplicate by cosine similarity
    stacked = torch.stack(patterns)
    unique_patterns = [stacked[0]]
    unique_classes = [classes[0]]
    for i in range(1, len(stacked)):
        cos_sims = torch.cosine_similarity(stacked[i].unsqueeze(0), torch.stack(unique_patterns))
        if cos_sims.max() < 0.999:
            unique_patterns.append(stacked[i])
            unique_classes.append(classes[i])

    patterns_tensor = torch.stack(unique_patterns)
    logger.info(f"Hopfield: {len(unique_patterns)} unique patterns "
                f"(from {len(patterns)} raw, {len(interactions)} interactions)")

    # Store in hierarchical banks
    hopfield.store(patterns_tensor, drug_classes=unique_classes)

    # Log class bank sizes
    for name, bank in hopfield.class_banks.items():
        if bank.count > 0:
            logger.info(f"  {name}: {bank.count} patterns")
    logger.info(f"  global: {hopfield.global_bank.count} patterns")

    return hopfield


def _run_epoch(
    model: PharmLoopModel,
    loader: DataLoader,
    criterion: PharmLoopLoss,
    optimizer: torch.optim.Optimizer | None,
    device: str,
    training: bool = True,
) -> dict[str, float]:
    """Run one training or validation epoch."""
    model.train(training)
    total_loss = 0.0
    total_answer = 0.0
    total_convergence = 0.0
    total_smoothness = 0.0
    total_do_no_harm = 0.0
    total_early_conv = 0.0
    convergence_count = 0
    total_known = 0
    total_samples = 0
    batches = 0

    for batch in loader:
        a_id = batch["drug_a_id"].to(device)
        a_feat = batch["drug_a_features"].to(device)
        b_id = batch["drug_b_id"].to(device)
        b_feat = batch["drug_b_features"].to(device)
        target_sev = batch["target_severity"].to(device)
        target_mech = batch["target_mechanisms"].to(device)
        target_flags = batch["target_flags"].to(device)
        is_unknown = batch["is_unknown"].to(device)

        output = model(a_id, a_feat, b_id, b_feat)

        losses = criterion(
            severity_logits=output["severity_logits"],
            mechanism_logits=output["mechanism_logits"],
            flag_logits=output["flag_logits"],
            final_v=output["trajectory"]["velocities"][-1],
            velocities=output["trajectory"]["velocities"],
            target_severity=target_sev,
            target_mechanisms=target_mech,
            target_flags=target_flags,
            is_unknown=is_unknown,
        )

        loss = losses["total"]

        if training and optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        bs = a_id.shape[0]
        total_loss += loss.item() * bs
        total_answer += losses["answer"].item() * bs
        total_convergence += losses["convergence"].item() * bs
        total_smoothness += losses["smoothness"].item() * bs
        total_do_no_harm += losses["do_no_harm"].item() * bs
        total_early_conv += losses.get("early_convergence", torch.tensor(0.0)).item() * bs

        known_mask = ~is_unknown
        total_known += known_mask.sum().item()
        if known_mask.any():
            convergence_count += output["converged"][known_mask].float().sum().item()

        total_samples += bs
        batches += 1

    if total_samples == 0:
        return {"total": 0.0, "convergence_rate": 0.0}

    known_count = max(1, total_known)

    return {
        "total": total_loss / total_samples,
        "answer": total_answer / total_samples,
        "convergence": total_convergence / total_samples,
        "smoothness": total_smoothness / total_samples,
        "do_no_harm": total_do_no_harm / total_samples,
        "early_convergence": total_early_conv / total_samples,
        "convergence_rate": convergence_count / known_count,
    }


def _compute_metrics(
    model: PharmLoopModel,
    loader: DataLoader,
    device: str,
) -> dict[str, float]:
    """
    Compute severity accuracy, mechanism accuracy, and false negative rate.

    NOTE: FNR on this synthetic dataset reflects convergence on probabilistic
    class-level interaction rules, NOT ground-truth clinical labels. A nonzero
    FNR here (e.g. 3-5%) may be inflated by noisy severity assignments from
    CLASS_INTERACTION_RULES. Real FNR will only be meaningful after validation
    against DrugBank data or pharmacist review (Phase 4b+).
    """
    model.eval()
    correct_sev = 0
    total_sev = 0
    correct_mech = 0
    total_mech = 0
    false_neg_severe = 0
    total_severe = 0

    with torch.no_grad():
        for batch in loader:
            a_id = batch["drug_a_id"].to(device)
            a_feat = batch["drug_a_features"].to(device)
            b_id = batch["drug_b_id"].to(device)
            b_feat = batch["drug_b_features"].to(device)
            target_sev = batch["target_severity"].to(device)
            target_mech = batch["target_mechanisms"].to(device)
            is_unknown = batch["is_unknown"].to(device)

            output = model(a_id, a_feat, b_id, b_feat)

            known_mask = ~is_unknown

            # Severity accuracy
            pred_sev = output["severity_logits"][known_mask].argmax(dim=-1)
            true_sev = target_sev[known_mask]
            correct_sev += (pred_sev == true_sev).sum().item()
            total_sev += known_mask.sum().item()

            # False negatives on severe/contraindicated
            for i in range(len(true_sev)):
                if true_sev[i].item() in (3, 4):  # severe or contraindicated
                    total_severe += 1
                    if pred_sev[i].item() == 0:  # predicted "none"
                        false_neg_severe += 1

            # Mechanism accuracy (at least one correct)
            mech_target = target_mech[known_mask]
            mech_pred = (torch.sigmoid(output["mechanism_logits"][known_mask]) > 0.5).float()
            for i in range(mech_target.shape[0]):
                if mech_target[i].sum() > 0:
                    total_mech += 1
                    pred_set = set(mech_pred[i].nonzero(as_tuple=True)[0].tolist())
                    true_set = set(mech_target[i].nonzero(as_tuple=True)[0].tolist())
                    if pred_set & true_set:
                        correct_mech += 1

    return {
        "severity_accuracy": correct_sev / max(1, total_sev),
        "mechanism_accuracy": correct_mech / max(1, total_mech),
        "false_negative_rate": false_neg_severe / max(1, total_severe),
        "total_severe": total_severe,
    }


def train_phase4a(
    data_dir: str = "./data/processed",
    checkpoint_dir: str = "./checkpoints",
    phase3_checkpoint: str = "./checkpoints/best_model_phase3.pt",
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 5e-4,
    warmup_epochs: int = 5,
    anneal_interval: int = 15,
    max_anneal_cycles: int = 3,
    device_str: str = "cpu",
    seed: int = 42,
) -> PharmLoopModel:
    """
    Phase 4a training with curriculum learning and Hopfield annealing.

    Stage 1 (15 epochs): High-confidence interactions only (clear mechanisms,
        no severity conflicts). Establishes good Hopfield attractors.
    Stage 2 (35 epochs): All interactions. Model has good priors from stage 1.
        Hopfield rebuilt every anneal_interval epochs.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    device = torch.device(device_str)

    data_path = Path(data_dir)
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # ── Load v2 data ──
    drugs_data, interactions_data = _load_v2_data(data_dir)
    num_drugs = drugs_data["num_drugs"]
    all_interactions = interactions_data["interactions"]
    logger.info(f"Loaded {num_drugs} drugs, {len(all_interactions)} interactions")

    drug_class_map = _build_drug_class_map(drugs_data)

    # ── Split data ──
    train_idx, val_idx, test_idx = _split_v2_data(all_interactions)
    logger.info(f"Split: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")

    # Save split for reproducibility
    with open(data_path / "split_v2.json", "w") as f:
        json.dump({
            "train_indices": train_idx,
            "val_indices": val_idx,
            "test_indices": test_idx,
        }, f)

    # ── Identify high-confidence interactions for curriculum stage 1 ──
    high_conf_idx = []
    for i in train_idx:
        ix = all_interactions[i]
        # High confidence: has clear mechanism labels or is severity "none"
        if ix["severity"] == "none" or len(ix["mechanisms"]) > 0:
            if ix.get("source") != "negative_sample" or random.random() < 0.5:
                high_conf_idx.append(i)
    logger.info(f"High-confidence training set: {len(high_conf_idx)} interactions")

    # ── Create data loaders ──
    drugs_path = data_path / "drugs_v2.json"
    interactions_path = data_path / "interactions_v2.json"

    high_conf_loader = create_dataloader(
        drugs_path, interactions_path, batch_size=batch_size,
        split_indices=high_conf_idx,
    )
    full_train_loader = create_dataloader(
        drugs_path, interactions_path, batch_size=batch_size,
        split_indices=train_idx,
    )
    val_loader = create_dataloader(
        drugs_path, interactions_path, batch_size=batch_size,
        shuffle=False, split_indices=val_idx,
    )
    test_loader = create_dataloader(
        drugs_path, interactions_path, batch_size=batch_size,
        shuffle=False, split_indices=test_idx,
    )

    # ── Build model ──
    hopfield = HierarchicalHopfield(input_dim=512, class_names=DRUG_CLASSES)
    model = PharmLoopModel(
        num_drugs=num_drugs,
        hopfield=hopfield,
        drug_class_map=drug_class_map,
    )
    model.to(device)

    # ── Transfer Phase 3 weights ──
    if Path(phase3_checkpoint).exists():
        original_drugs_path = data_path / "drugs.json"
        if original_drugs_path.exists():
            with open(original_drugs_path) as f:
                original_drugs = json.load(f)["drugs"]
            _transfer_phase3_weights(
                phase3_checkpoint, model, original_drugs, drugs_data["drugs"],
            )
        else:
            logger.warning("No original drugs.json found — skipping weight transfer")
    else:
        logger.warning(f"No Phase 3 checkpoint at {phase3_checkpoint} — training from scratch")

    # ── Build initial Hopfield ──
    logger.info("Building initial hierarchical Hopfield...")
    hopfield = _build_hierarchical_hopfield(
        model, all_interactions, drugs_data, drug_class_map,
    )
    # Replace the model's hopfield
    model.cell.hopfield = hopfield

    # ── Training ──
    criterion = PharmLoopLoss(
        convergence_weight=0.7,
        smoothness_weight=0.1,
        early_convergence_weight=0.05,
    )

    optimizer = Adam(model.parameters(), lr=lr)

    stage1_epochs = min(15, epochs)
    stage2_epochs = epochs - stage1_epochs
    best_val_loss = float("inf")
    global_epoch = 0

    # ── Stage 1: High-confidence curriculum ──
    logger.info(f"\n{'='*60}")
    logger.info(f"Stage 1: High-confidence curriculum ({stage1_epochs} epochs)")
    logger.info(f"{'='*60}")

    scheduler = CosineAnnealingLR(optimizer, T_max=stage1_epochs)

    for epoch in range(1, stage1_epochs + 1):
        global_epoch += 1
        train_metrics = _run_epoch(
            model, high_conf_loader, criterion, optimizer, device, training=True,
        )
        scheduler.step()
        val_metrics = _run_epoch(
            model, val_loader, criterion, None, device, training=False,
        )

        logger.info(
            f"Epoch {global_epoch} (S1 {epoch}/{stage1_epochs}) — "
            f"train: {train_metrics['total']:.4f} "
            f"val: {val_metrics['total']:.4f} "
            f"conv: {train_metrics['convergence_rate']:.1%}"
        )

        if val_metrics["total"] < best_val_loss:
            best_val_loss = val_metrics["total"]
            torch.save({
                "epoch": global_epoch, "stage": 1,
                "model_state_dict": model.state_dict(),
                "val_loss": best_val_loss, "phase": "4a",
            }, checkpoint_path / "best_model_phase4a.pt")

    # ── Stage 2: Full dataset with annealing ──
    logger.info(f"\n{'='*60}")
    logger.info(f"Stage 2: Full dataset ({stage2_epochs} epochs, "
                f"anneal every {anneal_interval})")
    logger.info(f"{'='*60}")

    optimizer = Adam(model.parameters(), lr=lr * 0.5)
    scheduler = CosineAnnealingLR(optimizer, T_max=stage2_epochs)

    anneal_count = 0
    for epoch in range(1, stage2_epochs + 1):
        global_epoch += 1

        # Hopfield annealing
        if epoch > 1 and (epoch - 1) % anneal_interval == 0 and anneal_count < max_anneal_cycles:
            anneal_count += 1
            logger.info(f"\n  Annealing cycle {anneal_count}: rebuilding Hopfield...")
            new_hopfield = _build_hierarchical_hopfield(
                model, all_interactions, drugs_data, drug_class_map,
            )
            model.cell.hopfield = new_hopfield

        train_metrics = _run_epoch(
            model, full_train_loader, criterion, optimizer, device, training=True,
        )
        scheduler.step()
        val_metrics = _run_epoch(
            model, val_loader, criterion, None, device, training=False,
        )

        # Compute detailed metrics every 5 epochs
        extra = ""
        if epoch % 5 == 0 or epoch == stage2_epochs:
            metrics = _compute_metrics(model, test_loader, device)
            extra = (
                f" | sev_acc: {metrics['severity_accuracy']:.1%}"
                f" mech_acc: {metrics['mechanism_accuracy']:.1%}"
                f" FNR: {metrics['false_negative_rate']:.1%}"
            )

        logger.info(
            f"Epoch {global_epoch} (S2 {epoch}/{stage2_epochs}) — "
            f"train: {train_metrics['total']:.4f} "
            f"val: {val_metrics['total']:.4f} "
            f"conv: {train_metrics['convergence_rate']:.1%}"
            f"{extra}"
        )

        if val_metrics["total"] < best_val_loss:
            best_val_loss = val_metrics["total"]
            torch.save({
                "epoch": global_epoch, "stage": 2,
                "model_state_dict": model.state_dict(),
                "val_loss": best_val_loss, "phase": "4a",
            }, checkpoint_path / "best_model_phase4a.pt")
            logger.info(f"  Saved best model (val_loss={best_val_loss:.4f})")

    # ── Final evaluation ──
    final_metrics = _compute_metrics(model, test_loader, device)
    logger.info(f"\n{'='*60}")
    logger.info(f"Phase 4a training complete. Total epochs: {global_epoch}")
    logger.info(f"Severity accuracy: {final_metrics['severity_accuracy']:.1%}")
    logger.info(f"Mechanism accuracy: {final_metrics['mechanism_accuracy']:.1%}")
    logger.info(f"False negative rate: {final_metrics['false_negative_rate']:.1%}")
    logger.info(f"{'='*60}")

    # Save final
    torch.save({
        "epoch": global_epoch,
        "model_state_dict": model.state_dict(),
        "val_loss": val_metrics["total"],
        "metrics": final_metrics,
        "phase": "4a",
    }, checkpoint_path / "final_model_phase4a.pt")

    return model


def main() -> None:
    train_phase4a(
        data_dir=os.getenv("PHARMLOOP_DATA_DIR", "./data/processed"),
        checkpoint_dir=os.getenv("PHARMLOOP_CHECKPOINT_DIR", "./checkpoints"),
        phase3_checkpoint=os.getenv(
            "PHARMLOOP_PHASE3_CHECKPOINT", "./checkpoints/best_model_phase3.pt",
        ),
        epochs=int(os.getenv("PHARMLOOP_EPOCHS", "50")),
        batch_size=int(os.getenv("PHARMLOOP_BATCH_SIZE", "32")),
    )


if __name__ == "__main__":
    main()
