"""
Context encoder training for Phase 4b.

Trains the context encoder end-to-end on context-annotated and
pharmacogenomic-annotated interaction examples.

Stage 1: Freeze base model, train only context encoder.
  Teaches the gate and projection to respond to context features
  without disrupting the base model.

Stage 2: Unfreeze all, fine-tune with context.
  Allows the oscillator and Hopfield to adapt to context-modulated
  initial states.

Usage:
    python -m training.train_context
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
from torch.utils.data import Dataset, DataLoader

from pharmloop.context import (
    CONTEXT_DIM,
    PGX_CYP2D6_OFFSET, PGX_CYP2C19_OFFSET,
    PGX_CYP2C9_VKORC1_OFFSET, PGX_HLA_OFFSET,
    PGX_METABOLIZER_MAP, PGX_VKORC1_MAP, PGX_HLA_MAP,
)
from pharmloop.hierarchical_hopfield import HierarchicalHopfield, DRUG_CLASSES
from pharmloop.model import PharmLoopModel
from pharmloop.output import SEVERITY_NAMES
from training.loss import PharmLoopLoss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger("pharmloop.train_context")


def encode_context_vector(context: dict) -> list[float]:
    """
    Convert a context dict to a 48-dim feature vector.

    Handles both clinical context (dose, route, timing, patient factors)
    and pharmacogenomic context (CYP2D6 status, etc.).

    Args:
        context: Dict with context keys (see pharmloop/context.py layout).

    Returns:
        List of 48 floats.
    """
    vec = [0.0] * CONTEXT_DIM

    # Drug A dosing (dims 0-3)
    vec[0] = context.get("dose_a_normalized", 0.0)
    vec[1] = context.get("frequency_a", 0.0)
    vec[2] = context.get("duration_a_days", 0.0) / 365.0
    vec[3] = float(context.get("is_loading_dose_a", False))

    # Drug B dosing (dims 4-7)
    vec[4] = context.get("dose_b_normalized", 0.0)
    vec[5] = context.get("frequency_b", 0.0)
    vec[6] = context.get("duration_b_days", 0.0) / 365.0
    vec[7] = float(context.get("is_loading_dose_b", False))

    # Route flags (dims 8-11)
    vec[8] = float(context.get("both_oral", False))
    vec[9] = float(context.get("any_iv", False))
    vec[10] = float(context.get("any_topical", False))
    vec[11] = float(context.get("any_inhaled", False))

    # Timing (dims 12-15)
    vec[12] = float(context.get("simultaneous", False))
    vec[13] = context.get("separated_hours_norm", 0.0)
    vec[14] = float(context.get("a_before_b", False))
    vec[15] = float(context.get("b_before_a", False))

    # Patient factors (dims 16-23)
    vec[16] = context.get("age_norm", 0.0)
    vec[17] = context.get("weight_norm", 0.0)
    vec[18] = context.get("renal_gfr_norm", 0.0)
    vec[19] = context.get("hepatic_child_pugh_norm", 0.0)
    vec[20] = float(context.get("pregnancy", False))
    vec[21] = float(context.get("pediatric", False))
    vec[22] = float(context.get("geriatric", False))
    vec[23] = float(context.get("genetic_pm", False))

    # Comedication burden (dims 24-27)
    vec[24] = context.get("total_drugs_norm", 0.0)
    vec[25] = context.get("cyp_inhibitor_count", 0.0)
    vec[26] = context.get("cyp_inducer_count", 0.0)
    vec[27] = context.get("protein_bound_count", 0.0)

    # Pharmacogenomic context (dims 32-47)
    # CYP2D6 (one-hot at offset 32-35)
    cyp2d6 = context.get("cyp2d6_status")
    if cyp2d6 and cyp2d6 in PGX_METABOLIZER_MAP:
        vec[PGX_CYP2D6_OFFSET + PGX_METABOLIZER_MAP[cyp2d6]] = 1.0

    # CYP2C19 (one-hot at offset 36-39)
    cyp2c19 = context.get("cyp2c19_status")
    if cyp2c19 and cyp2c19 in PGX_METABOLIZER_MAP:
        vec[PGX_CYP2C19_OFFSET + PGX_METABOLIZER_MAP[cyp2c19]] = 1.0

    # CYP2C9 (one-hot at offset 40-42)
    cyp2c9 = context.get("cyp2c9_status")
    if cyp2c9 and cyp2c9 in PGX_METABOLIZER_MAP:
        idx = PGX_METABOLIZER_MAP[cyp2c9]
        if idx < 3:  # only 3 slots for CYP2C9
            vec[PGX_CYP2C9_VKORC1_OFFSET + idx] = 1.0

    # VKORC1 (dim 43)
    vkorc1 = context.get("vkorc1_status")
    if vkorc1 == "sensitive":
        vec[PGX_CYP2C9_VKORC1_OFFSET + 3] = 1.0

    # HLA markers (dims 44-47)
    if context.get("hla_b5701") == "positive":
        vec[PGX_HLA_OFFSET + 0] = 1.0
    if context.get("hla_b1502") == "positive":
        vec[PGX_HLA_OFFSET + 1] = 1.0

    return vec


class ContextTrainingDataset(Dataset):
    """
    Dataset for context encoder training.

    Each sample provides a drug pair with and without context, plus
    the expected severity change. The model should learn that context
    shifts severity in the expected direction.

    Args:
        drugs_path: Path to drugs_v3.json (or v2).
        context_data_path: Path to context_training_data.json.
        pgx_data_path: Optional path to pharmacogenomic_examples.json.
    """

    def __init__(
        self,
        drugs_path: str | Path,
        context_data_path: str | Path,
        pgx_data_path: str | Path | None = None,
    ) -> None:
        with open(drugs_path) as f:
            drugs_data = json.load(f)

        self.drug_ids: dict[str, int] = {}
        self.drug_features: dict[int, list[float]] = {}
        for name, info in drugs_data["drugs"].items():
            self.drug_ids[name.lower()] = info["id"]
            self.drug_features[info["id"]] = info["features"]

        self.num_drugs = len(self.drug_ids)

        # Load context training examples
        self.samples: list[dict] = []
        with open(context_data_path) as f:
            context_examples = json.load(f)

        for ex in context_examples:
            self._add_example(ex, source="context")

        # Load PGx examples
        if pgx_data_path is not None:
            pgx_path = Path(pgx_data_path)
            if pgx_path.exists():
                with open(pgx_path) as f:
                    pgx_examples = json.load(f)
                for ex in pgx_examples:
                    self._add_pgx_example(ex)

        logger.info(f"Context dataset: {len(self.samples)} samples "
                     f"({len(context_examples)} context"
                     f"{f', {len(pgx_examples)} PGx' if pgx_data_path else ''})")

    def _add_example(self, ex: dict, source: str) -> None:
        """Add a context training example."""
        da = ex["drug_a"].lower()
        db = ex["drug_b"].lower()
        if da not in self.drug_ids or db not in self.drug_ids:
            return

        context_vec = encode_context_vector(ex.get("context", {}))
        sev_with = ex.get("severity_with_context", "unknown")
        sev_without = ex.get("severity_without_context", "unknown")

        if sev_with not in SEVERITY_NAMES or sev_without not in SEVERITY_NAMES:
            return

        self.samples.append({
            "drug_a_id": self.drug_ids[da],
            "drug_b_id": self.drug_ids[db],
            "drug_a_features": self.drug_features[self.drug_ids[da]],
            "drug_b_features": self.drug_features[self.drug_ids[db]],
            "context": context_vec,
            "target_severity_with": SEVERITY_NAMES.index(sev_with),
            "target_severity_without": SEVERITY_NAMES.index(sev_without),
            "source": source,
        })

    def _add_pgx_example(self, ex: dict) -> None:
        """Add a pharmacogenomic training example."""
        da = ex["drug_a"].lower()
        db = ex["drug_b"].lower()
        if da not in self.drug_ids or db not in self.drug_ids:
            return

        pgx_ctx = ex.get("pgx_context", {})
        context_vec = encode_context_vector(pgx_ctx)
        sev_with = ex.get("severity_with_pgx", "unknown")
        sev_without = ex.get("severity_without_pgx", "unknown")

        if sev_with not in SEVERITY_NAMES or sev_without not in SEVERITY_NAMES:
            return

        self.samples.append({
            "drug_a_id": self.drug_ids[da],
            "drug_b_id": self.drug_ids[db],
            "drug_a_features": self.drug_features[self.drug_ids[da]],
            "drug_b_features": self.drug_features[self.drug_ids[db]],
            "context": context_vec,
            "target_severity_with": SEVERITY_NAMES.index(sev_with),
            "target_severity_without": SEVERITY_NAMES.index(sev_without),
            "source": "pgx",
        })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        s = self.samples[idx]
        return {
            "drug_a_id": torch.tensor(s["drug_a_id"], dtype=torch.long),
            "drug_a_features": torch.tensor(s["drug_a_features"], dtype=torch.float32),
            "drug_b_id": torch.tensor(s["drug_b_id"], dtype=torch.long),
            "drug_b_features": torch.tensor(s["drug_b_features"], dtype=torch.float32),
            "context": torch.tensor(s["context"], dtype=torch.float32),
            "target_severity_with": torch.tensor(s["target_severity_with"], dtype=torch.long),
            "target_severity_without": torch.tensor(s["target_severity_without"], dtype=torch.long),
        }


def train_context(
    data_dir: str = "./data/processed",
    checkpoint_dir: str = "./checkpoints",
    base_checkpoint: str = "./checkpoints/best_model_phase4a.pt",
    context_data: str = "./data/raw/context_training_data.json",
    pgx_data: str = "./data/raw/pharmacogenomic_examples.json",
    stage1_epochs: int = 20,
    stage2_epochs: int = 15,
    batch_size: int = 16,
    lr: float = 1e-3,
    device_str: str = "cpu",
) -> PharmLoopModel:
    """
    Train the context encoder on context-annotated and PGx examples.

    Stage 1: Freeze base model, train only context encoder layers.
    Stage 2: Unfreeze all, fine-tune end-to-end.

    Args:
        data_dir: Directory containing drugs_v3.json (or v2).
        checkpoint_dir: Output directory for checkpoints.
        base_checkpoint: Path to Phase 4a/4b base model checkpoint.
        context_data: Path to context training data JSON.
        pgx_data: Path to pharmacogenomic examples JSON.
        stage1_epochs: Epochs for frozen-base stage.
        stage2_epochs: Epochs for full fine-tuning.
        batch_size: Training batch size.
        lr: Learning rate.
        device_str: Device ("cpu" or "cuda").

    Returns:
        Trained PharmLoopModel with context encoder.
    """
    torch.manual_seed(42)
    device = torch.device(device_str)
    data_path = Path(data_dir)
    ckpt_path = Path(checkpoint_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    # Find drugs file (prefer v3, fall back to v2)
    drugs_path = data_path / "drugs_v3.json"
    if not drugs_path.exists():
        drugs_path = data_path / "drugs_v2.json"

    with open(drugs_path) as f:
        drugs_data = json.load(f)
    num_drugs = drugs_data["num_drugs"]

    # Build drug class map
    drug_class_map: dict[int, str] = {}
    for name, info in drugs_data["drugs"].items():
        drug_class = info.get("class", "other")
        if drug_class in DRUG_CLASSES:
            drug_class_map[info["id"]] = drug_class
        else:
            drug_class_map[info["id"]] = "other"

    # Build model WITH context encoder
    # Detect Hopfield capacity from checkpoint to match buffer sizes
    class_cap = 500
    global_cap = 5000
    if Path(base_checkpoint).exists():
        ckpt_state = torch.load(base_checkpoint, map_location="cpu", weights_only=True)
        state_dict = ckpt_state.get("model_state_dict", {})
        for key, val in state_dict.items():
            if key.endswith("global_bank.stored_keys"):
                global_cap = val.shape[0]
            elif "class_banks." in key and key.endswith(".stored_keys"):
                class_cap = max(class_cap, val.shape[0])
        logger.info(f"Detected Hopfield capacity: class={class_cap}, global={global_cap}")

    hopfield = HierarchicalHopfield(
        input_dim=512, class_names=DRUG_CLASSES,
        class_capacity=class_cap, global_capacity=global_cap,
    )
    model = PharmLoopModel(
        num_drugs=num_drugs,
        hopfield=hopfield,
        drug_class_map=drug_class_map,
        use_context=True,
    )

    # Load base weights (strict=False to allow context encoder mismatch)
    if Path(base_checkpoint).exists():
        ckpt = torch.load(base_checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        logger.info(f"Loaded base model from {base_checkpoint}")
    else:
        logger.warning(f"No base checkpoint at {base_checkpoint} — training from scratch")

    model.to(device)

    # Load context dataset
    pgx_path = pgx_data if Path(pgx_data).exists() else None
    dataset = ContextTrainingDataset(
        str(drugs_path), context_data, pgx_path,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if len(dataset) == 0:
        logger.error("No context training examples found!")
        return model

    # Context-specific loss: severity cross-entropy on the with-context prediction
    criterion = nn.CrossEntropyLoss()

    # ── Stage 1: Freeze base, train context encoder only ──
    logger.info(f"\n{'='*60}")
    logger.info(f"Stage 1: Frozen base, context encoder only ({stage1_epochs} epochs)")
    logger.info(f"{'='*60}")

    # Freeze everything except context encoder
    for name, param in model.named_parameters():
        if "context_encoder" not in name:
            param.requires_grad = False

    ctx_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(ctx_params, lr=lr)
    logger.info(f"Training {sum(p.numel() for p in ctx_params)} context params")

    best_loss = float("inf")
    for epoch in range(1, stage1_epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in loader:
            a_id = batch["drug_a_id"].to(device)
            a_feat = batch["drug_a_features"].to(device)
            b_id = batch["drug_b_id"].to(device)
            b_feat = batch["drug_b_features"].to(device)
            ctx = batch["context"].to(device)
            target = batch["target_severity_with"].to(device)

            output = model(a_id, a_feat, b_id, b_feat, context=ctx)
            loss = criterion(output["severity_logits"], target)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(ctx_params, max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * a_id.shape[0]
            pred = output["severity_logits"].argmax(dim=-1)
            correct += (pred == target).sum().item()
            total += a_id.shape[0]

        avg_loss = total_loss / max(1, total)
        acc = correct / max(1, total)
        logger.info(f"Epoch {epoch}/{stage1_epochs} — loss: {avg_loss:.4f}, acc: {acc:.1%}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch, "stage": 1,
                "model_state_dict": model.state_dict(),
                "loss": best_loss, "phase": "4b_context",
            }, ckpt_path / "best_model_context.pt")

    # ── Stage 2: Unfreeze all, fine-tune ──
    logger.info(f"\n{'='*60}")
    logger.info(f"Stage 2: Full fine-tuning ({stage2_epochs} epochs)")
    logger.info(f"{'='*60}")

    for param in model.parameters():
        param.requires_grad = True

    optimizer = Adam(model.parameters(), lr=lr * 0.1)

    for epoch in range(1, stage2_epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in loader:
            a_id = batch["drug_a_id"].to(device)
            a_feat = batch["drug_a_features"].to(device)
            b_id = batch["drug_b_id"].to(device)
            b_feat = batch["drug_b_features"].to(device)
            ctx = batch["context"].to(device)
            target = batch["target_severity_with"].to(device)

            output = model(a_id, a_feat, b_id, b_feat, context=ctx)
            loss = criterion(output["severity_logits"], target)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * a_id.shape[0]
            pred = output["severity_logits"].argmax(dim=-1)
            correct += (pred == target).sum().item()
            total += a_id.shape[0]

        avg_loss = total_loss / max(1, total)
        acc = correct / max(1, total)
        logger.info(f"Epoch {epoch}/{stage2_epochs} — loss: {avg_loss:.4f}, acc: {acc:.1%}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": stage1_epochs + epoch, "stage": 2,
                "model_state_dict": model.state_dict(),
                "loss": best_loss, "phase": "4b_context",
            }, ckpt_path / "best_model_context.pt")

    logger.info(f"\nContext training complete. Best loss: {best_loss:.4f}")
    return model


def main() -> None:
    train_context(
        data_dir=os.getenv("PHARMLOOP_DATA_DIR", "./data/processed"),
        checkpoint_dir=os.getenv("PHARMLOOP_CHECKPOINT_DIR", "./checkpoints"),
        base_checkpoint=os.getenv(
            "PHARMLOOP_BASE_CHECKPOINT", "./checkpoints/best_model_phase4a.pt",
        ),
        context_data=os.getenv(
            "PHARMLOOP_CONTEXT_DATA", "./data/raw/context_training_data.json",
        ),
        pgx_data=os.getenv(
            "PHARMLOOP_PGX_DATA", "./data/raw/pharmacogenomic_examples.json",
        ),
    )


if __name__ == "__main__":
    main()
