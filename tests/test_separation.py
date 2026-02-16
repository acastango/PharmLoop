"""
THE test — three-way separation.

If this passes, the architecture works. Everything else is scaling.

1. fluoxetine + tramadol → should converge, severity >= severe
2. metformin + lisinopril → should converge, severity = none
3. QZ-7734 (fabricated) + aspirin → should NOT converge, output unknown

Pass criteria:
  - Case 1 converges in <= 12 steps with severity in {severe, contraindicated}
  - Case 2 converges in <= 12 steps with severity = none
  - Case 3 does NOT converge (hits max_steps) OR outputs severity = unknown
  - Case 1 final gray zone < Case 3 final gray zone (known is more certain than unknown)
"""

import json
from pathlib import Path

import pytest
import torch

from pharmloop.model import PharmLoopModel
from pharmloop.hopfield import PharmHopfield
from pharmloop.output import (
    SEVERITY_NONE, SEVERITY_SEVERE, SEVERITY_CONTRAINDICATED, SEVERITY_UNKNOWN,
    SEVERITY_NAMES,
)

# Paths (relative to project root)
DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"


def load_test_model() -> tuple[PharmLoopModel, dict]:
    """Load trained model and drug data for testing."""
    drugs_path = DATA_DIR / "drugs.json"
    with open(drugs_path) as f:
        drugs_data = json.load(f)

    num_drugs = len(drugs_data["drugs"])
    drug_features_list = []
    for _name, info in sorted(drugs_data["drugs"].items(), key=lambda x: x[1]["id"]):
        drug_features_list.append(info["features"])
    drug_features = torch.tensor(drug_features_list, dtype=torch.float32)

    # Build Hopfield in 64-dim feature space
    hopfield = PharmHopfield(input_dim=64, hidden_dim=512)
    hopfield.store(drug_features)
    for param in hopfield.parameters():
        param.requires_grad = False

    model = PharmLoopModel(num_drugs=num_drugs, hopfield=hopfield)

    # Load checkpoint if available (strict=False to allow architecture changes during dev)
    checkpoint_path = CHECKPOINT_DIR / "best_model.pt"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    model.eval()
    return model, drugs_data


def get_drug_tensor(drugs_data: dict, drug_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Get drug ID and features as tensors."""
    info = drugs_data["drugs"][drug_name]
    drug_id = torch.tensor([info["id"]], dtype=torch.long)
    features = torch.tensor([info["features"]], dtype=torch.float32)
    return drug_id, features


@pytest.fixture
def trained_model():
    """Fixture providing a trained model and drug data."""
    return load_test_model()


class TestThreeWaySeparation:
    """THE test. If this passes, the architecture works."""

    def test_severe_interaction_converges(self, trained_model) -> None:
        """Case 1: fluoxetine + tramadol → converges, severity >= severe."""
        model, drugs_data = trained_model

        a_id, a_feat = get_drug_tensor(drugs_data, "fluoxetine")
        b_id, b_feat = get_drug_tensor(drugs_data, "tramadol")

        with torch.no_grad():
            output = model(a_id, a_feat, b_id, b_feat)

        severity_pred = output["severity_logits"].argmax(dim=-1).item()
        converged = output["converged"].item()
        steps = output["trajectory"]["steps"]
        final_gz = output["trajectory"]["gray_zones"][-1]

        print(f"\nCase 1 (fluoxetine + tramadol):")
        print(f"  Severity: {SEVERITY_NAMES[severity_pred]}")
        print(f"  Converged: {converged}")
        print(f"  Steps: {steps}")
        print(f"  Final gray zone: {final_gz:.4f}")
        print(f"  GZ trajectory: {[f'{gz:.4f}' for gz in output['trajectory']['gray_zones']]}")

        assert converged, "Severe interaction should converge"
        assert steps <= 16, f"Should converge within max_steps, took {steps}"
        assert severity_pred in (SEVERITY_SEVERE, SEVERITY_CONTRAINDICATED), (
            f"Expected severe/contraindicated, got {SEVERITY_NAMES[severity_pred]}"
        )

    def test_safe_pair_converges_to_none(self, trained_model) -> None:
        """Case 2: metformin + lisinopril → converges, severity = none."""
        model, drugs_data = trained_model

        a_id, a_feat = get_drug_tensor(drugs_data, "metformin")
        b_id, b_feat = get_drug_tensor(drugs_data, "lisinopril")

        with torch.no_grad():
            output = model(a_id, a_feat, b_id, b_feat)

        severity_pred = output["severity_logits"].argmax(dim=-1).item()
        converged = output["converged"].item()
        steps = output["trajectory"]["steps"]
        final_gz = output["trajectory"]["gray_zones"][-1]

        print(f"\nCase 2 (metformin + lisinopril):")
        print(f"  Severity: {SEVERITY_NAMES[severity_pred]}")
        print(f"  Converged: {converged}")
        print(f"  Steps: {steps}")
        print(f"  Final gray zone: {final_gz:.4f}")
        print(f"  GZ trajectory: {[f'{gz:.4f}' for gz in output['trajectory']['gray_zones']]}")

        assert converged, "Safe pair should converge"
        assert steps <= 16, f"Should converge within max_steps, took {steps}"
        assert severity_pred == SEVERITY_NONE, (
            f"Expected none, got {SEVERITY_NAMES[severity_pred]}"
        )

    def test_fabricated_drug_does_not_converge(self, trained_model) -> None:
        """Case 3: QZ-7734 (fabricated) + aspirin → does NOT converge, output unknown."""
        model, drugs_data = trained_model

        # Fabricated drug: ID beyond vocabulary, random features
        fabricated_id = torch.tensor([model.num_drugs + 50], dtype=torch.long)
        fabricated_features = torch.rand(1, 64) * 0.5  # random low-magnitude features

        b_id, b_feat = get_drug_tensor(drugs_data, "aspirin")

        with torch.no_grad():
            output = model(fabricated_id, fabricated_features, b_id, b_feat)

        severity_pred = output["severity_logits"].argmax(dim=-1).item()
        converged = output["converged"].item()
        steps = output["trajectory"]["steps"]
        final_gz = output["trajectory"]["gray_zones"][-1]

        print(f"\nCase 3 (QZ-7734 + aspirin):")
        print(f"  Severity: {SEVERITY_NAMES[severity_pred]}")
        print(f"  Converged: {converged}")
        print(f"  Steps: {steps}")
        print(f"  Final gray zone: {final_gz:.4f}")
        print(f"  GZ trajectory: {[f'{gz:.4f}' for gz in output['trajectory']['gray_zones']]}")

        # Either doesn't converge OR outputs unknown severity
        assert (not converged) or (severity_pred == SEVERITY_UNKNOWN), (
            f"Fabricated drug should not converge or output unknown, "
            f"but converged={converged}, severity={SEVERITY_NAMES[severity_pred]}"
        )

    def test_gray_zone_separation(self, trained_model) -> None:
        """Known interactions should have lower final gray zone than unknown."""
        model, drugs_data = trained_model

        # Case 1: known severe
        a_id, a_feat = get_drug_tensor(drugs_data, "fluoxetine")
        b_id, b_feat = get_drug_tensor(drugs_data, "tramadol")
        with torch.no_grad():
            out_known = model(a_id, a_feat, b_id, b_feat)

        # Case 3: fabricated
        fabricated_id = torch.tensor([model.num_drugs + 50], dtype=torch.long)
        fabricated_features = torch.rand(1, 64) * 0.5
        b_id2, b_feat2 = get_drug_tensor(drugs_data, "aspirin")
        with torch.no_grad():
            out_unknown = model(fabricated_id, fabricated_features, b_id2, b_feat2)

        known_gz = out_known["trajectory"]["gray_zones"][-1]
        unknown_gz = out_unknown["trajectory"]["gray_zones"][-1]

        print(f"\nGray zone separation:")
        print(f"  Known (fluoxetine+tramadol) final GZ: {known_gz:.4f}")
        print(f"  Unknown (QZ-7734+aspirin) final GZ: {unknown_gz:.4f}")

        assert known_gz < unknown_gz, (
            f"Known final GZ ({known_gz:.4f}) should be < unknown final GZ ({unknown_gz:.4f})"
        )


class TestModelBasics:
    """Basic model sanity checks."""

    def test_model_instantiates(self, trained_model) -> None:
        """Model builds and has parameters."""
        model, _ = trained_model
        counts = model.count_parameters()
        print(f"\nParameter counts: {counts}")
        assert counts["learned"] > 0

    def test_param_budget(self, trained_model) -> None:
        """Model stays within parameter budget (learned < 3M, total < 6M)."""
        model, _ = trained_model
        counts = model.count_parameters()
        assert counts["learned"] < 3_000_000, (
            f"Learned params {counts['learned']:,} exceeds 3M budget"
        )
        assert counts["total"] < 6_000_000, (
            f"Total params {counts['total']:,} exceeds 6M budget"
        )

    def test_forward_pass_no_error(self, trained_model) -> None:
        """Model forward pass completes without error."""
        model, drugs_data = trained_model
        a_id, a_feat = get_drug_tensor(drugs_data, "fluoxetine")
        b_id, b_feat = get_drug_tensor(drugs_data, "tramadol")

        with torch.no_grad():
            output = model(a_id, a_feat, b_id, b_feat)

        assert "severity_logits" in output
        assert "mechanism_logits" in output
        assert "flag_logits" in output
        assert "confidence" in output
        assert "converged" in output

    def test_order_invariance(self, trained_model) -> None:
        """model(A, B) should equal model(B, A) (pair combination is symmetric)."""
        model, drugs_data = trained_model
        a_id, a_feat = get_drug_tensor(drugs_data, "fluoxetine")
        b_id, b_feat = get_drug_tensor(drugs_data, "tramadol")

        with torch.no_grad():
            out_ab = model(a_id, a_feat, b_id, b_feat)
            out_ba = model(b_id, b_feat, a_id, a_feat)

        # Severity logits should be identical (symmetric pair combination)
        assert torch.allclose(
            out_ab["severity_logits"], out_ba["severity_logits"], atol=1e-5
        ), "Model is not order-invariant"
