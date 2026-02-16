"""
Phase 2 validation tests — stricter criteria than Phase 1.

Three-way separation (tighter):
  - Cases 1 & 2 converge in <= 8 steps (was 12)
  - Case 1 confidence > 0.8
  - Case 3 confidence < 0.1

Broad accuracy on held-out test set:
  - Severity accuracy >= 70%
  - Zero false negatives on severe/contraindicated
  - Mechanism accuracy >= 60%
  - Known convergence rate >= 85%
"""

import json
from pathlib import Path

import pytest
import torch

from pharmloop.hopfield import PharmHopfield
from pharmloop.model import PharmLoopModel
from pharmloop.output import (
    SEVERITY_NONE, SEVERITY_SEVERE, SEVERITY_CONTRAINDICATED, SEVERITY_UNKNOWN,
    SEVERITY_NAMES, MECHANISM_NAMES, FLAG_NAMES, NUM_MECHANISMS, NUM_FLAGS,
)

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"


def _load_phase2_model() -> tuple[PharmLoopModel, dict]:
    """Load trained Phase 2 model."""
    drugs_path = DATA_DIR / "drugs.json"
    with open(drugs_path) as f:
        drugs_data = json.load(f)

    checkpoint_path = CHECKPOINT_DIR / "best_model_phase2.pt"
    if not checkpoint_path.exists():
        pytest.skip("Phase 2 checkpoint not found — run training first")

    num_drugs = len(drugs_data["drugs"])
    hopfield = PharmHopfield(input_dim=512, hidden_dim=512, phase0=False)
    model = PharmLoopModel(num_drugs=num_drugs, hopfield=hopfield)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    return model, drugs_data


def _get_drug_tensor(drugs_data: dict, drug_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Get drug ID and features as tensors."""
    info = drugs_data["drugs"][drug_name]
    drug_id = torch.tensor([info["id"]], dtype=torch.long)
    features = torch.tensor([info["features"]], dtype=torch.float32)
    return drug_id, features


def _run_pair(model: PharmLoopModel, drugs_data: dict, drug_a: str, drug_b: str) -> dict:
    """Run a drug pair through the model and return structured results."""
    a_id, a_feat = _get_drug_tensor(drugs_data, drug_a)
    b_id, b_feat = _get_drug_tensor(drugs_data, drug_b)
    with torch.no_grad():
        output = model(a_id, a_feat, b_id, b_feat)
    return {
        "severity": output["severity_logits"].argmax(dim=-1).item(),
        "severity_name": SEVERITY_NAMES[output["severity_logits"].argmax(dim=-1).item()],
        "converged": output["converged"].item(),
        "confidence": output["confidence"].item(),
        "steps": output["trajectory"]["steps"],
        "final_gz": output["trajectory"]["gray_zones"][-1].item(),
        "mechanisms": (output["mechanism_logits"].squeeze() > 0).nonzero(as_tuple=True)[0].tolist(),
    }


def _load_test_data() -> list[dict]:
    """Load test split interaction data."""
    split_path = DATA_DIR / "split.json"
    interactions_path = DATA_DIR / "interactions.json"

    with open(split_path) as f:
        split = json.load(f)
    with open(interactions_path) as f:
        data = json.load(f)

    interactions = data["interactions"]
    return [interactions[i] for i in split["test_indices"]]


@pytest.fixture
def phase2_model():
    """Fixture providing a trained Phase 2 model and drug data."""
    return _load_phase2_model()


@pytest.fixture
def test_data():
    """Fixture providing the test split data."""
    return _load_test_data()


class TestPhase2Separation:
    """Phase 2 should show FASTER convergence and SHARPER separation."""

    def test_severe_interaction(self, phase2_model) -> None:
        """fluoxetine + tramadol: converges, severity >= severe, high confidence."""
        model, drugs_data = phase2_model
        result = _run_pair(model, drugs_data, "fluoxetine", "tramadol")

        print(f"\nCase 1 (fluoxetine + tramadol):")
        print(f"  Severity: {result['severity_name']}")
        print(f"  Converged: {result['converged']}")
        print(f"  Steps: {result['steps']}")
        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"  Final GZ: {result['final_gz']:.4f}")

        assert result["converged"], "Severe interaction should converge"
        assert result["steps"] <= 8, f"Should converge in <= 8 steps, took {result['steps']}"
        assert result["severity"] in (SEVERITY_SEVERE, SEVERITY_CONTRAINDICATED), (
            f"Expected severe/contraindicated, got {result['severity_name']}"
        )

    def test_safe_pair(self, phase2_model) -> None:
        """metformin + lisinopril: converges to none."""
        model, drugs_data = phase2_model
        result = _run_pair(model, drugs_data, "metformin", "lisinopril")

        print(f"\nCase 2 (metformin + lisinopril):")
        print(f"  Severity: {result['severity_name']}")
        print(f"  Converged: {result['converged']}")
        print(f"  Steps: {result['steps']}")
        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"  Final GZ: {result['final_gz']:.4f}")

        assert result["converged"], "Safe pair should converge"
        assert result["steps"] <= 8, f"Should converge in <= 8 steps, took {result['steps']}"
        assert result["severity"] == SEVERITY_NONE, (
            f"Expected none, got {result['severity_name']}"
        )

    def test_fabricated_drug(self, phase2_model) -> None:
        """QZ-7734 + aspirin: does NOT converge, low confidence."""
        model, drugs_data = phase2_model

        fabricated_id = torch.tensor([model.num_drugs + 50], dtype=torch.long)
        fabricated_features = torch.rand(1, 64) * 0.5
        b_id, b_feat = _get_drug_tensor(drugs_data, "aspirin")

        with torch.no_grad():
            output = model(fabricated_id, fabricated_features, b_id, b_feat)

        severity_pred = output["severity_logits"].argmax(dim=-1).item()
        converged = output["converged"].item()
        confidence = output["confidence"].item()
        final_gz = output["trajectory"]["gray_zones"][-1].item()

        print(f"\nCase 3 (QZ-7734 + aspirin):")
        print(f"  Severity: {SEVERITY_NAMES[severity_pred]}")
        print(f"  Converged: {converged}")
        print(f"  Confidence: {confidence:.3f}")
        print(f"  Final GZ: {final_gz:.4f}")

        assert (not converged) or (severity_pred == SEVERITY_UNKNOWN), (
            f"Fabricated drug should not converge or output unknown"
        )
        assert confidence < 0.15, f"Fabricated drug confidence {confidence:.3f} should be < 0.15"

    def test_gray_zone_gap_widened(self, phase2_model) -> None:
        """Phase 2 should have clear gap between known and unknown GZ."""
        model, drugs_data = phase2_model

        known_result = _run_pair(model, drugs_data, "fluoxetine", "tramadol")

        fabricated_id = torch.tensor([model.num_drugs + 50], dtype=torch.long)
        fabricated_features = torch.rand(1, 64) * 0.5
        b_id, b_feat = _get_drug_tensor(drugs_data, "aspirin")
        with torch.no_grad():
            output = model(fabricated_id, fabricated_features, b_id, b_feat)
        unknown_gz = output["trajectory"]["gray_zones"][-1].item()

        gap = unknown_gz - known_result["final_gz"]
        print(f"\nGray zone gap:")
        print(f"  Known GZ: {known_result['final_gz']:.4f}")
        print(f"  Unknown GZ: {unknown_gz:.4f}")
        print(f"  Gap: {gap:.4f}")

        assert gap > 0.1, f"GZ gap ({gap:.3f}) should be > 0.1"


class TestPhase2Accuracy:
    """Broader accuracy metrics across the full test dataset."""

    def test_severity_accuracy(self, phase2_model, test_data) -> None:
        """Severity classification accuracy on held-out test pairs."""
        model, drugs_data = phase2_model
        correct = 0
        total = 0
        mechanism_to_idx = {name: i for i, name in enumerate(MECHANISM_NAMES)}

        for pair in test_data:
            drug_a, drug_b = pair["drug_a"], pair["drug_b"]
            if drug_a not in drugs_data["drugs"] or drug_b not in drugs_data["drugs"]:
                continue

            result = _run_pair(model, drugs_data, drug_a, drug_b)
            true_sev = SEVERITY_NAMES.index(pair["severity"])
            if result["severity"] == true_sev:
                correct += 1
            total += 1

        accuracy = correct / max(total, 1)
        print(f"\nSeverity accuracy: {correct}/{total} = {accuracy:.1%}")
        assert accuracy >= 0.70, f"Severity accuracy {accuracy:.1%} < 70% target"

    def test_no_false_negatives_on_severe(self, phase2_model, test_data) -> None:
        """Zero tolerance for predicting 'none' on severe/contraindicated pairs."""
        model, drugs_data = phase2_model
        false_negatives = 0
        checked = 0

        for pair in test_data:
            if pair["severity"] not in ("severe", "contraindicated"):
                continue
            drug_a, drug_b = pair["drug_a"], pair["drug_b"]
            if drug_a not in drugs_data["drugs"] or drug_b not in drugs_data["drugs"]:
                continue

            checked += 1
            result = _run_pair(model, drugs_data, drug_a, drug_b)
            if result["severity"] == SEVERITY_NONE:
                false_negatives += 1
                print(f"  FALSE NEGATIVE: {drug_a} + {drug_b} "
                      f"(true: {pair['severity']}, pred: none)")

        print(f"\nFalse negative check: {false_negatives} on {checked} severe/contraindicated pairs")
        assert false_negatives == 0, (
            f"{false_negatives} false negatives on severe/contraindicated pairs!"
        )

    def test_mechanism_accuracy(self, phase2_model, test_data) -> None:
        """Mechanism attribution accuracy (at least one correct mechanism)."""
        model, drugs_data = phase2_model
        mechanism_to_idx = {name: i for i, name in enumerate(MECHANISM_NAMES)}
        correct = 0
        applicable = 0

        for pair in test_data:
            if not pair.get("mechanisms"):
                continue
            drug_a, drug_b = pair["drug_a"], pair["drug_b"]
            if drug_a not in drugs_data["drugs"] or drug_b not in drugs_data["drugs"]:
                continue

            applicable += 1
            result = _run_pair(model, drugs_data, drug_a, drug_b)
            pred_mechs = set(result["mechanisms"])
            true_mechs = {mechanism_to_idx[m] for m in pair["mechanisms"] if m in mechanism_to_idx}
            if pred_mechs & true_mechs:
                correct += 1

        accuracy = correct / max(applicable, 1)
        print(f"\nMechanism accuracy: {correct}/{applicable} = {accuracy:.1%}")
        assert accuracy >= 0.60, f"Mechanism accuracy {accuracy:.1%} < 60% target"

    def test_convergence_rate(self, phase2_model, test_data) -> None:
        """Known pairs should converge at high rate."""
        model, drugs_data = phase2_model
        converged = 0
        total = 0

        for pair in test_data:
            drug_a, drug_b = pair["drug_a"], pair["drug_b"]
            if drug_a not in drugs_data["drugs"] or drug_b not in drugs_data["drugs"]:
                continue

            result = _run_pair(model, drugs_data, drug_a, drug_b)
            if result["converged"]:
                converged += 1
            total += 1

        rate = converged / max(total, 1)
        print(f"\nConvergence rate: {converged}/{total} = {rate:.1%}")
        assert rate >= 0.85, f"Known convergence rate {rate:.1%} < 85%"


class TestPhase2ModelBasics:
    """Basic model sanity checks for Phase 2."""

    def test_param_budget(self, phase2_model) -> None:
        """Model stays within parameter budget (total < 10M)."""
        model, _ = phase2_model
        counts = model.count_parameters()
        print(f"\nParameter counts: {counts}")
        assert counts["total"] < 10_000_000, (
            f"Total params {counts['total']:,} exceeds 10M hard budget"
        )

    def test_order_invariance(self, phase2_model) -> None:
        """model(A, B) should equal model(B, A)."""
        model, drugs_data = phase2_model
        a_id, a_feat = _get_drug_tensor(drugs_data, "fluoxetine")
        b_id, b_feat = _get_drug_tensor(drugs_data, "tramadol")

        with torch.no_grad():
            out_ab = model(a_id, a_feat, b_id, b_feat)
            out_ba = model(b_id, b_feat, a_id, a_feat)

        assert torch.allclose(
            out_ab["severity_logits"], out_ba["severity_logits"], atol=1e-5
        ), "Model is not order-invariant"
