"""Tests for Phase 2 Hopfield — retrieval quality in learned 512-dim space."""

import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from pharmloop.hopfield import PharmHopfield
from pharmloop.model import PharmLoopModel

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

    # Build Phase 2 Hopfield (512-dim, learned projections)
    hopfield = PharmHopfield(input_dim=512, hidden_dim=512, phase0=False)
    model = PharmLoopModel(num_drugs=num_drugs, hopfield=hopfield)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    return model, drugs_data


def _compute_pair_state(model: PharmLoopModel, drugs_data: dict, drug_a: str, drug_b: str) -> torch.Tensor:
    """Compute pair state for a given drug pair."""
    info_a = drugs_data["drugs"][drug_a]
    info_b = drugs_data["drugs"][drug_b]
    a_id = torch.tensor([info_a["id"]], dtype=torch.long)
    a_feat = torch.tensor([info_a["features"]], dtype=torch.float32)
    b_id = torch.tensor([info_b["id"]], dtype=torch.long)
    b_feat = torch.tensor([info_b["features"]], dtype=torch.float32)
    with torch.no_grad():
        return model.compute_pair_state(a_id, a_feat, b_id, b_feat)


@pytest.fixture
def phase2_model():
    """Fixture providing a trained Phase 2 model and drug data."""
    return _load_phase2_model()


class TestPhase2Hopfield:
    """Verify the Hopfield is actually helping."""

    def test_hopfield_has_patterns(self, phase2_model) -> None:
        """Phase 2 Hopfield has stored patterns in 512-dim space."""
        model, _ = phase2_model
        hopfield = model.cell.hopfield
        assert hopfield.count > 0, "Hopfield has no stored patterns"
        assert hopfield.input_dim == 512, f"Expected 512-dim, got {hopfield.input_dim}"
        assert not hopfield.phase0, "Phase 2 Hopfield should not be in phase0 mode"
        print(f"\nHopfield: {hopfield.count} patterns in {hopfield.input_dim}-dim space")

    def test_no_dimension_projection(self, phase2_model) -> None:
        """Phase 2 should not have dimension projection layers in oscillator."""
        model, _ = phase2_model
        assert model.cell.hopfield_query_proj is None, "hopfield_query_proj should be None in Phase 2"
        assert model.cell.hopfield_value_proj is None, "hopfield_value_proj should be None in Phase 2"

    def test_similar_pairs_retrieve_each_other(self, phase2_model) -> None:
        """SSRI+opioid pairs should retrieve patterns similar to other SSRI+opioid pairs."""
        model, drugs_data = phase2_model
        query = _compute_pair_state(model, drugs_data, "fluoxetine", "tramadol")
        retrieved = model.cell.hopfield.retrieve(query, beta=5.0)
        reference = _compute_pair_state(model, drugs_data, "sertraline", "codeine")

        cosine = F.cosine_similarity(retrieved, reference).item()
        print(f"\nSSRI+opioid retrieval: fluoxetine+tramadol → cosine to sertraline+codeine = {cosine:.3f}")
        assert cosine > 0.3, f"Similar pair retrieval cosine {cosine:.3f} too low"

    def test_dissimilar_pairs_dont_retrieve(self, phase2_model) -> None:
        """SSRI+opioid should NOT strongly retrieve unrelated pairs."""
        model, drugs_data = phase2_model
        query = _compute_pair_state(model, drugs_data, "fluoxetine", "tramadol")
        retrieved = model.cell.hopfield.retrieve(query, beta=5.0)
        reference = _compute_pair_state(model, drugs_data, "metformin", "lisinopril")

        cosine_to_dissimilar = F.cosine_similarity(retrieved, reference).item()
        cosine_to_self = F.cosine_similarity(retrieved, query).item()

        print(f"\nRetrieval similarity:")
        print(f"  To self (fluoxetine+tramadol): {cosine_to_self:.3f}")
        print(f"  To dissimilar (metformin+lisinopril): {cosine_to_dissimilar:.3f}")

        assert cosine_to_self > cosine_to_dissimilar, (
            "Retrieved pattern should be more similar to query than to dissimilar pair"
        )

    def test_retrieval_entropy_healthy(self, phase2_model) -> None:
        """Retrieval weights should not be collapsed or uniform."""
        model, _ = phase2_model
        hopfield = model.cell.hopfield
        n = hopfield.count

        # Use stored values as sample queries
        values = hopfield.stored_values[:n]
        q = hopfield.query_proj(values)
        keys = hopfield.stored_keys[:n]
        scores = q @ keys.T
        weights = torch.softmax(scores, dim=-1)
        entropy = -(weights * (weights + 1e-10).log()).sum(dim=-1).mean()
        max_entropy = torch.log(torch.tensor(float(n)))
        normalized = (entropy / max_entropy).item()

        print(f"\nRetrieval entropy: {entropy:.2f} / {max_entropy:.2f} = {normalized:.3f}")
        assert 0.05 < normalized < 0.95, (
            f"Retrieval entropy {normalized:.3f} is unhealthy (expected 0.05-0.95)"
        )
