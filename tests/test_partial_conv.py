"""Tests for PartialConvergenceAnalyzer."""

import json
from pathlib import Path

import pytest
import torch

from pharmloop.hopfield import PharmHopfield
from pharmloop.model import PharmLoopModel
from pharmloop.partial_convergence import PartialConvergenceAnalyzer

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"


def _load_model():
    """Load trained model (Phase 2 or 3)."""
    drugs_path = DATA_DIR / "drugs.json"
    with open(drugs_path) as f:
        drugs_data = json.load(f)

    # Try Phase 3 first, fall back to Phase 2
    for ckpt_name in ["best_model_phase3.pt", "best_model_phase2.pt"]:
        ckpt_path = CHECKPOINT_DIR / ckpt_name
        if ckpt_path.exists():
            break
    else:
        pytest.skip("No checkpoint found")

    num_drugs = len(drugs_data["drugs"])
    hopfield = PharmHopfield(input_dim=512, hidden_dim=512, phase0=False)
    model = PharmLoopModel(num_drugs=num_drugs, hopfield=hopfield)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    return model, drugs_data


@pytest.fixture
def model_and_data():
    return _load_model()


class TestPartialConvergenceAnalyzer:

    def test_analyzer_builds_from_output_head(self, model_and_data) -> None:
        """Analyzer can be built from a trained model's output head."""
        model, _ = model_and_data
        analyzer = PartialConvergenceAnalyzer(model.output_head)
        assert len(analyzer.severity_dims) > 0
        assert len(analyzer.mechanism_dims) > 0
        assert len(analyzer.flag_dims) > 0

    def test_known_pair_mostly_settled(self, model_and_data) -> None:
        """Known interaction should have mostly settled aspects."""
        model, drugs_data = model_and_data
        analyzer = PartialConvergenceAnalyzer(model.output_head)

        info_a = drugs_data["drugs"]["fluoxetine"]
        info_b = drugs_data["drugs"]["tramadol"]
        a_id = torch.tensor([info_a["id"]])
        a_feat = torch.tensor([info_a["features"]])
        b_id = torch.tensor([info_b["id"]])
        b_feat = torch.tensor([info_b["features"]])

        with torch.no_grad():
            output = model(a_id, a_feat, b_id, b_feat)

        result = analyzer.analyze(output["trajectory"]["velocities"][-1])
        print(f"\nfluoxetine+tramadol partial convergence:")
        print(f"  Settled: {result['settled_aspects']}")
        print(f"  Unsettled: {result['unsettled_aspects']}")
        print(f"  Severity GZ: {result['severity_gz']:.4f}")
        print(f"  Mechanism GZ: {result['mechanism_gz']:.4f}")
        print(f"  Flags GZ: {result['flags_gz']:.4f}")

        # At least severity should be settled for a known interaction
        assert len(result["settled_aspects"]) >= 1, "Known pair should have at least one settled aspect"

    def test_fabricated_drug_mostly_unsettled(self, model_and_data) -> None:
        """Fabricated drug should have mostly unsettled aspects."""
        model, drugs_data = model_and_data
        analyzer = PartialConvergenceAnalyzer(model.output_head)

        fab_id = torch.tensor([model.num_drugs + 50])
        fab_feat = torch.rand(1, 64) * 0.5
        info_b = drugs_data["drugs"]["aspirin"]
        b_id = torch.tensor([info_b["id"]])
        b_feat = torch.tensor([info_b["features"]])

        with torch.no_grad():
            output = model(fab_id, fab_feat, b_id, b_feat)

        result = analyzer.analyze(output["trajectory"]["velocities"][-1])
        print(f"\nQZ-7734+aspirin partial convergence:")
        print(f"  Settled: {result['settled_aspects']}")
        print(f"  Unsettled: {result['unsettled_aspects']}")
        print(f"  Severity GZ: {result['severity_gz']:.4f}")
        print(f"  Mechanism GZ: {result['mechanism_gz']:.4f}")
        print(f"  Flags GZ: {result['flags_gz']:.4f}")

        assert len(result["unsettled_aspects"]) >= 1, "Fabricated drug should have at least one unsettled aspect"

    def test_analyze_returns_expected_keys(self, model_and_data) -> None:
        """Analyze result has all expected keys."""
        model, _ = model_and_data
        analyzer = PartialConvergenceAnalyzer(model.output_head)
        fake_v = torch.randn(1, 512) * 0.1
        result = analyzer.analyze(fake_v)

        expected_keys = {
            "severity_settled", "mechanism_settled", "flags_settled",
            "settled_aspects", "unsettled_aspects", "partial_convergence",
            "severity_gz", "mechanism_gz", "flags_gz",
        }
        assert set(result.keys()) == expected_keys
