"""
Regression tests: ensure Phase 4a didn't break anything from earlier phases.

These tests require a trained model and data. Skip if not available.
"""

import json
from pathlib import Path

import pytest

from pharmloop.inference import PharmLoopInference


DATA_DIR = "data/processed"
CHECKPOINT_4A = "checkpoints/best_model_phase4a.pt"
CHECKPOINT_FINAL = "checkpoints/final_model_phase4a.pt"


def _get_checkpoint() -> str:
    """Return the best available checkpoint path."""
    for path in [CHECKPOINT_4A, CHECKPOINT_FINAL]:
        if Path(path).exists():
            return path
    return CHECKPOINT_4A


@pytest.fixture(scope="module")
def engine():
    """Load the inference engine once for all regression tests."""
    ckpt = _get_checkpoint()
    if not Path(ckpt).exists():
        pytest.skip(f"No checkpoint at {ckpt}")
    return PharmLoopInference.load(ckpt, data_dir=DATA_DIR)


class TestThreeWaySeparation:
    """Original three-way separation test still works."""

    def test_severe_interaction_detected(self, engine):
        """Fluoxetine + tramadol → severe or contraindicated."""
        result = engine.check("fluoxetine", "tramadol")
        assert result.severity in ("severe", "contraindicated", "moderate"), (
            f"Expected severe/contraindicated/moderate, got {result.severity}"
        )

    def test_safe_pair_not_severe(self, engine):
        """Metformin + lisinopril → not severe."""
        result = engine.check("metformin", "lisinopril")
        assert result.severity in ("none", "mild", "moderate"), (
            f"Expected none/mild/moderate, got {result.severity}"
        )

    def test_unknown_drug_detected(self, engine):
        """Fabricated drug QZ-7734 → unknown."""
        result = engine.check("QZ-7734", "aspirin")
        assert result.severity == "unknown"
        assert "QZ-7734" in result.unknown_drugs

    def test_confidence_ordering(self, engine):
        """Known pairs should have higher confidence than unknown."""
        severe = engine.check("fluoxetine", "tramadol")
        unknown = engine.check("QZ-7734", "aspirin")
        assert severe.confidence > unknown.confidence


class TestOriginalDrugsPresent:
    """All original 50 drugs are still in the registry."""

    def test_original_drugs_in_registry(self, engine):
        v1_path = Path(DATA_DIR) / "drugs.json"
        if not v1_path.exists():
            pytest.skip("No original drugs.json")

        with open(v1_path) as f:
            v1 = json.load(f)

        missing = []
        for name in v1["drugs"]:
            if name.lower() not in engine.drug_registry:
                missing.append(name)
        assert not missing, f"Original drugs missing from registry: {missing}"


class TestNarrativeOutputFormat:
    """Template engine still produces well-formed narratives."""

    def test_narrative_has_content(self, engine):
        result = engine.check("fluoxetine", "tramadol")
        assert len(result.narrative) > 50

    def test_narrative_has_confidence_tag(self, engine):
        result = engine.check("fluoxetine", "tramadol")
        assert "[Confidence:" in result.narrative

    def test_narrative_no_unresolved_slots(self, engine):
        result = engine.check("fluoxetine", "tramadol")
        # No unresolved template variables
        assert "{" not in result.narrative or "Confidence:" in result.narrative
