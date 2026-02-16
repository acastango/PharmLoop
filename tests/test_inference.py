"""Tests for PharmLoopInference end-to-end pipeline."""

import json
from pathlib import Path

import pytest

from pharmloop.inference import PharmLoopInference, InteractionResult

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"


def _get_checkpoint_path() -> str:
    """Find best available checkpoint."""
    for name in ["best_model_phase3.pt", "final_model_phase3.pt", "best_model_phase2.pt"]:
        path = CHECKPOINT_DIR / name
        if path.exists():
            return str(path)
    pytest.skip("No checkpoint found")


@pytest.fixture
def engine():
    """Fixture providing a loaded inference engine."""
    path = _get_checkpoint_path()
    return PharmLoopInference.load(path, str(DATA_DIR))


class TestInferencePipeline:

    def test_check_known_interaction(self, engine) -> None:
        """fluoxetine + tramadol returns complete result with narrative."""
        result = engine.check("fluoxetine", "tramadol")

        assert isinstance(result, InteractionResult)
        assert result.drug_a == "fluoxetine"
        assert result.drug_b == "tramadol"
        assert result.severity in ("severe", "contraindicated")
        assert result.confidence > 0
        assert result.narrative  # non-empty
        assert len(result.gray_zone_trajectory) > 0

        print(f"\n{'='*60}")
        print(f"fluoxetine + tramadol")
        print(f"{'='*60}")
        print(f"Severity: {result.severity}")
        print(f"Mechanisms: {result.mechanisms}")
        print(f"Flags: {result.flags}")
        print(f"Confidence: {result.confidence:.1%}")
        print(f"Converged: {result.converged}")
        print(f"Steps: {result.steps}")
        print(f"\nNarrative:\n{result.narrative}")

    def test_check_safe_pair(self, engine) -> None:
        """metformin + lisinopril returns none severity."""
        result = engine.check("metformin", "lisinopril")

        assert result.severity == "none"
        assert "no clinically significant" in result.narrative.lower()

        print(f"\n{'='*60}")
        print(f"metformin + lisinopril")
        print(f"{'='*60}")
        print(f"Severity: {result.severity}")
        print(f"\nNarrative:\n{result.narrative}")

    def test_check_unknown_drug(self, engine) -> None:
        """Unknown drug returns unknown result with clear narrative."""
        result = engine.check("QZ-7734", "aspirin")

        assert result.severity == "unknown"
        assert result.confidence == 0.0
        assert result.converged is False
        assert result.steps == 0
        assert "QZ-7734" in result.unknown_drugs
        assert "insufficient data" in result.narrative.lower()

        print(f"\n{'='*60}")
        print(f"QZ-7734 + aspirin")
        print(f"{'='*60}")
        print(f"\nNarrative:\n{result.narrative}")

    def test_check_both_unknown(self, engine) -> None:
        """Both drugs unknown returns unknown."""
        result = engine.check("fake_drug_1", "fake_drug_2")
        assert result.severity == "unknown"
        assert len(result.unknown_drugs) == 2

    def test_case_insensitive(self, engine) -> None:
        """Drug names should be case-insensitive."""
        result = engine.check("Fluoxetine", "TRAMADOL")
        assert result.severity in ("severe", "contraindicated")

    def test_result_has_all_fields(self, engine) -> None:
        """InteractionResult contains all expected fields."""
        result = engine.check("fluoxetine", "tramadol")
        assert hasattr(result, "drug_a")
        assert hasattr(result, "drug_b")
        assert hasattr(result, "severity")
        assert hasattr(result, "mechanisms")
        assert hasattr(result, "flags")
        assert hasattr(result, "confidence")
        assert hasattr(result, "converged")
        assert hasattr(result, "steps")
        assert hasattr(result, "partial_convergence")
        assert hasattr(result, "narrative")
        assert hasattr(result, "gray_zone_trajectory")

    def test_order_invariance(self, engine) -> None:
        """check(A, B) and check(B, A) should produce same severity."""
        r1 = engine.check("fluoxetine", "tramadol")
        r2 = engine.check("tramadol", "fluoxetine")
        assert r1.severity == r2.severity
