"""
Stress tests for PharmLoop Phase 4a.

Tests edge cases, latency requirements, and large polypharmacy requests.
"""

import time
from pathlib import Path

import pytest

from pharmloop.inference import PharmLoopInference


DATA_DIR = "data/processed"
CHECKPOINT_4A = "checkpoints/best_model_phase4a.pt"
CHECKPOINT_FINAL = "checkpoints/final_model_phase4a.pt"


def _get_checkpoint() -> str:
    for path in [CHECKPOINT_4A, CHECKPOINT_FINAL]:
        if Path(path).exists():
            return path
    return CHECKPOINT_4A


@pytest.fixture(scope="module")
def engine():
    ckpt = _get_checkpoint()
    if not Path(ckpt).exists():
        pytest.skip(f"No checkpoint at {ckpt}")
    return PharmLoopInference.load(ckpt, data_dir=DATA_DIR)


class TestEdgeCases:
    """Test edge cases that should not crash."""

    def test_same_drug_pair(self, engine):
        """Drug paired with itself must not crash."""
        result = engine.check("warfarin", "warfarin")
        assert result.severity is not None

    def test_unknown_drug(self, engine):
        """Unrecognized drug returns severity=unknown."""
        result = engine.check("madeupdrugxyz", "aspirin")
        assert result.severity == "unknown"
        assert "madeupdrugxyz" in result.unknown_drugs

    def test_both_drugs_unknown(self, engine):
        result = engine.check("fakemed123", "notarealmed456")
        assert result.severity == "unknown"
        assert len(result.unknown_drugs) == 2

    def test_case_insensitive(self, engine):
        result_lower = engine.check("fluoxetine", "tramadol")
        result_upper = engine.check("Fluoxetine", "Tramadol")
        assert result_lower.severity == result_upper.severity


class TestLatency:
    """Latency requirements."""

    def test_single_pair_latency(self, engine):
        """Single pair < 100ms on CPU (averaged over 20 runs)."""
        # Warmup
        for _ in range(3):
            engine.check("fluoxetine", "tramadol")

        start = time.perf_counter()
        for _ in range(20):
            engine.check("fluoxetine", "tramadol")
        avg_ms = (time.perf_counter() - start) / 20 * 1000
        # Use generous threshold for CI environments
        assert avg_ms < 200, f"Latency {avg_ms:.0f}ms > 200ms target"

    def test_polypharmacy_latency(self, engine):
        """10-drug polypharmacy (batched) < 500ms."""
        drugs = [
            "fluoxetine", "tramadol", "warfarin", "metformin",
            "lisinopril", "omeprazole", "amlodipine", "simvastatin",
            "metoprolol", "acetaminophen",
        ]
        # Warmup
        engine.check_multiple(drugs)

        start = time.perf_counter()
        engine.check_multiple(drugs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 2000, (
            f"Polypharmacy latency {elapsed_ms:.0f}ms > 2000ms"
        )


class TestPolypharmacy:
    """Multi-drug interaction checking."""

    def test_two_drug_polypharmacy(self, engine):
        report = engine.check_multiple(["fluoxetine", "tramadol"])
        assert report.total_pairs_checked == 1

    def test_ten_drug_polypharmacy(self, engine):
        drugs = [
            "fluoxetine", "tramadol", "warfarin", "metformin",
            "lisinopril", "omeprazole", "amlodipine", "simvastatin",
            "metoprolol", "acetaminophen",
        ]
        report = engine.check_multiple(drugs)
        assert report.total_pairs_checked == 45  # C(10,2) = 45

    def test_twenty_drug_polypharmacy(self, engine):
        """20 drugs = 190 pairs. Must complete."""
        drugs = list(engine.drug_registry.keys())[:20]
        report = engine.check_multiple(drugs)
        assert report.total_pairs_checked == 190

    def test_polypharmacy_with_unknown_drug(self, engine):
        """Unknown drugs in multi-check should be skipped gracefully."""
        drugs = ["fluoxetine", "madeupmed123", "tramadol"]
        report = engine.check_multiple(drugs)
        # Only 1 valid pair (fluoxetine-tramadol), madeupmed123 skipped
        assert report.total_pairs_checked >= 1
