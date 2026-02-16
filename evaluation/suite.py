"""
EvaluationSuite: comprehensive evaluation against held-out test data.

Measures safety, accuracy, calibration, unknown handling, and performance.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from pharmloop.inference import PharmLoopInference
from pharmloop.output import SEVERITY_NAMES


SEVERITY_ORDER = {
    "none": 0,
    "mild": 1,
    "moderate": 2,
    "severe": 3,
    "contraindicated": 4,
    "unknown": -1,
}


@dataclass
class EvaluationReport:
    """Results of running the full evaluation suite."""
    metrics: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        """Format metrics for display."""
        lines = ["PharmLoop Evaluation Report", "=" * 40]
        for key, value in sorted(self.metrics.items()):
            if "rate" in key or "accuracy" in key or "error" in key:
                lines.append(f"  {key}: {value:.1%}")
            elif "latency" in key or "steps" in key:
                lines.append(f"  {key}: {value:.1f}")
            else:
                lines.append(f"  {key}: {value:.4f}")
        return "\n".join(lines)


class EvaluationSuite:
    """Comprehensive evaluation against held-out test data."""

    def evaluate(
        self,
        engine: PharmLoopInference,
        test_data: list[dict],
    ) -> EvaluationReport:
        """
        Run all evaluation metrics.

        Args:
            engine: Loaded PharmLoopInference instance.
            test_data: List of interaction dicts with drug_a, drug_b,
                severity, mechanisms fields.

        Returns:
            EvaluationReport with all metrics.
        """
        metrics: dict[str, float] = {}

        # Safety (non-negotiable)
        metrics["false_negative_rate_severe"] = self._fnr_severe(engine, test_data)

        # Accuracy
        metrics["severity_accuracy_exact"] = self._severity_exact(engine, test_data)
        metrics["severity_accuracy_within_one"] = self._severity_within_one(engine, test_data)
        metrics["mechanism_accuracy_top1"] = self._mech_top1(engine, test_data)

        # Calibration
        metrics["confidence_calibration_error"] = self._calibration(engine, test_data)

        # Unknown handling
        metrics["unknown_detection_rate"] = self._unknown_detection(engine)

        # Performance
        metrics["avg_convergence_steps_known"] = self._avg_steps(engine, test_data)
        metrics["single_pair_latency_ms"] = self._latency_single(engine)
        metrics["ten_drug_polypharmacy_latency_ms"] = self._latency_poly(engine)

        return EvaluationReport(metrics=metrics)

    def _fnr_severe(
        self,
        engine: PharmLoopInference,
        test_data: list[dict],
    ) -> float:
        """False negative rate on severe/contraindicated interactions."""
        total_severe = 0
        false_negatives = 0

        for item in test_data:
            if item["severity"] in ("severe", "contraindicated"):
                total_severe += 1
                result = engine.check(item["drug_a"], item["drug_b"])
                if result.severity == "none":
                    false_negatives += 1

        return false_negatives / max(1, total_severe)

    def _severity_exact(
        self,
        engine: PharmLoopInference,
        test_data: list[dict],
    ) -> float:
        """Exact severity match accuracy."""
        correct = 0
        total = 0
        for item in test_data:
            if item["severity"] == "unknown":
                continue
            result = engine.check(item["drug_a"], item["drug_b"])
            total += 1
            if result.severity == item["severity"]:
                correct += 1
        return correct / max(1, total)

    def _severity_within_one(
        self,
        engine: PharmLoopInference,
        test_data: list[dict],
    ) -> float:
        """Severity accuracy allowing one level of tolerance."""
        correct = 0
        total = 0
        for item in test_data:
            if item["severity"] == "unknown":
                continue
            result = engine.check(item["drug_a"], item["drug_b"])
            total += 1
            pred_order = SEVERITY_ORDER.get(result.severity, -1)
            true_order = SEVERITY_ORDER.get(item["severity"], -1)
            if abs(pred_order - true_order) <= 1:
                correct += 1
        return correct / max(1, total)

    def _mech_top1(
        self,
        engine: PharmLoopInference,
        test_data: list[dict],
    ) -> float:
        """Top-1 mechanism accuracy (at least one correct mechanism)."""
        correct = 0
        total = 0
        for item in test_data:
            true_mechs = set(item.get("mechanisms", []))
            if not true_mechs:
                continue
            result = engine.check(item["drug_a"], item["drug_b"])
            total += 1
            pred_mechs = set(result.mechanisms)
            if pred_mechs & true_mechs:
                correct += 1
        return correct / max(1, total)

    def _calibration(
        self,
        engine: PharmLoopInference,
        test_data: list[dict],
    ) -> float:
        """
        Confidence calibration error.

        Bins predictions by confidence, computes |confidence - accuracy| per bin.
        """
        bins: dict[int, list[bool]] = {i: [] for i in range(10)}
        for item in test_data:
            if item["severity"] == "unknown":
                continue
            result = engine.check(item["drug_a"], item["drug_b"])
            bin_idx = min(9, int(result.confidence * 10))
            correct = result.severity == item["severity"]
            bins[bin_idx].append(correct)

        total_error = 0.0
        num_bins = 0
        for bin_idx, outcomes in bins.items():
            if not outcomes:
                continue
            expected_conf = (bin_idx + 0.5) / 10
            actual_acc = sum(outcomes) / len(outcomes)
            total_error += abs(expected_conf - actual_acc)
            num_bins += 1

        return total_error / max(1, num_bins)

    def _unknown_detection(self, engine: PharmLoopInference) -> float:
        """Detection rate for unknown drugs."""
        test_unknowns = [
            ("QZ-7734", "aspirin"),
            ("madeupdrugxyz", "metformin"),
            ("notarealmed", "warfarin"),
        ]
        detected = 0
        for drug_a, drug_b in test_unknowns:
            result = engine.check(drug_a, drug_b)
            if result.severity == "unknown" and result.unknown_drugs:
                detected += 1
        return detected / len(test_unknowns)

    def _avg_steps(
        self,
        engine: PharmLoopInference,
        test_data: list[dict],
    ) -> float:
        """Average convergence steps on known pairs."""
        total_steps = 0
        count = 0
        for item in test_data[:50]:  # Sample for speed
            if item["severity"] == "unknown":
                continue
            result = engine.check(item["drug_a"], item["drug_b"])
            total_steps += result.steps
            count += 1
        return total_steps / max(1, count)

    def _latency_single(self, engine: PharmLoopInference) -> float:
        """Single pair latency in milliseconds."""
        # Warmup
        engine.check("fluoxetine", "tramadol")

        start = time.perf_counter()
        for _ in range(20):
            engine.check("fluoxetine", "tramadol")
        return (time.perf_counter() - start) / 20 * 1000

    def _latency_poly(self, engine: PharmLoopInference) -> float:
        """10-drug polypharmacy latency in milliseconds."""
        drugs = [
            "fluoxetine", "tramadol", "warfarin", "metformin", "lisinopril",
            "omeprazole", "amlodipine", "simvastatin", "metoprolol", "acetaminophen",
        ]
        # Warmup
        engine.check_multiple(drugs)

        start = time.perf_counter()
        engine.check_multiple(drugs)
        return (time.perf_counter() - start) * 1000
