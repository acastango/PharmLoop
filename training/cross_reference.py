"""
Cross-reference evaluation: compare PharmLoop predictions against external databases.

Systematically compares severity, mechanism, and confidence predictions
against reference data (Lexicomp, Micromedex, Clinical Pharmacology).
Categorizes disagreements for pharmacist review.

Usage:
    evaluator = CrossReferenceEvaluator(engine)
    evaluator.load_reference("data/raw/reference_interactions.json")
    report = evaluator.evaluate()
    report.save("cross_reference_report.json")
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from pharmloop.output import SEVERITY_NAMES

logger = logging.getLogger("pharmloop.cross_reference")

SEVERITY_ORDER = {
    "none": 0,
    "mild": 1,
    "moderate": 2,
    "severe": 3,
    "contraindicated": 4,
    "unknown": -1,
}


@dataclass
class DisagreementRecord:
    """Record of a disagreement between PharmLoop and a reference database."""
    drug_a: str
    drug_b: str
    pharmloop_severity: str
    reference_severity: str
    reference_source: str
    disagreement_type: str
    pharmloop_mechanisms: list[str] = field(default_factory=list)
    reference_mechanisms: list[str] = field(default_factory=list)
    pharmloop_confidence: float = 0.0
    clinical_significance: str = ""  # filled by pharmacist review


@dataclass
class AgreementRecord:
    """Record of agreement between PharmLoop and a reference."""
    drug_a: str
    drug_b: str
    severity: str
    reference_source: str
    pharmloop_confidence: float = 0.0


@dataclass
class CrossReferenceReport:
    """Full cross-reference evaluation report."""
    total_compared: int = 0
    agreements: list[AgreementRecord] = field(default_factory=list)
    disagreements: list[DisagreementRecord] = field(default_factory=list)
    pharmloop_more_severe: int = 0
    pharmloop_less_severe: int = 0
    mechanism_differs: int = 0
    pharmloop_unknown: int = 0
    reference_no_interaction: int = 0

    @property
    def agreement_rate(self) -> float:
        """Fraction of pairs where severity matches."""
        if self.total_compared == 0:
            return 0.0
        return len(self.agreements) / self.total_compared

    @property
    def dangerous_disagreements(self) -> list[DisagreementRecord]:
        """Disagreements where PharmLoop is less severe than reference."""
        return [d for d in self.disagreements
                if d.disagreement_type == "pharmloop_less_severe"]

    def save(self, path: str | Path) -> None:
        """Save report to JSON."""
        data = {
            "total_compared": self.total_compared,
            "agreement_rate": self.agreement_rate,
            "agreements": len(self.agreements),
            "total_disagreements": len(self.disagreements),
            "pharmloop_more_severe": self.pharmloop_more_severe,
            "pharmloop_less_severe": self.pharmloop_less_severe,
            "mechanism_differs": self.mechanism_differs,
            "pharmloop_unknown": self.pharmloop_unknown,
            "reference_no_interaction": self.reference_no_interaction,
            "dangerous_disagreements": [
                {
                    "drug_a": d.drug_a,
                    "drug_b": d.drug_b,
                    "pharmloop_severity": d.pharmloop_severity,
                    "reference_severity": d.reference_severity,
                    "reference_source": d.reference_source,
                    "pharmloop_confidence": d.pharmloop_confidence,
                }
                for d in self.dangerous_disagreements
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def log_summary(self) -> None:
        """Log a summary of the cross-reference evaluation."""
        logger.info(f"Cross-reference: {self.total_compared} pairs compared")
        logger.info(f"  Agreement rate: {self.agreement_rate:.1%}")
        logger.info(f"  PharmLoop more severe: {self.pharmloop_more_severe}")
        logger.info(f"  PharmLoop less severe: {self.pharmloop_less_severe} "
                     f"(INVESTIGATE)")
        logger.info(f"  Mechanism differs: {self.mechanism_differs}")
        logger.info(f"  PharmLoop unknown: {self.pharmloop_unknown}")


class CrossReferenceEvaluator:
    """
    Evaluates PharmLoop predictions against reference interaction databases.

    Loads reference data and runs the inference engine against each pair,
    categorizing agreements and disagreements.

    Args:
        engine: PharmLoopInference engine (must be loaded).
    """

    def __init__(self, engine: object) -> None:
        self.engine = engine
        self.reference_data: list[dict] = []

    def load_reference(self, path: str | Path) -> int:
        """
        Load reference interaction data.

        Expected JSON format: list of dicts with:
          - drug_a: str
          - drug_b: str
          - severity: str (none/mild/moderate/severe/contraindicated)
          - mechanisms: list[str] (optional)
          - source: str (e.g., "lexicomp", "micromedex")

        Returns:
            Number of reference entries loaded.
        """
        with open(path) as f:
            self.reference_data = json.load(f)
        logger.info(f"Loaded {len(self.reference_data)} reference interactions")
        return len(self.reference_data)

    def evaluate(self) -> CrossReferenceReport:
        """
        Run cross-reference evaluation against all loaded reference data.

        For each reference pair, runs the PharmLoop engine and compares
        severity predictions. Categorizes disagreements.

        Returns:
            CrossReferenceReport with all comparisons.
        """
        report = CrossReferenceReport()

        for ref in self.reference_data:
            drug_a = ref["drug_a"]
            drug_b = ref["drug_b"]
            ref_severity = ref["severity"]
            ref_source = ref.get("source", "unknown")
            ref_mechanisms = ref.get("mechanisms", [])

            # Run PharmLoop
            result = self.engine.check(drug_a, drug_b)
            report.total_compared += 1

            pl_severity = result.severity
            pl_sev_order = SEVERITY_ORDER.get(pl_severity, -1)
            ref_sev_order = SEVERITY_ORDER.get(ref_severity, -1)

            if pl_severity == ref_severity:
                report.agreements.append(AgreementRecord(
                    drug_a=drug_a,
                    drug_b=drug_b,
                    severity=pl_severity,
                    reference_source=ref_source,
                    pharmloop_confidence=result.confidence,
                ))
                continue

            # Classify disagreement
            if pl_severity == "unknown":
                dtype = "pharmloop_unknown"
                report.pharmloop_unknown += 1
            elif ref_severity == "none" and pl_sev_order > 0:
                dtype = "reference_no_interaction"
                report.reference_no_interaction += 1
            elif pl_sev_order > ref_sev_order:
                dtype = "pharmloop_more_severe"
                report.pharmloop_more_severe += 1
            elif pl_sev_order < ref_sev_order:
                dtype = "pharmloop_less_severe"
                report.pharmloop_less_severe += 1
            else:
                dtype = "mechanism_differs"
                report.mechanism_differs += 1

            report.disagreements.append(DisagreementRecord(
                drug_a=drug_a,
                drug_b=drug_b,
                pharmloop_severity=pl_severity,
                reference_severity=ref_severity,
                reference_source=ref_source,
                disagreement_type=dtype,
                pharmloop_mechanisms=result.mechanisms,
                reference_mechanisms=ref_mechanisms,
                pharmloop_confidence=result.confidence,
            ))

        return report
