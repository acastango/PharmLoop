"""
Data quality report for the extraction pipeline.

Tracks completeness, ambiguity, and flags items for manual review.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path

logger = logging.getLogger("pharmloop.pipeline")


@dataclass
class DataQualityReport:
    """Track data quality issues during extraction."""
    total_drugs: int = 0
    drugs_with_complete_features: int = 0
    drugs_with_fallback_values: int = 0
    drugs_flagged_for_review: int = 0
    total_interactions: int = 0
    interactions_with_clear_mechanism: int = 0
    interactions_with_ambiguous_mechanism: int = 0
    severity_conflicts: int = 0
    severity_conflict_resolutions: list[dict] = field(default_factory=list)
    missing_feature_drugs: list[str] = field(default_factory=list)
    ambiguous_mechanism_pairs: list[str] = field(default_factory=list)

    @property
    def ambiguous_rate(self) -> float:
        """Fraction of interactions with ambiguous mechanism."""
        if self.total_interactions == 0:
            return 0.0
        return self.interactions_with_ambiguous_mechanism / self.total_interactions

    def log_summary(self) -> None:
        """Log a human-readable summary."""
        logger.info(f"\n{'='*60}")
        logger.info("Data Quality Report")
        logger.info(f"{'='*60}")
        logger.info(f"Drugs:        {self.total_drugs}")
        logger.info(f"  Complete:   {self.drugs_with_complete_features}")
        logger.info(f"  Fallback:   {self.drugs_with_fallback_values}")
        logger.info(f"  Review:     {self.drugs_flagged_for_review}")
        logger.info(f"Interactions: {self.total_interactions}")
        logger.info(f"  Clear mech: {self.interactions_with_clear_mechanism}")
        logger.info(f"  Ambiguous:  {self.interactions_with_ambiguous_mechanism} "
                     f"({self.ambiguous_rate:.1%})")
        logger.info(f"  Sev conflicts: {self.severity_conflicts}")
        logger.info(f"{'='*60}")

    def save(self, path: str | Path) -> None:
        """Save report to JSON."""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    def passes_quality_gate(self) -> bool:
        """Check if data meets Phase 4a quality thresholds."""
        checks = [
            (self.total_drugs >= 280, f"Need >= 280 drugs, got {self.total_drugs}"),
            (self.total_interactions >= 1200,
             f"Need >= 1200 interactions, got {self.total_interactions}"),
            (self.ambiguous_rate < 0.10,
             f"Ambiguous mechanism rate {self.ambiguous_rate:.1%} >= 10%"),
        ]
        passed = True
        for ok, msg in checks:
            if not ok:
                logger.warning(f"QUALITY GATE FAIL: {msg}")
                passed = False
            else:
                logger.info(f"QUALITY GATE PASS: {msg}")
        return passed


def validate_feature_continuity(
    original_drugs: dict,
    pipeline_drugs: dict,
    max_drift: float = 0.15,
) -> tuple[bool, list[str]]:
    """
    Verify pipeline-extracted features are close to hand-curated originals.

    Some drift is expected (different normalization, more precise values),
    but large deviations indicate a pipeline bug.

    Returns:
        (passed, list of drift warnings)
    """
    import torch

    warnings: list[str] = []
    passed = True

    for name, original in original_drugs.items():
        if name not in pipeline_drugs:
            warnings.append(f"MISSING: {name} not in pipeline output")
            passed = False
            continue

        orig_feat = torch.tensor(original["features"])
        pipe_feat = torch.tensor(pipeline_drugs[name]["features"])
        drift = (orig_feat - pipe_feat).abs().mean().item()

        if drift > max_drift:
            warnings.append(
                f"DRIFT: {name} mean feature drift = {drift:.3f} (max {max_drift})"
            )
            passed = False

    return passed, warnings
