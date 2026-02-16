"""Tests for BasicPolypharmacyAnalyzer."""

from dataclasses import dataclass

import pytest

from pharmloop.polypharmacy import BasicPolypharmacyAnalyzer, PolypharmacyReport


@dataclass
class FakeInteractionResult:
    """Minimal mock of InteractionResult for testing."""
    severity: str
    mechanisms: list[str]
    confidence: float = 0.8
    flags: list[str] = None

    def __post_init__(self):
        if self.flags is None:
            self.flags = []


class TestBasicPolypharmacyAnalyzer:
    """Tests for rule-based polypharmacy pattern detection."""

    def setup_method(self):
        self.analyzer = BasicPolypharmacyAnalyzer()

    def test_no_alerts_for_safe_medications(self):
        results = {
            ("metformin", "lisinopril"): FakeInteractionResult(
                severity="none", mechanisms=[]
            ),
            ("metformin", "amlodipine"): FakeInteractionResult(
                severity="none", mechanisms=[]
            ),
            ("lisinopril", "amlodipine"): FakeInteractionResult(
                severity="mild", mechanisms=["hypotension"]
            ),
        }
        report = self.analyzer.analyze(
            ["metformin", "lisinopril", "amlodipine"], results
        )
        assert len(report.multi_drug_alerts) == 0

    def test_serotonergic_alert_triggered(self):
        results = {
            ("fluoxetine", "tramadol"): FakeInteractionResult(
                severity="severe", mechanisms=["serotonergic"]
            ),
            ("fluoxetine", "trazodone"): FakeInteractionResult(
                severity="moderate", mechanisms=["serotonergic"]
            ),
            ("tramadol", "trazodone"): FakeInteractionResult(
                severity="moderate", mechanisms=["serotonergic"]
            ),
        }
        report = self.analyzer.analyze(
            ["fluoxetine", "tramadol", "trazodone"], results
        )
        sero_alerts = [a for a in report.multi_drug_alerts
                       if a.pattern == "additive_serotonergic"]
        assert len(sero_alerts) == 1
        alert = sero_alerts[0]
        assert "fluoxetine" in alert.involved_drugs
        assert "tramadol" in alert.involved_drugs
        assert "trazodone" in alert.involved_drugs
        assert alert.pair_count >= 2

    def test_qt_prolongation_alert(self):
        results = {
            ("amiodarone", "haloperidol"): FakeInteractionResult(
                severity="severe", mechanisms=["qt_prolongation"]
            ),
            ("amiodarone", "ondansetron"): FakeInteractionResult(
                severity="moderate", mechanisms=["qt_prolongation"]
            ),
            ("haloperidol", "ondansetron"): FakeInteractionResult(
                severity="moderate", mechanisms=["qt_prolongation"]
            ),
        }
        report = self.analyzer.analyze(
            ["amiodarone", "haloperidol", "ondansetron"], results
        )
        qt_alerts = [a for a in report.multi_drug_alerts
                     if a.pattern == "additive_qt_prolongation"]
        assert len(qt_alerts) == 1

    def test_cyp_inhibition_alert(self):
        results = {
            ("fluconazole", "clarithromycin"): FakeInteractionResult(
                severity="moderate", mechanisms=["cyp_inhibition"]
            ),
            ("fluconazole", "simvastatin"): FakeInteractionResult(
                severity="severe", mechanisms=["cyp_inhibition"]
            ),
            ("clarithromycin", "simvastatin"): FakeInteractionResult(
                severity="severe", mechanisms=["cyp_inhibition"]
            ),
        }
        report = self.analyzer.analyze(
            ["fluconazole", "clarithromycin", "simvastatin"], results
        )
        cyp_alerts = [a for a in report.multi_drug_alerts
                      if a.pattern == "additive_cyp_inhibition"]
        assert len(cyp_alerts) == 1

    def test_multiple_patterns_detected(self):
        """Both serotonergic and CYP alerts fire simultaneously."""
        results = {
            ("fluoxetine", "tramadol"): FakeInteractionResult(
                severity="severe",
                mechanisms=["serotonergic", "cyp_inhibition"],
            ),
            ("fluoxetine", "paroxetine"): FakeInteractionResult(
                severity="moderate",
                mechanisms=["serotonergic", "cyp_inhibition"],
            ),
            ("tramadol", "paroxetine"): FakeInteractionResult(
                severity="severe",
                mechanisms=["serotonergic", "cyp_inhibition"],
            ),
        }
        report = self.analyzer.analyze(
            ["fluoxetine", "tramadol", "paroxetine"], results
        )
        patterns = {a.pattern for a in report.multi_drug_alerts}
        assert "additive_serotonergic" in patterns
        assert "additive_cyp_inhibition" in patterns

    def test_highest_severity_reported(self):
        results = {
            ("drug_a", "drug_b"): FakeInteractionResult(
                severity="severe", mechanisms=["serotonergic"]
            ),
            ("drug_a", "drug_c"): FakeInteractionResult(
                severity="mild", mechanisms=[]
            ),
            ("drug_b", "drug_c"): FakeInteractionResult(
                severity="moderate", mechanisms=[]
            ),
        }
        report = self.analyzer.analyze(
            ["drug_a", "drug_b", "drug_c"], results
        )
        assert report.highest_severity == "severe"

    def test_empty_results_handled(self):
        report = self.analyzer.analyze(["drug_a", "drug_b"], {})
        assert report.total_pairs_checked == 0
        assert report.highest_severity == "none"

    def test_pairwise_results_ranked_by_severity(self):
        results = {
            ("a", "b"): FakeInteractionResult(
                severity="mild", mechanisms=[], confidence=0.9
            ),
            ("a", "c"): FakeInteractionResult(
                severity="severe", mechanisms=[], confidence=0.8
            ),
            ("b", "c"): FakeInteractionResult(
                severity="none", mechanisms=[], confidence=0.95
            ),
        }
        report = self.analyzer.analyze(["a", "b", "c"], results)
        # Severe should be first
        assert report.pairwise_results[0][1].severity == "severe"
