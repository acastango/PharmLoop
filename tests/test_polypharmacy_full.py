"""Tests for FullPolypharmacyAnalyzer: cascade, renal, saturation, bleeding."""

import pytest

from pharmloop.polypharmacy_full import (
    FullPolypharmacyAnalyzer,
    CYP2D6_INHIB_DIM,
    CYP2D6_SUBSTRATE_DIM,
    CYP3A4_INHIB_DIM,
    CYP3A4_SUBSTRATE_DIM,
    NEPHROTOXICITY_DIM,
    RENAL_ELIMINATION_DIM,
    BLEEDING_RISK_DIM,
)


@pytest.fixture
def analyzer():
    return FullPolypharmacyAnalyzer()


def _make_features(overrides: dict[int, float]) -> list[float]:
    """Build a 64-dim feature vector with specific dims set."""
    feats = [0.0] * 64
    for dim, val in overrides.items():
        feats[dim] = val
    return feats


class FakeInteractionResult:
    """Minimal InteractionResult for testing."""
    def __init__(self, severity="moderate", mechanisms=None, drug_a="", drug_b=""):
        self.severity = severity
        self.mechanisms = mechanisms or []
        self.drug_a = drug_a
        self.drug_b = drug_b
        self.confidence = 0.8


class TestCascadeDetection:
    """Test CYP cascade interaction detection."""

    def test_cyp2d6_cascade(self, analyzer):
        """Detect cascade when CYP2D6 inhibitor + 2 substrates present."""
        features = {
            "fluoxetine": _make_features({CYP2D6_INHIB_DIM: 0.9}),
            "codeine": _make_features({CYP2D6_SUBSTRATE_DIM: 0.8}),
            "tramadol": _make_features({CYP2D6_SUBSTRATE_DIM: 0.7}),
        }
        pairwise = {
            ("fluoxetine", "codeine"): FakeInteractionResult(severity="severe"),
            ("fluoxetine", "tramadol"): FakeInteractionResult(severity="severe"),
            ("codeine", "tramadol"): FakeInteractionResult(severity="moderate"),
        }
        report = analyzer.analyze(
            list(features.keys()), pairwise, drug_features=features,
        )
        cascade_alerts = [a for a in report.multi_drug_alerts if "cascade_cyp2d6" in a.pattern]
        assert len(cascade_alerts) >= 1
        assert "fluoxetine" in cascade_alerts[0].involved_drugs

    def test_cyp3a4_cascade(self, analyzer):
        """Detect cascade for CYP3A4."""
        features = {
            "ketoconazole": _make_features({CYP3A4_INHIB_DIM: 0.9}),
            "simvastatin": _make_features({CYP3A4_SUBSTRATE_DIM: 0.8}),
            "midazolam": _make_features({CYP3A4_SUBSTRATE_DIM: 0.7}),
        }
        pairwise = {
            ("ketoconazole", "simvastatin"): FakeInteractionResult(severity="severe"),
            ("ketoconazole", "midazolam"): FakeInteractionResult(severity="severe"),
            ("simvastatin", "midazolam"): FakeInteractionResult(severity="none"),
        }
        report = analyzer.analyze(
            list(features.keys()), pairwise, drug_features=features,
        )
        cascade_alerts = [a for a in report.multi_drug_alerts if "cascade_cyp3a4" in a.pattern]
        assert len(cascade_alerts) >= 1

    def test_no_cascade_single_substrate(self, analyzer):
        """No cascade with only 1 substrate."""
        features = {
            "fluoxetine": _make_features({CYP2D6_INHIB_DIM: 0.9}),
            "codeine": _make_features({CYP2D6_SUBSTRATE_DIM: 0.8}),
        }
        pairwise = {
            ("fluoxetine", "codeine"): FakeInteractionResult(severity="severe"),
        }
        report = analyzer.analyze(
            list(features.keys()), pairwise, drug_features=features,
        )
        cascade_alerts = [a for a in report.multi_drug_alerts if "cascade" in a.pattern]
        assert len(cascade_alerts) == 0


class TestRenalCascade:
    """Test renal risk cascade detection."""

    def test_nephrotoxic_plus_renal_drug(self, analyzer):
        """Detect renal cascade with nephrotoxic + renally-eliminated drugs."""
        features = {
            "gentamicin": _make_features({NEPHROTOXICITY_DIM: 0.8}),
            "metformin": _make_features({RENAL_ELIMINATION_DIM: 0.9}),
        }
        pairwise = {
            ("gentamicin", "metformin"): FakeInteractionResult(
                severity="moderate", mechanisms=["nephrotoxicity"],
                drug_a="gentamicin", drug_b="metformin",
            ),
        }
        report = analyzer.analyze(
            list(features.keys()), pairwise, drug_features=features,
        )
        renal_alerts = [a for a in report.multi_drug_alerts if "renal" in a.pattern]
        assert len(renal_alerts) >= 1

    def test_no_renal_cascade_without_renal_drug(self, analyzer):
        """No renal cascade if no renally-eliminated drugs."""
        features = {
            "gentamicin": _make_features({NEPHROTOXICITY_DIM: 0.8}),
            "amiodarone": _make_features({}),
        }
        pairwise = {
            ("gentamicin", "amiodarone"): FakeInteractionResult(severity="moderate"),
        }
        report = analyzer.analyze(
            list(features.keys()), pairwise, drug_features=features,
        )
        renal_alerts = [a for a in report.multi_drug_alerts if "renal" in a.pattern]
        assert len(renal_alerts) == 0


class TestMetabolicSaturation:
    """Test metabolic pathway saturation detection."""

    def test_saturation_detected(self, analyzer):
        """Detect saturation: inhibitor + 2 pure substrates on same enzyme."""
        features = {
            "ketoconazole": _make_features({CYP3A4_INHIB_DIM: 0.9}),
            "simvastatin": _make_features({CYP3A4_SUBSTRATE_DIM: 0.8}),
            "cyclosporine": _make_features({CYP3A4_SUBSTRATE_DIM: 0.7}),
        }
        pairwise = {
            ("ketoconazole", "simvastatin"): FakeInteractionResult(severity="severe"),
            ("ketoconazole", "cyclosporine"): FakeInteractionResult(severity="severe"),
            ("simvastatin", "cyclosporine"): FakeInteractionResult(severity="moderate"),
        }
        report = analyzer.analyze(
            list(features.keys()), pairwise, drug_features=features,
        )
        sat_alerts = [a for a in report.multi_drug_alerts if "saturation" in a.pattern]
        assert len(sat_alerts) >= 1


class TestFeatureBleedingRisk:
    """Test feature-based bleeding risk detection."""

    def test_triple_bleeding_risk(self, analyzer):
        """Detect additive bleeding when 3+ drugs have high bleeding_risk."""
        features = {
            "warfarin": _make_features({BLEEDING_RISK_DIM: 0.9}),
            "aspirin": _make_features({BLEEDING_RISK_DIM: 0.8}),
            "clopidogrel": _make_features({BLEEDING_RISK_DIM: 0.7}),
        }
        pairwise = {
            ("warfarin", "aspirin"): FakeInteractionResult(severity="severe"),
            ("warfarin", "clopidogrel"): FakeInteractionResult(severity="severe"),
            ("aspirin", "clopidogrel"): FakeInteractionResult(severity="moderate"),
        }
        report = analyzer.analyze(
            list(features.keys()), pairwise, drug_features=features,
        )
        bleeding_alerts = [a for a in report.multi_drug_alerts
                          if "feature_bleeding" in a.pattern]
        assert len(bleeding_alerts) == 1
        assert len(bleeding_alerts[0].involved_drugs) == 3

    def test_no_bleeding_with_two_drugs(self, analyzer):
        """No feature-based bleeding alert with only 2 drugs."""
        features = {
            "warfarin": _make_features({BLEEDING_RISK_DIM: 0.9}),
            "aspirin": _make_features({BLEEDING_RISK_DIM: 0.8}),
        }
        pairwise = {
            ("warfarin", "aspirin"): FakeInteractionResult(severity="severe"),
        }
        report = analyzer.analyze(
            list(features.keys()), pairwise, drug_features=features,
        )
        bleeding_alerts = [a for a in report.multi_drug_alerts
                          if "feature_bleeding" in a.pattern]
        assert len(bleeding_alerts) == 0


class TestNoFeatures:
    """Test behavior when drug_features is None."""

    def test_falls_back_to_basic(self, analyzer):
        """Without features, only basic additive patterns (serotonergic, QT, CYP) detected."""
        pairwise = {
            ("fluoxetine", "tramadol"): FakeInteractionResult(
                severity="severe", mechanisms=["serotonergic"],
            ),
            ("fluoxetine", "lithium"): FakeInteractionResult(
                severity="moderate", mechanisms=["serotonergic"],
            ),
            ("tramadol", "lithium"): FakeInteractionResult(
                severity="moderate", mechanisms=["serotonergic"],
            ),
        }
        report = analyzer.analyze(
            ["fluoxetine", "tramadol", "lithium"], pairwise, drug_features=None,
        )
        # Should detect additive serotonergic from mechanism labels (basic pattern)
        sero_alerts = [a for a in report.multi_drug_alerts
                      if "serotonergic" in a.pattern]
        assert len(sero_alerts) >= 1


class TestSkippedDrugsPassthrough:
    """Test that skipped_drugs passes through correctly."""

    def test_skipped_drugs_in_report(self, analyzer):
        report = analyzer.analyze(
            ["warfarin", "aspirin", "unknown_drug"],
            {("warfarin", "aspirin"): FakeInteractionResult(severity="severe")},
            skipped_drugs=["unknown_drug"],
        )
        assert report.skipped_drugs == ["unknown_drug"]
