"""Tests for ClinicalNarrator template engine."""

import pytest

from pharmloop.output import SEVERITY_NAMES, MECHANISM_NAMES, FLAG_NAMES
from pharmloop.templates import ClinicalNarrator


@pytest.fixture
def narrator():
    return ClinicalNarrator()


class TestTemplateCompleteness:
    """Verify every model output maps to a valid template."""

    def test_all_severities_have_templates(self, narrator) -> None:
        for severity in SEVERITY_NAMES:
            assert severity in narrator.SEVERITY_TEMPLATES, (
                f"No template for severity: {severity}"
            )

    def test_all_mechanisms_have_explanations(self, narrator) -> None:
        for mechanism in MECHANISM_NAMES:
            assert mechanism in narrator.MECHANISM_EXPLANATIONS, (
                f"No template for mechanism: {mechanism}"
            )

    def test_all_flags_have_recommendations(self, narrator) -> None:
        for flag in FLAG_NAMES:
            assert flag in narrator.FLAG_RECOMMENDATIONS, (
                f"No template for flag: {flag}"
            )

    def test_all_severities_have_actions(self, narrator) -> None:
        for severity in SEVERITY_NAMES:
            assert severity in narrator.ACTIONS, (
                f"No action for severity: {severity}"
            )


class TestTemplateRendering:
    """Verify templates render correctly without unresolved slots."""

    def test_no_unresolved_format_slots(self, narrator) -> None:
        """No output should contain {A} or {B} or other unresolved slots."""
        for severity in SEVERITY_NAMES:
            for mechanism in MECHANISM_NAMES:
                output = narrator.narrate(
                    "fluoxetine", "tramadol", severity,
                    [mechanism], ["monitor_serotonin_syndrome"],
                    confidence=0.85, converged=True, steps=8,
                )
                assert "{" not in output, (
                    f"Unresolved slot in severity={severity}, mechanism={mechanism}: {output[:100]}"
                )

    def test_unknown_severity_produces_insufficient_data(self, narrator) -> None:
        output = narrator.narrate(
            "QZ-7734", "aspirin", "unknown", [], [],
            confidence=0.08, converged=False, steps=16,
        )
        assert "insufficient data" in output.lower()
        assert "serotonin" not in output.lower()

    def test_none_severity_clean(self, narrator) -> None:
        output = narrator.narrate(
            "metformin", "lisinopril", "none", [], [],
            confidence=0.95, converged=True, steps=5,
        )
        assert "no clinically significant interaction" in output.lower()

    def test_severe_includes_action(self, narrator) -> None:
        output = narrator.narrate(
            "fluoxetine", "tramadol", "severe",
            ["serotonergic"], ["monitor_serotonin_syndrome", "avoid_combination"],
            confidence=0.90, converged=True, steps=7,
        )
        assert "serotonin" in output.lower()
        assert "benefit" in output.lower()

    def test_contraindicated_includes_warning(self, narrator) -> None:
        output = narrator.narrate(
            "fluoxetine", "sertraline", "contraindicated",
            ["serotonergic"], ["avoid_combination"],
            confidence=0.95, converged=True, steps=4,
        )
        assert "CONTRAINDICATED" in output
        assert "should not be used together" in output

    def test_low_confidence_qualifier(self, narrator) -> None:
        output = narrator.narrate(
            "drug_a", "drug_b", "moderate",
            ["cyp_inhibition"], [],
            confidence=0.3, converged=True, steps=12,
        )
        assert "limited confidence" in output.lower()

    def test_high_confidence_no_qualifier(self, narrator) -> None:
        output = narrator.narrate(
            "drug_a", "drug_b", "moderate",
            ["cyp_inhibition"], [],
            confidence=0.8, converged=True, steps=6,
        )
        assert "limited confidence" not in output.lower()

    def test_partial_convergence_report(self, narrator) -> None:
        partial = {
            "partial_convergence": True,
            "settled_aspects": ["severity"],
            "unsettled_aspects": ["mechanism", "clinical flags"],
        }
        output = narrator.narrate(
            "drug_a", "drug_b", "moderate",
            [], [], confidence=0.6, converged=True, steps=10,
            partial_convergence=partial,
        )
        assert "severity" in output.lower()
        assert "could not be fully determined" in output.lower()

    def test_metadata_line_converged(self, narrator) -> None:
        output = narrator.narrate(
            "drug_a", "drug_b", "none", [], [],
            confidence=0.9, converged=True, steps=5,
        )
        assert "Converged in 5 steps" in output
        assert "Confidence: 90%" in output

    def test_metadata_line_not_converged(self, narrator) -> None:
        output = narrator.narrate(
            "drug_a", "drug_b", "unknown", [], [],
            confidence=0.05, converged=False, steps=16,
        )
        assert "Did not converge" in output

    def test_render_all_paths(self, narrator) -> None:
        """Render every severity x mechanism combination for human review."""
        for severity in SEVERITY_NAMES:
            for mechanism in MECHANISM_NAMES:
                output = narrator.narrate(
                    "drug_A", "drug_B", severity,
                    [mechanism], [],
                    confidence=0.75, converged=True, steps=10,
                )
                print(f"\n{'='*60}")
                print(f"Severity: {severity} | Mechanism: {mechanism}")
                print(f"{'='*60}")
                print(output)
