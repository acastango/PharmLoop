"""
Extract drug-drug interactions from DrugBank XML or generate them
from pharmacological class interaction rules.

Maps DrugBank interaction descriptions to PharmLoop severity and
mechanism vocabulary using rule-based keyword classification.
"""

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger("pharmloop.pipeline")


# ── Keyword → mechanism mapping ──
MECHANISM_KEYWORDS: dict[str, str] = {
    "serotonin syndrome": "serotonergic",
    "serotonergic": "serotonergic",
    "serotonin": "serotonergic",
    "cyp3a4 inhibit": "cyp_inhibition",
    "cyp2d6 inhibit": "cyp_inhibition",
    "cyp2c9 inhibit": "cyp_inhibition",
    "cyp2c19 inhibit": "cyp_inhibition",
    "cyp1a2 inhibit": "cyp_inhibition",
    "enzyme inhibit": "cyp_inhibition",
    "inhibits the metabolism": "cyp_inhibition",
    "increase.*concentration": "cyp_inhibition",
    "cyp3a4 induc": "cyp_induction",
    "cyp inducer": "cyp_induction",
    "enzyme induc": "cyp_induction",
    "decrease.*concentration": "cyp_induction",
    "decrease.*efficacy": "cyp_induction",
    "qt prolong": "qt_prolongation",
    "torsades": "qt_prolongation",
    "arrhythmi": "qt_prolongation",
    "bleeding": "bleeding_risk",
    "hemorrhag": "bleeding_risk",
    "anticoagulant effect": "bleeding_risk",
    "inr increase": "bleeding_risk",
    "cns depress": "cns_depression",
    "sedati": "cns_depression",
    "respiratory depression": "cns_depression",
    "drowsiness": "cns_depression",
    "nephrotox": "nephrotoxicity",
    "renal": "nephrotoxicity",
    "kidney": "nephrotoxicity",
    "hepatotox": "hepatotoxicity",
    "liver": "hepatotoxicity",
    "hepatic injury": "hepatotoxicity",
    "hypotens": "hypotension",
    "blood pressure decrease": "hypotension",
    "hyperkalem": "hyperkalemia",
    "potassium increase": "hyperkalemia",
    "seizure": "seizure_risk",
    "convuls": "seizure_risk",
    "lower.*seizure threshold": "seizure_risk",
    "immunosuppress": "immunosuppression",
    "absorption": "absorption_altered",
    "bioavailability decrease": "absorption_altered",
    "protein binding": "protein_binding_displacement",
    "displace.*protein": "protein_binding_displacement",
    "electrolyte": "electrolyte_imbalance",
    "hyponatremia": "electrolyte_imbalance",
    "hypokalemia": "electrolyte_imbalance",
    "hypomagnesemia": "electrolyte_imbalance",
}

# ── Severity keyword mapping ──
SEVERITY_KEYWORDS: dict[str, str] = {
    "contraindicated": "contraindicated",
    "do not use": "contraindicated",
    "avoid combination": "contraindicated",
    "life-threatening": "contraindicated",
    "fatal": "contraindicated",
    "serious": "severe",
    "major": "severe",
    "significant": "severe",
    "dangerous": "severe",
    "moderate": "moderate",
    "caution": "moderate",
    "monitor": "moderate",
    "minor": "mild",
    "minimal": "mild",
    "unlikely": "mild",
}

# Severity ordering for conflict resolution (higher = more severe)
SEVERITY_ORDER = {
    "none": 0,
    "mild": 1,
    "moderate": 2,
    "severe": 3,
    "contraindicated": 4,
}

# ── Mechanism → flag mapping ──
MECHANISM_TO_FLAGS: dict[str, list[str]] = {
    "serotonergic": ["monitor_serotonin_syndrome"],
    "cyp_inhibition": ["monitor_drug_levels"],
    "cyp_induction": ["monitor_drug_levels"],
    "qt_prolongation": ["monitor_qt_interval"],
    "bleeding_risk": ["monitor_bleeding", "monitor_inr"],
    "cns_depression": ["monitor_cns_depression"],
    "nephrotoxicity": ["monitor_renal_function"],
    "hepatotoxicity": ["monitor_hepatic_function"],
    "hypotension": ["monitor_blood_pressure"],
    "hyperkalemia": ["monitor_electrolytes"],
    "seizure_risk": [],
    "immunosuppression": ["monitor_drug_levels"],
    "absorption_altered": ["separate_administration"],
    "protein_binding_displacement": ["monitor_drug_levels"],
    "electrolyte_imbalance": ["monitor_electrolytes"],
}


@dataclass
class ExtractionResult:
    """Result of interaction extraction."""
    interactions: list[dict]
    stats: dict = field(default_factory=dict)


class InteractionExtractor:
    """
    Extract drug-drug interactions from DrugBank XML descriptions.

    Maps DrugBank interaction text to PharmLoop severity and mechanism
    vocabulary using rule-based keyword classification.
    """

    def extract_from_text(
        self,
        drug_a: str,
        drug_b: str,
        description: str,
    ) -> dict | None:
        """
        Parse a single interaction description into PharmLoop format.

        Args:
            drug_a: First drug name.
            drug_b: Second drug name.
            description: Interaction description text from DrugBank.

        Returns:
            Dict with severity, mechanisms, flags, or None if unparseable.
        """
        desc_lower = description.lower()

        # Extract mechanisms
        mechanisms = self._extract_mechanisms(desc_lower)
        if not mechanisms:
            mechanisms = ["cyp_inhibition"]  # default fallback

        # Extract severity
        severity = self._extract_severity(desc_lower)

        # Derive flags from mechanisms
        flags = self._derive_flags(mechanisms, severity)

        return {
            "drug_a": drug_a.lower(),
            "drug_b": drug_b.lower(),
            "severity": severity,
            "mechanisms": mechanisms,
            "flags": flags,
            "source": "DrugBank",
            "notes": description[:200],
        }

    def _extract_mechanisms(self, text: str) -> list[str]:
        """Extract mechanisms from description text via keyword matching."""
        found: set[str] = set()
        for keyword, mechanism in MECHANISM_KEYWORDS.items():
            if re.search(keyword, text):
                found.add(mechanism)
        return sorted(found)

    def _extract_severity(self, text: str) -> str:
        """Extract severity from description text. Defaults to moderate."""
        best_severity = "moderate"
        best_order = SEVERITY_ORDER["moderate"]

        for keyword, severity in SEVERITY_KEYWORDS.items():
            if keyword in text:
                order = SEVERITY_ORDER.get(severity, 2)
                if order > best_order:
                    best_severity = severity
                    best_order = order

        return best_severity

    def _derive_flags(self, mechanisms: list[str], severity: str) -> list[str]:
        """Derive monitoring flags from mechanisms and severity."""
        flags: set[str] = set()
        for mech in mechanisms:
            for flag in MECHANISM_TO_FLAGS.get(mech, []):
                flags.add(flag)

        if severity in ("severe", "contraindicated"):
            flags.add("avoid_combination")

        return sorted(flags)

    @staticmethod
    def resolve_severity_conflict(
        severity_a: str,
        severity_b: str,
    ) -> str:
        """
        Resolve conflicting severities from multiple sources.

        Always takes the MORE SEVERE rating. DO NO HARM.
        """
        order_a = SEVERITY_ORDER.get(severity_a, 2)
        order_b = SEVERITY_ORDER.get(severity_b, 2)
        return severity_a if order_a >= order_b else severity_b
