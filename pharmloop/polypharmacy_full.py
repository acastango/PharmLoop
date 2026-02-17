"""
FullPolypharmacyAnalyzer: extended polypharmacy analysis with cascade detection.

Inherits the three additive patterns from BasicPolypharmacyAnalyzer, adds:
  1. CYP cascade interactions (inhibitor + multiple substrates)
  2. Renal risk cascades (nephrotoxic + renally-eliminated drugs)
  3. Metabolic pathway saturation (multi-substrate + inhibitor)
  4. Feature-based bleeding risk (uses drug features, not just mechanisms)

Zero learned parameters — pure rule-based analysis.
"""

from pharmloop.polypharmacy import (
    BasicPolypharmacyAnalyzer,
    MultiDrugAlert,
    PolypharmacyReport,
)

# Feature dimension indices for CYP enzymes (matching CLASS_FEATURE_PRIORS layout)
# Inhibition dims
CYP1A2_INHIB_DIM = 0
CYP2C9_INHIB_DIM = 1
CYP2C19_INHIB_DIM = 2
CYP2D6_INHIB_DIM = 3
CYP3A4_INHIB_DIM = 4

# Induction dims
CYP2C9_INDUCTION_DIM = 6
CYP2C19_INDUCTION_DIM = 7

# Substrate dims
CYP2D6_SUBSTRATE_DIM = 8
CYP3A4_SUBSTRATE_DIM = 9

# Other relevant dims
RENAL_ELIMINATION_DIM = 24
NEPHROTOXICITY_DIM = 44
BLEEDING_RISK_DIM = 42

# CYP enzyme definitions for cascade detection:
# (enzyme_name, inhibition_dim, induction_dim_or_None, substrate_dim_or_None)
CYP_ENZYMES = {
    "CYP1A2": {"inhib": CYP1A2_INHIB_DIM, "induction": None, "substrate": None},
    "CYP2C9": {"inhib": CYP2C9_INHIB_DIM, "induction": CYP2C9_INDUCTION_DIM, "substrate": None},
    "CYP2C19": {"inhib": CYP2C19_INHIB_DIM, "induction": CYP2C19_INDUCTION_DIM, "substrate": None},
    "CYP2D6": {"inhib": CYP2D6_INHIB_DIM, "induction": None, "substrate": CYP2D6_SUBSTRATE_DIM},
    "CYP3A4": {"inhib": CYP3A4_INHIB_DIM, "induction": None, "substrate": CYP3A4_SUBSTRATE_DIM},
}

# Threshold for considering a drug as having a relevant property
FEATURE_THRESHOLD = 0.5


class FullPolypharmacyAnalyzer(BasicPolypharmacyAnalyzer):
    """
    Extended polypharmacy analysis with cascade and saturation detection.

    Inherits the three additive patterns from BasicPolypharmacyAnalyzer,
    adds cascade interactions, renal risk chains, metabolic saturation,
    and feature-based risk detection.

    All pattern detection is rule-based with zero learned parameters.
    Drug features are used to identify pharmacological properties
    (CYP substrate/inhibitor status, renal elimination, etc.).

    Args:
        feature_threshold: Minimum feature value to consider a drug
            as having a property (default 0.5).
    """

    def __init__(self, feature_threshold: float = FEATURE_THRESHOLD) -> None:
        super().__init__()
        self.feature_threshold = feature_threshold

    def analyze(
        self,
        drug_names: list[str],
        pairwise_results: dict[tuple[str, str], object],
        skipped_drugs: list[str] | None = None,
        drug_features: dict[str, list[float]] | None = None,
    ) -> PolypharmacyReport:
        """
        Analyze pairwise results with extended cascade pattern detection.

        Runs BasicPolypharmacyAnalyzer first (additive serotonergic, QT,
        CYP inhibition), then adds cascade, renal, saturation, and
        feature-based bleeding risk patterns.

        Args:
            drug_names: List of all drug names in the medication list.
            pairwise_results: Dict mapping (drug_a, drug_b) -> InteractionResult.
            skipped_drugs: Drug names not in registry (passed through).
            drug_features: Dict mapping drug_name -> 64-dim feature vector.
                If None, feature-based patterns are skipped.

        Returns:
            PolypharmacyReport with all detected patterns.
        """
        # Run basic additive patterns first
        report = super().analyze(drug_names, pairwise_results, skipped_drugs)

        if drug_features is None:
            return report

        # Add cascade detection
        cascades = self._detect_cascades(drug_names, drug_features)
        report.multi_drug_alerts.extend(cascades)

        # Add renal risk chains
        renal_chains = self._detect_renal_cascades(
            drug_names, pairwise_results, drug_features,
        )
        report.multi_drug_alerts.extend(renal_chains)

        # Add metabolic saturation
        saturation = self._detect_metabolic_saturation(drug_names, drug_features)
        report.multi_drug_alerts.extend(saturation)

        # Feature-based bleeding risk check
        bleeding = self._detect_feature_bleeding_risk(drug_names, drug_features)
        report.multi_drug_alerts.extend(bleeding)

        return report

    def _detect_cascades(
        self,
        drug_names: list[str],
        drug_features: dict[str, list[float]],
    ) -> list[MultiDrugAlert]:
        """
        Detect CYP cascade interactions.

        For each CYP enzyme with a known substrate dimension:
          1. Find all inhibitors of that enzyme in the drug list
          2. Find all substrates of that enzyme
          3. If inhibitor + >=2 substrates present, flag cascade
             (multiple substrates compete for reduced enzyme capacity)

        Args:
            drug_names: Drug names in the medication list.
            drug_features: Drug name -> 64-dim feature vector.

        Returns:
            List of MultiDrugAlert for detected cascades.
        """
        alerts: list[MultiDrugAlert] = []
        threshold = self.feature_threshold

        for enzyme, dims in CYP_ENZYMES.items():
            sub_dim = dims["substrate"]
            inhib_dim = dims["inhib"]

            if sub_dim is None:
                continue

            inhibitors = [
                d for d in drug_names
                if d in drug_features
                and len(drug_features[d]) > inhib_dim
                and drug_features[d][inhib_dim] > threshold
            ]
            substrates = [
                d for d in drug_names
                if d in drug_features
                and len(drug_features[d]) > sub_dim
                and drug_features[d][sub_dim] > threshold
            ]

            # Need at least 1 inhibitor and 2+ substrates
            if inhibitors and len(substrates) >= 2:
                alerts.append(MultiDrugAlert(
                    pattern=f"cascade_{enzyme.lower()}",
                    alert_text=(
                        f"Metabolic cascade risk: {', '.join(inhibitors)} inhibit "
                        f"{enzyme}, which metabolizes {', '.join(substrates)}. "
                        f"Multiple substrates competing for reduced enzyme capacity "
                        f"may lead to unpredictable drug level elevations."
                    ),
                    involved_drugs=sorted(set(inhibitors + substrates)),
                    involved_pairs=[],
                    trigger_mechanism="cyp_inhibition",
                    pair_count=len(inhibitors) * len(substrates),
                ))

        return alerts

    def _detect_renal_cascades(
        self,
        drug_names: list[str],
        pairwise_results: dict[tuple[str, str], object],
        drug_features: dict[str, list[float]],
    ) -> list[MultiDrugAlert]:
        """
        Detect nephrotoxic drug + renally-eliminated drug combinations.

        If nephrotoxic drugs coexist with renally-cleared drugs, renal
        damage from the nephrotoxic agents reduces clearance of the
        renally-eliminated drugs, causing accumulation.

        Args:
            drug_names: Drug names in the medication list.
            pairwise_results: Pairwise interaction results.
            drug_features: Drug name -> 64-dim feature vector.

        Returns:
            List of MultiDrugAlert for detected renal cascades.
        """
        alerts: list[MultiDrugAlert] = []
        threshold = self.feature_threshold

        # Find nephrotoxic drugs from feature vectors.
        # We use features only (not pairwise mechanism labels) because
        # "nephrotoxicity" as an interaction mechanism doesn't tell us
        # WHICH drug is nephrotoxic — adding both would misclassify
        # renally-eliminated drugs as nephrotoxic.
        nephrotoxic = set()
        for d in drug_names:
            if d in drug_features and len(drug_features[d]) > NEPHROTOXICITY_DIM:
                if drug_features[d][NEPHROTOXICITY_DIM] > threshold:
                    nephrotoxic.add(d)

        # Find renally-cleared drugs
        renally_cleared = set()
        for d in drug_names:
            if d in drug_features and len(drug_features[d]) > RENAL_ELIMINATION_DIM:
                if drug_features[d][RENAL_ELIMINATION_DIM] > threshold:
                    renally_cleared.add(d)

        # If nephrotoxic + non-nephrotoxic renally cleared drugs coexist
        non_overlap_renal = renally_cleared - nephrotoxic
        if nephrotoxic and non_overlap_renal:
            alerts.append(MultiDrugAlert(
                pattern="renal_cascade",
                alert_text=(
                    f"Renal cascade risk: {', '.join(sorted(nephrotoxic))} may "
                    f"impair renal function, reducing clearance of renally-"
                    f"eliminated drugs ({', '.join(sorted(non_overlap_renal))}). "
                    f"Monitor renal function and consider dose adjustment for "
                    f"renally-cleared medications."
                ),
                involved_drugs=sorted(nephrotoxic | non_overlap_renal),
                involved_pairs=[],
                trigger_mechanism="nephrotoxicity",
                pair_count=len(nephrotoxic) * len(non_overlap_renal),
            ))

        return alerts

    def _detect_metabolic_saturation(
        self,
        drug_names: list[str],
        drug_features: dict[str, list[float]],
    ) -> list[MultiDrugAlert]:
        """
        Detect metabolic pathway saturation.

        When >=2 substrates and >=1 inhibitor share the same CYP enzyme,
        the substrates compete for the (reduced) enzyme capacity. Drug
        levels rise unpredictably for both substrates.

        This differs from cascade detection: cascade checks for inhibitor +
        multiple substrates; saturation specifically checks for the inhibitor
        being a DIFFERENT drug from the substrates.

        Args:
            drug_names: Drug names in the medication list.
            drug_features: Drug name -> 64-dim feature vector.

        Returns:
            List of MultiDrugAlert for detected saturation.
        """
        alerts: list[MultiDrugAlert] = []
        threshold = self.feature_threshold

        for enzyme, dims in CYP_ENZYMES.items():
            sub_dim = dims["substrate"]
            inhib_dim = dims["inhib"]

            if sub_dim is None:
                continue

            inhibitors = set()
            substrates = set()

            for d in drug_names:
                if d not in drug_features:
                    continue
                feats = drug_features[d]
                if len(feats) > inhib_dim and feats[inhib_dim] > threshold:
                    inhibitors.add(d)
                if len(feats) > sub_dim and feats[sub_dim] > threshold:
                    substrates.add(d)

            # Pure substrates (not also inhibitors)
            pure_substrates = substrates - inhibitors

            # Need at least 1 inhibitor and 2+ pure substrates
            if inhibitors and len(pure_substrates) >= 2:
                alerts.append(MultiDrugAlert(
                    pattern=f"metabolic_saturation_{enzyme.lower()}",
                    alert_text=(
                        f"Metabolic saturation risk for {enzyme}: "
                        f"{', '.join(sorted(inhibitors))} inhibit {enzyme} while "
                        f"{', '.join(sorted(pure_substrates))} are substrates. "
                        f"With inhibited enzyme capacity, substrates compete for "
                        f"remaining metabolism. Both substrate drug levels may "
                        f"rise unpredictably. Monitor drug levels closely."
                    ),
                    involved_drugs=sorted(inhibitors | pure_substrates),
                    involved_pairs=[],
                    trigger_mechanism="cyp_inhibition",
                    pair_count=len(inhibitors) * len(pure_substrates),
                ))

        return alerts

    def _detect_feature_bleeding_risk(
        self,
        drug_names: list[str],
        drug_features: dict[str, list[float]],
    ) -> list[MultiDrugAlert]:
        """
        Detect additive bleeding risk from drug features.

        Some antiplatelet/anticoagulant interactions are classified as
        "protein_binding_displacement" but the real risk is bleeding.
        This checks the bleeding_risk feature dimension directly.

        Args:
            drug_names: Drug names in the medication list.
            drug_features: Drug name -> 64-dim feature vector.

        Returns:
            List of MultiDrugAlert if 3+ drugs have high bleeding_risk feature.
        """
        alerts: list[MultiDrugAlert] = []
        threshold = self.feature_threshold

        bleeding_drugs = [
            d for d in drug_names
            if d in drug_features
            and len(drug_features[d]) > BLEEDING_RISK_DIM
            and drug_features[d][BLEEDING_RISK_DIM] > threshold
        ]

        if len(bleeding_drugs) >= 3:
            alerts.append(MultiDrugAlert(
                pattern="feature_bleeding_risk",
                alert_text=(
                    f"High additive bleeding risk: {len(bleeding_drugs)} drugs "
                    f"in this medication list have significant bleeding risk "
                    f"({', '.join(sorted(bleeding_drugs))}). Combined bleeding "
                    f"risk exceeds individual drug risks. Monitor for signs of "
                    f"bleeding and review necessity of each agent."
                ),
                involved_drugs=sorted(bleeding_drugs),
                involved_pairs=[],
                trigger_mechanism="bleeding_risk",
                pair_count=len(bleeding_drugs),
            ))

        return alerts
