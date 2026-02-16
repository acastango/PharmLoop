"""
Parse DrugBank XML → 64-dim pharmacological feature vectors.

Maps DrugBank fields to the PharmLoop feature layout:
  Dims 0-9:   CYP enzyme interactions
  Dims 10-19: Receptor/target activity
  Dims 20-29: Pharmacokinetic parameters
  Dims 30-39: Drug class encoding
  Dims 40-49: Risk flags
  Dims 50-59: Transporter interactions
  Dims 60-63: Physical/chemical properties

When DrugBank XML is not available, falls back to synthetic feature
generation based on drug class priors. Synthetic features are
pharmacologically plausible defaults — NOT random.
"""

import json
import logging
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("pharmloop.pipeline")

DRUGBANK_NS = "{http://www.drugbank.ca}"
FEATURE_DIM = 64

# ── CYP enzyme indices (dims 0-9) ──
CYP_ENZYMES = [
    "CYP1A2", "CYP2B6", "CYP2C8", "CYP2C9", "CYP2C19",
    "CYP2D6", "CYP3A4", "combined_strength", "overall_burden", "reserved",
]

# ── Receptor indices (dims 10-19) ──
RECEPTORS = [
    "5-HT_reuptake", "NE_reuptake", "DA_reuptake", "mu_opioid",
    "GABA-A", "beta-1", "alpha-1", "H1", "muscarinic", "Na_channel",
]

# ── PK parameter indices (dims 20-29) ──
PK_PARAMS = [
    "half_life_norm", "protein_binding", "Vd_norm", "bioavailability",
    "renal_elim", "hepatic_elim", "active_metabolites", "prodrug",
    "narrow_therapeutic_index", "food_effect",
]

# ── Drug class flags (dims 30-39) ──
CLASS_FLAGS = [
    "SSRI", "opioid", "anticoagulant", "antihypertensive", "antidiabetic",
    "antiarrhythmic", "antibiotic", "antifungal", "PPI_GI", "statin",
]

# ── Risk flags (dims 40-49) ──
RISK_FLAGS = [
    "serotonergic", "QT_prolongation", "bleeding", "CNS_depression",
    "nephrotoxicity", "hepatotoxicity", "hypotension", "hyperkalemia",
    "seizure", "immunosuppression",
]

# ── Transporter indices (dims 50-59) ──
TRANSPORTERS = [
    "P-gp_sub", "P-gp_inh", "OATP1B1_sub", "OATP1B3_sub",
    "OCT2_sub", "OAT1_sub", "OAT3_sub", "BCRP_sub", "MRP2_sub",
    "transporter_inh_strength",
]

# ── Physical properties (dims 60-63) ──
PHYSICAL = ["MW_norm", "logP_norm", "charge_pH7.4", "solubility"]


@dataclass
class ExtractionStats:
    """Track extraction statistics."""
    total_parsed: int = 0
    features_complete: int = 0
    features_partial: int = 0
    features_fallback: int = 0
    drugs_skipped: int = 0
    warnings: list[str] = field(default_factory=list)


class DrugBankParser:
    """
    Parse DrugBank XML → 64-dim pharmacological feature vectors.

    Outputs the same drugs.json format used by the model.
    Falls back to class-based synthetic features when XML is unavailable.
    """

    def __init__(
        self,
        drugbank_xml_path: str | None = None,
        target_drugs: list[str] | None = None,
    ) -> None:
        self.tree = None
        self.root = None
        self.target_drugs = set(d.lower() for d in target_drugs) if target_drugs else None
        self.stats = ExtractionStats()

        if drugbank_xml_path and Path(drugbank_xml_path).exists():
            logger.info(f"Parsing DrugBank XML: {drugbank_xml_path}")
            self.tree = ET.parse(drugbank_xml_path)
            self.root = self.tree.getroot()

    @property
    def has_xml(self) -> bool:
        """Whether real DrugBank XML is available."""
        return self.root is not None

    def extract_all(self) -> dict[str, dict]:
        """
        Extract features for target drugs.

        Returns dict mapping drug names to {id, name, class, features}.
        Uses DrugBank XML if available, otherwise returns empty
        (caller should use synthetic fallback).
        """
        if not self.has_xml:
            logger.warning("No DrugBank XML available. Use SyntheticFeatureGenerator.")
            return {}

        drugs: dict[str, dict] = {}
        for drug_elem in self.root.findall(f".//{DRUGBANK_NS}drug"):
            if not self._is_approved_small_molecule(drug_elem):
                continue

            name = self._get_name(drug_elem)
            if name is None:
                continue
            name = name.lower()

            if self.target_drugs and name not in self.target_drugs:
                continue

            self.stats.total_parsed += 1
            features = self._extract_features(drug_elem)

            if features is not None:
                drugs[name] = {
                    "id": len(drugs),
                    "name": name,
                    "class": self._get_class(drug_elem),
                    "features": features,
                    "drugbank_id": self._get_drugbank_id(drug_elem),
                }
                self.stats.features_complete += 1

        return drugs

    def _is_approved_small_molecule(self, elem: ET.Element) -> bool:
        """Check if drug is an approved small molecule."""
        drug_type = elem.get("type", "")
        if drug_type != "small molecule":
            return False
        groups = elem.find(f"{DRUGBANK_NS}groups")
        if groups is None:
            return False
        return any(
            g.text == "approved"
            for g in groups.findall(f"{DRUGBANK_NS}group")
        )

    def _get_name(self, elem: ET.Element) -> str | None:
        """Get drug name."""
        name_elem = elem.find(f"{DRUGBANK_NS}name")
        return name_elem.text if name_elem is not None else None

    def _get_drugbank_id(self, elem: ET.Element) -> str:
        """Get DrugBank ID."""
        id_elem = elem.find(f"{DRUGBANK_NS}drugbank-id[@primary='true']")
        if id_elem is not None:
            return id_elem.text or ""
        return ""

    def _get_class(self, elem: ET.Element) -> str:
        """Determine drug class from categories."""
        categories = elem.find(f"{DRUGBANK_NS}categories")
        if categories is None:
            return "other"
        cat_texts = [
            c.find(f"{DRUGBANK_NS}category").text.lower()
            for c in categories.findall(f"{DRUGBANK_NS}category")
            if c.find(f"{DRUGBANK_NS}category") is not None
        ]
        cat_str = " ".join(cat_texts)

        if "serotonin" in cat_str or "ssri" in cat_str or "snri" in cat_str:
            return "ssri_snri"
        if "opioid" in cat_str:
            return "opioid"
        if "anticoagulant" in cat_str or "antiplatelet" in cat_str:
            return "anticoagulant"
        if "antihypertensive" in cat_str or "diuretic" in cat_str:
            return "antihypertensive"
        if "statin" in cat_str or "lipid" in cat_str:
            return "statin_lipid"
        if "antidiabetic" in cat_str or "hypoglycemic" in cat_str:
            return "antidiabetic"
        if "antibiotic" in cat_str or "antimicrobial" in cat_str:
            return "antibiotic"
        if "antiepileptic" in cat_str or "anticonvulsant" in cat_str:
            return "antiepileptic"
        if "immunosuppressant" in cat_str:
            return "immunosuppressant"
        if "antiarrhythmic" in cat_str:
            return "cardiac"
        if "antipsychotic" in cat_str or "anxiolytic" in cat_str:
            return "cns_psych"
        if "nsaid" in cat_str or "analgesic" in cat_str:
            return "nsaid_analgesic"
        return "other"

    def _extract_features(self, elem: ET.Element) -> list[float] | None:
        """Extract 64-dim feature vector from a DrugBank drug element."""
        features = [0.0] * FEATURE_DIM

        # CYP interactions (dims 0-9)
        enzymes = self._get_enzyme_data(elem)
        for i, cyp in enumerate(CYP_ENZYMES[:7]):
            features[i] = enzymes.get(cyp, 0.0)

        # Receptor activity (dims 10-19)
        targets = self._get_target_data(elem)
        for i, receptor in enumerate(RECEPTORS):
            features[10 + i] = targets.get(receptor, 0.0)

        # PK params (dims 20-29)
        pk = self._get_pk_data(elem)
        features[20] = self._normalize_half_life(pk.get("half_life"))
        features[21] = pk.get("protein_binding", 0.5)
        features[22] = self._normalize_vd(pk.get("volume_of_distribution"))
        features[23] = pk.get("bioavailability", 0.5)
        features[24] = float(pk.get("renal_elimination", False))
        features[25] = float(pk.get("hepatic_metabolism", True))
        features[26] = float(pk.get("active_metabolites", False))
        features[27] = float(pk.get("prodrug", False))
        features[28] = float(pk.get("narrow_therapeutic_index", False))
        features[29] = pk.get("food_effect", 0.0)

        return features

    def _get_enzyme_data(self, elem: ET.Element) -> dict[str, float]:
        """Extract CYP enzyme interaction data."""
        result: dict[str, float] = {}
        enzymes = elem.find(f"{DRUGBANK_NS}enzymes")
        if enzymes is None:
            return result
        for enzyme in enzymes.findall(f"{DRUGBANK_NS}enzyme"):
            name_elem = enzyme.find(f"{DRUGBANK_NS}name")
            actions = enzyme.find(f"{DRUGBANK_NS}actions")
            if name_elem is None or name_elem.text is None:
                continue
            name = name_elem.text.upper()
            if actions is not None:
                for action in actions.findall(f"{DRUGBANK_NS}action"):
                    if action.text and "inhibit" in action.text.lower():
                        result[name] = max(result.get(name, 0.0), 0.8)
                    elif action.text and "induc" in action.text.lower():
                        result[name] = max(result.get(name, 0.0), 0.7)
                    elif action.text and "substrate" in action.text.lower():
                        result[name] = max(result.get(name, 0.0), 0.5)
        return result

    def _get_target_data(self, elem: ET.Element) -> dict[str, float]:
        """Extract receptor/target activity data."""
        return {}  # Requires detailed target parsing — placeholder

    def _get_pk_data(self, elem: ET.Element) -> dict:
        """Extract pharmacokinetic parameters."""
        pk: dict = {}
        half_life = elem.find(f"{DRUGBANK_NS}half-life")
        if half_life is not None and half_life.text:
            pk["half_life"] = self._parse_half_life(half_life.text)

        protein_binding = elem.find(f"{DRUGBANK_NS}protein-binding")
        if protein_binding is not None and protein_binding.text:
            pk["protein_binding"] = self._parse_percentage(protein_binding.text)

        return pk

    @staticmethod
    def _normalize_half_life(hours: float | None) -> float:
        """Normalize half-life to 0-1 using log scale."""
        if hours is None:
            return 0.5
        return min(1.0, math.log1p(hours) / math.log1p(100))

    @staticmethod
    def _normalize_vd(vd: float | None) -> float:
        """Normalize volume of distribution to 0-1."""
        if vd is None:
            return 0.5
        return min(1.0, vd / 500.0)

    @staticmethod
    def _parse_half_life(text: str) -> float | None:
        """Parse half-life text to hours."""
        import re
        match = re.search(r"([\d.]+)", text)
        if match:
            return float(match.group(1))
        return None

    @staticmethod
    def _parse_percentage(text: str) -> float:
        """Parse percentage text to 0-1 float."""
        import re
        match = re.search(r"([\d.]+)", text)
        if match:
            val = float(match.group(1))
            return val / 100.0 if val > 1.0 else val
        return 0.5
