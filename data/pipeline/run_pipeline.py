"""
Phase 4a data pipeline orchestrator.

Reads the target drug list, extracts features (from DrugBank XML or
synthetic generation), extracts interactions, runs quality checks,
and produces drugs_v2.json + interactions_v2.json.

Usage:
    python -m data.pipeline.run_pipeline [--drugbank path/to/drugbank.xml]
"""

import argparse
import hashlib
import json
import logging
import random
from itertools import combinations
from pathlib import Path

import torch

from data.pipeline.drugbank_parser import DrugBankParser
from data.pipeline.interaction_extractor import (
    InteractionExtractor,
    MECHANISM_TO_FLAGS,
    SEVERITY_ORDER,
)
from data.pipeline.quality_report import DataQualityReport, validate_feature_continuity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger("pharmloop.pipeline")

# ── Class-level pharmacological priors ──
# These define default feature ranges per drug class.
# Used for synthetic feature generation when DrugBank XML is not available.
# Each value is (dim_index, default_value) or (dim_index, low, high) for range.

CLASS_FEATURE_PRIORS: dict[str, dict[int, float | tuple[float, float]]] = {
    "ssri_snri": {
        5: (0.3, 1.0),   # CYP2D6 inhibition
        6: (0.2, 0.5),   # CYP3A4
        10: (0.7, 1.0),  # 5-HT reuptake
        11: (0.0, 0.5),  # NE reuptake (higher for SNRIs)
        20: (0.5, 1.0),  # half-life
        21: (0.85, 0.98), # protein binding
        25: (0.8, 1.0),  # hepatic elimination
        30: 1.0,          # SSRI class flag
        40: (0.7, 1.0),  # serotonergic risk
        48: (0.1, 0.3),  # seizure risk
        50: (0.3, 0.8),  # P-gp substrate
    },
    "opioid": {
        8: (0.5, 0.9),   # CYP2D6 substrate
        9: (0.3, 0.7),   # CYP3A4 substrate
        13: (0.6, 1.0),  # mu-opioid
        20: (0.3, 0.8),  # half-life
        25: (0.7, 1.0),  # hepatic elimination
        31: 1.0,          # opioid class flag
        43: (0.5, 0.9),  # CNS depression
        48: (0.1, 0.3),  # seizure risk
    },
    "anticoagulant": {
        3: (0.1, 0.5),   # CYP2C9 (warfarin)
        21: (0.9, 0.99),  # protein binding
        28: 1.0,          # NTI
        32: 1.0,          # anticoagulant class flag
        42: (0.7, 1.0),  # bleeding risk
    },
    "antihypertensive": {
        20: (0.3, 0.7),  # half-life
        24: (0.3, 0.7),  # renal elimination
        33: 1.0,          # antihypertensive class flag
        46: (0.5, 0.9),  # hypotension risk
        47: (0.1, 0.5),  # hyperkalemia risk (ACEi/ARBs)
    },
    "statin_lipid": {
        6: (0.3, 0.8),   # CYP3A4 (most statins)
        9: (0.5, 0.9),   # CYP3A4 substrate
        21: (0.9, 0.99),  # protein binding
        25: (0.8, 1.0),  # hepatic elimination
        39: 1.0,          # statin class flag
        45: (0.2, 0.5),  # hepatotoxicity
        50: (0.2, 0.6),  # P-gp substrate
        52: (0.3, 0.7),  # OATP1B1 substrate
    },
    "antidiabetic": {
        20: (0.3, 0.6),  # half-life
        34: 1.0,          # antidiabetic class flag
    },
    "antibiotic": {
        20: (0.2, 0.5),  # half-life (most short)
        24: (0.3, 0.7),  # renal elimination
        36: 1.0,          # antibiotic class flag
        44: (0.1, 0.3),  # nephrotoxicity (aminoglycosides)
    },
    "antiepileptic": {
        6: (0.3, 0.8),   # CYP3A4
        9: (0.5, 0.9),   # CYP3A4 substrate
        20: (0.5, 0.9),  # half-life
        25: (0.8, 1.0),  # hepatic elimination
        28: (0.5, 1.0),  # NTI (phenytoin, carbamazepine)
        43: (0.3, 0.7),  # CNS depression
        48: 0.0,          # seizure risk (they treat seizures)
    },
    "immunosuppressant": {
        6: (0.2, 0.5),   # CYP3A4
        9: (0.7, 1.0),   # CYP3A4 substrate
        20: (0.4, 0.8),  # half-life
        28: (0.5, 1.0),  # NTI
        44: (0.3, 0.6),  # nephrotoxicity
        49: (0.7, 1.0),  # immunosuppression
    },
    "cardiac": {
        20: (0.5, 1.0),  # half-life (amiodarone very long)
        28: (0.5, 1.0),  # NTI
        35: 1.0,          # antiarrhythmic class flag
        41: (0.3, 0.8),  # QT prolongation
    },
    "cns_psych": {
        5: (0.1, 0.5),   # CYP2D6
        9: (0.3, 0.7),   # CYP3A4 substrate
        20: (0.3, 0.8),  # half-life
        21: (0.85, 0.98), # protein binding
        25: (0.8, 1.0),  # hepatic elimination
        41: (0.1, 0.5),  # QT prolongation
        43: (0.5, 0.9),  # CNS depression
        46: (0.2, 0.5),  # hypotension
    },
    "nsaid_analgesic": {
        17: (0.5, 1.0),  # COX inhibition
        20: (0.2, 0.5),  # half-life
        21: (0.9, 0.99),  # protein binding
        42: (0.3, 0.7),  # bleeding risk
        44: (0.2, 0.5),  # nephrotoxicity
    },
}

# ── Class-level interaction rules ──
# (class_a, class_b) → list of (severity, mechanisms, probability)
# probability controls how often this interaction appears for pairs in these classes.

CLASS_INTERACTION_RULES: list[dict] = [
    # Serotonergic combinations
    {"classes": ("ssri_snri", "ssri_snri"), "severity": "severe",
     "mechanisms": ["serotonergic"], "prob": 0.30,
     "flags": ["monitor_serotonin_syndrome", "avoid_combination"]},
    {"classes": ("ssri_snri", "opioid"), "severity": "severe",
     "mechanisms": ["serotonergic", "cyp_inhibition"], "prob": 0.25,
     "flags": ["monitor_serotonin_syndrome", "avoid_combination"]},

    # QT prolongation combinations
    {"classes": ("cardiac", "cardiac"), "severity": "severe",
     "mechanisms": ["qt_prolongation"], "prob": 0.20,
     "flags": ["monitor_qt_interval", "avoid_combination"]},
    {"classes": ("cardiac", "cns_psych"), "severity": "moderate",
     "mechanisms": ["qt_prolongation"], "prob": 0.10,
     "flags": ["monitor_qt_interval"]},
    {"classes": ("cardiac", "antibiotic"), "severity": "moderate",
     "mechanisms": ["qt_prolongation"], "prob": 0.08,
     "flags": ["monitor_qt_interval"]},

    # CYP interactions
    {"classes": ("ssri_snri", "antiepileptic"), "severity": "moderate",
     "mechanisms": ["cyp_inhibition"], "prob": 0.15,
     "flags": ["monitor_drug_levels"]},
    {"classes": ("ssri_snri", "cns_psych"), "severity": "moderate",
     "mechanisms": ["cyp_inhibition", "cns_depression"], "prob": 0.12,
     "flags": ["monitor_drug_levels", "monitor_cns_depression"]},
    {"classes": ("antibiotic", "statin_lipid"), "severity": "moderate",
     "mechanisms": ["cyp_inhibition"], "prob": 0.12,
     "flags": ["monitor_drug_levels", "reduce_statin_dose"]},
    {"classes": ("antibiotic", "immunosuppressant"), "severity": "severe",
     "mechanisms": ["cyp_inhibition"], "prob": 0.20,
     "flags": ["monitor_drug_levels", "avoid_combination"]},
    {"classes": ("antibiotic", "anticoagulant"), "severity": "moderate",
     "mechanisms": ["cyp_inhibition", "bleeding_risk"], "prob": 0.12,
     "flags": ["monitor_inr", "monitor_bleeding"]},
    {"classes": ("antibiotic", "cardiac"), "severity": "moderate",
     "mechanisms": ["cyp_inhibition"], "prob": 0.10,
     "flags": ["monitor_drug_levels"]},
    {"classes": ("antiepileptic", "statin_lipid"), "severity": "moderate",
     "mechanisms": ["cyp_induction"], "prob": 0.12,
     "flags": ["monitor_drug_levels"]},
    {"classes": ("antiepileptic", "anticoagulant"), "severity": "moderate",
     "mechanisms": ["cyp_induction"], "prob": 0.15,
     "flags": ["monitor_inr"]},
    {"classes": ("antiepileptic", "immunosuppressant"), "severity": "severe",
     "mechanisms": ["cyp_induction"], "prob": 0.18,
     "flags": ["monitor_drug_levels"]},

    # Bleeding risk
    {"classes": ("anticoagulant", "nsaid_analgesic"), "severity": "severe",
     "mechanisms": ["bleeding_risk"], "prob": 0.35,
     "flags": ["monitor_bleeding", "monitor_inr", "avoid_combination"]},
    {"classes": ("anticoagulant", "anticoagulant"), "severity": "severe",
     "mechanisms": ["bleeding_risk"], "prob": 0.30,
     "flags": ["monitor_bleeding", "monitor_inr", "avoid_combination"]},
    {"classes": ("anticoagulant", "ssri_snri"), "severity": "moderate",
     "mechanisms": ["bleeding_risk"], "prob": 0.15,
     "flags": ["monitor_bleeding"]},

    # CNS depression
    {"classes": ("opioid", "cns_psych"), "severity": "severe",
     "mechanisms": ["cns_depression"], "prob": 0.20,
     "flags": ["monitor_cns_depression", "avoid_combination"]},
    {"classes": ("opioid", "opioid"), "severity": "contraindicated",
     "mechanisms": ["cns_depression"], "prob": 0.25,
     "flags": ["monitor_cns_depression", "avoid_combination"]},
    {"classes": ("cns_psych", "cns_psych"), "severity": "moderate",
     "mechanisms": ["cns_depression"], "prob": 0.08,
     "flags": ["monitor_cns_depression"]},

    # Nephrotoxicity
    {"classes": ("nsaid_analgesic", "antihypertensive"), "severity": "moderate",
     "mechanisms": ["nephrotoxicity", "hypotension"], "prob": 0.12,
     "flags": ["monitor_renal_function", "monitor_blood_pressure"]},
    {"classes": ("nsaid_analgesic", "anticoagulant"), "severity": "severe",
     "mechanisms": ["bleeding_risk", "nephrotoxicity"], "prob": 0.20,
     "flags": ["monitor_bleeding", "monitor_renal_function"]},
    {"classes": ("antibiotic", "nsaid_analgesic"), "severity": "mild",
     "mechanisms": ["nephrotoxicity"], "prob": 0.06,
     "flags": ["monitor_renal_function"]},

    # Hyperkalemia
    {"classes": ("antihypertensive", "antihypertensive"), "severity": "moderate",
     "mechanisms": ["hyperkalemia", "hypotension"], "prob": 0.08,
     "flags": ["monitor_electrolytes", "monitor_blood_pressure"]},

    # Immunosuppressant combos
    {"classes": ("immunosuppressant", "immunosuppressant"), "severity": "moderate",
     "mechanisms": ["immunosuppression", "nephrotoxicity"], "prob": 0.25,
     "flags": ["monitor_drug_levels", "monitor_renal_function"]},

    # Statin + azole antifungals
    {"classes": ("statin_lipid", "antibiotic"), "severity": "moderate",
     "mechanisms": ["cyp_inhibition"], "prob": 0.10,
     "flags": ["reduce_statin_dose"]},

    # Antidiabetic combos
    {"classes": ("antidiabetic", "antidiabetic"), "severity": "mild",
     "mechanisms": ["hypotension"], "prob": 0.08,
     "flags": ["monitor_blood_glucose"]},
    {"classes": ("antidiabetic", "antibiotic"), "severity": "mild",
     "mechanisms": ["cyp_inhibition"], "prob": 0.05,
     "flags": ["monitor_blood_glucose"]},

    # Absorption interactions
    {"classes": ("antibiotic", "antidiabetic"), "severity": "mild",
     "mechanisms": ["absorption_altered"], "prob": 0.05,
     "flags": ["separate_administration"]},

    # Electrolyte imbalance
    {"classes": ("antihypertensive", "cardiac"), "severity": "moderate",
     "mechanisms": ["electrolyte_imbalance", "hypotension"], "prob": 0.08,
     "flags": ["monitor_electrolytes", "monitor_blood_pressure"]},
]


def _deterministic_random(drug_a: str, drug_b: str, salt: str = "") -> float:
    """Deterministic pseudo-random float from drug pair names."""
    key = f"{min(drug_a, drug_b)}:{max(drug_a, drug_b)}:{salt}"
    h = hashlib.sha256(key.encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def _drug_random(drug_name: str, dim: int) -> float:
    """Deterministic pseudo-random float for a drug+dimension."""
    key = f"{drug_name}:{dim}"
    h = hashlib.sha256(key.encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def parse_drug_list(drug_list_path: str) -> list[dict]:
    """
    Parse top_300_drugs.txt into list of {name, primary_class, secondary_class}.
    """
    drugs: list[dict] = []
    seen = set()
    with open(drug_list_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("|")
            name = parts[0].strip().lower()
            if name in seen:
                continue
            seen.add(name)
            primary_class = parts[1].strip() if len(parts) > 1 else "other"
            secondary_class = parts[2].strip() if len(parts) > 2 else None
            drugs.append({
                "name": name,
                "primary_class": primary_class,
                "secondary_class": secondary_class,
            })
    return drugs


def generate_synthetic_features(
    drug_entry: dict,
    existing_features: dict[str, list[float]] | None = None,
) -> list[float]:
    """
    Generate pharmacologically plausible 64-dim feature vector.

    If the drug exists in existing_features (original 50), returns those
    features unchanged. Otherwise generates based on class priors with
    deterministic per-drug variation.
    """
    name = drug_entry["name"]

    # Preserve original 50 drug features exactly
    if existing_features and name in existing_features:
        return existing_features[name]

    features = [0.0] * 64
    primary = drug_entry["primary_class"]
    secondary = drug_entry.get("secondary_class")

    # Apply class priors
    for cls in [primary, secondary]:
        if cls is None or cls not in CLASS_FEATURE_PRIORS:
            continue
        for dim, val in CLASS_FEATURE_PRIORS[cls].items():
            if isinstance(val, tuple):
                low, high = val
                r = _drug_random(name, dim)
                features[dim] = max(features[dim], low + r * (high - low))
            else:
                features[dim] = max(features[dim], val)

    # Add general PK variation
    for dim in [20, 21, 22, 23]:  # half-life, protein binding, Vd, bioavail
        if features[dim] == 0.0:
            features[dim] = 0.3 + _drug_random(name, dim) * 0.4

    # Physical properties
    features[60] = 0.2 + _drug_random(name, 60) * 0.6  # MW norm
    features[61] = 0.2 + _drug_random(name, 61) * 0.6  # logP norm
    features[62] = -0.5 + _drug_random(name, 62) * 1.0  # charge
    features[63] = 0.3 + _drug_random(name, 63) * 0.5  # solubility

    # Round to 2 decimal places for cleanliness
    features = [round(f, 2) for f in features]
    return features


def generate_interactions(
    drugs: list[dict],
    original_interactions: list[dict] | None = None,
) -> list[dict]:
    """
    Generate interactions based on class-level rules.

    Preserves all original 209 interactions exactly. Generates new
    interactions for new drug pairs using class interaction rules.

    Returns list of interaction dicts in PharmLoop format.
    """
    interactions: list[dict] = []
    seen_pairs: set[tuple[str, str]] = set()

    # Preserve original interactions
    if original_interactions:
        for ix in original_interactions:
            pair = (min(ix["drug_a"], ix["drug_b"]),
                    max(ix["drug_a"], ix["drug_b"]))
            seen_pairs.add(pair)
            interactions.append(ix)

    # Build class lookup
    drug_classes: dict[str, list[str]] = {}
    for d in drugs:
        classes = [d["primary_class"]]
        if d.get("secondary_class"):
            classes.append(d["secondary_class"])
        drug_classes[d["name"]] = classes

    drug_names = [d["name"] for d in drugs]

    # Generate new interactions from class rules
    for drug_a, drug_b in combinations(drug_names, 2):
        pair = (min(drug_a, drug_b), max(drug_a, drug_b))
        if pair in seen_pairs:
            continue

        classes_a = drug_classes.get(drug_a, [])
        classes_b = drug_classes.get(drug_b, [])

        # Check each rule
        best_rule = None
        best_severity_order = -1

        for rule in CLASS_INTERACTION_RULES:
            rule_a, rule_b = rule["classes"]
            matches = (
                (rule_a in classes_a and rule_b in classes_b) or
                (rule_b in classes_a and rule_a in classes_b)
            )
            if not matches:
                continue

            # Deterministic check: does this pair trigger this rule?
            r = _deterministic_random(drug_a, drug_b, rule["mechanisms"][0])
            if r > rule["prob"]:
                continue

            sev_order = SEVERITY_ORDER.get(rule["severity"], 2)
            if sev_order > best_severity_order:
                best_severity_order = sev_order
                best_rule = rule

        if best_rule is not None:
            seen_pairs.add(pair)
            interactions.append({
                "drug_a": drug_a,
                "drug_b": drug_b,
                "severity": best_rule["severity"],
                "mechanisms": best_rule["mechanisms"],
                "flags": best_rule["flags"],
                "source": "class_rule",
                "notes": f"Generated from class rule: "
                         f"{best_rule['classes'][0]}×{best_rule['classes'][1]}",
            })

    # Also add "none" severity for same-class pairs that didn't trigger rules
    # (up to a limit, to provide negative examples)
    none_count = 0
    max_none = len(interactions) // 2  # roughly balance
    for drug_a, drug_b in combinations(drug_names, 2):
        if none_count >= max_none:
            break
        pair = (min(drug_a, drug_b), max(drug_a, drug_b))
        if pair in seen_pairs:
            continue

        # Only add none-pairs for drugs in different classes (more realistic)
        classes_a = set(drug_classes.get(drug_a, []))
        classes_b = set(drug_classes.get(drug_b, []))
        if classes_a & classes_b:
            continue

        r = _deterministic_random(drug_a, drug_b, "none_sample")
        if r < 0.02:  # very sparse sampling of negatives
            seen_pairs.add(pair)
            interactions.append({
                "drug_a": drug_a,
                "drug_b": drug_b,
                "severity": "none",
                "mechanisms": [],
                "flags": [],
                "source": "negative_sample",
                "notes": "No known interaction between these classes.",
            })
            none_count += 1

    return interactions


def run_pipeline(
    drug_list_path: str,
    output_dir: str,
    drugbank_xml_path: str | None = None,
    original_data_dir: str | None = None,
) -> DataQualityReport:
    """
    Run the full Phase 4a data pipeline.

    Args:
        drug_list_path: Path to top_300_drugs.txt.
        output_dir: Directory for output files.
        drugbank_xml_path: Optional path to DrugBank XML.
        original_data_dir: Path to Phase 1-3 data (for preservation).

    Returns:
        DataQualityReport with extraction statistics.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report = DataQualityReport()

    # ── Load original data for preservation ──
    original_drugs: dict = {}
    original_interactions: list[dict] = []
    original_features: dict[str, list[float]] = {}

    if original_data_dir:
        orig_path = Path(original_data_dir)
        drugs_file = orig_path / "drugs.json"
        if drugs_file.exists():
            with open(drugs_file) as f:
                orig_data = json.load(f)
            original_drugs = orig_data.get("drugs", {})
            original_features = {
                name: info["features"]
                for name, info in original_drugs.items()
            }
            logger.info(f"Loaded {len(original_drugs)} original drugs for preservation")

        ix_file = orig_path / "interactions.json"
        if ix_file.exists():
            with open(ix_file) as f:
                ix_data = json.load(f)
            original_interactions = ix_data.get("interactions", [])
            logger.info(f"Loaded {len(original_interactions)} original interactions")

    # ── Parse target drug list ──
    drug_entries = parse_drug_list(drug_list_path)
    logger.info(f"Target drug list: {len(drug_entries)} drugs")

    # ── Try DrugBank XML extraction ──
    parser = DrugBankParser(drugbank_xml_path, [d["name"] for d in drug_entries])
    xml_drugs = parser.extract_all()
    if xml_drugs:
        logger.info(f"Extracted {len(xml_drugs)} drugs from DrugBank XML")
    else:
        logger.info("No DrugBank XML — using synthetic feature generation")

    # ── Build drugs_v2.json ──
    drugs_v2: dict[str, dict] = {}
    for i, entry in enumerate(drug_entries):
        name = entry["name"]

        if name in xml_drugs:
            # Use real DrugBank features
            drug_info = xml_drugs[name]
            drug_info["id"] = i
            drugs_v2[name] = drug_info
            report.drugs_with_complete_features += 1
        else:
            # Synthetic features
            features = generate_synthetic_features(entry, original_features)
            drugs_v2[name] = {
                "id": i,
                "name": name,
                "class": entry["primary_class"],
                "features": features,
            }
            if name in original_features:
                report.drugs_with_complete_features += 1
            else:
                report.drugs_with_fallback_values += 1

    report.total_drugs = len(drugs_v2)
    logger.info(f"Built {report.total_drugs} drug entries")

    # ── Validate original drug feature continuity ──
    if original_drugs:
        passed, drift_warnings = validate_feature_continuity(
            original_drugs, drugs_v2, max_drift=0.15,
        )
        for w in drift_warnings:
            logger.warning(w)
            report.missing_feature_drugs.append(w)
        if passed:
            logger.info("Original drug feature continuity: PASSED")
        else:
            logger.warning("Original drug feature continuity: SOME DRIFT DETECTED")

    # ── Generate interactions_v2.json ──
    interactions_v2 = generate_interactions(drug_entries, original_interactions)
    report.total_interactions = len(interactions_v2)

    # Count mechanism quality
    for ix in interactions_v2:
        if ix["severity"] == "none" or len(ix["mechanisms"]) > 0:
            report.interactions_with_clear_mechanism += 1
        else:
            report.interactions_with_ambiguous_mechanism += 1
            report.ambiguous_mechanism_pairs.append(
                f"{ix['drug_a']}+{ix['drug_b']}"
            )

    logger.info(f"Built {report.total_interactions} interactions")

    # ── Severity distribution ──
    sev_dist: dict[str, int] = {}
    for ix in interactions_v2:
        sev_dist[ix["severity"]] = sev_dist.get(ix["severity"], 0) + 1
    logger.info(f"Severity distribution: {sev_dist}")

    # ── Write outputs ──
    drugs_output = {
        "num_drugs": len(drugs_v2),
        "feature_dim": 64,
        "sources": (
            "Phase 1-3 hand-curated + Phase 4a synthetic generation "
            "from pharmacological class priors"
        ),
        "drugs": drugs_v2,
    }
    with open(output_path / "drugs_v2.json", "w") as f:
        json.dump(drugs_output, f, indent=2)

    interactions_output = {
        "num_interactions": len(interactions_v2),
        "interactions": interactions_v2,
    }
    with open(output_path / "interactions_v2.json", "w") as f:
        json.dump(interactions_output, f, indent=2)

    # ── Quality report ──
    report.log_summary()
    report.save(output_path / "quality_report.json")

    # ── Quality gate check ──
    report.passes_quality_gate()

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4a data pipeline")
    parser.add_argument("--drugbank", type=str, default=None,
                        help="Path to DrugBank XML (optional)")
    parser.add_argument("--drug-list", type=str,
                        default="data/raw/top_300_drugs.txt")
    parser.add_argument("--output-dir", type=str,
                        default="data/processed")
    parser.add_argument("--original-data", type=str,
                        default="data/processed")
    args = parser.parse_args()

    run_pipeline(
        drug_list_path=args.drug_list,
        output_dir=args.output_dir,
        drugbank_xml_path=args.drugbank,
        original_data_dir=args.original_data,
    )


if __name__ == "__main__":
    main()
