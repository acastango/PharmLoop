"""Tests for the Phase 4a data pipeline."""

import json
from pathlib import Path

import pytest


DATA_DIR = Path("data/processed")
RAW_DIR = Path("data/raw")


class TestDrugList:
    """Tests for the top 300 drugs list."""

    def test_drug_list_exists(self):
        assert (RAW_DIR / "top_300_drugs.txt").exists()

    def test_drug_list_has_enough_drugs(self):
        drugs = _load_drug_list()
        assert len(drugs) >= 280, f"Only {len(drugs)} drugs, expected ≥280"

    def test_drug_classes_are_valid(self):
        valid_classes = {
            "ssri_snri", "opioid", "anticoagulant", "antihypertensive",
            "statin_lipid", "antidiabetic", "antibiotic", "antiepileptic",
            "immunosuppressant", "cardiac", "cns_psych", "nsaid_analgesic",
        }
        drugs = _load_drug_list()
        for name, primary_class, *_ in drugs:
            assert primary_class in valid_classes, (
                f"Drug {name} has invalid class {primary_class}"
            )


class TestV2Data:
    """Tests for v2 JSON data files."""

    def test_drugs_v2_exists(self):
        assert (DATA_DIR / "drugs_v2.json").exists()

    def test_interactions_v2_exists(self):
        assert (DATA_DIR / "interactions_v2.json").exists()

    def test_drugs_v2_has_correct_structure(self):
        with open(DATA_DIR / "drugs_v2.json") as f:
            data = json.load(f)
        assert "drugs" in data
        assert "num_drugs" in data
        assert data["num_drugs"] >= 280

    def test_drugs_v2_has_features(self):
        with open(DATA_DIR / "drugs_v2.json") as f:
            data = json.load(f)
        for name, info in data["drugs"].items():
            assert "features" in info, f"Drug {name} missing features"
            assert len(info["features"]) == 64, (
                f"Drug {name} has {len(info['features'])} features, expected 64"
            )

    def test_original_50_drugs_preserved(self):
        """All original 50 drugs from Phase 1 are in v2."""
        if not (DATA_DIR / "drugs.json").exists():
            pytest.skip("No original drugs.json to compare against")

        with open(DATA_DIR / "drugs.json") as f:
            v1 = json.load(f)
        with open(DATA_DIR / "drugs_v2.json") as f:
            v2 = json.load(f)

        for name in v1["drugs"]:
            assert name in v2["drugs"], f"Original drug {name} missing from v2"

    def test_interactions_v2_quality(self):
        with open(DATA_DIR / "interactions_v2.json") as f:
            data = json.load(f)
        interactions = data["interactions"]
        assert len(interactions) >= 1200, (
            f"Only {len(interactions)} interactions, expected ≥1200"
        )

        # Check severity distribution
        severities = [ix["severity"] for ix in interactions]
        assert "severe" in severities or "contraindicated" in severities, (
            "No severe/contraindicated interactions found"
        )
        assert "none" in severities, "No safe (none) interactions found"

    def test_interactions_have_mechanisms(self):
        with open(DATA_DIR / "interactions_v2.json") as f:
            data = json.load(f)
        interactions = data["interactions"]

        with_mechs = sum(1 for ix in interactions if ix.get("mechanisms"))
        ratio = with_mechs / len(interactions)
        assert ratio > 0.5, (
            f"Only {ratio:.0%} interactions have mechanisms, expected >50%"
        )


def _load_drug_list() -> list[tuple]:
    """Parse the drug list file."""
    drugs = []
    with open(RAW_DIR / "top_300_drugs.txt") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("|")
            drugs.append(tuple(parts))
    return drugs
