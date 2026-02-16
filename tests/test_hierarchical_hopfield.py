"""Tests for HierarchicalHopfield memory."""

import torch
import pytest

from pharmloop.hierarchical_hopfield import HierarchicalHopfield, DRUG_CLASSES
from pharmloop.hopfield import PharmHopfield


class TestHierarchicalHopfieldInterface:
    """Test that HierarchicalHopfield has PharmHopfield-compatible interface."""

    def test_has_input_dim(self):
        hh = HierarchicalHopfield(input_dim=512)
        assert hh.input_dim == 512

    def test_has_count(self):
        hh = HierarchicalHopfield(input_dim=512)
        assert hh.count == 0

    def test_has_store(self):
        hh = HierarchicalHopfield(input_dim=512)
        assert hasattr(hh, "store")

    def test_has_retrieve(self):
        hh = HierarchicalHopfield(input_dim=512)
        assert hasattr(hh, "retrieve")

    def test_has_clear(self):
        hh = HierarchicalHopfield(input_dim=512)
        assert hasattr(hh, "clear")


class TestHierarchicalHopfieldStorage:
    """Test pattern storage in class-specific and global banks."""

    def test_store_increases_global_count(self):
        hh = HierarchicalHopfield(input_dim=512)
        patterns = torch.randn(10, 512)
        hh.store(patterns)
        assert hh.count == 10

    def test_store_with_classes_fills_class_banks(self):
        hh = HierarchicalHopfield(input_dim=512)
        patterns = torch.randn(3, 512)
        classes = [
            ("ssri_snri", "opioid"),
            ("ssri_snri", "anticoagulant"),
            ("opioid", "cardiac"),
        ]
        hh.store(patterns, drug_classes=classes)

        # Global should have all 3
        assert hh.global_bank.count == 3
        # ssri_snri bank should have 2 (patterns 0 and 1)
        assert hh.class_banks["ssri_snri"].count == 2
        # opioid bank should have 2 (patterns 0 and 2)
        assert hh.class_banks["opioid"].count == 2
        # anticoagulant should have 1
        assert hh.class_banks["anticoagulant"].count == 1

    def test_clear_empties_all_banks(self):
        hh = HierarchicalHopfield(input_dim=512)
        patterns = torch.randn(5, 512)
        classes = [("ssri_snri", "opioid")] * 5
        hh.store(patterns, drug_classes=classes)
        hh.clear()
        assert hh.count == 0
        assert hh.class_banks["ssri_snri"].count == 0


class TestHierarchicalHopfieldRetrieval:
    """Test retrieval from hierarchical banks."""

    def test_retrieve_without_classes_uses_global(self):
        hh = HierarchicalHopfield(input_dim=512)
        patterns = torch.randn(5, 512)
        hh.store(patterns)

        query = torch.randn(1, 512)
        result = hh.retrieve(query, beta=1.0)
        assert result.shape == (1, 512)

    def test_retrieve_with_classes_blends_class_and_global(self):
        hh = HierarchicalHopfield(input_dim=512)

        # Store distinctive patterns in ssri_snri
        ssri_patterns = torch.ones(3, 512)
        ssri_classes = [("ssri_snri", "ssri_snri")] * 3
        hh.store(ssri_patterns, drug_classes=ssri_classes)

        # Store different patterns globally
        other_patterns = -torch.ones(3, 512)
        hh.global_bank.store(other_patterns)

        query = torch.ones(1, 512) * 0.5
        hh._current_classes = ("ssri_snri", "ssri_snri")
        result = hh.retrieve(query, beta=1.0)
        assert result.shape == (1, 512)

    def test_empty_class_bank_falls_back_to_global(self):
        hh = HierarchicalHopfield(input_dim=512)
        patterns = torch.randn(5, 512)
        hh.store(patterns)  # No class routing

        hh._current_classes = ("statin_lipid", "statin_lipid")
        query = torch.randn(1, 512)
        result = hh.retrieve(query, beta=1.0)
        assert result.shape == (1, 512)

    def test_class_bank_capacity_limit(self):
        hh = HierarchicalHopfield(input_dim=512, class_capacity=5)
        patterns = torch.randn(10, 512)
        classes = [("ssri_snri", "ssri_snri")] * 10
        hh.store(patterns, drug_classes=classes)

        # Class bank should be capped at 5
        assert hh.class_banks["ssri_snri"].count == 5
        # Global should have all 10
        assert hh.global_bank.count == 10

    def test_all_drug_classes_have_banks(self):
        hh = HierarchicalHopfield(input_dim=512)
        for cls in DRUG_CLASSES:
            assert cls in hh.class_banks, f"Missing bank for {cls}"
