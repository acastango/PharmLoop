"""Tests for DrugResolver: brand name resolution and fuzzy matching."""

import pytest

from pharmloop.drug_resolver import DrugResolver, ResolvedDrug


@pytest.fixture
def registry():
    """Simple drug registry for testing."""
    return {
        "fluoxetine": {"id": 0},
        "sertraline": {"id": 1},
        "warfarin": {"id": 2},
        "metformin": {"id": 3},
        "acetaminophen": {"id": 4},
        "amoxicillin_clavulanate": {"id": 5},
    }


@pytest.fixture
def brand_map():
    """Brand → generic map for testing."""
    return {
        "prozac": "fluoxetine",
        "zoloft": "sertraline",
        "coumadin": "warfarin",
        "glucophage": "metformin",
        "tylenol": "acetaminophen",
        "augmentin": "amoxicillin_clavulanate",
    }


@pytest.fixture
def resolver(registry, brand_map):
    return DrugResolver(registry, brand_map)


class TestDrugResolver:
    """Tests for drug name resolution."""

    def test_exact_match_lowercase(self, resolver):
        result = resolver.resolve("fluoxetine")
        assert result.resolved == "fluoxetine"
        assert result.method == "exact"
        assert result.confidence == 1.0

    def test_exact_match_uppercase(self, resolver):
        result = resolver.resolve("FLUOXETINE")
        assert result.resolved == "fluoxetine"
        assert result.method == "exact"

    def test_exact_match_mixed_case(self, resolver):
        result = resolver.resolve("Warfarin")
        assert result.resolved == "warfarin"
        assert result.method == "exact"

    def test_brand_name(self, resolver):
        result = resolver.resolve("Prozac")
        assert result.resolved == "fluoxetine"
        assert result.method == "brand"
        assert result.confidence == 1.0

    def test_brand_name_case_insensitive(self, resolver):
        result = resolver.resolve("COUMADIN")
        assert result.resolved == "warfarin"
        assert result.method == "brand"

    def test_fuzzy_match(self, resolver):
        # "fluoxetin" is close to "fluoxetine"
        result = resolver.resolve("fluoxetin")
        assert result.resolved == "fluoxetine"
        assert result.method == "fuzzy"
        assert result.confidence > 0

    def test_unknown_drug(self, resolver):
        result = resolver.resolve("xyz123unknown")
        assert result.resolved is None
        assert result.method == "unknown"
        assert result.confidence == 0.0

    def test_whitespace_handling(self, resolver):
        result = resolver.resolve("  fluoxetine  ")
        assert result.resolved == "fluoxetine"
        assert result.method == "exact"

    def test_separator_normalization(self, resolver):
        result = resolver.resolve("amoxicillin/clavulanate")
        assert result.resolved == "amoxicillin_clavulanate"
        assert result.method == "exact"

    def test_resolve_many(self, resolver):
        results = resolver.resolve_many(["Prozac", "Coumadin", "xyz123"])
        assert len(results) == 3
        assert results[0].resolved == "fluoxetine"
        assert results[1].resolved == "warfarin"
        assert results[2].resolved is None


class TestDrugResolverLoad:
    """Tests for DrugResolver.load factory method."""

    def test_load_without_brand_file(self, registry, tmp_path):
        resolver = DrugResolver.load(registry, brand_names_path=None)
        # Should still do exact match
        result = resolver.resolve("fluoxetine")
        assert result.resolved == "fluoxetine"

    def test_load_with_brand_file(self, registry, tmp_path):
        brand_file = tmp_path / "brands.json"
        brand_file.write_text('{"Prozac": "fluoxetine"}')

        resolver = DrugResolver.load(registry, brand_names_path=str(brand_file))
        result = resolver.resolve("Prozac")
        assert result.resolved == "fluoxetine"
        assert result.method == "brand"

    def test_load_missing_brand_file(self, registry):
        resolver = DrugResolver.load(
            registry, brand_names_path="/nonexistent/file.json"
        )
        # Should work but without brand matching
        result = resolver.resolve("Prozac")
        assert result.method in ("fuzzy", "unknown")
