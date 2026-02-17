"""
Drug name resolution: brand names, normalization, and fuzzy matching.

Pharmacists often use brand names (Prozac, Lipitor) rather than generics.
This module resolves drug names through three layers:

  1. Exact match against the drug registry (case-insensitive)
  2. Brand → generic mapping (loaded from brand_names.json)
  3. Fuzzy match for typos (difflib with 0.85 cutoff)

Usage:
    resolver = DrugResolver(drug_registry, brand_map)
    resolved = resolver.resolve("Prozac")  # → "fluoxetine"
    resolved = resolver.resolve("fluoxetin")  # → "fluoxetine" (fuzzy)
    resolved = resolver.resolve("xyz123")  # → None (unknown)
"""

import difflib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ResolvedDrug:
    """Result of drug name resolution."""
    original: str
    resolved: str | None
    method: str  # "exact", "brand", "fuzzy", "unknown"
    confidence: float  # 1.0 for exact/brand, <1.0 for fuzzy


class DrugResolver:
    """
    Resolves drug names to canonical generic names in the registry.

    Applies three resolution strategies in order:
      1. Exact match (case-insensitive, underscore-normalized)
      2. Brand name lookup
      3. Fuzzy matching with configurable cutoff

    Args:
        drug_registry: Set or dict of known generic drug names.
        brand_map: Dict of brand_name -> generic_name.
        fuzzy_cutoff: Minimum similarity ratio for fuzzy matching (default 0.85).
    """

    def __init__(
        self,
        drug_registry: set[str] | dict[str, object],
        brand_map: dict[str, str] | None = None,
        fuzzy_cutoff: float = 0.85,
    ) -> None:
        # Normalize registry to lowercase set
        if isinstance(drug_registry, dict):
            self._registry = {k.lower() for k in drug_registry.keys()}
        else:
            self._registry = {k.lower() for k in drug_registry}

        self._brand_map = brand_map or {}
        self._fuzzy_cutoff = fuzzy_cutoff

        # Pre-compute sorted registry list for difflib
        self._registry_list = sorted(self._registry)

    @classmethod
    def load(
        cls,
        drug_registry: set[str] | dict[str, object],
        brand_names_path: str | Path | None = None,
        fuzzy_cutoff: float = 0.85,
    ) -> "DrugResolver":
        """
        Create a DrugResolver, optionally loading brand names from JSON.

        Args:
            drug_registry: Known generic drug names.
            brand_names_path: Path to brand_names.json (optional).
            fuzzy_cutoff: Minimum similarity for fuzzy matching.

        Returns:
            Configured DrugResolver instance.
        """
        brand_map: dict[str, str] = {}
        if brand_names_path is not None:
            path = Path(brand_names_path)
            if path.exists():
                with open(path) as f:
                    raw = json.load(f)
                # Normalize keys to lowercase
                brand_map = {k.lower(): v.lower() for k, v in raw.items()}

        return cls(drug_registry, brand_map, fuzzy_cutoff)

    def resolve(self, name: str) -> ResolvedDrug:
        """
        Resolve a drug name to a canonical generic name.

        Tries exact match, then brand lookup, then fuzzy matching.

        Args:
            name: Drug name (brand or generic, any case).

        Returns:
            ResolvedDrug with resolution result and method used.
        """
        normalized = self._normalize(name)

        # 1. Exact match
        if normalized in self._registry:
            return ResolvedDrug(
                original=name, resolved=normalized,
                method="exact", confidence=1.0,
            )

        # 2. Brand name lookup
        generic = self._brand_map.get(normalized)
        if generic and generic in self._registry:
            return ResolvedDrug(
                original=name, resolved=generic,
                method="brand", confidence=1.0,
            )

        # 3. Fuzzy match
        close = difflib.get_close_matches(
            normalized, self._registry_list,
            n=1, cutoff=self._fuzzy_cutoff,
        )
        if close:
            return ResolvedDrug(
                original=name, resolved=close[0],
                method="fuzzy", confidence=self._fuzzy_cutoff,
            )

        # Unknown
        return ResolvedDrug(
            original=name, resolved=None,
            method="unknown", confidence=0.0,
        )

    def resolve_many(self, names: list[str]) -> list[ResolvedDrug]:
        """Resolve a list of drug names."""
        return [self.resolve(name) for name in names]

    def _normalize(self, name: str) -> str:
        """
        Normalize drug name for matching.

        Lowercases, strips whitespace, replaces common separators
        with underscores.
        """
        result = name.strip().lower()
        # Replace common separators with underscore
        for sep in ["/", "-", " "]:
            result = result.replace(sep, "_")
        # Remove trailing punctuation
        result = result.rstrip(".")
        return result
