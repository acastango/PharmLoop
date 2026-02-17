"""
Two-level Hierarchical Hopfield memory for scaled drug interaction retrieval.

Class-specific banks: sharp retrieval within drug categories (~100-200
patterns each). Global bank: fallback for cross-class or novel pairs
(all patterns).

The interface is compatible with PharmHopfield — the OscillatorCell
doesn't need to know whether it's talking to a flat or hierarchical bank.
"""

import torch
import torch.nn as nn
from torch import Tensor

from pharmloop.hopfield import PharmHopfield

# Drug class taxonomy — ~12 classes covering major pharmacological categories
DRUG_CLASSES = [
    "ssri_snri",
    "opioid",
    "anticoagulant",
    "antihypertensive",
    "statin_lipid",
    "antidiabetic",
    "antibiotic",
    "antiepileptic",
    "immunosuppressant",
    "cardiac",
    "cns_psych",
    "nsaid_analgesic",
]


class HierarchicalHopfield(nn.Module):
    """
    Two-level Hopfield memory for scaled drug interaction retrieval.

    Class-specific banks keep retrieval sharp within drug categories.
    Global bank catches cross-class interactions.

    The interface matches PharmHopfield for backward compatibility with
    the OscillatorCell — it has input_dim, count, store(), retrieve(),
    and clear().

    Args:
        input_dim: Dimension of stored patterns (512 for Phase 2+).
        class_names: List of drug class names for class-specific banks.
        class_capacity: Max patterns per class bank.
        global_capacity: Max patterns in the global bank.
    """

    def __init__(
        self,
        input_dim: int,
        class_names: list[str] | None = None,
        class_capacity: int = 500,
        global_capacity: int = 5000,
    ) -> None:
        super().__init__()
        self._input_dim = input_dim

        if class_names is None:
            class_names = DRUG_CLASSES

        # Class-specific banks — sharp retrieval within drug categories
        self.class_banks = nn.ModuleDict({
            name: PharmHopfield(input_dim, input_dim, max_capacity=class_capacity, phase0=False)
            for name in class_names
        })

        # Global bank — catches cross-class interactions and serves as fallback
        self.global_bank = PharmHopfield(
            input_dim, input_dim, max_capacity=global_capacity, phase0=False,
        )

        # Learned gating: how much to trust class-specific vs global retrieval
        self.combine_gate = nn.Sequential(
            nn.Linear(input_dim * 2, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Temporary storage for drug classes (set by PharmLoopModel.forward).
        # Can be a single tuple for uniform batches (backward compat) or
        # a list of tuples for per-item routing in mixed batches.
        self._current_classes: tuple[str, str] | list[tuple[str, str]] | None = None

    @property
    def input_dim(self) -> int:
        """Input dimension — matches PharmHopfield interface."""
        return self._input_dim

    @property
    def count(self) -> int:
        """Total patterns in global bank — matches PharmHopfield interface."""
        return self.global_bank.count

    def store(
        self,
        patterns: Tensor,
        drug_classes: list[tuple[str, str]] | None = None,
    ) -> None:
        """
        Store patterns in global bank AND relevant class banks.

        Args:
            patterns: (N, input_dim) pair patterns to store.
            drug_classes: Optional list of (class_a, class_b) for each pattern.
                If provided, patterns are also stored in class-specific banks.
        """
        # Always store in global
        self.global_bank.store(patterns)

        # Also store in class-specific banks
        if drug_classes is not None:
            for i, (cls_a, cls_b) in enumerate(drug_classes):
                pattern = patterns[i:i + 1]
                for cls in set([cls_a, cls_b]):
                    if cls in self.class_banks:
                        bank = self.class_banks[cls]
                        if bank.count + 1 <= bank.max_capacity:
                            bank.store(pattern)

    def retrieve(
        self,
        query: Tensor,
        beta: float = 1.0,
        drug_classes: tuple[str, str] | list[tuple[str, str]] | None = None,
    ) -> Tensor:
        """
        Hierarchical retrieval: class-specific + global, gated.

        Supports per-item class routing for mixed batches. If drug_classes
        is a list of (class_a, class_b) tuples (one per batch item), items
        are grouped by their class pair and each group gets its own
        class-specific retrieval. If a single tuple, it's applied uniformly.

        Falls back to _current_classes if not provided, then to global-only.

        Args:
            query: (batch, input_dim) query tensor.
            beta: Inverse temperature for retrieval sharpness.
            drug_classes: Single (class_a, class_b) tuple, list of per-item
                tuples, or None.

        Returns:
            (batch, input_dim) retrieved patterns.
        """
        # Use stored classes if not explicitly provided
        if drug_classes is None:
            drug_classes = self._current_classes

        # Global retrieval (always)
        global_retrieved = self.global_bank.retrieve(query, beta=beta)

        if drug_classes is None:
            return global_retrieved

        # Normalize: single tuple → uniform for all items
        if isinstance(drug_classes, tuple):
            return self._retrieve_for_class_pair(
                query, global_retrieved, drug_classes, beta,
            )

        # Per-item routing: group items by class pair, retrieve per group
        batch_size = query.shape[0]
        assert len(drug_classes) == batch_size

        # Group by normalized class pair (sorted so (A,B) == (B,A))
        groups: dict[tuple[str, str], list[int]] = {}
        for i, (cls_a, cls_b) in enumerate(drug_classes):
            key = (min(cls_a, cls_b), max(cls_a, cls_b))
            groups.setdefault(key, []).append(i)

        result = global_retrieved.clone()
        for cls_pair, indices in groups.items():
            idx = torch.tensor(indices, device=query.device, dtype=torch.long)
            group_result = self._retrieve_for_class_pair(
                query[idx], global_retrieved[idx], cls_pair, beta,
            )
            result[idx] = group_result

        return result

    def _retrieve_for_class_pair(
        self,
        query: Tensor,
        global_retrieved: Tensor,
        drug_classes: tuple[str, str],
        beta: float,
    ) -> Tensor:
        """
        Class-specific retrieval for a single (class_a, class_b) pair.

        Args:
            query: (N, input_dim) query tensor for this group.
            global_retrieved: (N, input_dim) pre-computed global retrieval.
            drug_classes: (class_a, class_b) tuple.
            beta: Inverse temperature.

        Returns:
            (N, input_dim) gated blend of class-specific and global retrieval.
        """
        class_a, class_b = drug_classes
        class_retrievals = []
        for cls in set([class_a, class_b]):
            if cls in self.class_banks and self.class_banks[cls].count > 0:
                class_retrievals.append(
                    self.class_banks[cls].retrieve(query, beta=beta)
                )

        if not class_retrievals:
            return global_retrieved

        class_retrieved = torch.stack(class_retrievals).mean(dim=0)

        # Learned gating: blend class-specific and global
        combined = torch.cat([class_retrieved, global_retrieved], dim=-1)
        gate = self.combine_gate(combined)  # (N, 1)
        return gate * class_retrieved + (1 - gate) * global_retrieved

    def clear(self) -> None:
        """Remove all stored patterns from all banks."""
        self.global_bank.clear()
        for bank in self.class_banks.values():
            bank.clear()
        self._current_classes = None
