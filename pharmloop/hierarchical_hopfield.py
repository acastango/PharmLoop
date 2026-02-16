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

        # Temporary storage for drug classes (set by PharmLoopModel.forward)
        self._current_classes: tuple[str, str] | None = None

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
        drug_classes: tuple[str, str] | None = None,
    ) -> Tensor:
        """
        Hierarchical retrieval: class-specific + global, gated.

        If drug_classes not provided, checks _current_classes (set by
        PharmLoopModel), falls back to global-only.

        Args:
            query: (batch, input_dim) query tensor.
            beta: Inverse temperature for retrieval sharpness.
            drug_classes: Optional (class_a, class_b) tuple.

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

        # Class-specific retrieval
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
        gate = self.combine_gate(combined)  # (batch, 1)
        return gate * class_retrieved + (1 - gate) * global_retrieved

    def clear(self) -> None:
        """Remove all stored patterns from all banks."""
        self.global_bank.clear()
        for bank in self.class_banks.values():
            bank.clear()
        self._current_classes = None
