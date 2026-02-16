"""
Modern continuous Hopfield network for drug interaction pattern storage and retrieval.

Phase 0: initialized from raw 64-dim pharmacological feature vectors (no learned projections).
Phase 2: rebuilt with learned projections in 512-dim space.

Stored patterns are nn.Buffers — they don't receive gradients.
They are a database of verified interactions, not trainable weights.
"""

import torch
import torch.nn as nn
from torch import Tensor


# Maximum number of patterns the memory bank can hold
MAX_CAPACITY = 5000


class PharmHopfield(nn.Module):
    """
    Modern continuous Hopfield network for drug interaction patterns.

    Uses exponential energy function for sharp pattern retrieval.
    Query and key projections are learned; stored patterns are buffers.

    Args:
        input_dim: Dimension of input patterns (64 for Phase 0, 512 for Phase 2).
        hidden_dim: Internal projection dimension for queries/keys.
        max_capacity: Maximum number of stored patterns.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 512,
                 max_capacity: int = MAX_CAPACITY, phase0: bool = False) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim if not phase0 else input_dim
        self.max_capacity = max_capacity
        self.phase0 = phase0

        if phase0:
            # Phase 0: no learned projections, operate in raw feature space
            self.query_proj = nn.Identity()
            self.key_proj = nn.Identity()
            # Stored keys are in input_dim space
            self.register_buffer("stored_keys", torch.zeros(max_capacity, input_dim))
        else:
            # Phase 2: learned projections in hidden_dim space
            self.query_proj = nn.Linear(input_dim, self.hidden_dim)
            self.key_proj = nn.Linear(input_dim, self.hidden_dim)
            self.register_buffer("stored_keys", torch.zeros(max_capacity, self.hidden_dim))

        # Stored values always in input_dim space
        self.register_buffer("stored_values", torch.zeros(max_capacity, input_dim))
        self.register_buffer("num_stored", torch.tensor(0, dtype=torch.long))

    @property
    def count(self) -> int:
        """Number of patterns currently stored."""
        return self.num_stored.item()

    def store(self, patterns: Tensor) -> None:
        """
        Add patterns to the memory bank.

        Args:
            patterns: Tensor of shape (N, input_dim) — patterns to store.

        Raises:
            RuntimeError: If adding patterns would exceed max_capacity.
        """
        assert patterns.dim() == 2 and patterns.shape[1] == self.input_dim, (
            f"Expected patterns of shape (N, {self.input_dim}), got {patterns.shape}"
        )

        n = patterns.shape[0]
        current = self.count
        if current + n > self.max_capacity:
            raise RuntimeError(
                f"Cannot store {n} patterns: {current} stored, capacity {self.max_capacity}"
            )

        with torch.no_grad():
            # Project patterns to key space and store
            # In phase0 mode, key_proj is Identity so keys == patterns
            keys = self.key_proj(patterns)
            self.stored_keys[current:current + n] = keys.detach()
            self.stored_values[current:current + n] = patterns.detach()
            self.num_stored.fill_(current + n)

    def retrieve(self, query: Tensor, beta: float = 1.0) -> Tensor:
        """
        Retrieve from memory using softmax attention.

        retrieval = softmax(beta * query @ keys.T) @ values

        Args:
            query: Tensor of shape (batch, input_dim).
            beta: Inverse temperature controlling retrieval sharpness.
                  High beta → nearest neighbor, low beta → blended retrieval.

        Returns:
            Retrieved patterns of shape (batch, input_dim).
        """
        assert query.dim() == 2 and query.shape[1] == self.input_dim, (
            f"Expected query of shape (batch, {self.input_dim}), got {query.shape}"
        )

        n = self.count
        if n == 0:
            # No stored patterns — return zeros
            return torch.zeros_like(query)

        # Project query
        q = self.query_proj(query)  # (batch, hidden_dim)

        # Active stored keys and values
        keys = self.stored_keys[:n]    # (n, hidden_dim)
        values = self.stored_values[:n]  # (n, input_dim)

        # Attention scores with inverse temperature
        scores = beta * (q @ keys.T)  # (batch, n)
        weights = torch.softmax(scores, dim=-1)  # (batch, n)

        # Weighted retrieval
        retrieved = weights @ values  # (batch, input_dim)
        return retrieved

    def clear(self) -> None:
        """Remove all stored patterns."""
        self.stored_keys.zero_()
        self.stored_values.zero_()
        self.num_stored.fill_(0)
