"""Tests for PharmHopfield — retrieval accuracy, capacity, and basic operations."""

import pytest
import torch

from pharmloop.hopfield import PharmHopfield, MAX_CAPACITY


class TestPharmHopfield:
    """Test suite for the Hopfield memory bank."""

    def test_init(self) -> None:
        """Module initializes with correct dimensions and zero stored patterns."""
        hopfield = PharmHopfield(input_dim=64, hidden_dim=512)
        assert hopfield.count == 0
        assert hopfield.input_dim == 64
        assert hopfield.hidden_dim == 512

    def test_store_and_count(self) -> None:
        """Storing patterns increments the count correctly."""
        hopfield = PharmHopfield(input_dim=64, hidden_dim=256)
        patterns = torch.randn(10, 64)
        hopfield.store(patterns)
        assert hopfield.count == 10

        # Store more
        more = torch.randn(5, 64)
        hopfield.store(more)
        assert hopfield.count == 15

    def test_store_shape_check(self) -> None:
        """Storing patterns with wrong shape raises AssertionError."""
        hopfield = PharmHopfield(input_dim=64)
        with pytest.raises(AssertionError):
            hopfield.store(torch.randn(10, 32))  # wrong dim

    def test_store_capacity_overflow(self) -> None:
        """Exceeding max capacity raises RuntimeError."""
        hopfield = PharmHopfield(input_dim=8, max_capacity=20)
        hopfield.store(torch.randn(15, 8))
        with pytest.raises(RuntimeError):
            hopfield.store(torch.randn(10, 8))  # would exceed 20

    def test_retrieve_empty(self) -> None:
        """Retrieving from empty memory returns zeros."""
        hopfield = PharmHopfield(input_dim=64)
        query = torch.randn(3, 64)
        result = hopfield.retrieve(query, beta=1.0)
        assert result.shape == (3, 64)
        assert (result == 0).all()

    def test_retrieve_shape(self) -> None:
        """Retrieved patterns have correct shape."""
        hopfield = PharmHopfield(input_dim=64, hidden_dim=128)
        hopfield.store(torch.randn(20, 64))
        query = torch.randn(5, 64)
        result = hopfield.retrieve(query, beta=1.0)
        assert result.shape == (5, 64)

    def test_retrieve_high_beta_is_sharper(self) -> None:
        """High beta produces sharper (less blended) retrieval than low beta."""
        hopfield = PharmHopfield(input_dim=8, hidden_dim=32)
        p1 = torch.ones(1, 8) * 1.0
        p2 = torch.ones(1, 8) * -1.0
        hopfield.store(torch.cat([p1, p2], dim=0))

        query = torch.randn(1, 8)
        result_sharp = hopfield.retrieve(query, beta=100.0)
        result_blended = hopfield.retrieve(query, beta=0.001)

        # High beta should produce a result closer to one of the stored patterns
        # (less blended), while low beta should be closer to the mean (≈0)
        dist_sharp_to_mean = result_sharp.norm()
        dist_blended_to_mean = result_blended.norm()
        assert dist_sharp_to_mean > dist_blended_to_mean, (
            "High beta should produce sharper (further from mean) retrieval"
        )

    def test_retrieve_low_beta_blended(self) -> None:
        """With very low beta, retrieval blends all patterns."""
        hopfield = PharmHopfield(input_dim=8, hidden_dim=32)
        p1 = torch.ones(1, 8)
        p2 = -torch.ones(1, 8)
        hopfield.store(torch.cat([p1, p2], dim=0))

        query = torch.randn(1, 8)
        result = hopfield.retrieve(query, beta=0.001)
        # With near-uniform weights, result should be close to mean of patterns (≈0)
        assert result.abs().mean() < 1.0  # not exactly zero but blended

    def test_clear(self) -> None:
        """Clearing memory resets count to zero."""
        hopfield = PharmHopfield(input_dim=8)
        hopfield.store(torch.randn(10, 8))
        assert hopfield.count == 10
        hopfield.clear()
        assert hopfield.count == 0

    def test_works_in_512dim(self) -> None:
        """Hopfield works in 512-dim space (for Phase 2)."""
        hopfield = PharmHopfield(input_dim=512, hidden_dim=512)
        hopfield.store(torch.randn(50, 512))
        query = torch.randn(4, 512)
        result = hopfield.retrieve(query, beta=1.0)
        assert result.shape == (4, 512)

    def test_gradients_flow_through_query_proj(self) -> None:
        """Gradients flow through the query projection during retrieval."""
        hopfield = PharmHopfield(input_dim=16, hidden_dim=32)
        hopfield.store(torch.randn(5, 16))
        query = torch.randn(2, 16, requires_grad=True)
        result = hopfield.retrieve(query, beta=1.0)
        loss = result.sum()
        loss.backward()
        assert query.grad is not None
        assert hopfield.query_proj.weight.grad is not None
