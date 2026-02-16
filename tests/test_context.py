"""Tests for ContextEncoder — validates the mechanism works, not clinical accuracy."""

import torch
import pytest

from pharmloop.context import ContextEncoder, CONTEXT_DIM
from pharmloop.hopfield import PharmHopfield
from pharmloop.model import PharmLoopModel


class TestContextEncoder:

    def test_gate_starts_near_zero(self) -> None:
        """Gate output should be near-zero at initialization (backward compatible)."""
        ctx = ContextEncoder()
        dummy_pair = torch.randn(2, 512)
        dummy_context = torch.randn(2, CONTEXT_DIM)

        with torch.no_grad():
            gate_out = ctx.gate(dummy_context)

        # Sigmoid(-2.0) ≈ 0.12, so gate values should be small
        assert gate_out.mean().item() < 0.2, (
            f"Gate should start near-zero, got mean={gate_out.mean().item():.3f}"
        )

    def test_modulation_is_small_at_init(self) -> None:
        """Modulated pair_state ≈ original pair_state at initialization."""
        ctx = ContextEncoder()
        pair_state = torch.randn(2, 512)
        context = torch.randn(2, CONTEXT_DIM)

        with torch.no_grad():
            modulated = ctx(pair_state, context)

        diff = (modulated - pair_state).norm() / pair_state.norm()
        print(f"\nRelative modulation at init: {diff.item():.4f}")
        assert diff.item() < 0.5, f"Modulation too large at init: {diff.item():.3f}"

    def test_different_context_different_output(self) -> None:
        """Different context inputs should produce different modulations."""
        ctx = ContextEncoder()
        pair_state = torch.randn(1, 512)
        ctx_low = torch.zeros(1, CONTEXT_DIM)
        ctx_low[0, 4] = 0.2  # low dose B
        ctx_high = torch.zeros(1, CONTEXT_DIM)
        ctx_high[0, 4] = 0.9  # high dose B

        with torch.no_grad():
            out_low = ctx(pair_state, ctx_low)
            out_high = ctx(pair_state, ctx_high)

        diff = (out_low - out_high).norm().item()
        print(f"\nDiff between low-dose and high-dose context: {diff:.4f}")
        assert diff > 0, "Different contexts should produce different outputs"

    def test_zero_context_near_identity(self) -> None:
        """Zero context vector should produce near-identity modulation."""
        ctx = ContextEncoder()
        pair_state = torch.randn(2, 512)
        zero_ctx = torch.zeros(2, CONTEXT_DIM)

        with torch.no_grad():
            modulated = ctx(pair_state, zero_ctx)

        diff = (modulated - pair_state).norm() / pair_state.norm()
        print(f"\nZero-context relative diff: {diff.item():.4f}")
        # Even with zero input, the gate bias means some small modulation
        assert diff.item() < 0.5

    def test_gradients_flow(self) -> None:
        """Gradients should flow through the context encoder."""
        ctx = ContextEncoder()
        pair_state = torch.randn(2, 512, requires_grad=True)
        context = torch.randn(2, CONTEXT_DIM)

        modulated = ctx(pair_state, context)
        loss = modulated.sum()
        loss.backward()

        assert pair_state.grad is not None
        assert pair_state.grad.abs().sum() > 0


class TestContextInModel:

    def test_model_without_context_matches_phase2(self) -> None:
        """Model with use_context=False should not have a context_encoder."""
        hopfield = PharmHopfield(input_dim=512, hidden_dim=512, phase0=False)
        model = PharmLoopModel(num_drugs=50, hopfield=hopfield, use_context=False)
        assert model.context_encoder is None

    def test_model_with_context_has_encoder(self) -> None:
        """Model with use_context=True should have a context_encoder."""
        hopfield = PharmHopfield(input_dim=512, hidden_dim=512, phase0=False)
        model = PharmLoopModel(num_drugs=50, hopfield=hopfield, use_context=True)
        assert model.context_encoder is not None

    def test_forward_without_context_works(self) -> None:
        """Forward pass without context should work regardless of use_context."""
        hopfield = PharmHopfield(input_dim=512, hidden_dim=512, phase0=False)
        model = PharmLoopModel(num_drugs=50, hopfield=hopfield, use_context=True)
        model.eval()

        a_id = torch.tensor([0])
        a_feat = torch.randn(1, 64)
        b_id = torch.tensor([1])
        b_feat = torch.randn(1, 64)

        with torch.no_grad():
            output = model(a_id, a_feat, b_id, b_feat)  # no context

        assert "severity_logits" in output
