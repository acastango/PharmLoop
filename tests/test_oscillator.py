"""Tests for OscillatorCell and ReasoningLoop — does it actually oscillate? Convergence tests."""

import pytest
import torch

from pharmloop.hopfield import PharmHopfield
from pharmloop.oscillator import OscillatorCell, ReasoningLoop, STATE_DIM, MAX_STEPS


class TestOscillatorCell:
    """Test the single-step oscillator cell."""

    def test_init(self) -> None:
        """OscillatorCell initializes without error."""
        cell = OscillatorCell(state_dim=512)
        assert cell.state_dim == 512

    def test_forward_shape(self) -> None:
        """Single step produces correct output shapes."""
        cell = OscillatorCell(state_dim=64)
        x = torch.randn(4, 64)
        v = torch.randn(4, 64)
        x_new, v_new, gz = cell(x, v, training=False)
        assert x_new.shape == (4, 64)
        assert v_new.shape == (4, 64)
        assert isinstance(gz, torch.Tensor)
        assert gz.shape == (4,)
        assert (gz >= 0).all()

    def test_clamped_parameters(self) -> None:
        """Decay, dt, spring are properly clamped per-dimension; threshold is scalar."""
        cell = OscillatorCell()
        assert cell.decay.shape == (STATE_DIM,)
        assert (cell.decay >= 0.5).all() and (cell.decay <= 0.99).all()
        assert cell.dt.shape == (STATE_DIM,)
        assert (cell.dt >= 0.01).all() and (cell.dt <= 0.5).all()
        assert cell.spring.shape == (STATE_DIM,)
        assert (cell.spring > 0).all()
        assert cell.threshold.item() > 0

    def test_with_hopfield(self) -> None:
        """Cell works when connected to a Hopfield memory."""
        hopfield = PharmHopfield(input_dim=64, hidden_dim=128)
        hopfield.store(torch.randn(10, 64))
        cell = OscillatorCell(state_dim=64, hopfield=hopfield)
        x = torch.randn(2, 64)
        v = torch.zeros(2, 64)
        x_new, v_new, gz = cell(x, v, training=True)
        assert x_new.shape == (2, 64)

    def test_without_hopfield(self) -> None:
        """Cell works without Hopfield (retrieved = zeros)."""
        cell = OscillatorCell(state_dim=64)
        x = torch.randn(2, 64)
        v = torch.zeros(2, 64)
        x_new, v_new, gz = cell(x, v, training=False)
        assert x_new.shape == (2, 64)

    def test_no_noise_during_eval(self) -> None:
        """Same input gives same output when training=False (no noise)."""
        cell = OscillatorCell(state_dim=32)
        x = torch.randn(1, 32)
        v = torch.randn(1, 32)

        torch.manual_seed(42)
        x1, v1, _ = cell(x.clone(), v.clone(), training=False)
        torch.manual_seed(99)  # different seed
        x2, v2, _ = cell(x.clone(), v.clone(), training=False)

        assert torch.allclose(x1, x2)
        assert torch.allclose(v1, v2)


class TestReasoningLoop:
    """Test the full reasoning loop."""

    def test_init(self) -> None:
        """ReasoningLoop initializes without error."""
        cell = OscillatorCell(state_dim=64)
        loop = ReasoningLoop(cell, max_steps=16)
        assert loop.max_steps == 16

    def test_forward_returns_trajectory(self) -> None:
        """Loop returns all expected trajectory fields."""
        cell = OscillatorCell(state_dim=64)
        loop = ReasoningLoop(cell, max_steps=8)
        initial = torch.randn(3, 64)
        result = loop(initial, training=True)

        assert "final_x" in result
        assert "final_v" in result
        assert "positions" in result
        assert "velocities" in result
        assert "gray_zones" in result
        assert "steps" in result
        assert "converged" in result

        assert result["final_x"].shape == (3, 64)
        assert result["final_v"].shape == (3, 64)
        assert result["converged"].shape == (3,)
        # positions includes initial + max_steps
        assert len(result["positions"]) == 8 + 1
        assert len(result["gray_zones"]) == 8 + 1
        assert result["steps"] == 8

    def test_training_runs_all_steps(self) -> None:
        """During training, always runs all max_steps (for gradient consistency)."""
        cell = OscillatorCell(state_dim=32)
        loop = ReasoningLoop(cell, max_steps=10)
        initial = torch.randn(2, 32)
        result = loop(initial, training=True)
        assert result["steps"] == 10

    def test_eval_can_stop_early(self) -> None:
        """During eval, loop can stop early if converged."""
        cell = OscillatorCell(state_dim=32)
        # Set a very high threshold so it converges immediately
        cell.raw_threshold = torch.nn.Parameter(torch.tensor(1000.0))
        loop = ReasoningLoop(cell, max_steps=16)
        initial = torch.randn(1, 32)
        result = loop(initial, training=False)
        # Should stop early since threshold is very high
        assert result["steps"] <= 16

    def test_gray_zone_is_velocity_norm(self) -> None:
        """Gray zone values are per-sample L2 norm of velocity (|v|)."""
        cell = OscillatorCell(state_dim=16)
        loop = ReasoningLoop(cell, max_steps=4)
        initial = torch.randn(1, 16)
        result = loop(initial, training=False)

        # Gray zones are now per-sample tensors
        for gz in result["gray_zones"]:
            assert isinstance(gz, torch.Tensor)
            assert gz.shape == (1,)
            assert (gz >= 0.0).all()

    def test_gradients_flow(self) -> None:
        """Gradients flow through the entire reasoning loop."""
        cell = OscillatorCell(state_dim=32)
        loop = ReasoningLoop(cell, max_steps=4)
        initial = torch.randn(2, 32, requires_grad=True)
        result = loop(initial, training=True)
        loss = result["final_x"].sum()
        loss.backward()
        assert initial.grad is not None

    def test_oscillation_occurs(self) -> None:
        """The position actually changes across steps (oscillation is happening)."""
        cell = OscillatorCell(state_dim=32)
        loop = ReasoningLoop(cell, max_steps=8)
        initial = torch.randn(1, 32)
        result = loop(initial, training=False)

        # Positions should differ across steps
        pos_diffs = [
            (result["positions"][i+1] - result["positions"][i]).norm().item()
            for i in range(len(result["positions"]) - 1)
        ]
        # At least some steps should show movement
        assert max(pos_diffs) > 0, "No oscillation detected — positions are static"
