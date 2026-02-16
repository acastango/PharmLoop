"""
Oscillatory reasoning core — the heart of PharmLoop.

State = (position x, velocity v) in 512-dim space.
  - x = current belief about the drug pair interaction
  - v = rate/direction of belief change
  - Gray zone = |v| — this IS the uncertainty (not a side computation)

Update rule (damped driven oscillator):
  force = spring * evidence_transform(cat(x, hopfield_retrieved))
  noise = randn * noise_gate(|v|) * 0.1
  v(t+1) = clamp(decay, 0.5, 0.99) * v(t) + force + noise
  x(t+1) = x(t) + clamp(dt, 0.01, 0.5) * v(t+1)

Convergence: |v| drops below learned threshold → stop, output answer.
Non-convergence: max_steps reached with |v| still high → output UNKNOWN.
"""

import torch
import torch.nn as nn
from torch import Tensor

from pharmloop.hopfield import PharmHopfield


# Oscillator constants
STATE_DIM = 512
MAX_STEPS = 16
NOISE_SCALE = 0.1


class OscillatorCell(nn.Module):
    """
    Single step of the damped driven oscillator.

    Takes current (x, v) and Hopfield memory, produces next (x, v).
    The force term transforms concatenated [x, hopfield_retrieved] through
    a learned network, producing evidence that drives the oscillation.

    Args:
        state_dim: Dimension of position/velocity state (default 512).
        hopfield: PharmHopfield memory module for evidence retrieval.
    """

    def __init__(self, state_dim: int = STATE_DIM, hopfield: PharmHopfield | None = None) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.hopfield = hopfield

        # Projection for querying Hopfield when its input_dim != state_dim
        # Phase 0: Hopfield is 64-dim, state is 512-dim, so we need a projection
        # Phase 2: Hopfield is rebuilt in 512-dim, so this becomes identity-like
        if hopfield is not None and hopfield.input_dim != state_dim:
            self.hopfield_query_proj = nn.Linear(state_dim, hopfield.input_dim)
            self.hopfield_value_proj = nn.Linear(hopfield.input_dim, state_dim)
        else:
            self.hopfield_query_proj = None
            self.hopfield_value_proj = None

        # Learned oscillator parameters — per-dimension so different aspects
        # of the belief state can settle at different rates
        self.raw_decay = nn.Parameter(torch.ones(state_dim) * 0.9)      # damping coefficient (clamped to [0.5, 0.99])
        self.raw_dt = nn.Parameter(torch.ones(state_dim) * 0.1)          # time step (clamped to [0.01, 0.5])
        self.raw_spring = nn.Parameter(torch.ones(state_dim) * 0.5)      # spring constant (positive)
        self.raw_threshold = nn.Parameter(torch.tensor(0.15))             # convergence threshold for |v| (scalar — operates on norm)

        # Evidence transform: cat(x, hopfield_retrieved) → force
        # Input is 2 * state_dim because we concatenate x and retrieved
        self.evidence_transform = nn.Sequential(
            nn.Linear(state_dim * 2, state_dim),
            nn.Tanh(),
            nn.Linear(state_dim, state_dim),
        )

        # Noise gate: per-dimension |v| → per-dimension noise scale
        # High uncertainty dimensions get more exploration, settled dimensions stay stable
        self.noise_gate = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.GELU(),
            nn.Linear(128, state_dim),
            nn.Sigmoid(),
        )

        # Hopfield beta modulation: gray zone → beta (high uncertainty → lower beta → broader retrieval)
        self.beta_mod = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus(),  # beta must be positive
        )

    @property
    def decay(self) -> Tensor:
        """Damping coefficient, clamped to [0.5, 0.99]."""
        return self.raw_decay.clamp(0.5, 0.99)

    @property
    def dt(self) -> Tensor:
        """Time step, clamped to [0.01, 0.5]."""
        return self.raw_dt.clamp(0.01, 0.5)

    @property
    def spring(self) -> Tensor:
        """Spring constant, kept positive."""
        return self.raw_spring.abs() + 0.01

    @property
    def threshold(self) -> Tensor:
        """Convergence threshold for |v|, kept positive."""
        return self.raw_threshold.abs() + 1e-4

    def forward(
        self,
        x: Tensor,
        v: Tensor,
        training: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        One step of the damped driven oscillator.

        Args:
            x: Position (belief) tensor of shape (batch, state_dim).
            v: Velocity tensor of shape (batch, state_dim).
            training: Whether to inject noise.

        Returns:
            Tuple of (new_x, new_v, gz_per_sample) where gz_per_sample = |v_new| per sample (batch,).
        """
        batch = x.shape[0]
        assert x.shape == (batch, self.state_dim)
        assert v.shape == (batch, self.state_dim)

        # Compute gray zone: per-dimension |v| for noise gating, scalar for beta_mod
        gz_per_dim = torch.abs(v)                              # (batch, state_dim)
        gz_scalar = v.norm(dim=-1, keepdim=True)               # (batch, 1) — for beta_mod

        # Hopfield retrieval with beta modulated by gray zone
        if self.hopfield is not None and self.hopfield.count > 0:
            beta = self.beta_mod(gz_scalar).mean()  # scalar beta
            # Project state to Hopfield's input space if dimensions differ
            if self.hopfield_query_proj is not None:
                query = self.hopfield_query_proj(x)  # (batch, hopfield.input_dim)
            else:
                query = x
            retrieved = self.hopfield.retrieve(query, beta=beta.item())  # (batch, hopfield.input_dim)
            # Project retrieved patterns back to state space if needed
            if self.hopfield_value_proj is not None:
                retrieved = self.hopfield_value_proj(retrieved)  # (batch, state_dim)

        else:
            retrieved = torch.zeros_like(x)

        # Evidence transform
        evidence_input = torch.cat([x, retrieved], dim=-1)  # (batch, state_dim * 2)
        force = self.spring * self.evidence_transform(evidence_input)  # (batch, state_dim)

        # Noise (only during training, gated by per-dimension gray zone)
        if training:
            noise_scale = self.noise_gate(gz_per_dim)  # (batch, state_dim)
            noise = torch.randn_like(v) * noise_scale * NOISE_SCALE
        else:
            noise = torch.zeros_like(v)

        # Damped oscillator update
        v_new = self.decay * v + force + noise
        x_new = x + self.dt * v_new

        gz_per_sample = v_new.norm(dim=-1)  # (batch,) — per-sample gray zone of NEW velocity
        return x_new, v_new, gz_per_sample


class ReasoningLoop(nn.Module):
    """
    Runs the OscillatorCell until convergence or max_steps.

    Tracks full trajectory of positions, velocities, and gray zones for
    loss computation (smoothness penalty) and visualization.

    Args:
        cell: The OscillatorCell to iterate.
        max_steps: Maximum number of oscillation steps before declaring UNKNOWN.
    """

    def __init__(self, cell: OscillatorCell, max_steps: int = MAX_STEPS) -> None:
        super().__init__()
        self.cell = cell
        self.max_steps = max_steps

        # Learned initial velocity projection: gives the oscillator a non-zero kick
        # so there is something to damp (known pairs) or fail to damp (unknown)
        self.initial_v_proj = nn.Linear(cell.state_dim, cell.state_dim)

    def forward(
        self,
        initial_state: Tensor,
        training: bool = True,
    ) -> dict[str, Tensor | list[Tensor] | int]:
        """
        Run the oscillator from initial_state until convergence or max_steps.

        Args:
            initial_state: Tensor of shape (batch, state_dim) — initial position.
                           Initial velocity is zero (no prior momentum).

        Returns:
            Dictionary with:
              - "final_x": Final position tensor (batch, state_dim).
              - "final_v": Final velocity tensor (batch, state_dim).
              - "positions": List of position tensors at each step.
              - "velocities": List of velocity tensors at each step.
              - "gray_zones": List of per-sample gray zone tensors (batch,) at each step.
              - "steps": Number of steps taken.
              - "converged": Whether oscillator converged (bool tensor per sample).
        """
        batch = initial_state.shape[0]
        state_dim = initial_state.shape[1]

        x = initial_state
        # Non-zero initial velocity: gives the oscillator a perturbation to damp.
        # Known pairs should damp this quickly; unknown pairs will fail to damp.
        v = self.initial_v_proj(initial_state)

        positions: list[Tensor] = [x]
        velocities: list[Tensor] = [v]
        gray_zones: list[Tensor] = [v.norm(dim=-1)]  # (batch,) tensor

        steps = 0
        # During training, always run all steps (for consistent gradient computation)
        # During eval, can stop early on convergence
        for step in range(self.max_steps):
            x, v, gz = self.cell(x, v, training=training)
            positions.append(x)
            velocities.append(v)
            gray_zones.append(gz)  # gz is (batch,) tensor
            steps = step + 1

            # Early stopping only during inference, and only after a minimum number of steps
            # to give the oscillator time to settle (not just trivially pass threshold)
            min_eval_steps = 3
            if not training and steps >= min_eval_steps:
                if gz.mean().item() < self.cell.threshold.item():
                    break

        # Determine convergence per sample
        final_gz_per_sample = v.norm(dim=-1)  # (batch,)
        converged = final_gz_per_sample < self.cell.threshold.detach()  # (batch,)

        return {
            "final_x": x,
            "final_v": v,
            "positions": positions,
            "velocities": velocities,
            "gray_zones": gray_zones,
            "steps": steps,
            "converged": converged,
        }
