"""
Output head: maps converged oscillator state to structured predictions.

Severity: 6 classes (none / mild / moderate / severe / contraindicated / unknown)
Mechanism: multi-label (serotonergic, CYP inhibition, QT prolongation, etc.)
Flags: binary clinical monitoring flags

Confidence is NOT predicted by this head. It comes from convergence dynamics:
  confidence = f(final_gray_zone, steps_to_converge, trajectory_smoothness)
This is a formula, not a neural network.
"""

import torch
import torch.nn as nn
from torch import Tensor


# Severity class indices
SEVERITY_NONE = 0
SEVERITY_MILD = 1
SEVERITY_MODERATE = 2
SEVERITY_SEVERE = 3
SEVERITY_CONTRAINDICATED = 4
SEVERITY_UNKNOWN = 5
NUM_SEVERITY_CLASSES = 6

SEVERITY_NAMES = ["none", "mild", "moderate", "severe", "contraindicated", "unknown"]

# Mechanism vocabulary
MECHANISM_NAMES = [
    "serotonergic",
    "cyp_inhibition",
    "cyp_induction",
    "qt_prolongation",
    "bleeding_risk",
    "cns_depression",
    "nephrotoxicity",
    "hepatotoxicity",
    "hypotension",
    "hyperkalemia",
    "seizure_risk",
    "immunosuppression",
    "absorption_altered",
    "protein_binding_displacement",
    "electrolyte_imbalance",
]
NUM_MECHANISMS = len(MECHANISM_NAMES)

# Clinical flag vocabulary
FLAG_NAMES = [
    "monitor_serotonin_syndrome",
    "monitor_inr",
    "monitor_qt_interval",
    "monitor_renal_function",
    "monitor_hepatic_function",
    "monitor_blood_pressure",
    "monitor_blood_glucose",
    "monitor_electrolytes",
    "monitor_drug_levels",
    "monitor_cns_depression",
    # --- previously missing ---
    "avoid_combination",
    "monitor_bleeding",
    "monitor_digoxin_levels",
    "monitor_lithium_levels",
    "monitor_cyclosporine_levels",
    "monitor_theophylline_levels",
    "reduce_statin_dose",
    "separate_administration",
]
NUM_FLAGS = len(FLAG_NAMES)  # now 18

# State dimension from the oscillator
STATE_DIM = 512


class TrajectoryMechanismHead(nn.Module):
    """
    Reads mechanism signal from the oscillation trajectory, not just
    the final position.

    Takes the last K positions from the trajectory, attention-pools them,
    and runs through a small MLP. This captures mechanism information that
    emerges during oscillation but may not survive to the final position.

    Zero additional oscillator parameters — this only adds readout capacity.
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        num_mechanisms: int = NUM_MECHANISMS,
        trajectory_window: int = 4,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.trajectory_window = trajectory_window

        # Attention pool over trajectory window
        self.step_attention = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # MLP for mechanism classification
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_mechanisms),
        )

    def forward(self, positions: list[Tensor]) -> Tensor:
        """
        Classify mechanisms from trajectory positions.

        Args:
            positions: List of (batch, state_dim) tensors from trajectory.

        Returns:
            (batch, num_mechanisms) mechanism logits.
        """
        window = positions[-self.trajectory_window:]
        stacked = torch.stack(window, dim=1)  # (batch, K, state_dim)

        attn_scores = self.step_attention(stacked).squeeze(-1)  # (batch, K)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (batch, K)
        pooled = (stacked * attn_weights.unsqueeze(-1)).sum(dim=1)  # (batch, state_dim)

        return self.mlp(pooled)


class OutputHead(nn.Module):
    """
    Maps converged oscillator state to structured drug interaction predictions.

    Does NOT predict confidence — that comes from convergence dynamics.

    Args:
        state_dim: Dimension of the oscillator state (default 512).
        num_mechanisms: Number of mechanism labels (default 15).
        num_flags: Number of clinical flag labels (default 18).
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        num_mechanisms: int = NUM_MECHANISMS,
        num_flags: int = NUM_FLAGS,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim

        # Severity: 6-class classification
        self.severity_head = nn.Linear(state_dim, NUM_SEVERITY_CLASSES)

        # Mechanism: trajectory-aware multi-label classification
        self.mechanism_head = TrajectoryMechanismHead(state_dim, num_mechanisms)

        # Clinical flags: binary classification
        self.flags_head = nn.Linear(state_dim, num_flags)

    def forward(
        self,
        state: Tensor,
        positions: list[Tensor] | None = None,
    ) -> dict[str, Tensor]:
        """
        Produce structured predictions from oscillator state.

        Args:
            state: (batch, state_dim) — converged oscillator position.
            positions: List of (batch, state_dim) trajectory positions.
                       If None, falls back to using [state] for mechanism head.

        Returns:
            Dictionary with severity_logits, mechanism_logits, flag_logits.
        """
        batch = state.shape[0]
        assert state.shape == (batch, self.state_dim), (
            f"Expected state shape ({batch}, {self.state_dim}), got {state.shape}"
        )

        severity_logits = self.severity_head(state)
        flag_logits = self.flags_head(state)

        if positions is not None:
            mechanism_logits = self.mechanism_head(positions)
        else:
            mechanism_logits = self.mechanism_head([state])

        return {
            "severity_logits": severity_logits,
            "mechanism_logits": mechanism_logits,
            "flag_logits": flag_logits,
        }


def compute_confidence(
    gray_zones: list[Tensor],
    converged: Tensor,
    max_steps: int = 16,
) -> Tensor:
    """
    Compute per-sample confidence from convergence dynamics (a formula, not a neural network).

    confidence = f(final_gray_zone, steps_to_converge, trajectory_smoothness)

    Components:
      - Final gray zone: lower |v| at end → higher confidence
      - Speed: converging in fewer steps → higher confidence
      - Smoothness: smooth decay of |v| → higher confidence (vs chaotic oscillation)

    Args:
        gray_zones: List of per-sample gray zone tensors (batch,) at each step.
        converged: Boolean tensor of shape (batch,) — whether each sample converged.
        max_steps: Maximum number of oscillator steps.

    Returns:
        Tensor of shape (batch,) with confidence scores in [0, 1].
    """
    batch = converged.shape[0]
    device = converged.device

    # Final gray zone component: lower → more confident (per-sample)
    final_gz = gray_zones[-1]  # (batch,)
    gz_confidence = (1.0 - final_gz * 5.0).clamp(min=0.0)

    # Speed component: fewer steps → more confident (same for all samples in batch)
    steps_taken = len(gray_zones) - 1  # subtract initial state
    speed_confidence = max(0.0, 1.0 - steps_taken / max_steps)
    speed_confidence = torch.full((batch,), speed_confidence, device=device)

    # Per-sample smoothness: measure how monotonically |v| decreases
    if len(gray_zones) >= 3:
        gz_stack = torch.stack(gray_zones, dim=0)          # (steps, batch)
        first_d = gz_stack[1:] - gz_stack[:-1]             # (steps-1, batch)
        second_d = first_d[1:] - first_d[:-1]              # (steps-2, batch)
        roughness = second_d.abs().mean(dim=0)              # (batch,)
        smoothness_confidence = (1.0 - roughness * 10.0).clamp(min=0.0)
    else:
        smoothness_confidence = torch.full((batch,), 0.5, device=device)

    # Combined confidence (per-sample)
    raw = 0.4 * gz_confidence + 0.3 * speed_confidence + 0.3 * smoothness_confidence

    # Non-converged samples get forced low confidence
    raw[~converged] = raw[~converged].clamp(max=0.1)
    return raw
