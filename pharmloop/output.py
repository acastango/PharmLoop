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
]
NUM_FLAGS = len(FLAG_NAMES)

# State dimension from the oscillator
STATE_DIM = 512


class OutputHead(nn.Module):
    """
    Maps converged oscillator state to structured drug interaction predictions.

    Does NOT predict confidence — that comes from convergence dynamics.

    Args:
        state_dim: Dimension of the oscillator state (default 512).
        num_mechanisms: Number of mechanism labels (default 15).
        num_flags: Number of clinical flag labels (default 10).
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

        # Mechanism: multi-label classification
        self.mechanism_head = nn.Linear(state_dim, num_mechanisms)

        # Clinical flags: binary classification
        self.flags_head = nn.Linear(state_dim, num_flags)

    def forward(self, state: Tensor) -> dict[str, Tensor]:
        """
        Produce structured predictions from oscillator state.

        Args:
            state: Tensor of shape (batch, state_dim) — converged oscillator position.

        Returns:
            Dictionary with:
              - "severity_logits": (batch, 6) — raw logits for severity classification.
              - "mechanism_logits": (batch, num_mechanisms) — raw logits for multi-label.
              - "flag_logits": (batch, num_flags) — raw logits for binary flags.
        """
        batch = state.shape[0]
        assert state.shape == (batch, self.state_dim), (
            f"Expected state shape ({batch}, {self.state_dim}), got {state.shape}"
        )

        return {
            "severity_logits": self.severity_head(state),      # (batch, 6)
            "mechanism_logits": self.mechanism_head(state),     # (batch, num_mechanisms)
            "flag_logits": self.flags_head(state),              # (batch, num_flags)
        }


def compute_confidence(
    gray_zones: list[float],
    converged: Tensor,
    max_steps: int = 16,
) -> Tensor:
    """
    Compute confidence from convergence dynamics (a formula, not a neural network).

    confidence = f(final_gray_zone, steps_to_converge, trajectory_smoothness)

    Components:
      - Final gray zone: lower |v| at end → higher confidence
      - Speed: converging in fewer steps → higher confidence
      - Smoothness: smooth decay of |v| → higher confidence (vs chaotic oscillation)

    Args:
        gray_zones: List of gray zone values (|v|) at each step.
        converged: Boolean tensor of shape (batch,) — whether each sample converged.
        max_steps: Maximum number of oscillator steps.

    Returns:
        Tensor of shape (batch,) with confidence scores in [0, 1].
    """
    # Final gray zone component: lower → more confident
    final_gz = gray_zones[-1] if gray_zones else 1.0
    gz_confidence = max(0.0, 1.0 - final_gz * 5.0)  # scale so gz=0.2 → confidence=0

    # Speed component: fewer steps → more confident
    steps_taken = len(gray_zones) - 1  # subtract initial state
    speed_confidence = max(0.0, 1.0 - steps_taken / max_steps)

    # Smoothness component: measure how monotonically |v| decreases
    if len(gray_zones) >= 3:
        diffs = [gray_zones[i + 1] - gray_zones[i] for i in range(len(gray_zones) - 1)]
        second_diffs = [diffs[i + 1] - diffs[i] for i in range(len(diffs) - 1)]
        roughness = sum(abs(d) for d in second_diffs) / len(second_diffs)
        smoothness_confidence = max(0.0, 1.0 - roughness * 10.0)
    else:
        smoothness_confidence = 0.5

    # Combined confidence
    raw_confidence = 0.4 * gz_confidence + 0.3 * speed_confidence + 0.3 * smoothness_confidence

    # Non-converged samples get forced low confidence
    batch = converged.shape[0]
    confidence = torch.full((batch,), raw_confidence, device=converged.device)
    confidence[~converged] = confidence[~converged].clamp(max=0.1)

    return confidence
