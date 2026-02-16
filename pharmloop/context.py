"""
Context encoder: modulates pair state with dose, route, timing, and patient factors.

Some interactions are context-dependent:
  - Warfarin + acetaminophen: safe at low doses, risky at high chronic doses
  - CYP interactions scale with dose
  - Some interactions are mitigated by separating administration times

The context encoder uses gated additive modulation: when context is absent or
irrelevant, the gate stays near zero and output ≈ pair_state. When context
is informative, the gate opens and shifts the starting point of oscillation.

~200K params.
"""

import torch
import torch.nn as nn
from torch import Tensor

CONTEXT_DIM = 32

# Context feature layout:
# Dims 0-3:   Drug A dosing (dose_normalized, frequency, duration_days, is_loading_dose)
# Dims 4-7:   Drug B dosing (same layout)
# Dims 8-11:  Route flags (both_oral, any_iv, any_topical, any_inhaled)
# Dims 12-15: Timing (simultaneous, separated_hours_norm, a_before_b, b_before_a)
# Dims 16-23: Patient factors (age_norm, weight_norm, renal_gfr_norm,
#             hepatic_child_pugh_norm, pregnancy, pediatric, geriatric, genetic_pm)
# Dims 24-27: Comedication burden (total_drugs_norm, cyp_inhibitor_count,
#             cyp_inducer_count, protein_bound_count)
# Dims 28-31: Reserved


class ContextEncoder(nn.Module):
    """
    Encodes contextual factors into a modulation signal for the pair state.

    Does NOT replace the pair state — it MODULATES it. The base interaction
    profile comes from the drug pair; context adjusts it.

    Args:
        context_dim: Dimension of context feature vector (default 32).
        state_dim: Dimension of pair state / oscillator state (default 512).
    """

    def __init__(self, context_dim: int = CONTEXT_DIM, state_dim: int = 512) -> None:
        super().__init__()
        self.context_dim = context_dim

        self.context_proj = nn.Sequential(
            nn.Linear(context_dim, 128),
            nn.GELU(),
            nn.Linear(128, state_dim),
        )

        self.gate = nn.Sequential(
            nn.Linear(context_dim, 128),
            nn.GELU(),
            nn.Linear(128, state_dim),
            nn.Sigmoid(),
        )

        # Initialize gate bias to -2.0 so sigmoid outputs ~0.12 at init.
        # Model starts very close to Phase 2 behavior.
        with torch.no_grad():
            self.gate[-2].bias.fill_(-2.0)

    def forward(self, pair_state: Tensor, context: Tensor) -> Tensor:
        """
        Modulate pair state with context.

        Args:
            pair_state: (batch, state_dim) from drug pair encoding.
            context: (batch, context_dim) structured context features.

        Returns:
            (batch, state_dim) context-modulated pair state.
        """
        ctx_signal = self.context_proj(context)  # (batch, state_dim)
        gate = self.gate(context)  # (batch, state_dim)

        # Gated additive modulation
        return pair_state + gate * ctx_signal
