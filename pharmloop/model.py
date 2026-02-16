"""
PharmLoopModel — full pipeline from drug pairs to interaction predictions.

Pipeline:
  (drug_a_id, drug_a_features, drug_b_id, drug_b_features)
    → encode both drugs
    → combine pair state
    → oscillatory reasoning loop
    → output head
    → {severity, mechanisms, flags, confidence, converged, trajectory}
"""

import torch
import torch.nn as nn
from torch import Tensor

from pharmloop.encoder import DrugEncoder, FUSED_DIM
from pharmloop.hopfield import PharmHopfield
from pharmloop.oscillator import OscillatorCell, ReasoningLoop, STATE_DIM, MAX_STEPS
from pharmloop.output import OutputHead, compute_confidence


class PharmLoopModel(nn.Module):
    """
    Complete PharmLoop pipeline: drug pair → interaction prediction.

    Encodes two drugs, combines them into a pair state, runs the oscillatory
    reasoning loop, and produces structured output.

    Args:
        num_drugs: Number of known drugs in the vocabulary.
        feature_dim: Dimension of structured feature vectors (default 64).
        state_dim: Dimension of oscillator state (default 512).
        hopfield: Pre-initialized Hopfield memory (can be None initially).
        max_steps: Maximum oscillator steps before UNKNOWN.
    """

    def __init__(
        self,
        num_drugs: int,
        feature_dim: int = 64,
        state_dim: int = STATE_DIM,
        hopfield: PharmHopfield | None = None,
        max_steps: int = MAX_STEPS,
    ) -> None:
        super().__init__()
        self.num_drugs = num_drugs
        self.state_dim = state_dim

        # Drug encoder (shared for both drugs in a pair)
        self.encoder = DrugEncoder(num_drugs, feature_dim)

        # Pair combination: concat both encodings → project to state_dim
        self.pair_combine = nn.Sequential(
            nn.Linear(FUSED_DIM * 2, state_dim),
            nn.LayerNorm(state_dim),
            nn.GELU(),
            nn.Linear(state_dim, state_dim),
        )

        # Oscillatory reasoning
        self.cell = OscillatorCell(state_dim, hopfield)
        self.reasoning_loop = ReasoningLoop(self.cell, max_steps)

        # Output head
        self.output_head = OutputHead(state_dim)

    def forward(
        self,
        drug_a_id: Tensor,
        drug_a_features: Tensor,
        drug_b_id: Tensor,
        drug_b_features: Tensor,
    ) -> dict[str, Tensor | list | bool]:
        """
        Full forward pass: drug pair → interaction prediction.

        Args:
            drug_a_id: (batch,) integer tensor — drug A vocabulary index.
            drug_a_features: (batch, 64) — drug A structured features.
            drug_b_id: (batch,) integer tensor — drug B vocabulary index.
            drug_b_features: (batch, 64) — drug B structured features.

        Returns:
            Dictionary with:
              - "severity_logits": (batch, 6) severity class logits.
              - "mechanism_logits": (batch, num_mechanisms) mechanism logits.
              - "flag_logits": (batch, num_flags) flag logits.
              - "confidence": (batch,) confidence scores from dynamics.
              - "converged": (batch,) boolean convergence flags.
              - "trajectory": dict with positions, velocities, gray_zones, steps.
        """
        batch = drug_a_id.shape[0]

        # Encode both drugs (shared encoder)
        enc_a = self.encoder(drug_a_id, drug_a_features)  # (batch, 512)
        enc_b = self.encoder(drug_b_id, drug_b_features)  # (batch, 512)

        # Combine pair: make it order-invariant by also adding the reverse
        pair_forward = torch.cat([enc_a, enc_b], dim=-1)   # (batch, 1024)
        pair_reverse = torch.cat([enc_b, enc_a], dim=-1)   # (batch, 1024)
        initial_state = self.pair_combine(pair_forward) + self.pair_combine(pair_reverse)
        initial_state = initial_state / 2.0  # average for symmetry

        assert initial_state.shape == (batch, self.state_dim)

        # Run oscillatory reasoning
        trajectory = self.reasoning_loop(initial_state, training=self.training)

        # Output head on final position
        predictions = self.output_head(trajectory["final_x"])

        # Compute confidence from dynamics
        confidence = compute_confidence(
            trajectory["gray_zones"],
            trajectory["converged"],
            max_steps=self.reasoning_loop.max_steps,
        )

        return {
            "severity_logits": predictions["severity_logits"],
            "mechanism_logits": predictions["mechanism_logits"],
            "flag_logits": predictions["flag_logits"],
            "confidence": confidence,
            "converged": trajectory["converged"],
            "trajectory": {
                "positions": trajectory["positions"],
                "velocities": trajectory["velocities"],
                "gray_zones": trajectory["gray_zones"],
                "steps": trajectory["steps"],
            },
        }

    def count_parameters(self) -> dict[str, int]:
        """Count learned parameters and buffer sizes."""
        learned = sum(p.numel() for p in self.parameters() if p.requires_grad)
        buffers = sum(b.numel() for b in self.buffers())
        total = learned + buffers
        return {"learned": learned, "buffers": buffers, "total": total}
