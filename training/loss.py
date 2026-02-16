"""
Multi-objective loss function for PharmLoop.

L_total = L_answer + L_convergence + L_smoothness + L_do_no_harm

- L_answer: cross-entropy (severity) + BCE (mechanism, flags)
- L_convergence: reward fast convergence on known, non-convergence on unknown
- L_smoothness: penalize chaotic oscillation (second derivative of gray zone trajectory)
- L_do_no_harm: 10x penalty for false-none on severe, 50x on contraindicated
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from pharmloop.output import SEVERITY_NONE, SEVERITY_SEVERE, SEVERITY_CONTRAINDICATED, SEVERITY_UNKNOWN


# DO NO HARM penalty multipliers
PENALTY_FALSE_NONE_SEVERE = 10.0
PENALTY_FALSE_NONE_CONTRAINDICATED = 50.0


class PharmLoopLoss(nn.Module):
    """
    Multi-objective loss with DO NO HARM asymmetry.

    Returns component losses separately (for logging) plus weighted total.

    Args:
        convergence_weight: Weight for L_convergence (default 0.5).
        smoothness_weight: Weight for L_smoothness (default 0.1).
        mechanism_weight: Weight for L_mechanism relative to L_severity (default 0.5).
        flags_weight: Weight for L_flags relative to L_severity (default 0.3).
    """

    def __init__(
        self,
        convergence_weight: float = 0.5,
        smoothness_weight: float = 0.1,
        mechanism_weight: float = 0.5,
        flags_weight: float = 0.3,
    ) -> None:
        super().__init__()
        self.convergence_weight = convergence_weight
        self.smoothness_weight = smoothness_weight
        self.mechanism_weight = mechanism_weight
        self.flags_weight = flags_weight

    def forward(
        self,
        severity_logits: Tensor,
        mechanism_logits: Tensor,
        flag_logits: Tensor,
        final_v: Tensor,
        velocities: list[Tensor],
        target_severity: Tensor,
        target_mechanisms: Tensor,
        target_flags: Tensor,
        is_unknown: Tensor,
    ) -> dict[str, Tensor]:
        """
        Compute all loss components.

        Args:
            severity_logits: (batch, 6) raw severity logits.
            mechanism_logits: (batch, num_mechanisms) raw mechanism logits.
            flag_logits: (batch, num_flags) raw flag logits.
            final_v: (batch, state_dim) final velocity tensor (carries gradients).
            velocities: List of velocity tensors at each step (for smoothness).
            target_severity: (batch,) integer severity class targets.
            target_mechanisms: (batch, num_mechanisms) binary mechanism targets.
            target_flags: (batch, num_flags) binary flag targets.
            is_unknown: (batch,) boolean — whether this sample is an unknown/fabricated pair.

        Returns:
            Dictionary with "total", "answer", "convergence", "smoothness", "do_no_harm".
        """
        batch = severity_logits.shape[0]
        device = severity_logits.device

        # === L_answer: classification losses ===
        l_severity = F.cross_entropy(severity_logits, target_severity, reduction="none")  # (batch,)
        l_mechanism = F.binary_cross_entropy_with_logits(
            mechanism_logits, target_mechanisms, reduction="none"
        ).mean(dim=-1)  # (batch,)
        l_flags = F.binary_cross_entropy_with_logits(
            flag_logits, target_flags, reduction="none"
        ).mean(dim=-1)  # (batch,)

        l_answer = l_severity + self.mechanism_weight * l_mechanism + self.flags_weight * l_flags
        # Don't penalize answer loss for unknown samples (they have no ground-truth answer)
        l_answer = l_answer * (~is_unknown).float()
        l_answer = l_answer.mean()

        # === L_convergence: differentiable, uses final_v tensor ===
        # Gray zone per sample = L2 norm of final velocity
        final_gz = final_v.norm(dim=-1)  # (batch,) — carries gradients

        known_mask = (~is_unknown).float()   # (batch,)
        unknown_mask = is_unknown.float()    # (batch,)

        # Known pairs: penalize high |v| (want convergence)
        l_conv_known = (final_gz * known_mask).sum() / (known_mask.sum() + 1e-8)

        # Unknown pairs: penalize low |v| (want non-convergence)
        # Use hinge: max(0, target_gz - |v|) where target_gz is a minimum desired gz
        target_gz = 0.5  # unknown pairs should maintain at least this much gray zone
        l_conv_unknown = (F.relu(target_gz - final_gz) * unknown_mask).sum() / (unknown_mask.sum() + 1e-8)

        l_convergence = l_conv_known + l_conv_unknown

        # === L_smoothness: second derivative penalty on per-sample |v| trajectory ===
        l_smoothness = torch.tensor(0.0, device=device)
        if len(velocities) >= 3:
            # Compute per-sample gray zone at each step as tensors (preserves gradients)
            gz_steps = [v.norm(dim=-1) for v in velocities]  # list of (batch,) tensors
            gz_stack = torch.stack(gz_steps, dim=0)  # (steps, batch)
            first_deriv = gz_stack[1:] - gz_stack[:-1]  # (steps-1, batch)
            second_deriv = first_deriv[1:] - first_deriv[:-1]  # (steps-2, batch)
            l_smoothness = (second_deriv ** 2).mean()

        # === L_do_no_harm: asymmetric penalty for dangerous false negatives ===
        severity_preds = severity_logits.argmax(dim=-1)  # (batch,)

        # False-none on severe: predicted none but actually severe
        false_none_severe = (
            (severity_preds == SEVERITY_NONE) & (target_severity == SEVERITY_SEVERE)
        ).float()
        # False-none on contraindicated: predicted none but actually contraindicated
        false_none_contra = (
            (severity_preds == SEVERITY_NONE) & (target_severity == SEVERITY_CONTRAINDICATED)
        ).float()

        # Apply to severity loss (boost the cross-entropy for these cases)
        harm_penalty = (
            PENALTY_FALSE_NONE_SEVERE * false_none_severe
            + PENALTY_FALSE_NONE_CONTRAINDICATED * false_none_contra
        )
        l_do_no_harm = (harm_penalty * F.cross_entropy(
            severity_logits, target_severity, reduction="none"
        )).mean()

        # === Total ===
        l_total = (
            l_answer
            + self.convergence_weight * l_convergence
            + self.smoothness_weight * l_smoothness
            + l_do_no_harm
        )

        return {
            "total": l_total,
            "answer": l_answer,
            "convergence": l_convergence,
            "smoothness": l_smoothness,
            "do_no_harm": l_do_no_harm,
        }
