"""
Partial convergence analysis: per-dimension velocity inspection.

Maps oscillator dimensions to semantic categories (severity, mechanism, flags)
using the output head weights as a guide. Reports which aspects of the prediction
have settled and which are still uncertain.

Zero parameters — pure analysis of trained model state.
"""

import torch
import torch.nn as nn
from torch import Tensor


class PartialConvergenceAnalyzer:
    """
    Analyzes per-dimension velocity to determine which aspects of the
    prediction have settled and which are still uncertain.

    Uses L1 norm of output head weights to identify which oscillator
    dimensions each head relies on most.

    Args:
        output_head: Trained OutputHead instance.
        convergence_threshold: Per-dimension velocity threshold for "settled."
    """

    def __init__(self, output_head: nn.Module, convergence_threshold: float = 0.05) -> None:
        self.threshold = convergence_threshold

        self.severity_dims = self._important_dims(output_head.severity_head)
        self.mechanism_dims = self._important_dims(output_head.mechanism_head)
        self.flag_dims = self._important_dims(output_head.flags_head)

    def _important_dims(self, head: nn.Module) -> Tensor:
        """
        Get top-30% most important input dimensions from a head's weights.

        Handles both nn.Linear and TrajectoryMechanismHead.
        """
        if isinstance(head, nn.Linear):
            weight = head.weight.data
        elif hasattr(head, "mlp"):
            # TrajectoryMechanismHead: use first MLP layer
            weight = head.mlp[0].weight.data
        else:
            return torch.arange(512)

        importance = weight.abs().sum(dim=0)  # (state_dim,)
        k = max(1, int(0.3 * importance.shape[0]))
        _, top_indices = importance.topk(k)
        return top_indices

    def analyze(self, final_v: Tensor) -> dict:
        """
        Analyze partial convergence from final velocity.

        Args:
            final_v: (batch, state_dim) final velocity tensor.

        Returns:
            Dict with per-aspect convergence info.
        """
        per_dim_gz = final_v.abs().mean(dim=0)  # (state_dim,)

        severity_gz = per_dim_gz[self.severity_dims].mean().item()
        mechanism_gz = per_dim_gz[self.mechanism_dims].mean().item()
        flag_gz = per_dim_gz[self.flag_dims].mean().item()

        severity_settled = severity_gz < self.threshold
        mechanism_settled = mechanism_gz < self.threshold
        flags_settled = flag_gz < self.threshold

        settled = []
        unsettled = []
        for name, is_settled in [
            ("severity", severity_settled),
            ("mechanism", mechanism_settled),
            ("clinical flags", flags_settled),
        ]:
            (settled if is_settled else unsettled).append(name)

        return {
            "severity_settled": severity_settled,
            "mechanism_settled": mechanism_settled,
            "flags_settled": flags_settled,
            "settled_aspects": settled,
            "unsettled_aspects": unsettled,
            "partial_convergence": bool(settled) and bool(unsettled),
            "severity_gz": severity_gz,
            "mechanism_gz": mechanism_gz,
            "flags_gz": flag_gz,
        }
