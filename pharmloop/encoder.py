"""
Drug encoder: maps a single drug to a 512-dim pharmacological state vector.

Two pathways fused:
  1. Learned identity embedding — captures drug-specific quirks not in the feature vector.
  2. Structured 64-dim feature vector — explicit pharmacological profile (CYP, receptors, PK, etc.).

The structured features are the ANCHOR. The identity embedding learns whatever the features
don't capture. For unknown/fabricated drug IDs, the identity embedding is random (never trained),
which is what makes fabricated drugs fail to converge in the oscillator.
"""

import torch
import torch.nn as nn
from torch import Tensor


# Embedding dimensions
IDENTITY_DIM = 256
FEATURE_DIM = 64
FEATURE_PROJ_DIM = 256
FUSED_DIM = 512

# Padding slots for unknown drug IDs
UNKNOWN_PADDING = 100


class DrugEncoder(nn.Module):
    """
    Encodes a single drug into a 512-dim pharmacological state.

    Args:
        num_drugs: Number of known drugs in the vocabulary.
        feature_dim: Dimension of structured feature vectors (default 64).
    """

    def __init__(self, num_drugs: int, feature_dim: int = FEATURE_DIM) -> None:
        super().__init__()
        self.num_drugs = num_drugs
        self.feature_dim = feature_dim
        self.total_vocab = num_drugs + UNKNOWN_PADDING

        # Pathway 1: learned identity embedding
        self.identity_embedding = nn.Embedding(self.total_vocab, IDENTITY_DIM)

        # Pathway 2: structured feature projection
        self.feature_proj = nn.Linear(feature_dim, FEATURE_PROJ_DIM)

        # Fusion: concat(identity_256, feature_256) → 512 → 512
        self.fusion = nn.Sequential(
            nn.Linear(IDENTITY_DIM + FEATURE_PROJ_DIM, FUSED_DIM),
            nn.LayerNorm(FUSED_DIM),
            nn.GELU(),
            nn.Linear(FUSED_DIM, FUSED_DIM),
        )

    def forward(self, drug_id: Tensor, features: Tensor) -> Tensor:
        """
        Encode a drug into 512-dim state.

        Args:
            drug_id: Integer tensor of shape (batch,) — drug vocabulary index.
                     IDs >= num_drugs map to untrained padding embeddings.
            features: Tensor of shape (batch, 64) — structured feature vector.

        Returns:
            Tensor of shape (batch, 512) — encoded drug state.
        """
        batch = drug_id.shape[0]
        assert features.shape == (batch, self.feature_dim), (
            f"Expected features shape ({batch}, {self.feature_dim}), got {features.shape}"
        )

        # Clamp IDs to valid range (unknown IDs map to padding slots)
        safe_id = drug_id.clamp(0, self.total_vocab - 1)

        # Pathway 1: identity embedding
        identity = self.identity_embedding(safe_id)  # (batch, 256)

        # Pathway 2: structured features
        feat_proj = self.feature_proj(features)  # (batch, 256)

        # Fuse
        combined = torch.cat([identity, feat_proj], dim=-1)  # (batch, 512)
        encoded = self.fusion(combined)  # (batch, 512)

        assert encoded.shape == (batch, FUSED_DIM)
        return encoded
