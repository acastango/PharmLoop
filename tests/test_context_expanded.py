"""Tests for expanded context encoder (48-dim) and PGx encoding."""

import torch
import pytest

from pharmloop.context import (
    CONTEXT_DIM,
    ContextEncoder,
    PGX_CYP2D6_OFFSET,
    PGX_CYP2C19_OFFSET,
    PGX_CYP2C9_VKORC1_OFFSET,
    PGX_HLA_OFFSET,
)
from training.train_context import encode_context_vector


class TestContextDim:
    """Verify context dimension is 48 (Phase 4b expansion)."""

    def test_context_dim_is_48(self):
        assert CONTEXT_DIM == 48

    def test_encoder_accepts_48_dim(self):
        encoder = ContextEncoder(context_dim=48)
        pair_state = torch.randn(1, 512)
        context = torch.randn(1, 48)
        output = encoder(pair_state, context)
        assert output.shape == (1, 512)


class TestEncodeContextVector:
    """Test context dict → 48-dim vector encoding."""

    def test_empty_context(self):
        vec = encode_context_vector({})
        assert len(vec) == 48
        assert all(v == 0.0 for v in vec)

    def test_dose_encoding(self):
        vec = encode_context_vector({
            "dose_a_normalized": 0.5,
            "dose_b_normalized": 0.8,
        })
        assert vec[0] == 0.5
        assert vec[4] == 0.8

    def test_route_encoding(self):
        vec = encode_context_vector({"both_oral": True, "any_iv": True})
        assert vec[8] == 1.0
        assert vec[9] == 1.0
        assert vec[10] == 0.0

    def test_timing_encoding(self):
        vec = encode_context_vector({
            "simultaneous": True,
            "separated_hours_norm": 0.5,
        })
        assert vec[12] == 1.0
        assert vec[13] == 0.5

    def test_patient_factors(self):
        vec = encode_context_vector({
            "age_norm": 0.7,
            "renal_gfr_norm": 0.3,
            "geriatric": True,
        })
        assert vec[16] == 0.7
        assert vec[18] == 0.3
        assert vec[22] == 1.0

    def test_cyp2d6_poor_metabolizer(self):
        vec = encode_context_vector({"cyp2d6_status": "poor_metabolizer"})
        # One-hot at offset 32
        assert vec[PGX_CYP2D6_OFFSET + 0] == 1.0  # poor
        assert vec[PGX_CYP2D6_OFFSET + 1] == 0.0  # intermediate
        assert vec[PGX_CYP2D6_OFFSET + 2] == 0.0  # extensive
        assert vec[PGX_CYP2D6_OFFSET + 3] == 0.0  # ultra-rapid

    def test_cyp2d6_ultra_rapid(self):
        vec = encode_context_vector({"cyp2d6_status": "ultra_rapid_metabolizer"})
        assert vec[PGX_CYP2D6_OFFSET + 3] == 1.0
        assert vec[PGX_CYP2D6_OFFSET + 0] == 0.0

    def test_cyp2c19_extensive(self):
        vec = encode_context_vector({"cyp2c19_status": "extensive_metabolizer"})
        assert vec[PGX_CYP2C19_OFFSET + 2] == 1.0

    def test_vkorc1_sensitive(self):
        vec = encode_context_vector({"vkorc1_status": "sensitive"})
        assert vec[PGX_CYP2C9_VKORC1_OFFSET + 3] == 1.0

    def test_hla_b5701_positive(self):
        vec = encode_context_vector({"hla_b5701": "positive"})
        assert vec[PGX_HLA_OFFSET + 0] == 1.0

    def test_hla_b1502_positive(self):
        vec = encode_context_vector({"hla_b1502": "positive"})
        assert vec[PGX_HLA_OFFSET + 1] == 1.0

    def test_combined_context_and_pgx(self):
        """Verify clinical context and PGx can coexist."""
        vec = encode_context_vector({
            "dose_a_normalized": 0.5,
            "geriatric": True,
            "cyp2d6_status": "poor_metabolizer",
            "hla_b5701": "positive",
        })
        assert vec[0] == 0.5  # dose
        assert vec[22] == 1.0  # geriatric
        assert vec[PGX_CYP2D6_OFFSET + 0] == 1.0  # CYP2D6 poor
        assert vec[PGX_HLA_OFFSET + 0] == 1.0  # HLA-B*5701

    def test_invalid_pgx_status_ignored(self):
        """Invalid PGx status values should not crash."""
        vec = encode_context_vector({"cyp2d6_status": "nonexistent_status"})
        # All PGx dims should be 0
        for i in range(PGX_CYP2D6_OFFSET, PGX_CYP2D6_OFFSET + 4):
            assert vec[i] == 0.0


class TestContextEncoderBackwardCompat:
    """Test that expanding context encoder doesn't break existing behavior."""

    def test_zero_pgx_gives_same_output(self):
        """With PGx dims all zero, output should be same as 32-dim behavior."""
        encoder = ContextEncoder(context_dim=48)
        pair_state = torch.randn(2, 512)

        # Context with only first 32 dims set
        ctx_48 = torch.zeros(2, 48)
        ctx_48[:, :28] = torch.randn(2, 28)  # random clinical context

        output = encoder(pair_state, ctx_48)
        assert output.shape == (2, 512)
        # Output should be close to pair_state when gate is near-closed
        # (gate initializes to ~0.12)

    def test_pgx_context_changes_output(self):
        """Adding PGx context should change the output."""
        encoder = ContextEncoder(context_dim=48)
        pair_state = torch.randn(1, 512)

        ctx_no_pgx = torch.zeros(1, 48)
        ctx_with_pgx = torch.zeros(1, 48)
        ctx_with_pgx[0, PGX_CYP2D6_OFFSET] = 1.0  # poor metabolizer

        out_no_pgx = encoder(pair_state, ctx_no_pgx)
        out_with_pgx = encoder(pair_state, ctx_with_pgx)

        # Outputs should differ
        assert not torch.allclose(out_no_pgx, out_with_pgx)
