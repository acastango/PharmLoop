"""
Inference pipeline: drug names in, clinical narrative out.

Supports both Phase 3 (50 drugs, flat Hopfield) and Phase 4a
(300 drugs, hierarchical Hopfield, polypharmacy).

Usage:
    engine = PharmLoopInference.load("checkpoints/best_model_phase4a.pt",
                                     data_dir="data/processed")
    result = engine.check("fluoxetine", "tramadol")
    print(result.narrative)

    report = engine.check_multiple(["fluoxetine", "tramadol", "warfarin"])
    print(report.highest_severity, report.multi_drug_alerts)
"""

import json
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import torch
from torch import Tensor

from pharmloop.context import CONTEXT_DIM
from pharmloop.hierarchical_hopfield import HierarchicalHopfield, DRUG_CLASSES
from pharmloop.hopfield import PharmHopfield
from pharmloop.model import PharmLoopModel
from pharmloop.output import SEVERITY_NAMES, MECHANISM_NAMES, FLAG_NAMES
from pharmloop.partial_convergence import PartialConvergenceAnalyzer
from pharmloop.polypharmacy import BasicPolypharmacyAnalyzer, PolypharmacyReport
from pharmloop.templates import ClinicalNarrator


@dataclass
class InteractionResult:
    """Complete result of a drug interaction check."""
    drug_a: str
    drug_b: str
    severity: str
    mechanisms: list[str]
    flags: list[str]
    confidence: float
    converged: bool
    steps: int
    partial_convergence: dict | None
    narrative: str
    gray_zone_trajectory: list[float]
    unknown_drugs: list[str] = field(default_factory=list)


class PharmLoopInference:
    """
    Complete inference pipeline: drug names → clinical narrative.

    Wraps the trained model, drug registry, template engine, and
    partial convergence analyzer into a single callable interface.
    """

    def __init__(
        self,
        model: PharmLoopModel,
        drug_registry: dict[str, dict],
        narrator: ClinicalNarrator,
        convergence_analyzer: PartialConvergenceAnalyzer,
        polypharmacy_analyzer: BasicPolypharmacyAnalyzer | None = None,
    ) -> None:
        self.model = model
        self.drug_registry = drug_registry
        self.narrator = narrator
        self.analyzer = convergence_analyzer
        self.polypharmacy_analyzer = polypharmacy_analyzer or BasicPolypharmacyAnalyzer()
        self.model.eval()

    @classmethod
    def load(
        cls,
        checkpoint_path: str,
        data_dir: str = "data/processed",
    ) -> "PharmLoopInference":
        """
        Load a trained model and build the inference engine.

        Tries v2 data (drugs_v2.json, hierarchical Hopfield) first,
        falls back to v1 (drugs.json, flat Hopfield) for backward compat.

        Args:
            checkpoint_path: Path to model checkpoint (.pt file).
            data_dir: Path to data directory containing drugs*.json.

        Returns:
            Ready-to-use PharmLoopInference instance.
        """
        data_path = Path(data_dir)

        # Try v2 data first (Phase 4a+), fall back to v1
        v2_path = data_path / "drugs_v2.json"
        v1_path = data_path / "drugs.json"

        if v2_path.exists():
            drugs_path = v2_path
        elif v1_path.exists():
            drugs_path = v1_path
        else:
            raise FileNotFoundError(
                f"No drug data found in {data_dir}. "
                f"Expected drugs_v2.json or drugs.json."
            )

        with open(drugs_path) as f:
            drugs_data = json.load(f)

        num_drugs = drugs_data.get("num_drugs", len(drugs_data["drugs"]))

        # Build drug class map and choose Hopfield type
        drug_class_map: dict[int, str] = {}
        is_v2 = "num_drugs" in drugs_data  # v2 has num_drugs key

        if is_v2:
            for name, info in drugs_data["drugs"].items():
                drug_class = info.get("class", "other")
                if drug_class in DRUG_CLASSES:
                    drug_class_map[info["id"]] = drug_class
                else:
                    drug_class_map[info["id"]] = "other"
            hopfield = HierarchicalHopfield(input_dim=512, class_names=DRUG_CLASSES)
        else:
            hopfield = PharmHopfield(input_dim=512, hidden_dim=512, phase0=False)

        model = PharmLoopModel(
            num_drugs=num_drugs,
            hopfield=hopfield,
            drug_class_map=drug_class_map if is_v2 else None,
        )

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        model.eval()

        # Build drug registry (name → {id, features, class})
        drug_registry = {}
        for name, info in drugs_data["drugs"].items():
            drug_registry[name.lower()] = {
                "id": info["id"],
                "features": info["features"],
                "class": info.get("class", "other"),
            }

        narrator = ClinicalNarrator()
        analyzer = PartialConvergenceAnalyzer(model.output_head)
        polypharmacy = BasicPolypharmacyAnalyzer()

        return cls(model, drug_registry, narrator, analyzer, polypharmacy)

    def check(
        self,
        drug_a_name: str,
        drug_b_name: str,
        context: dict | None = None,
    ) -> InteractionResult:
        """
        Check interaction between two drugs.

        Args:
            drug_a_name: Drug name (must be in registry or returns unknown).
            drug_b_name: Drug name.
            context: Optional dict with dose, route, timing, patient info.

        Returns:
            InteractionResult with all predictions and clinical narrative.
        """
        drug_a = self.drug_registry.get(drug_a_name.lower())
        drug_b = self.drug_registry.get(drug_b_name.lower())

        if drug_a is None or drug_b is None:
            return self._handle_unknown(drug_a_name, drug_b_name, drug_a, drug_b)

        a_id = torch.tensor([drug_a["id"]], dtype=torch.long)
        a_feat = torch.tensor([drug_a["features"]], dtype=torch.float32)
        b_id = torch.tensor([drug_b["id"]], dtype=torch.long)
        b_feat = torch.tensor([drug_b["features"]], dtype=torch.float32)

        ctx_tensor = self._encode_context(context) if context else None

        with torch.no_grad():
            output = self.model(a_id, a_feat, b_id, b_feat, context=ctx_tensor)

        severity_idx = output["severity_logits"].argmax(dim=-1).item()
        severity = SEVERITY_NAMES[severity_idx]
        confidence = output["confidence"].item()
        converged = output["converged"].item()
        steps = output["trajectory"]["steps"]

        # Mechanisms above threshold
        mech_probs = torch.sigmoid(output["mechanism_logits"]).squeeze()
        mechanisms = [
            MECHANISM_NAMES[i] for i, p in enumerate(mech_probs) if p.item() > 0.5
        ]

        # Flags above threshold
        flag_probs = torch.sigmoid(output["flag_logits"]).squeeze()
        flags = [
            FLAG_NAMES[i] for i, p in enumerate(flag_probs) if p.item() > 0.5
        ]

        # Partial convergence
        partial = self.analyzer.analyze(output["trajectory"]["velocities"][-1])

        # Narrative
        narrative = self.narrator.narrate(
            drug_a_name, drug_b_name, severity,
            mechanisms, flags, confidence, converged, steps, partial,
        )

        return InteractionResult(
            drug_a=drug_a_name,
            drug_b=drug_b_name,
            severity=severity,
            mechanisms=mechanisms,
            flags=flags,
            confidence=confidence,
            converged=converged,
            steps=steps,
            partial_convergence=partial,
            narrative=narrative,
            gray_zone_trajectory=[gz.item() for gz in output["trajectory"]["gray_zones"]],
        )

    def check_multiple(
        self,
        drug_names: list[str],
        context: dict | None = None,
    ) -> PolypharmacyReport:
        """
        Check all pairwise interactions in one batched forward pass.

        10 drugs = 45 pairs. Batched: one forward pass, ~100ms.
        Sequential: 45 forward passes, ~2.2 seconds.

        Args:
            drug_names: List of 2-20 drug names.
            context: Optional context dict (same for all pairs).

        Returns:
            PolypharmacyReport with ranked pairs and multi-drug alerts.
        """
        pairs = list(combinations(drug_names, 2))

        # Build batched tensors for known drugs
        a_ids, a_feats, b_ids, b_feats = [], [], [], []
        valid_pairs: list[tuple[str, str]] = []
        unknown_drugs_in_list: list[str] = []

        for drug_a, drug_b in pairs:
            da = self.drug_registry.get(drug_a.lower())
            db = self.drug_registry.get(drug_b.lower())
            if da is None:
                if drug_a not in unknown_drugs_in_list:
                    unknown_drugs_in_list.append(drug_a)
                continue
            if db is None:
                if drug_b not in unknown_drugs_in_list:
                    unknown_drugs_in_list.append(drug_b)
                continue
            a_ids.append(da["id"])
            a_feats.append(da["features"])
            b_ids.append(db["id"])
            b_feats.append(db["features"])
            valid_pairs.append((drug_a, drug_b))

        if not valid_pairs:
            return PolypharmacyReport(
                drugs=drug_names,
                total_pairs_checked=0,
                highest_severity="unknown",
                pairwise_results=[],
                multi_drug_alerts=[],
            )

        # Batch forward pass
        a_ids_t = torch.tensor(a_ids, dtype=torch.long)
        a_feats_t = torch.tensor(a_feats, dtype=torch.float32)
        b_ids_t = torch.tensor(b_ids, dtype=torch.long)
        b_feats_t = torch.tensor(b_feats, dtype=torch.float32)

        with torch.no_grad():
            output = self.model(a_ids_t, a_feats_t, b_ids_t, b_feats_t)

        # Unbatch into individual InteractionResults
        pairwise_results: dict[tuple[str, str], InteractionResult] = {}
        for i, (drug_a, drug_b) in enumerate(valid_pairs):
            result = self._unbatch_single(output, i, drug_a, drug_b)
            pairwise_results[(drug_a, drug_b)] = result

        # Run polypharmacy pattern detection
        return self.polypharmacy_analyzer.analyze(drug_names, pairwise_results)

    def _unbatch_single(
        self,
        output: dict,
        idx: int,
        drug_a: str,
        drug_b: str,
    ) -> InteractionResult:
        """Extract a single InteractionResult from batched model output."""
        severity_idx = output["severity_logits"][idx].argmax(dim=-1).item()
        severity = SEVERITY_NAMES[severity_idx]
        confidence = output["confidence"][idx].item()
        converged = output["converged"][idx].item()
        steps = output["trajectory"]["steps"]

        mech_probs = torch.sigmoid(output["mechanism_logits"][idx])
        mechanisms = [
            MECHANISM_NAMES[j] for j, p in enumerate(mech_probs) if p.item() > 0.5
        ]

        flag_probs = torch.sigmoid(output["flag_logits"][idx])
        flags = [
            FLAG_NAMES[j] for j, p in enumerate(flag_probs) if p.item() > 0.5
        ]

        partial = self.analyzer.analyze(
            output["trajectory"]["velocities"][-1][idx:idx + 1]
        )

        narrative = self.narrator.narrate(
            drug_a, drug_b, severity, mechanisms, flags,
            confidence, converged, steps, partial,
        )

        gz_trajectory = [gz[idx].item() for gz in output["trajectory"]["gray_zones"]]

        return InteractionResult(
            drug_a=drug_a,
            drug_b=drug_b,
            severity=severity,
            mechanisms=mechanisms,
            flags=flags,
            confidence=confidence,
            converged=converged,
            steps=steps,
            partial_convergence=partial,
            narrative=narrative,
            gray_zone_trajectory=gz_trajectory,
        )

    def _handle_unknown(
        self,
        drug_a_name: str,
        drug_b_name: str,
        drug_a: dict | None,
        drug_b: dict | None,
    ) -> InteractionResult:
        """Handle case where one or both drugs are not in registry."""
        unknown_drugs = []
        if drug_a is None:
            unknown_drugs.append(drug_a_name)
        if drug_b is None:
            unknown_drugs.append(drug_b_name)

        narrative = self.narrator.narrate(
            drug_a_name, drug_b_name, "unknown", [], [],
            confidence=0.0, converged=False, steps=0,
        )

        return InteractionResult(
            drug_a=drug_a_name,
            drug_b=drug_b_name,
            severity="unknown",
            mechanisms=[],
            flags=[],
            confidence=0.0,
            converged=False,
            steps=0,
            partial_convergence=None,
            narrative=narrative,
            gray_zone_trajectory=[],
            unknown_drugs=unknown_drugs,
        )

    def _encode_context(self, context: dict) -> Tensor:
        """
        Convert context dict to 32-dim feature tensor.

        Maps context keys to the feature layout defined in context.py.
        Unknown keys are silently ignored. Missing keys default to 0.
        """
        vec = torch.zeros(1, CONTEXT_DIM)

        # Drug A dosing (dims 0-3)
        vec[0, 0] = context.get("dose_a_normalized", 0.0)
        vec[0, 1] = context.get("frequency_a", 0.0)
        vec[0, 2] = context.get("duration_a_days", 0.0) / 365.0  # normalize
        vec[0, 3] = float(context.get("is_loading_dose_a", False))

        # Drug B dosing (dims 4-7)
        vec[0, 4] = context.get("dose_b_normalized", 0.0)
        vec[0, 5] = context.get("frequency_b", 0.0)
        vec[0, 6] = context.get("duration_b_days", 0.0) / 365.0
        vec[0, 7] = float(context.get("is_loading_dose_b", False))

        # Route flags (dims 8-11)
        vec[0, 8] = float(context.get("both_oral", False))
        vec[0, 9] = float(context.get("any_iv", False))
        vec[0, 10] = float(context.get("any_topical", False))
        vec[0, 11] = float(context.get("any_inhaled", False))

        # Timing (dims 12-15)
        vec[0, 12] = float(context.get("simultaneous", False))
        vec[0, 13] = context.get("separated_hours_norm", 0.0)
        vec[0, 14] = float(context.get("a_before_b", False))
        vec[0, 15] = float(context.get("b_before_a", False))

        # Patient factors (dims 16-23)
        vec[0, 16] = context.get("age_norm", 0.0)
        vec[0, 17] = context.get("weight_norm", 0.0)
        vec[0, 18] = context.get("renal_gfr_norm", 0.0)
        vec[0, 19] = context.get("hepatic_child_pugh_norm", 0.0)
        vec[0, 20] = float(context.get("pregnancy", False))
        vec[0, 21] = float(context.get("pediatric", False))
        vec[0, 22] = float(context.get("geriatric", False))
        vec[0, 23] = float(context.get("genetic_pm", False))

        # Comedication burden (dims 24-27)
        vec[0, 24] = context.get("total_drugs_norm", 0.0)
        vec[0, 25] = context.get("cyp_inhibitor_count", 0.0)
        vec[0, 26] = context.get("cyp_inducer_count", 0.0)
        vec[0, 27] = context.get("protein_bound_count", 0.0)

        return vec
