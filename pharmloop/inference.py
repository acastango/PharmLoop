"""
Inference pipeline: drug names in, clinical narrative out.

Supports Phase 3 (50 drugs, flat Hopfield), Phase 4a (300 drugs,
hierarchical Hopfield), and Phase 4b (500+ drugs, full polypharmacy,
drug resolution, pharmacogenomics).

Usage:
    engine = PharmLoopInference.load("checkpoints/best_model_phase4b.pt",
                                     data_dir="data/processed")
    result = engine.check("fluoxetine", "tramadol")
    print(result.narrative)

    # Brand names work too (Phase 4b+)
    result = engine.check("Prozac", "Ultram")
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
from pharmloop.drug_resolver import DrugResolver, ResolvedDrug
from pharmloop.hierarchical_hopfield import HierarchicalHopfield, DRUG_CLASSES
from pharmloop.hopfield import PharmHopfield
from pharmloop.model import PharmLoopModel
from pharmloop.output import SEVERITY_NAMES, MECHANISM_NAMES, FLAG_NAMES
from pharmloop.partial_convergence import PartialConvergenceAnalyzer
from pharmloop.polypharmacy import BasicPolypharmacyAnalyzer, PolypharmacyReport
from pharmloop.polypharmacy_full import FullPolypharmacyAnalyzer
from pharmloop.templates import ClinicalNarrator
from training.train_context import encode_context_vector


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

    Wraps the trained model, drug registry, template engine,
    partial convergence analyzer, drug resolver, and polypharmacy
    analyzer into a single callable interface.

    Phase 4b adds:
      - Drug name resolution (brand names, fuzzy matching)
      - Full polypharmacy analyzer with cascade detection
      - Pharmacogenomic context encoding
    """

    def __init__(
        self,
        model: PharmLoopModel,
        drug_registry: dict[str, dict],
        narrator: ClinicalNarrator,
        convergence_analyzer: PartialConvergenceAnalyzer,
        polypharmacy_analyzer: BasicPolypharmacyAnalyzer | None = None,
        drug_resolver: DrugResolver | None = None,
    ) -> None:
        self.model = model
        self.drug_registry = drug_registry
        self.narrator = narrator
        self.analyzer = convergence_analyzer
        self.polypharmacy_analyzer = polypharmacy_analyzer or BasicPolypharmacyAnalyzer()
        self.drug_resolver = drug_resolver
        self.model.eval()

    @classmethod
    def load(
        cls,
        checkpoint_path: str,
        data_dir: str = "data/processed",
        brand_names_path: str | None = None,
    ) -> "PharmLoopInference":
        """
        Load a trained model and build the inference engine.

        Tries v3 data first (Phase 4b), then v2 (Phase 4a), then v1 (Phase 3).
        If brand_names_path is provided or found in data/raw/, enables drug
        name resolution.

        Args:
            checkpoint_path: Path to model checkpoint (.pt file).
            data_dir: Path to data directory containing drugs*.json.
            brand_names_path: Optional path to brand_names.json.

        Returns:
            Ready-to-use PharmLoopInference instance.
        """
        data_path = Path(data_dir)

        # Try v3 (Phase 4b), then v2 (Phase 4a), then v1 (Phase 1-3)
        v3_path = data_path / "drugs_v3.json"
        v2_path = data_path / "drugs_v2.json"
        v1_path = data_path / "drugs.json"

        if v3_path.exists():
            drugs_path = v3_path
        elif v2_path.exists():
            drugs_path = v2_path
        elif v1_path.exists():
            drugs_path = v1_path
        else:
            raise FileNotFoundError(
                f"No drug data found in {data_dir}. "
                f"Expected drugs_v3.json, drugs_v2.json, or drugs.json."
            )

        with open(drugs_path) as f:
            drugs_data = json.load(f)

        num_drugs = drugs_data.get("num_drugs", len(drugs_data["drugs"]))

        # Build drug class map and choose Hopfield type
        drug_class_map: dict[int, str] = {}
        is_v2_plus = "num_drugs" in drugs_data  # v2+ has num_drugs key

        if is_v2_plus:
            for name, info in drugs_data["drugs"].items():
                drug_class = info.get("class", "other")
                if drug_class in DRUG_CLASSES:
                    drug_class_map[info["id"]] = drug_class
                else:
                    drug_class_map[info["id"]] = "other"
            hopfield = HierarchicalHopfield(input_dim=512, class_names=DRUG_CLASSES)
        else:
            hopfield = PharmHopfield(input_dim=512, hidden_dim=512, phase0=False)

        # Detect if checkpoint has context encoder weights
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        has_context = any(
            k.startswith("context_encoder.") for k in checkpoint["model_state_dict"]
        )

        model = PharmLoopModel(
            num_drugs=num_drugs,
            hopfield=hopfield,
            drug_class_map=drug_class_map if is_v2_plus else None,
            use_context=has_context,
        )

        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        model.eval()

        # Build drug registry (name → {id, features, class})
        drug_registry: dict[str, dict] = {}
        for name, info in drugs_data["drugs"].items():
            drug_registry[name.lower()] = {
                "id": info["id"],
                "features": info["features"],
                "class": info.get("class", "other"),
            }

        # Build drug resolver (brand names + fuzzy matching)
        if brand_names_path is None:
            # Auto-detect brand_names.json
            for candidate in [
                data_path.parent / "raw" / "brand_names.json",
                data_path / "brand_names.json",
            ]:
                if candidate.exists():
                    brand_names_path = str(candidate)
                    break

        resolver = DrugResolver.load(
            drug_registry, brand_names_path=brand_names_path,
        )

        narrator = ClinicalNarrator()
        analyzer = PartialConvergenceAnalyzer(model.output_head)

        # Use FullPolypharmacyAnalyzer for Phase 4b (v3 data)
        is_v3 = v3_path.exists()
        polypharmacy: BasicPolypharmacyAnalyzer
        if is_v3:
            polypharmacy = FullPolypharmacyAnalyzer()
        else:
            polypharmacy = BasicPolypharmacyAnalyzer()

        return cls(
            model, drug_registry, narrator, analyzer,
            polypharmacy, drug_resolver=resolver,
        )

    def resolve_drug(self, name: str) -> str | None:
        """
        Resolve a drug name through brand lookup and fuzzy matching.

        Returns the canonical generic name, or None if unresolvable.
        """
        if self.drug_resolver is None:
            return name.lower() if name.lower() in self.drug_registry else None
        result = self.drug_resolver.resolve(name)
        return result.resolved

    def check(
        self,
        drug_a_name: str,
        drug_b_name: str,
        context: dict | None = None,
    ) -> InteractionResult:
        """
        Check interaction between two drugs.

        Supports brand names and fuzzy matching when a DrugResolver
        is configured (Phase 4b+).

        Args:
            drug_a_name: Drug name (generic or brand).
            drug_b_name: Drug name (generic or brand).
            context: Optional dict with dose, route, timing, patient info,
                and/or pharmacogenomic status.

        Returns:
            InteractionResult with all predictions and clinical narrative.
        """
        # Resolve drug names (brand → generic, fuzzy matching)
        resolved_a = self.resolve_drug(drug_a_name)
        resolved_b = self.resolve_drug(drug_b_name)
        drug_a = self.drug_registry.get(resolved_a) if resolved_a else None
        drug_b = self.drug_registry.get(resolved_b) if resolved_b else None

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
        # Resolve all drug names first
        resolved_names: list[str | None] = []
        for name in drug_names:
            resolved_names.append(self.resolve_drug(name))

        pairs = list(combinations(range(len(drug_names)), 2))

        # Build batched tensors for known drugs
        a_ids, a_feats, b_ids, b_feats = [], [], [], []
        valid_pairs: list[tuple[str, str]] = []
        unknown_drugs_in_list: list[str] = []

        for i, j in pairs:
            ra = resolved_names[i]
            rb = resolved_names[j]
            da = self.drug_registry.get(ra) if ra else None
            db = self.drug_registry.get(rb) if rb else None
            drug_a = drug_names[i]
            drug_b = drug_names[j]
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
                skipped_drugs=unknown_drugs_in_list,
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

        # Build drug features map for full polypharmacy analyzer
        drug_features: dict[str, list[float]] | None = None
        if isinstance(self.polypharmacy_analyzer, FullPolypharmacyAnalyzer):
            drug_features = {}
            for name, resolved in zip(drug_names, resolved_names):
                if resolved and resolved in self.drug_registry:
                    drug_features[name] = self.drug_registry[resolved]["features"]

        # Run polypharmacy pattern detection
        if drug_features is not None:
            return self.polypharmacy_analyzer.analyze(
                drug_names, pairwise_results,
                skipped_drugs=unknown_drugs_in_list,
                drug_features=drug_features,
            )
        return self.polypharmacy_analyzer.analyze(
            drug_names, pairwise_results, skipped_drugs=unknown_drugs_in_list,
        )

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
        Convert context dict to 48-dim feature tensor.

        Maps context keys to the feature layout defined in context.py,
        including pharmacogenomic status (Phase 4b+). Uses the shared
        encode_context_vector function for consistency with training.
        """
        vec_list = encode_context_vector(context)
        return torch.tensor([vec_list], dtype=torch.float32)
