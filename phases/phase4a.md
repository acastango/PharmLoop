# Phase 4a — Minimum Viable Scale

## Prerequisites
Phase 3 validated:
- [ ] Mechanism accuracy >= 60%
- [ ] Severity accuracy >= 90% maintained
- [ ] Zero false negatives on severe/contraindicated
- [ ] Average convergence steps <= 10
- [ ] Template engine covers all output paths
- [ ] Inference pipeline works end-to-end
- [ ] Context encoder modulates output in expected direction
- [ ] Param budget < 10M

---

## What Phase 4a Accomplishes

Phases 1-3 proved the architecture on 50 drugs and 209 interactions.
Phase 4a scales to a **useful** system — enough drugs to cover the most
commonly prescribed medications, enough interactions to be clinically
relevant, with an API that a pharmacist-facing tool can actually call.

- **~300 drugs** covering the top prescribed medications in the US
- **~1500 verified interactions** from DrugBank
- **Hierarchical Hopfield** to keep retrieval sharp at scale
- **Basic polypharmacy** — pairwise scan + the 3 most dangerous additive patterns
- **REST API** — a service other software can talk to
- **Evaluation framework** to measure how good (and how safe) the system is

Phase 4a is the first version you could put in front of a pharmacist and
get meaningful feedback. That feedback drives Phase 4b.

---

## Step 1: Data Pipeline

### 1.1 Scope: Top 300 Drugs

Don't try to parse all of DrugBank. Target the ~300 most commonly
prescribed drugs in the United States. These account for the vast majority
of real-world interaction checks.

Sources for the drug list:
- ClinCalc top 200 prescribed drugs (publicly available)
- Supplement with the top drugs in high-interaction categories:
  anticoagulants, antiepileptics, immunosuppressants, antiarrhythmics,
  narrow therapeutic index drugs

This gives broad coverage of what pharmacists actually encounter while
keeping the data manageable enough to quality-check.

### 1.2 Feature Extraction Pipeline

Build an automated pipeline that reads DrugBank XML and produces the same
64-dim feature vector format the model expects.

**File: `data/pipeline/drugbank_parser.py`**

```python
class DrugBankParser:
    """
    Parse DrugBank XML → 64-dim pharmacological feature vectors.

    Maps DrugBank fields to the PharmLoop feature layout:
      Dims 0-9:   CYP enzyme interactions
      Dims 10-19: Receptor/target activity
      Dims 20-29: Pharmacokinetic parameters
      Dims 30-39: Drug class encoding
      Dims 40-49: Risk flags
      Dims 50-59: Transporter interactions
      Dims 60-63: Physical/chemical properties

    Outputs the same drugs.json format used by the model.
    """

    def __init__(self, drugbank_xml_path: str, target_drugs: list[str] | None = None):
        self.tree = ET.parse(drugbank_xml_path)
        self.root = self.tree.getroot()
        self.target_drugs = set(d.lower() for d in target_drugs) if target_drugs else None

    def extract_all(self) -> dict:
        """Extract features for target drugs (or all approved small molecules)."""
        drugs = {}
        for drug_elem in self.root.findall('.//drug'):
            if not self._is_approved_small_molecule(drug_elem):
                continue
            name = self._get_name(drug_elem).lower()
            if self.target_drugs and name not in self.target_drugs:
                continue
            features = self._extract_features(drug_elem)
            if features is not None:
                drugs[name] = {
                    "id": len(drugs),
                    "name": name,
                    "class": self._get_class(drug_elem),
                    "features": features,
                    "drugbank_id": self._get_id(drug_elem),
                }
        return drugs

    def _extract_features(self, drug_elem) -> list[float] | None:
        """Extract 64-dim feature vector from a drug element."""
        features = [0.0] * 64

        # Dims 0-9: CYP interactions
        enzymes = self._get_enzyme_interactions(drug_elem)
        features[0] = enzymes.get("CYP1A2", {}).get("inhibition", 0.0)
        features[1] = enzymes.get("CYP2C9", {}).get("inhibition", 0.0)
        features[2] = enzymes.get("CYP2C19", {}).get("inhibition", 0.0)
        features[3] = enzymes.get("CYP2D6", {}).get("inhibition", 0.0)
        features[4] = enzymes.get("CYP3A4", {}).get("inhibition", 0.0)
        features[5] = enzymes.get("CYP1A2", {}).get("induction", 0.0)
        features[6] = enzymes.get("CYP2C9", {}).get("induction", 0.0)
        features[7] = enzymes.get("CYP2C19", {}).get("induction", 0.0)
        features[8] = enzymes.get("CYP2D6", {}).get("substrate", 0.0)
        features[9] = enzymes.get("CYP3A4", {}).get("substrate", 0.0)

        # Dims 10-19: Receptor/target activity
        targets = self._get_target_activity(drug_elem)
        features[10] = targets.get("serotonin_reuptake", 0.0)
        features[11] = targets.get("norepinephrine_reuptake", 0.0)
        features[12] = targets.get("dopamine_reuptake", 0.0)
        features[13] = targets.get("mu_opioid", 0.0)
        features[14] = targets.get("gaba_a", 0.0)
        features[15] = targets.get("sodium_channel", 0.0)
        features[16] = targets.get("potassium_channel", 0.0)
        features[17] = targets.get("cox_inhibition", 0.0)
        features[18] = targets.get("ace_inhibition", 0.0)
        features[19] = targets.get("hmg_coa_inhibition", 0.0)

        # Dims 20-29: PK parameters (normalized 0-1)
        pk = self._get_pk_params(drug_elem)
        features[20] = self._normalize_half_life(pk.get("half_life_hours"))
        features[21] = pk.get("oral_bioavailability", 0.5)
        features[22] = self._normalize_vd(pk.get("volume_of_distribution"))
        features[23] = pk.get("protein_binding", 0.5)
        features[24] = self._normalize_clearance(pk.get("clearance"))
        features[25] = float(pk.get("renal_elimination", 0.0) > 0.3)
        features[26] = float(pk.get("hepatic_metabolism", 0.0) > 0.3)
        features[27] = pk.get("active_metabolites", 0.0)
        features[28] = self._narrow_therapeutic_index(drug_elem)
        features[29] = pk.get("nonlinear_pk", 0.0)

        # Dims 30-63: class encoding, risk flags, transporters, physical props
        # (similar extraction for each dimension block)

        return features
```

### 1.3 Interaction Extraction

```python
class InteractionExtractor:
    """
    Extract drug-drug interactions from DrugBank.

    Maps DrugBank interaction descriptions to PharmLoop severity and
    mechanism vocabulary using rule-based keyword classification.
    """

    MECHANISM_KEYWORDS = {
        "serotonin": "serotonergic",
        "serotonin syndrome": "serotonergic",
        "CYP3A4 inhibit": "cyp_inhibition",
        "CYP2D6 inhibit": "cyp_inhibition",
        "CYP3A4 induc": "cyp_induction",
        "QT prolong": "qt_prolongation",
        "bleeding": "bleeding_risk",
        "CNS depress": "cns_depression",
        "sedati": "cns_depression",
        "nephrotox": "nephrotoxicity",
        "hepatotox": "hepatotoxicity",
        "hypotens": "hypotension",
        "hyperkalem": "hyperkalemia",
        "seizure": "seizure_risk",
    }

    SEVERITY_MAP = {
        "contraindicated": "contraindicated",
        "major": "severe",
        "moderate": "moderate",
        "minor": "mild",
    }

    def extract(self, drugbank_interactions: list,
                known_drugs: set[str]) -> list[dict]:
        """
        Convert DrugBank interactions to PharmLoop format.
        Only includes interactions where BOTH drugs are in our drug set.
        """
        ...
```

### 1.4 Data Quality Controls

```python
@dataclass
class DataQualityReport:
    """Track data quality issues during extraction."""
    total_drugs: int
    drugs_with_complete_features: int
    drugs_with_fallback_values: int
    drugs_flagged_for_review: int
    total_interactions: int
    interactions_with_clear_mechanism: int
    interactions_with_ambiguous_mechanism: int
    severity_conflicts: int
    severity_conflict_resolutions: list[dict]
```

Quality rules:
- **Missing CYP data:** Default to 0.0 and flag. Don't invent data.
- **Missing PK params:** Use drug-class-average fallbacks. Log which drugs.
- **Ambiguous interactions:** Classify to most-likely mechanism for that
  drug class. Flag for manual review.
- **Severity conflicts:** Same pair, different severity from different
  sources → take the MORE SEVERE rating. Always conservative.

### 1.5 Validation: Original 50 Drugs Must Match

The pipeline must reproduce feature vectors for the original 50 drugs that
are close to the hand-curated Phase 1 values. Run a comparison:

```python
def validate_feature_continuity(original_drugs: dict, pipeline_drugs: dict,
                                 max_drift: float = 0.15) -> bool:
    """
    Verify pipeline-extracted features are close to hand-curated originals.

    Some drift is expected (different normalization, more precise values),
    but large deviations indicate a pipeline bug.
    """
    for name, original in original_drugs.items():
        if name not in pipeline_drugs:
            print(f"WARNING: {name} missing from pipeline output")
            continue
        orig_feat = torch.tensor(original["features"])
        pipe_feat = torch.tensor(pipeline_drugs[name]["features"])
        drift = (orig_feat - pipe_feat).abs().mean().item()
        if drift > max_drift:
            print(f"DRIFT: {name} mean feature drift = {drift:.3f} (max {max_drift})")
            return False
    return True
```

If drift exceeds threshold, either the pipeline has a bug or the hand-curated
values had errors. Investigate each case — don't just accept pipeline output
blindly.

### 1.6 Pipeline Output

```
data/
  raw/
    drugbank_full.xml            ← DrugBank XML (not committed)
    top_300_drugs.txt            ← Curated target drug list
  pipeline/
    drugbank_parser.py
    interaction_extractor.py
    quality_report.py
    run_pipeline.py              ← Orchestrator
  processed/
    drugs_v2.json                ← ~300 drugs with 64-dim features
    interactions_v2.json         ← ~1500 verified interactions
    quality_report.json
    manual_review_queue.json
    drugs.json                   ← Phase 1-3 original (KEEP for regression)
    interactions.json            ← Phase 1-3 original (KEEP for regression)
```

---

## Step 2: Encoder Scaling

### 2.1 Feature-Dominant Encoding

At 300 drugs, most drugs appear in only 5-15 interactions. Identity
embeddings for these drugs are undertrained. Scale identity embedding
influence by training frequency:

```python
class DrugEncoder(nn.Module):
    def __init__(self, num_drugs, feature_dim=64, min_appearances=5):
        super().__init__()
        self.identity_embedding = nn.Embedding(num_drugs, 256, padding_idx=0)
        self.feature_proj = nn.Linear(feature_dim, 256)
        self.fusion = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU(),
        )
        self.register_buffer("drug_counts", torch.zeros(num_drugs, dtype=torch.long))
        self.min_appearances = min_appearances

    def forward(self, drug_id, features):
        identity = self.identity_embedding(drug_id)
        feat_enc = self.feature_proj(features)

        # At inference: suppress identity for rarely-seen drugs
        if not self.training:
            counts = self.drug_counts[drug_id].float()
            identity_weight = torch.sigmoid((counts - self.min_appearances) / 2.0)
            identity = identity * identity_weight.unsqueeze(-1)

        fused = torch.cat([identity, feat_enc], dim=-1)
        return self.fusion(fused)
```

### 2.2 Embedding Table Sizing

Allocate headroom for future drug additions without retraining:

```python
num_drugs_allocated = max(1024, len(drugs) * 3)  # ~900 slots for ~300 drugs
```

### 2.3 Weight Transfer from Phase 3

The original 50 drugs keep their trained identity embeddings. New drugs
get random-initialized embeddings that are suppressed by the feature-dominant
gate until they accumulate enough training appearances.

```python
def transfer_encoder_weights(phase3_model, phase4a_model, original_drug_ids: dict):
    """
    Transfer Phase 3 encoder weights. Original drug embeddings are preserved;
    new drug embeddings are random-initialized.
    """
    # Copy feature_proj and fusion (architecture unchanged)
    phase4a_model.encoder.feature_proj.load_state_dict(
        phase3_model.encoder.feature_proj.state_dict()
    )
    phase4a_model.encoder.fusion.load_state_dict(
        phase3_model.encoder.fusion.state_dict()
    )

    # Copy identity embeddings for original drugs only
    with torch.no_grad():
        for name, orig_id in original_drug_ids.items():
            new_id = new_drug_registry[name]["id"]
            phase4a_model.encoder.identity_embedding.weight[new_id] = \
                phase3_model.encoder.identity_embedding.weight[orig_id]
```

---

## Step 3: Hierarchical Hopfield

### 3.1 Why Hierarchy

Phase 2 stored ~250 patterns. Phase 4a targets ~1500 interactions. With
severity amplification, that's ~1800 patterns. Softmax over 1800 keys
makes retrieval much softer than over 250 — every query retrieves a blurry
average rather than a sharp match.

Solution: two-level hierarchy. Class-specific banks keep retrieval sharp
within drug categories. Global bank catches cross-class interactions.

### 3.2 Drug Class Taxonomy

Define ~12 drug classes based on pharmacological category. Each drug can
belong to 1-2 classes:

```python
DRUG_CLASSES = [
    "ssri_snri",           # SSRIs, SNRIs, serotonergic agents
    "opioid",              # opioid agonists, partial agonists
    "anticoagulant",       # warfarin, DOACs, heparins, antiplatelets
    "antihypertensive",    # ACEi, ARB, CCB, beta blockers, diuretics
    "statin_lipid",        # statins, fibrates, lipid-lowering
    "antidiabetic",        # metformin, sulfonylureas, insulin, SGLT2i
    "antibiotic",          # fluoroquinolones, macrolides, azoles, etc.
    "antiepileptic",       # carbamazepine, phenytoin, valproate, etc.
    "immunosuppressant",   # cyclosporine, tacrolimus, mycophenolate
    "cardiac",             # antiarrhythmics, digoxin, nitrates
    "cns_psych",           # benzodiazepines, antipsychotics, lithium
    "nsaid_analgesic",     # NSAIDs, acetaminophen, non-opioid analgesics
]
```

### 3.3 Implementation

```python
class HierarchicalHopfield(nn.Module):
    """
    Two-level Hopfield memory for scaled drug interaction retrieval.

    Class-specific banks: sharp retrieval within drug categories (~100-200
    patterns each). Global bank: fallback for cross-class or novel pairs
    (all ~1800 patterns).

    The interface is the same as PharmHopfield — the OscillatorCell doesn't
    know it's talking to a hierarchy.
    """

    def __init__(self, input_dim: int, class_names: list[str],
                 class_capacity: int = 500, global_capacity: int = 5000):
        super().__init__()

        self.class_banks = nn.ModuleDict({
            name: PharmHopfield(input_dim, input_dim, max_capacity=class_capacity)
            for name in class_names
        })

        self.global_bank = PharmHopfield(input_dim, input_dim,
                                          max_capacity=global_capacity)

        # Learned gating: how much to trust class-specific vs global
        self.combine_gate = nn.Sequential(
            nn.Linear(input_dim * 2, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    @property
    def input_dim(self) -> int:
        return self.global_bank.input_dim

    @property
    def count(self) -> int:
        return self.global_bank.count

    def store(self, patterns: Tensor,
              drug_classes: list[tuple[str, str]] | None = None) -> None:
        """
        Store patterns in global bank AND relevant class banks.

        Args:
            patterns: (N, input_dim) pair patterns to store.
            drug_classes: Optional list of (class_a, class_b) for each pattern.
                          If provided, patterns are also stored in class banks.
        """
        # Always store in global
        self.global_bank.store(patterns)

        # Also store in class-specific banks
        if drug_classes is not None:
            for i, (cls_a, cls_b) in enumerate(drug_classes):
                pattern = patterns[i:i+1]
                for cls in set([cls_a, cls_b]):
                    if cls in self.class_banks:
                        self.class_banks[cls].store(pattern)

    def retrieve(self, query: Tensor, beta: float = 1.0,
                 drug_classes: tuple[str, str] | None = None) -> Tensor:
        """
        Hierarchical retrieval: class-specific + global, gated.

        If drug_classes not provided, falls back to global-only.
        """
        # Global retrieval (always)
        global_retrieved = self.global_bank.retrieve(query, beta=beta)

        if drug_classes is None:
            return global_retrieved

        # Class-specific retrieval
        class_a, class_b = drug_classes
        class_retrievals = []
        for cls in set([class_a, class_b]):
            if cls in self.class_banks and self.class_banks[cls].count > 0:
                class_retrievals.append(
                    self.class_banks[cls].retrieve(query, beta=beta)
                )

        if not class_retrievals:
            return global_retrieved

        class_retrieved = torch.stack(class_retrievals).mean(dim=0)

        # Learned gating
        combined = torch.cat([class_retrieved, global_retrieved], dim=-1)
        gate = self.combine_gate(combined)
        return gate * class_retrieved + (1 - gate) * global_retrieved

    def clear(self) -> None:
        self.global_bank.clear()
        for bank in self.class_banks.values():
            bank.clear()
```

### 3.4 Integration with OscillatorCell

The OscillatorCell currently calls `self.hopfield.retrieve(query, beta=...)`.
The HierarchicalHopfield needs drug class information passed through.

Two options:
1. Pass drug classes through the entire forward path (invasive)
2. Store drug classes as context on the model and read them in the cell

Go with option 2 — minimal invasion:

```python
# In PharmLoopModel.forward, before running the reasoning loop:
if isinstance(self.cell.hopfield, HierarchicalHopfield):
    # Store drug classes as temporary context for the oscillator
    drug_a_class = self.drug_registry[drug_a_id]["class"]
    drug_b_class = self.drug_registry[drug_b_id]["class"]
    self.cell.hopfield._current_classes = (drug_a_class, drug_b_class)

# In OscillatorCell.forward, modify the retrieve call:
if hasattr(self.hopfield, '_current_classes'):
    retrieved = self.hopfield.retrieve(
        query, beta=beta.item(),
        drug_classes=self.hopfield._current_classes,
    )
else:
    retrieved = self.hopfield.retrieve(query, beta=beta.item())
```

Not elegant, but it avoids changing the OscillatorCell signature and keeps
backward compatibility with the flat PharmHopfield.

### 3.5 Hopfield Build Protocol

```python
def build_hierarchical_hopfield(
    model: PharmLoopModel,
    interactions: list[dict],
    drug_registry: dict,
) -> HierarchicalHopfield:
    """
    Build hierarchical Hopfield from trained encoder.
    Same as Phase 2 rebuild, but stores into class banks too.
    """
    hopfield = HierarchicalHopfield(
        input_dim=512,
        class_names=DRUG_CLASSES,
    )

    # Identity-init projections on all banks
    for bank in [hopfield.global_bank] + list(hopfield.class_banks.values()):
        if hasattr(bank, 'query_proj') and isinstance(bank.query_proj, nn.Linear):
            with torch.no_grad():
                nn.init.eye_(bank.query_proj.weight)
                nn.init.zeros_(bank.query_proj.bias)
                nn.init.eye_(bank.key_proj.weight)
                nn.init.zeros_(bank.key_proj.bias)

    # Compute and store pair patterns
    model.eval()
    patterns = []
    classes = []
    with torch.no_grad():
        for interaction in interactions:
            pair_state = model.compute_pair_state(interaction)
            patterns.append(pair_state)
            cls_a = drug_registry[interaction["drug_a"]]["class"]
            cls_b = drug_registry[interaction["drug_b"]]["class"]
            classes.append((cls_a, cls_b))

            # Severity amplification
            sev = interaction["severity"]
            if sev in ("severe", "contraindicated"):
                for _ in range(2):
                    noisy = pair_state + torch.randn_like(pair_state) * 0.01
                    patterns.append(noisy)
                    classes.append((cls_a, cls_b))

    patterns_tensor = torch.stack(patterns)
    hopfield.store(patterns_tensor, drug_classes=classes)

    return hopfield
```

---

## Step 4: Basic Polypharmacy

### 4.1 Scope for Phase 4a

Full polypharmacy analysis with all pattern types is Phase 4b.
Phase 4a implements:

1. **Pairwise scan** — check all (n choose 2) pairs, batched for speed
2. **Three additive patterns** — the most dangerous and most common:
   - Additive serotonergic risk (multiple serotonergics)
   - Additive QT prolongation (multiple QT prolongers)
   - Additive CYP inhibition (multiple inhibitors of same enzyme)
3. **Ranked report** — severity-ranked pairwise results + alerts

That's it. No cascade detection, no renal risk chains, no complex
multi-drug reasoning. Those need more validation data and pharmacist
feedback to get right.

### 4.2 Implementation

```python
class BasicPolypharmacyAnalyzer:
    """
    Phase 4a polypharmacy: pairwise scan + three additive patterns.

    Zero learned parameters. Rule-based detection of the three most
    dangerous multi-drug patterns.
    """

    ADDITIVE_PATTERNS = {
        "additive_serotonergic": {
            "trigger_mechanism": "serotonergic",
            "min_count": 2,  # 2+ serotonergic pairs → alert
            "alert": (
                "Multiple serotonergic agents detected in this medication list. "
                "Risk of serotonin syndrome is significantly elevated with "
                "multiple serotonergic drugs. Consider reducing serotonergic "
                "burden or increasing monitoring frequency."
            ),
            "severity_bump": True,  # bump highest serotonergic pair by one level
        },
        "additive_qt_prolongation": {
            "trigger_mechanism": "qt_prolongation",
            "min_count": 2,
            "alert": (
                "Multiple QT-prolonging agents detected. Combined QT "
                "prolongation risk exceeds the sum of individual risks. "
                "Obtain baseline ECG and monitor QTc interval closely. "
                "Consider alternatives where possible."
            ),
            "severity_bump": True,
        },
        "additive_cyp_inhibition": {
            "trigger_mechanism": "cyp_inhibition",
            "min_count": 2,
            "alert": (
                "Multiple CYP inhibitors detected. Combined enzyme inhibition "
                "may significantly increase substrate drug levels beyond what "
                "any single inhibitor would cause. Monitor drug levels for "
                "all CYP substrates in this medication list."
            ),
            "severity_bump": True,
        },
    }

    def analyze(
        self,
        drug_names: list[str],
        pairwise_results: dict[tuple[str, str], InteractionResult],
    ) -> PolypharmacyReport:
        """
        Analyze pairwise results for the three additive patterns.
        """
        # Count mechanism occurrences across all pairs
        mechanism_counts: dict[str, int] = {}
        mechanism_pairs: dict[str, list[tuple[str, str]]] = {}
        for pair, result in pairwise_results.items():
            for mech in result.mechanisms:
                mechanism_counts[mech] = mechanism_counts.get(mech, 0) + 1
                mechanism_pairs.setdefault(mech, []).append(pair)

        # Check patterns
        alerts = []
        for pattern_name, pattern in self.ADDITIVE_PATTERNS.items():
            trigger_mech = pattern["trigger_mechanism"]
            count = mechanism_counts.get(trigger_mech, 0)
            if count >= pattern["min_count"]:
                involved_pairs = mechanism_pairs.get(trigger_mech, [])
                involved_drugs = set()
                for a, b in involved_pairs:
                    involved_drugs.add(a)
                    involved_drugs.add(b)

                alerts.append(MultiDrugAlert(
                    pattern=pattern_name,
                    alert_text=pattern["alert"],
                    involved_drugs=sorted(involved_drugs),
                    involved_pairs=involved_pairs,
                    trigger_mechanism=trigger_mech,
                    pair_count=count,
                ))

        # Rank pairwise results by severity × confidence
        ranked_pairs = sorted(
            pairwise_results.items(),
            key=lambda x: (
                SEVERITY_ORDER.get(x[1].severity, 0),
                x[1].confidence,
            ),
            reverse=True,
        )

        highest = ranked_pairs[0][1].severity if ranked_pairs else "none"

        return PolypharmacyReport(
            drugs=drug_names,
            total_pairs_checked=len(pairwise_results),
            highest_severity=highest,
            pairwise_results=ranked_pairs,
            multi_drug_alerts=alerts,
        )


@dataclass
class MultiDrugAlert:
    pattern: str
    alert_text: str
    involved_drugs: list[str]
    involved_pairs: list[tuple[str, str]]
    trigger_mechanism: str
    pair_count: int


@dataclass
class PolypharmacyReport:
    drugs: list[str]
    total_pairs_checked: int
    highest_severity: str
    pairwise_results: list[tuple[tuple[str, str], InteractionResult]]
    multi_drug_alerts: list[MultiDrugAlert]
```

### 4.3 Batched Multi-Drug Inference

For speed, batch all pairs into a single forward pass:

```python
def check_multiple_batched(
    self,
    drug_names: list[str],
    context: dict | None = None,
) -> PolypharmacyReport:
    """
    Check all pairwise interactions in one batched forward pass.

    10 drugs = 45 pairs. Batched: one forward pass, ~100ms.
    Sequential: 45 forward passes, ~2.2 seconds.
    """
    from itertools import combinations
    pairs = list(combinations(drug_names, 2))

    # Stack all pair tensors
    a_ids, a_feats, b_ids, b_feats = [], [], [], []
    valid_pairs = []
    for drug_a, drug_b in pairs:
        da = self.drug_registry.get(drug_a.lower())
        db = self.drug_registry.get(drug_b.lower())
        if da is None or db is None:
            continue  # skip unknown drugs, handle separately
        a_ids.append(da["id"])
        a_feats.append(da["features"])
        b_ids.append(db["id"])
        b_feats.append(db["features"])
        valid_pairs.append((drug_a, drug_b))

    if not valid_pairs:
        return PolypharmacyReport(drugs=drug_names, total_pairs_checked=0,
                                  highest_severity="unknown",
                                  pairwise_results=[], multi_drug_alerts=[])

    # Batch forward pass
    a_ids_t = torch.tensor(a_ids, dtype=torch.long)
    a_feats_t = torch.tensor(a_feats, dtype=torch.float32)
    b_ids_t = torch.tensor(b_ids, dtype=torch.long)
    b_feats_t = torch.tensor(b_feats, dtype=torch.float32)

    with torch.no_grad():
        output = self.model(a_ids_t, a_feats_t, b_ids_t, b_feats_t)

    # Unbatch into individual InteractionResults
    pairwise_results = {}
    for i, (drug_a, drug_b) in enumerate(valid_pairs):
        result = self._unbatch_single(output, i, drug_a, drug_b)
        pairwise_results[(drug_a, drug_b)] = result

    # Run polypharmacy pattern detection
    return self.polypharmacy_analyzer.analyze(drug_names, pairwise_results)
```

---

## Step 5: REST API

### 5.1 Endpoints

Minimal but complete:

```
POST /check          → single pair interaction check
POST /check-multiple → polypharmacy check (2-20 drugs)
GET  /drugs          → list available drugs
GET  /health         → service health + model info
```

### 5.2 Implementation

**File: `api/server.py`**

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="PharmLoop", version="0.4a.0",
              description="Drug interaction checking with oscillatory reasoning")

engine: PharmLoopInference = None

@app.on_event("startup")
def load_model():
    global engine
    engine = PharmLoopInference.load(
        checkpoint_path=os.environ["PHARMLOOP_CHECKPOINT"],
        data_dir=os.environ.get("PHARMLOOP_DATA_DIR", "data/processed"),
    )

class PairCheckRequest(BaseModel):
    drug_a: str
    drug_b: str
    context: dict | None = None

class PairCheckResponse(BaseModel):
    drug_a: str
    drug_b: str
    severity: str
    mechanisms: list[str]
    flags: list[str]
    confidence: float
    converged: bool
    steps: int
    narrative: str
    partial_convergence: dict | None = None
    unknown_drugs: list[str] | None = None

class MultiCheckRequest(BaseModel):
    drugs: list[str]  # 2-20 drug names
    context: dict | None = None

class MultiCheckResponse(BaseModel):
    drugs: list[str]
    total_pairs_checked: int
    highest_severity: str
    pairwise_results: list[PairCheckResponse]
    multi_drug_alerts: list[dict]

@app.post("/check", response_model=PairCheckResponse)
def check_pair(req: PairCheckRequest):
    result = engine.check(req.drug_a, req.drug_b, context=req.context)
    return PairCheckResponse(
        drug_a=result.drug_a, drug_b=result.drug_b,
        severity=result.severity, mechanisms=result.mechanisms,
        flags=result.flags, confidence=result.confidence,
        converged=result.converged, steps=result.steps,
        narrative=result.narrative,
        partial_convergence=result.partial_convergence,
        unknown_drugs=result.unknown_drugs,
    )

@app.post("/check-multiple", response_model=MultiCheckResponse)
def check_multiple(req: MultiCheckRequest):
    if len(req.drugs) < 2:
        raise HTTPException(400, "Need at least 2 drugs")
    if len(req.drugs) > 20:
        raise HTTPException(400, "Maximum 20 drugs per request")
    report = engine.check_multiple(req.drugs, context=req.context)
    return MultiCheckResponse(
        drugs=report.drugs,
        total_pairs_checked=report.total_pairs_checked,
        highest_severity=report.highest_severity,
        pairwise_results=[...],  # convert each InteractionResult
        multi_drug_alerts=[alert.__dict__ for alert in report.multi_drug_alerts],
    )

@app.get("/drugs")
def list_drugs():
    return {
        "drugs": sorted(engine.drug_registry.keys()),
        "count": len(engine.drug_registry),
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": engine is not None,
        "num_drugs": len(engine.drug_registry),
        "hopfield_patterns": engine.model.cell.hopfield.count,
        "version": "0.4a.0",
    }
```

### 5.3 Docker Deployment

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PHARMLOOP_CHECKPOINT=/app/checkpoints/phase4a_best.pt
ENV PHARMLOOP_DATA_DIR=/app/data/processed
EXPOSE 8000
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Step 6: Evaluation Framework

### 6.1 Metrics

```python
class EvaluationSuite:
    """Comprehensive evaluation against held-out test data."""

    def evaluate(self, engine: PharmLoopInference,
                 test_data: list[dict]) -> EvaluationReport:
        metrics = {}

        # Safety (non-negotiable)
        metrics["false_negative_rate_severe"] = self._fnr_severe(engine, test_data)

        # Accuracy
        metrics["severity_accuracy_exact"] = self._severity_exact(engine, test_data)
        metrics["severity_accuracy_within_one"] = self._severity_within_one(engine, test_data)
        metrics["mechanism_accuracy_top1"] = self._mech_top1(engine, test_data)

        # Calibration
        metrics["confidence_calibration_error"] = self._calibration(engine, test_data)

        # Unknown handling
        metrics["unknown_detection_rate"] = self._unknown_detection(engine, test_data)

        # Performance
        metrics["avg_convergence_steps_known"] = self._avg_steps(engine, test_data, known=True)
        metrics["single_pair_latency_ms"] = self._latency_single(engine)
        metrics["ten_drug_polypharmacy_latency_ms"] = self._latency_poly(engine)

        return EvaluationReport(metrics=metrics)
```

### 6.2 Regression Tests

Every test from Phases 1-3 must still pass:

```python
class RegressionTests:
    """Ensure Phase 4a didn't break anything from earlier phases."""

    def test_three_way_separation(self, engine):
        """Original three-way test still works."""
        severe = engine.check("fluoxetine", "tramadol")
        safe = engine.check("metformin", "lisinopril")
        unknown = engine.check("QZ-7734", "aspirin")

        assert severe.severity in ("severe", "contraindicated")
        assert safe.severity == "none"
        assert unknown.severity == "unknown"
        assert severe.confidence > unknown.confidence
        assert safe.confidence > unknown.confidence

    def test_original_50_drugs_present(self, engine):
        """All original 50 drugs are in the registry."""
        original_drugs = load_original_drugs()
        for name in original_drugs:
            assert name in engine.drug_registry, f"Original drug {name} missing"

    def test_narrative_output_format(self, engine):
        """Template engine still produces well-formed narratives."""
        result = engine.check("fluoxetine", "tramadol")
        assert "serotonin" in result.narrative.lower()
        assert "[Confidence:" in result.narrative
        assert "{" not in result.narrative  # no unresolved template slots
```

### 6.3 Stress Tests

```python
class StressTests:

    def test_same_drug_pair(self, engine):
        """Drug paired with itself."""
        result = engine.check("warfarin", "warfarin")
        # Should not crash. Severity is implementation-dependent.

    def test_unknown_drug(self, engine):
        """Drug not in registry."""
        result = engine.check("madeupdrugxyz", "aspirin")
        assert result.severity == "unknown"
        assert result.unknown_drugs == ["madeupdrugxyz"]

    def test_single_pair_latency(self, engine):
        """Single pair < 100ms on CPU."""
        import time
        start = time.perf_counter()
        for _ in range(100):
            engine.check("fluoxetine", "tramadol")
        avg_ms = (time.perf_counter() - start) / 100 * 1000
        assert avg_ms < 100, f"Latency {avg_ms:.0f}ms > 100ms target"

    def test_polypharmacy_latency(self, engine):
        """10-drug polypharmacy (batched) < 500ms."""
        drugs = ["fluoxetine", "tramadol", "warfarin", "metformin",
                 "lisinopril", "omeprazole", "amlodipine", "simvastatin",
                 "metoprolol", "acetaminophen"]
        import time
        start = time.perf_counter()
        engine.check_multiple(drugs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 500, f"Polypharmacy latency {elapsed_ms:.0f}ms > 500ms"

    def test_twenty_drug_polypharmacy(self, engine):
        """20-drug polypharmacy = 190 pairs. Should still complete."""
        drugs = list(engine.drug_registry.keys())[:20]
        result = engine.check_multiple(drugs)
        assert result.total_pairs_checked == 190
```

---

## Step 7: Training Protocol

### 7.1 Configuration

```python
config = {
    "epochs": 50,
    "batch_size": 32,
    "lr": 5e-4,
    "lr_schedule": "cosine",
    "warmup_epochs": 5,
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "test_ratio": 0.1,
    "anneal_interval": 15,
    "max_anneal_cycles": 3,
}
```

### 7.2 Curriculum Learning

Not all interactions are equally well-documented. Train in two stages:

1. **High-confidence first (15 epochs):** Only pairs with clear mechanism
   mapping and no severity conflicts. Establishes good Hopfield attractors.
2. **All interactions (35 epochs):** Add ambiguous and singly-sourced pairs.
   The model has good priors from stage 1.

### 7.3 Hopfield Annealing (Same as Phase 2)

Rebuild the hierarchical Hopfield every 15 epochs with the current encoder:
```
Cycle 1: Build Hopfield → Train 15 epochs
Cycle 2: Rebuild Hopfield → Train 15 epochs
Cycle 3: Rebuild Hopfield → Train 15 epochs
Stop when pattern drift < 0.01
```

---

## Implementation Files

### New Files
```
data/
  raw/
    top_300_drugs.txt
  pipeline/
    drugbank_parser.py
    interaction_extractor.py
    quality_report.py
    run_pipeline.py
  processed/
    drugs_v2.json
    interactions_v2.json
    quality_report.json

pharmloop/
    hierarchical_hopfield.py
    polypharmacy.py

api/
    server.py
    Dockerfile

training/
    train_phase4a.py
    evaluate.py

tests/
    test_pipeline.py
    test_hierarchical_hopfield.py
    test_polypharmacy.py
    test_api.py
    test_evaluation.py
    test_stress.py
    test_regression.py
```

### Modified Files
```
pharmloop/encoder.py          ← drug_counts buffer, feature-dominant scaling
pharmloop/model.py            ← HierarchicalHopfield integration, drug class context
pharmloop/inference.py        ← check_multiple batched, polypharmacy analyzer
training/data_loader.py       ← v2 data format, curriculum support
```

---

## Implementation Sequence

### 7.1 Data pipeline (do first)
1. Curate top_300_drugs.txt drug list
2. Implement DrugBank parser
3. Implement interaction extractor
4. Run pipeline → drugs_v2.json + interactions_v2.json
5. Run quality report, review flagged items
6. Validate original 50 drugs match

### 7.2 Encoder + Hopfield scaling
7. Add feature-dominant encoding to encoder.py
8. Implement HierarchicalHopfield
9. Implement Hopfield build protocol with class routing
10. Transfer Phase 3 weights, verify backward compat

### 7.3 Training
11. Implement train_phase4a.py with curriculum
12. Train on expanded dataset (50 epochs, 3 annealing cycles)
13. Run evaluation suite

### 7.4 Polypharmacy
14. Implement BasicPolypharmacyAnalyzer
15. Implement batched check_multiple
16. Test the three additive patterns

### 7.5 API
17. Implement FastAPI server
18. Implement all endpoints
19. Docker build
20. Integration tests

### 7.6 Evaluation
21. Run full evaluation suite
22. Run regression tests
23. Run stress tests
24. Generate evaluation report for pharmacist review

---

## Validation Criteria (must pass before Phase 4b)

### Data
- [ ] >= 280 drugs extracted with valid features
- [ ] >= 1200 interactions with severity and mechanism labels
- [ ] Quality report: < 10% ambiguous mechanisms
- [ ] Original 50 drugs present with features within drift threshold

### Model
- [ ] Severity accuracy >= 80% on expanded test set
- [ ] Zero false negatives on severe/contraindicated
- [ ] Mechanism accuracy >= 50% on expanded test set
- [ ] Unknown detection rate >= 85%
- [ ] Param budget < 10M

### Polypharmacy
- [ ] check_multiple works for 2-20 drugs
- [ ] Detects additive serotonergic pattern
- [ ] Detects additive QT prolongation pattern
- [ ] Detects additive CYP inhibition pattern

### Performance
- [ ] Single pair inference < 100ms on CPU
- [ ] 10-drug polypharmacy (batched) < 500ms on CPU
- [ ] API endpoints respond correctly

### Regression
- [ ] Three-way separation test passes
- [ ] All original 50 drugs queryable
- [ ] Template engine output well-formed
- [ ] No severity accuracy regression on original test set

---

## What NOT to Do in Phase 4a

- Don't try to parse all of DrugBank — target 300, not 2500
- Don't implement complex multi-drug patterns (cascade, renal chain) — Phase 4b
- Don't build a frontend UI — the API is the deliverable
- Don't fully train the context encoder on expanded data — Phase 4b
- Don't optimize for production deployment (load balancing, caching) — Phase 4b
- Don't exceed the 10M param budget
- Don't remove the original dataset — keep for regression testing

---

## What Phase 4a Enables

With a working API over 300 drugs, you can:

1. **Get pharmacist feedback.** Show real interaction checks to real pharmacists.
   Where does PharmLoop agree with their intuition? Where does it disagree?
   What interactions is it missing? What's the narrative quality?

2. **Identify data gaps.** Which drug classes have sparse interaction data?
   Where are the mechanism labels unreliable? Which severity ratings need
   pharmacist review?

3. **Prioritize Phase 4b work.** Maybe polypharmacy patterns matter more than
   expanding to 500 drugs. Maybe the context encoder matters more than
   hierarchical Hopfield tuning. Real feedback tells you where to invest.

4. **Demo the architecture.** The oscillatory reasoning — convergence speed,
   gray zone trajectories, partial convergence, UNKNOWN for fabricated drugs
   — is visible through the API. People can see that this is not a lookup table.
