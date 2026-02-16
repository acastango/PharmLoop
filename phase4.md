# Phase 4 — Drug Set Expansion, Polypharmacy, API, and Evaluation

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

## What Phase 4 Accomplishes

Phases 1-3 proved the architecture, built the Hopfield, added clinical output,
and established the inference pipeline — all on 50 drugs and 209 interactions.
Phase 4 scales to real-world pharmacology:

- **500+ drugs** from DrugBank with automated feature extraction
- **2000+ verified interactions** from DrugBank + FDA + clinical literature
- **Polypharmacy reasoning** — checking N drugs simultaneously, not just pairs
- **REST API** for clinical system integration
- **Evaluation framework** against gold-standard interaction databases
- **Hopfield at scale** — managing thousands of patterns efficiently

After Phase 4, PharmLoop is a deployable service that a clinical system
can call to check drug interactions in real time with auditable reasoning.

---

## Step 1: DrugBank Data Pipeline

### 1.1 The Problem

The current 50 drugs and their 64-dim feature vectors were hand-curated.
This doesn't scale. DrugBank (open-access subset) contains ~2500 approved
drugs with structured pharmacological data that maps directly to our
feature dimensions.

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

    def __init__(self, drugbank_xml_path: str):
        self.tree = ET.parse(drugbank_xml_path)
        self.root = self.tree.getroot()

    def extract_all(self) -> dict:
        """Extract features for all approved small-molecule drugs."""
        drugs = {}
        for drug_elem in self.root.findall('.//drug'):
            if not self._is_approved_small_molecule(drug_elem):
                continue
            name = self._get_name(drug_elem).lower()
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

        # ... dims 30-63: class encoding, risk flags, transporters, physical props
        # (similar extraction for each dimension block)

        return features
```

### 1.3 Interaction Extraction

DrugBank contains pairwise interaction data with severity levels and
mechanism descriptions. Map these to PharmLoop's vocabulary:

```python
class InteractionExtractor:
    """
    Extract drug-drug interactions from DrugBank.

    Maps DrugBank interaction descriptions to PharmLoop severity and
    mechanism vocabulary using rule-based classification.

    NOT an LLM — uses keyword matching against a curated mapping table.
    """

    # DrugBank description keywords → PharmLoop mechanism
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
        # ... comprehensive keyword table
    }

    # DrugBank severity terms → PharmLoop severity
    SEVERITY_MAP = {
        "contraindicated": "contraindicated",
        "major": "severe",
        "moderate": "moderate",
        "minor": "mild",
    }

    def extract(self, drugbank_interactions: list) -> list[dict]:
        """Convert DrugBank interaction records to PharmLoop format."""
        ...
```

### 1.4 Data Quality Controls

Not all DrugBank entries are complete. The pipeline must flag and handle:

- **Missing CYP data:** Default to 0.0 (unknown, not "no interaction").
  Flag these drugs so the model knows features are incomplete.
- **Missing PK parameters:** Use drug-class-average values as fallback.
  Log which drugs used fallback values.
- **Ambiguous interactions:** DrugBank descriptions that don't clearly map
  to a mechanism get classified as the generic most-likely mechanism for
  that drug class. Flag for manual review.
- **Duplicate interactions:** Same pair from multiple sources with
  conflicting severity → take the MORE SEVERE rating. Conservative.

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
    severity_conflicts: int  # same pair, different severity from different sources
    severity_conflict_resolutions: list[dict]  # how each conflict was resolved
```

### 1.5 Pipeline Output

```
data/
  raw/
    drugbank_full.xml            ← DrugBank XML (not committed, downloaded)
  pipeline/
    drugbank_parser.py           ← XML → feature vectors
    interaction_extractor.py     ← XML → interaction pairs
    quality_report.py            ← Data quality analysis
    run_pipeline.py              ← Orchestrator: XML → drugs.json + interactions.json
  processed/
    drugs_v2.json                ← 500+ drugs with 64-dim features
    interactions_v2.json         ← 2000+ verified interactions
    quality_report.json          ← Data quality metrics
    manual_review_queue.json     ← Flagged items needing pharmacist review
    drugs.json                   ← Phase 1-3 original (kept for regression testing)
    interactions.json            ← Phase 1-3 original (kept for regression testing)
```

---

## Step 2: Embedding and Encoder Scaling

### 2.1 The Problem

The current identity embedding table is `nn.Embedding(num_drugs, 256)`.
At 50 drugs that's 12.8K params. At 500 drugs that's 128K. At 2500 it's
640K. Still within budget, but identity embeddings for hundreds of drugs
will be undertrained — most drugs appear in only a handful of interactions.

### 2.2 Solution: Feature-Dominant Encoding with Sparse Identity

For drugs with few training interactions (< 5 pair appearances), the
identity embedding is unreliable noise. The 64-dim structured features
ARE the identity for sparse drugs — that's the whole point of having them.

Modify the encoder to scale identity embedding influence by training
frequency:

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

        # Track how many times each drug appears in training
        # Used to scale identity embedding influence
        self.register_buffer("drug_counts", torch.zeros(num_drugs, dtype=torch.long))
        self.min_appearances = min_appearances

    def forward(self, drug_id, features):
        identity = self.identity_embedding(drug_id)
        feat_enc = self.feature_proj(features)

        # Scale identity embedding by confidence based on training appearances
        # Drugs seen < min_appearances times: rely more on features
        if not self.training:
            counts = self.drug_counts[drug_id].float()  # (batch,)
            # Sigmoid ramp: 0 at 0 appearances, ~1 at 10+ appearances
            identity_weight = torch.sigmoid((counts - self.min_appearances) / 2.0)
            identity = identity * identity_weight.unsqueeze(-1)

        fused = torch.cat([identity, feat_enc], dim=-1)
        return self.fusion(fused)
```

During training, update the count buffer:
```python
# In training loop:
for drug_id in batch_drug_a_ids:
    model.encoder.drug_counts[drug_id] += 1
for drug_id in batch_drug_b_ids:
    model.encoder.drug_counts[drug_id] += 1
```

### 2.3 Embedding Table Size

Reserve embedding slots for future growth:

```python
# Allocate 4x current drug count for growth headroom
num_drugs_allocated = max(2048, len(drugs) * 4)
```

New drugs added via the growth protocol get the next available ID.
Embedding is random-initialized → the identity_weight ramp suppresses
it until the drug accumulates enough training appearances.

---

## Step 3: Hopfield at Scale

### 3.1 Capacity Planning

Phase 2 stored ~250 pair patterns. Phase 4 targets 2000+ interactions.
With severity amplification (2 extra copies for severe/contraindicated):

```
Base interactions:        2000
Severe amplification:     ~400 (assume 20% severe × 2 extra copies)
Total patterns:           ~2400
Capacity used:            2400 / 5000 = 48%
```

Still well within capacity. But retrieval quality degrades as the bank
fills — softmax over 2400 keys is much softer than over 250. Two mitigations:

### 3.2 Hierarchical Retrieval

Instead of one flat Hopfield, organize into a two-level hierarchy:

```
Level 1: Drug-class Hopfield banks (coarse)
  - SSRI interactions:           ~150 patterns
  - Opioid interactions:         ~120 patterns
  - Anticoagulant interactions:  ~100 patterns
  - Antihypertensive interactions: ~80 patterns
  - ... ~15 class-based banks

Level 2: Global Hopfield (fine)
  - All 2400 patterns
  - Used as fallback when class-based retrieval is ambiguous
```

At query time:
1. Determine the drug classes of both drugs in the pair
2. Retrieve from the relevant class-specific bank(s)
3. If the pair spans two classes (e.g., SSRI + opioid), retrieve from both
4. If no class-specific bank matches, fall back to global bank
5. Combine retrievals with learned weighting

```python
class HierarchicalHopfield(nn.Module):
    """
    Two-level Hopfield memory for scaled drug interaction retrieval.

    Class-specific banks give sharper retrieval for within-class interactions.
    Global bank provides fallback for cross-class or novel combinations.
    """

    def __init__(self, input_dim: int, class_names: list[str],
                 class_capacity: int = 500, global_capacity: int = 5000):
        super().__init__()

        # Per-class Hopfield banks
        self.class_banks = nn.ModuleDict({
            name: PharmHopfield(input_dim, input_dim, max_capacity=class_capacity)
            for name in class_names
        })

        # Global fallback bank
        self.global_bank = PharmHopfield(input_dim, input_dim,
                                         max_capacity=global_capacity)

        # Learned weighting between class-specific and global retrieval
        self.combine_gate = nn.Sequential(
            nn.Linear(input_dim * 2, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def retrieve(self, query: Tensor, drug_classes: tuple[str, str],
                 beta: float = 1.0) -> Tensor:
        """
        Hierarchical retrieval: class-specific first, global fallback.
        """
        class_a, class_b = drug_classes

        # Retrieve from relevant class banks
        class_retrievals = []
        for cls in set([class_a, class_b]):
            if cls in self.class_banks and self.class_banks[cls].count > 0:
                class_retrievals.append(
                    self.class_banks[cls].retrieve(query, beta=beta)
                )

        if class_retrievals:
            # Average class-specific retrievals
            class_retrieved = torch.stack(class_retrievals).mean(dim=0)
        else:
            class_retrieved = torch.zeros_like(query)

        # Global retrieval
        global_retrieved = self.global_bank.retrieve(query, beta=beta)

        # Learned gating between class-specific and global
        combined_input = torch.cat([class_retrieved, global_retrieved], dim=-1)
        gate = self.combine_gate(combined_input)
        retrieved = gate * class_retrieved + (1 - gate) * global_retrieved

        return retrieved
```

### 3.3 Incremental Hopfield Rebuild

With 2000+ interactions, rebuilding the entire Hopfield from scratch after
each annealing cycle is expensive. Instead, use incremental updates:

```python
def incremental_hopfield_rebuild(model, new_interactions, batch_size=64):
    """
    Update Hopfield patterns for new/changed interactions only.
    Don't rebuild the entire bank.
    """
    model.eval()
    with torch.no_grad():
        for batch in chunked(new_interactions, batch_size):
            # Compute pair states
            pair_states = model.compute_pair_state_batch(batch)
            # Find existing patterns for these pairs (by pair index)
            # Replace in-place rather than append
            for i, (pair, state) in enumerate(zip(batch, pair_states)):
                existing_idx = hopfield_pattern_registry.get(pair.key)
                if existing_idx is not None:
                    model.hopfield.stored_keys[existing_idx] = state
                    model.hopfield.stored_values[existing_idx] = state
                else:
                    model.hopfield.store(state.unsqueeze(0))
                    hopfield_pattern_registry[pair.key] = model.hopfield.count - 1
```

---

## Step 4: Polypharmacy

### 4.1 The Problem

Real patients take 5-15 medications. Checking all pairs is O(n²) — for
10 drugs that's 45 pairs. But multi-drug interactions are more than the
sum of pairwise interactions. Three drugs together can create emergent
risks that no pair exhibits alone:

- Drug A inhibits CYP3A4 (mild alone)
- Drug B is a CYP3A4 substrate (no issue by itself)
- Drug C also inhibits CYP3A4 (mild alone)
- A + B = moderate (elevated B levels)
- C + B = moderate (elevated B levels)
- A + C + B = SEVERE (A + C together strongly inhibit CYP3A4, dramatically
  elevating B levels)

### 4.2 Approach: Pairwise Base + Multi-Drug Modifiers

The oscillatory pair-checking is the foundation. Polypharmacy adds a
second pass that identifies multi-drug interaction patterns:

```
Phase 1: Pairwise scan
  For all (n choose 2) drug pairs:
    Run PharmLoop inference → collect severity, mechanisms, confidence
  Result: pairwise interaction matrix

Phase 2: Multi-drug pattern detection
  Analyze the pairwise matrix for dangerous patterns:
    - Additive CYP inhibition (multiple inhibitors of same enzyme)
    - Additive receptor effects (multiple serotonergics)
    - Cascade interactions (A inhibits metabolism of B, B inhibits metabolism of C)
    - Shared risk factors (multiple QT prolongers, multiple bleeders)
  Result: multi-drug alerts on top of pairwise results

Phase 3: Aggregate risk report
  Combine pairwise + multi-drug into a single clinical report
  Rank all interactions by severity × confidence
  Flag multi-drug patterns as separate alerts
```

### 4.3 Multi-Drug Pattern Detector

```python
class PolypharmacyAnalyzer:
    """
    Detects multi-drug interaction patterns from pairwise results.

    Zero learned parameters. Rule-based pattern matching against
    known dangerous multi-drug scenarios.

    This is NOT trying to predict novel multi-drug interactions.
    It's identifying KNOWN dangerous patterns (additive CYP inhibition,
    serotonin syndrome from 3+ serotonergics, etc.) that are worse
    than any single pair would suggest.
    """

    # Known dangerous multi-drug patterns
    MULTI_DRUG_PATTERNS = {
        "additive_cyp_inhibition": {
            "description": "Multiple inhibitors of the same CYP enzyme",
            "trigger": lambda mechs: sum(1 for m in mechs if "cyp_inhibition" in m) >= 2,
            "severity_modifier": +1,  # bump severity by one level
            "alert": (
                "Multiple CYP inhibitors present. Combined inhibition may be "
                "significantly greater than any single inhibitor. Consider "
                "therapeutic drug monitoring for all CYP substrates."
            ),
        },
        "serotonin_syndrome_risk": {
            "description": "Three or more serotonergic agents",
            "trigger": lambda mechs: sum(1 for m in mechs if "serotonergic" in m) >= 2,
            "severity_modifier": +1,
            "alert": (
                "Multiple serotonergic agents present. Risk of serotonin "
                "syndrome is significantly elevated with three or more "
                "serotonergic drugs. Consider reducing serotonergic burden."
            ),
        },
        "additive_qt_prolongation": {
            "description": "Multiple QT-prolonging agents",
            "trigger": lambda mechs: sum(1 for m in mechs if "qt_prolongation" in m) >= 2,
            "severity_modifier": +1,
            "alert": (
                "Multiple QT-prolonging agents present. Combined QT "
                "prolongation risk is greater than additive. Obtain ECG "
                "and monitor QTc interval closely."
            ),
        },
        "triple_bleed_risk": {
            "description": "Three or more agents affecting hemostasis",
            "trigger": lambda mechs: sum(1 for m in mechs if "bleeding" in m) >= 2,
            "severity_modifier": +1,
            "alert": (
                "Multiple agents affecting hemostasis. Combined bleeding "
                "risk is significantly elevated. Monitor closely for signs "
                "of hemorrhage."
            ),
        },
        "renal_cascade": {
            "description": "Nephrotoxic agent + renal-eliminated drug + ACE/ARB",
            "trigger": "custom",  # requires specific drug class checks
            "severity_modifier": +1,
            "alert": (
                "Nephrotoxic cascade detected: renal impairment from one "
                "agent may reduce elimination of another, compounding toxicity."
            ),
        },
    }

    def analyze(
        self,
        drug_names: list[str],
        pairwise_results: dict[tuple[str, str], InteractionResult],
    ) -> PolypharmacyReport:
        """
        Analyze all pairwise results for multi-drug patterns.

        Args:
            drug_names: List of all drug names being checked.
            pairwise_results: Dict mapping (drug_a, drug_b) → InteractionResult.

        Returns:
            PolypharmacyReport with pairwise results + multi-drug alerts.
        """
        # Collect all active mechanisms across all pairs
        all_mechanisms = []
        for result in pairwise_results.values():
            all_mechanisms.extend(result.mechanisms)

        # Check each multi-drug pattern
        alerts = []
        for pattern_name, pattern in self.MULTI_DRUG_PATTERNS.items():
            if callable(pattern["trigger"]) and pattern["trigger"](all_mechanisms):
                alerts.append(MultiDrugAlert(
                    pattern=pattern_name,
                    description=pattern["description"],
                    alert_text=pattern["alert"],
                    severity_modifier=pattern["severity_modifier"],
                    involved_drugs=self._find_involved_drugs(
                        pattern_name, pairwise_results
                    ),
                ))

        # Rank pairwise results by severity × confidence
        ranked_pairs = sorted(
            pairwise_results.items(),
            key=lambda x: (
                SEVERITY_ORDER[x[1].severity],
                x[1].confidence,
            ),
            reverse=True,
        )

        return PolypharmacyReport(
            drugs=drug_names,
            pairwise_results=ranked_pairs,
            multi_drug_alerts=alerts,
            total_pairs_checked=len(pairwise_results),
            highest_severity=ranked_pairs[0][1].severity if ranked_pairs else "none",
        )
```

### 4.4 Polypharmacy Inference

```python
class PharmLoopInference:
    ...

    def check_multiple(
        self,
        drug_names: list[str],
        context: dict | None = None,
    ) -> PolypharmacyReport:
        """
        Check all pairwise interactions among a list of drugs,
        then analyze for multi-drug patterns.

        Args:
            drug_names: List of 2+ drug names.
            context: Optional shared context (patient factors apply to all).

        Returns:
            PolypharmacyReport with ranked pairwise results + multi-drug alerts.
        """
        from itertools import combinations

        pairwise_results = {}
        for drug_a, drug_b in combinations(drug_names, 2):
            result = self.check(drug_a, drug_b, context=context)
            pairwise_results[(drug_a, drug_b)] = result

        # Multi-drug pattern analysis
        report = self.polypharmacy_analyzer.analyze(drug_names, pairwise_results)
        return report
```

---

## Step 5: REST API

### 5.1 Design

Minimal FastAPI service. Three endpoints:

```
POST /check          → single pair interaction check
POST /check-multiple → polypharmacy check (list of drugs)
GET  /drugs          → list available drugs
GET  /health         → service health check
```

### 5.2 Implementation

**File: `api/server.py`**

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="PharmLoop", version="0.4.0")

# Load model at startup
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
    partial_convergence: dict | None
    gray_zone_trajectory: list[float]
    unknown_drugs: list[str] | None

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
    return PairCheckResponse(**result.__dict__)

@app.post("/check-multiple", response_model=MultiCheckResponse)
def check_multiple(req: MultiCheckRequest):
    if len(req.drugs) < 2:
        raise HTTPException(400, "Need at least 2 drugs")
    if len(req.drugs) > 20:
        raise HTTPException(400, "Maximum 20 drugs per request")
    report = engine.check_multiple(req.drugs, context=req.context)
    return MultiCheckResponse(...)

@app.get("/drugs")
def list_drugs():
    return {"drugs": sorted(engine.drug_registry.keys()),
            "count": len(engine.drug_registry)}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": engine is not None,
        "num_drugs": len(engine.drug_registry),
        "hopfield_patterns": engine.model.cell.hopfield.count,
    }
```

### 5.3 Inference Performance

Single pair inference should be < 50ms on CPU (16 oscillator steps through
a ~3M param model is trivial). Polypharmacy with 10 drugs = 45 pairs =
~2.2 seconds sequentially. For faster multi-drug checks, batch all 45 pairs
into a single forward pass:

```python
def check_multiple_batched(self, drug_names, context=None):
    """Batch all pairs into one forward pass for speed."""
    pairs = list(combinations(drug_names, 2))

    # Stack all pair tensors into batch
    a_ids = torch.stack([self._get_id(p[0]) for p in pairs])
    a_feats = torch.stack([self._get_feat(p[0]) for p in pairs])
    b_ids = torch.stack([self._get_id(p[1]) for p in pairs])
    b_feats = torch.stack([self._get_feat(p[1]) for p in pairs])

    with torch.no_grad():
        output = self.model(a_ids, a_feats, b_ids, b_feats)

    # Unbatch results
    ...
```

This brings 10-drug polypharmacy down to ~100ms — a single forward pass
with batch_size=45.

---

## Step 6: Evaluation Framework

### 6.1 Gold Standard Benchmarks

Evaluate PharmLoop against established interaction databases:

- **DrugBank held-out test set:** Interactions not in training data
- **FDA Adverse Event Reporting System (FAERS):** Real-world signal
- **Clinical Pharmacology (Elsevier):** Expert-curated severity ratings
- **Lexicomp/Micromedex cross-reference:** Industry standard ratings

### 6.2 Evaluation Metrics

```python
class EvaluationSuite:
    """
    Comprehensive evaluation against gold-standard databases.
    """

    def evaluate(self, model, test_data) -> EvaluationReport:
        metrics = {}

        # 1. Severity accuracy (exact match)
        metrics["severity_accuracy"] = self._severity_accuracy(model, test_data)

        # 2. Severity accuracy (within one level)
        #    Predicting "moderate" for a "severe" is less bad than "none"
        metrics["severity_accuracy_relaxed"] = self._severity_within_one(model, test_data)

        # 3. False negative rate on severe/contraindicated
        #    THE critical safety metric
        metrics["false_negative_rate_severe"] = self._fnr_severe(model, test_data)

        # 4. False positive rate on none
        #    Predicting interactions that don't exist (annoying but not dangerous)
        metrics["false_positive_rate_none"] = self._fpr_none(model, test_data)

        # 5. Mechanism accuracy (top-1 and top-3)
        metrics["mechanism_accuracy_top1"] = self._mech_top1(model, test_data)
        metrics["mechanism_accuracy_top3"] = self._mech_top3(model, test_data)

        # 6. Confidence calibration
        #    When model says 90% confidence, is it right 90% of the time?
        metrics["calibration_error"] = self._calibration(model, test_data)

        # 7. Unknown detection rate
        #    What fraction of truly unknown drugs get "unknown" output?
        metrics["unknown_detection_rate"] = self._unknown_detection(model, test_data)

        # 8. Convergence statistics
        metrics["avg_steps_known"] = self._avg_steps(model, test_data, known=True)
        metrics["avg_steps_unknown"] = self._avg_steps(model, test_data, known=False)

        # 9. Comparison with existing tools
        #    Where PharmLoop disagrees with Lexicomp/Micromedex, who's right?
        metrics["agreement_with_lexicomp"] = self._cross_reference(model, test_data, "lexicomp")
        metrics["agreement_with_micromedex"] = self._cross_reference(model, test_data, "micromedex")

        return EvaluationReport(metrics=metrics)
```

### 6.3 Stress Tests

```python
class StressTests:
    """Edge cases and adversarial tests."""

    def test_same_drug_twice(self, model):
        """Drug + itself should predict based on overdose risk, not interaction."""
        result = model.check("warfarin", "warfarin")
        # Should either flag as "not applicable" or converge to "none"

    def test_prodrug_and_metabolite(self, model):
        """codeine + morphine: codeine IS morphine after CYP2D6."""
        result = model.check("codeine", "morphine")
        # Should detect the overlapping pharmacology

    def test_similar_drugs_same_class(self, model):
        """Two SSRIs: fluoxetine + sertraline."""
        result = model.check("fluoxetine", "sertraline")
        # Should flag serotonergic risk (additive)

    def test_all_50_original_drugs_still_work(self, model):
        """Regression: original Phase 1-3 test cases still pass."""
        ...

    def test_inference_latency(self, model):
        """Single pair should complete in < 100ms on CPU."""
        import time
        start = time.perf_counter()
        for _ in range(100):
            model.check("fluoxetine", "tramadol")
        elapsed = (time.perf_counter() - start) / 100
        assert elapsed < 0.1, f"Latency {elapsed:.3f}s exceeds 100ms target"

    def test_polypharmacy_latency(self, model):
        """10-drug check should complete in < 500ms (batched)."""
        drugs = ["fluoxetine", "tramadol", "warfarin", "metformin",
                 "lisinopril", "omeprazole", "amlodipine", "simvastatin",
                 "metoprolol", "acetaminophen"]
        import time
        start = time.perf_counter()
        model.check_multiple(drugs)
        elapsed = time.perf_counter() - start
        assert elapsed < 0.5, f"Polypharmacy latency {elapsed:.3f}s exceeds 500ms"
```

---

## Step 7: Retraining Protocol

### 7.1 Training with Expanded Data

With 500+ drugs and 2000+ interactions, training changes significantly:

```python
# Phase 4 training config
config = {
    "epochs": 50,
    "batch_size": 32,             # can go bigger with more data
    "lr": 5e-4,
    "lr_schedule": "cosine",
    "warmup_epochs": 5,

    # Split
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "test_ratio": 0.1,

    # Hopfield rebuild every N epochs
    "anneal_interval": 15,
    "max_anneal_cycles": 3,

    # Curriculum: start with well-documented pairs, add sparse pairs later
    "curriculum": True,
    "curriculum_phase1_epochs": 15,   # well-documented pairs only (>= 3 sources)
    "curriculum_phase2_epochs": 35,   # all pairs
}
```

### 7.2 Curriculum Learning

Not all interactions are equally well-documented. Some have multiple
independent sources (DrugBank + FDA + literature). Others have only a
single DrugBank entry. Train in two stages:

1. **High-confidence interactions first (15 epochs):** Only pairs with
   >= 3 source citations. This establishes good attractors in the Hopfield.
2. **All interactions (35 epochs):** Add the singly-sourced pairs. The
   model now has good priors from the high-confidence data and is less
   likely to be misled by noisy single-source entries.

### 7.3 Validation During Training

- **Primary metric:** Severity accuracy on held-out test set
- **Safety metric:** False negative rate on severe/contraindicated (must be 0)
- **Secondary:** Mechanism accuracy, convergence speed, confidence calibration
- **Early stopping:** On validation severity accuracy, patience=10

---

## Implementation Files

### New Files
```
data/
  pipeline/
    drugbank_parser.py            ← DrugBank XML → feature vectors
    interaction_extractor.py      ← DrugBank XML → interaction pairs
    quality_report.py             ← Data quality analysis
    run_pipeline.py               ← Pipeline orchestrator
  processed/
    drugs_v2.json                 ← Expanded drug set
    interactions_v2.json          ← Expanded interactions
    quality_report.json
    manual_review_queue.json

pharmloop/
    polypharmacy.py               ← PolypharmacyAnalyzer + reports
    hierarchical_hopfield.py      ← HierarchicalHopfield (two-level)

api/
    server.py                     ← FastAPI service
    models.py                     ← Pydantic request/response models

training/
    train_phase4.py               ← Training with expanded data + curriculum
    evaluate.py                   ← EvaluationSuite

tests/
    test_pipeline.py              ← Data pipeline tests
    test_polypharmacy.py          ← Multi-drug pattern detection
    test_hierarchical_hopfield.py ← Two-level retrieval
    test_api.py                   ← API endpoint tests
    test_evaluation.py            ← Evaluation suite tests
    test_stress.py                ← Edge cases and latency
```

### Modified Files
```
pharmloop/encoder.py              ← Drug count tracking, feature-dominant scaling
pharmloop/model.py                ← HierarchicalHopfield integration
pharmloop/inference.py            ← check_multiple(), batched inference
training/data_loader.py           ← Support v2 data format, curriculum
```

---

## Implementation Sequence

### 8.1 Data pipeline (do first — everything depends on data)
1. Implement DrugBank parser
2. Implement interaction extractor
3. Run pipeline, generate drugs_v2.json + interactions_v2.json
4. Generate quality report, review flagged items
5. Verify original 50 drugs are present and features match

### 8.2 Encoder scaling
6. Add drug_counts buffer and feature-dominant scaling to encoder
7. Expand embedding table with growth headroom
8. Verify backward compatibility with Phase 3 checkpoint

### 8.3 Hopfield scaling
9. Implement HierarchicalHopfield
10. Build class-specific banks from drug metadata
11. Implement incremental rebuild protocol
12. Verify retrieval quality at scale

### 8.4 Retrain on expanded data
13. Implement curriculum learning in train_phase4.py
14. Train Phase 4 model (50 epochs with annealing)
15. Evaluate severity accuracy, false negatives, mechanism accuracy
16. Run full evaluation suite

### 8.5 Polypharmacy
17. Implement PolypharmacyAnalyzer
18. Implement check_multiple in inference pipeline
19. Implement batched inference for speed
20. Test multi-drug pattern detection

### 8.6 API
21. Implement FastAPI server
22. Implement all endpoints
23. Latency testing
24. Integration test: full request/response cycle

### 8.7 Evaluation
25. Implement EvaluationSuite
26. Run against held-out test set
27. Cross-reference with external databases where available
28. Run stress tests
29. Generate final evaluation report

---

## Validation Criteria (must pass before Phase 5)

### Data
- [ ] 500+ drugs extracted from DrugBank
- [ ] 2000+ interactions with severity and mechanism labels
- [ ] Quality report generated, no critical data issues unresolved
- [ ] Original 50 drugs preserved with matching features (regression check)

### Model
- [ ] Severity accuracy >= 85% on expanded test set
- [ ] Zero false negatives on severe/contraindicated (expanded test set)
- [ ] Mechanism accuracy >= 55% on expanded test set
- [ ] Confidence calibration error < 0.15
- [ ] Unknown detection rate >= 90%
- [ ] Model total params < 10M

### Polypharmacy
- [ ] check_multiple works for 2-20 drugs
- [ ] Correctly detects additive CYP inhibition pattern
- [ ] Correctly detects multi-serotonergic pattern
- [ ] Correctly detects additive QT prolongation pattern

### Performance
- [ ] Single pair inference < 100ms on CPU
- [ ] 10-drug polypharmacy (batched) < 500ms on CPU
- [ ] API responds within 200ms for single pair

### API
- [ ] All 4 endpoints functional
- [ ] Correct error handling for unknown drugs
- [ ] Drug list endpoint returns all available drugs
- [ ] Health endpoint reports model status

### Regression
- [ ] Three-way separation test still passes (fluoxetine+tramadol, metformin+lisinopril, fabricated)
- [ ] Phase 3 template output unchanged for original test cases
- [ ] No severity accuracy regression on original 50-drug test set

---

## What NOT to Do in Phase 4

- Don't build a frontend UI (separate project, not core PharmLoop)
- Don't try to predict truly novel multi-drug interactions — the analyzer detects KNOWN patterns
- Don't train on FAERS data directly (too noisy — use for evaluation only)
- Don't add a text decoder for narrative (templates are the right call)
- Don't over-optimize latency before correctness is validated
- Don't remove the original 50-drug dataset — keep it for regression testing
- Don't exceed the 10M param budget — if you need more capacity, optimize existing params first
- Don't try to replicate Lexicomp/Micromedex exactly — they have different methodologies and PharmLoop has a different (dynamical) approach. Disagreements are expected and informative.

---

## Phase 5+ Roadmap (Not Scoped Yet)

For reference, here's what comes after Phase 4. These are NOT commitments:

- **Phase 5: Clinical Integration** — EHR integration, FHIR-compatible output,
  real-time alerting in prescription workflows
- **Phase 6: Pharmacogenomics** — CYP2D6 poor/rapid metabolizer status,
  HLA typing for hypersensitivity, genetic risk factors in context encoder
- **Phase 7: Temporal Dynamics** — model drug accumulation over time,
  loading doses, steady state interactions, washout periods
- **Phase 8: Pediatric/Geriatric Specialization** — age-adjusted PK models,
  weight-based dosing context, organ maturation factors
- **Phase 9: Regulatory Submission** — FDA 510(k) or De Novo pathway for
  clinical decision support software (Class II medical device)
