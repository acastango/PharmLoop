# Phase 4b — Production Scale

## Prerequisites
Phase 4a validated:
- [ ] >= 280 drugs with valid features
- [ ] >= 1200 interactions
- [ ] Severity accuracy >= 80% on expanded test set
- [ ] Zero false negatives on severe/contraindicated
- [ ] Polypharmacy: three additive patterns detected correctly
- [ ] API functional with < 100ms single-pair latency
- [ ] Regression tests pass on original 50-drug test set
- [ ] **Pharmacist feedback collected on Phase 4a output**

The last item is the most important. Phase 4b should be shaped by
what pharmacists actually said about Phase 4a, not by what we
predicted they'd want. This spec describes the LIKELY scope of 4b
but the priorities should be reordered based on real feedback.

---

## What Phase 4b Accomplishes

Phase 4a delivered a minimum viable system: 300 drugs, basic polypharmacy,
REST API. Phase 4b takes it to production quality:

- **500-600 drugs** — complete coverage of commonly prescribed medications
- **Full polypharmacy analyzer** — cascade interactions, renal risk chains,
  metabolic pathway saturation
- **Context encoder fully trained** — dose, route, timing, patient factors
  actually influence predictions
- **Pharmacogenomic context** — CYP2D6 metabolizer status, relevant genetic
  polymorphisms
- **Cross-reference evaluation** — systematic comparison with Lexicomp,
  Micromedex, Clinical Pharmacology
- **Production hardening** — caching, request batching, monitoring, error handling

---

## Step 1: Drug Set Expansion to 500-600

### 1.1 What to Add

Phase 4a covered the top 300. Phase 4b adds:

- **Remaining high-interaction drugs:** drugs that appear as interaction
  partners in DrugBank but weren't in our top 300 list. If drug X interacts
  with 15 drugs already in our set, we should include X.
- **Specialty medications:** immunosuppressants (tacrolimus, sirolimus),
  antiepileptics (levetiracetam, lacosamide), oncology supportive care
  (ondansetron, dexamethasone), HIV antiretrovirals (ritonavir, efavirenz)
- **OTC medications:** more NSAIDs, antihistamines, PPIs, supplements
  (St. John's Wort, grapefruit extract — these are real interaction risks)
- **Herbal/supplement interactions:** St. John's Wort (strong CYP3A4 inducer),
  ginkgo (bleeding risk), valerian (CNS depression) — if DrugBank has data

### 1.2 Pipeline Reuse

The Phase 4a pipeline (`drugbank_parser.py`, `interaction_extractor.py`)
already works. Just expand `top_300_drugs.txt` to `full_drug_list.txt`
with 500-600 entries and rerun:

```bash
python data/pipeline/run_pipeline.py \
    --drugbank data/raw/drugbank_full.xml \
    --target-drugs data/raw/full_drug_list.txt \
    --output data/processed/drugs_v3.json \
    --interactions data/processed/interactions_v3.json
```

Expected output: ~550 drugs, ~3000+ interactions (more drugs = quadratically
more interaction pairs discovered).

### 1.3 Feature Gap Analysis

Run the quality report on the expanded set. Drugs added in 4b are likely
to have sparser DrugBank data than the top 300. Identify:
- Drugs with > 30% zero-filled features → flag for manual review
- Drug classes with < 3 representatives → may need manual feature curation
- Interactions from single source with no severity → classify conservatively

---

## Step 2: Full Polypharmacy Analyzer

### 2.1 New Patterns Beyond Phase 4a

Phase 4a detected three additive patterns. Phase 4b adds:

**Cascade interactions:**
```
Drug A inhibits CYP3A4
Drug B is metabolized by CYP3A4 → produces metabolite B'
Drug C interacts with metabolite B'

A alone + C alone = nothing
A + B = moderate (elevated B levels)
A + B + C = severe (elevated B levels + B' interaction with C)
```

Detection: trace CYP substrate→inhibitor chains through the pairwise
results. If A inhibits the enzyme that metabolizes B, AND B has an
interaction with C, flag the A→B→C chain.

**Renal risk cascade:**
```
Drug A is nephrotoxic (mild)
Drug B is renally eliminated
Drug C is also nephrotoxic (mild)

A + B = moderate (reduced B clearance)
A + C = moderate (additive nephrotoxicity)
A + B + C = severe (A+C damage kidneys → B accumulates → toxicity)
```

Detection: identify drugs with `nephrotoxicity` mechanism AND drugs with
high renal elimination (feature dim 25). If both are present with a
renally-eliminated drug, flag the cascade.

**Metabolic pathway saturation:**
```
Drug A: CYP3A4 substrate (high affinity)
Drug B: CYP3A4 substrate (moderate affinity)
Drug C: CYP3A4 inhibitor

C inhibits 3A4 → A and B compete for remaining enzyme capacity
→ Both A and B levels rise, but unpredictably
```

Detection: count CYP3A4 substrates and inhibitors. If >= 2 substrates
and >= 1 inhibitor for the same enzyme, flag saturation risk.

**Additive bleeding risk (expanded):**
Phase 4a detected additive bleeding from mechanism labels. Phase 4b
additionally checks drug features: if 3+ drugs have `bleeding_risk`
feature flag > 0.5, alert even if the pairwise mechanism labels don't
all say "bleeding_risk" (some antiplatelet interactions are classified
as "protein_binding_displacement" but the real risk is bleeding).

### 2.2 Implementation

```python
class FullPolypharmacyAnalyzer(BasicPolypharmacyAnalyzer):
    """
    Extended polypharmacy analysis with cascade and saturation detection.

    Inherits the three additive patterns from BasicPolypharmacyAnalyzer,
    adds cascade interactions, renal risk chains, metabolic saturation,
    and feature-based risk detection.
    """

    def analyze(self, drug_names, pairwise_results, drug_features=None):
        # Run basic additive patterns first
        report = super().analyze(drug_names, pairwise_results)

        # Add cascade detection
        cascades = self._detect_cascades(drug_names, pairwise_results, drug_features)
        report.multi_drug_alerts.extend(cascades)

        # Add renal risk chains
        renal_chains = self._detect_renal_cascades(drug_names, pairwise_results, drug_features)
        report.multi_drug_alerts.extend(renal_chains)

        # Add metabolic saturation
        saturation = self._detect_metabolic_saturation(drug_names, drug_features)
        report.multi_drug_alerts.extend(saturation)

        # Feature-based bleeding risk check
        bleeding = self._detect_feature_bleeding_risk(drug_names, drug_features)
        report.multi_drug_alerts.extend(bleeding)

        return report

    def _detect_cascades(self, drug_names, pairwise_results, drug_features):
        """
        Detect CYP substrate → inhibitor chains.

        For each CYP enzyme (3A4, 2D6, 2C9, etc.):
          1. Find all inhibitors of that enzyme in the drug list
          2. Find all substrates of that enzyme
          3. For each substrate, check if any of its interaction partners
             are also in the drug list
          4. If inhibitor + substrate + substrate's partner all present,
             flag the cascade
        """
        alerts = []
        # CYP feature dimensions: 0-4 = inhibition, 5-7 = induction, 8-9 = substrate
        cyp_enzymes = {
            "CYP1A2": (0, None, None),  # (inhibition_dim, induction_dim, substrate_dim)
            "CYP2C9": (1, 6, None),
            "CYP2C19": (2, 7, None),
            "CYP2D6": (3, None, 8),
            "CYP3A4": (4, None, 9),
        }

        if drug_features is None:
            return alerts

        for enzyme, (inhib_dim, _, sub_dim) in cyp_enzymes.items():
            if sub_dim is None:
                continue

            inhibitors = [d for d in drug_names
                          if drug_features.get(d, [0]*64)[inhib_dim] > 0.5]
            substrates = [d for d in drug_names
                          if drug_features.get(d, [0]*64)[sub_dim] > 0.5]

            if inhibitors and len(substrates) >= 2:
                alerts.append(MultiDrugAlert(
                    pattern=f"cascade_{enzyme.lower()}",
                    alert_text=(
                        f"Metabolic cascade risk: {', '.join(inhibitors)} inhibit "
                        f"{enzyme}, which metabolizes {', '.join(substrates)}. "
                        f"Multiple substrates competing for reduced enzyme capacity "
                        f"may lead to unpredictable drug level elevations."
                    ),
                    involved_drugs=sorted(set(inhibitors + substrates)),
                    involved_pairs=[],
                    trigger_mechanism="cyp_inhibition",
                    pair_count=len(inhibitors) * len(substrates),
                ))

        return alerts

    def _detect_renal_cascades(self, drug_names, pairwise_results, drug_features):
        """Detect nephrotoxic drug + renally-eliminated drug combinations."""
        alerts = []
        if drug_features is None:
            return alerts

        nephrotoxic = [d for d in drug_names
                       if any(r.mechanisms and "nephrotoxicity" in r.mechanisms
                              for r in pairwise_results.values()
                              if d in r.drug_a or d in r.drug_b)]

        renally_cleared = [d for d in drug_names
                           if drug_features.get(d, [0]*64)[25] > 0.5]  # dim 25 = renal_elimination

        # If nephrotoxic + renally cleared drugs coexist
        overlap_risk = set(nephrotoxic) & set(renally_cleared)
        non_overlap_renal = set(renally_cleared) - set(nephrotoxic)

        if nephrotoxic and non_overlap_renal:
            alerts.append(MultiDrugAlert(
                pattern="renal_cascade",
                alert_text=(
                    f"Renal cascade risk: {', '.join(nephrotoxic)} may impair "
                    f"renal function, reducing clearance of renally-eliminated "
                    f"drugs ({', '.join(non_overlap_renal)}). Monitor renal "
                    f"function and consider dose adjustment for renally-cleared "
                    f"medications."
                ),
                involved_drugs=sorted(set(nephrotoxic) | non_overlap_renal),
                involved_pairs=[],
                trigger_mechanism="nephrotoxicity",
                pair_count=len(nephrotoxic) * len(non_overlap_renal),
            ))

        return alerts
```

---

## Step 3: Context Encoder — Full Training

### 3.1 Status from Phase 3

Phase 3 validated that the context encoder MECHANISM works — gate modulation
changes output in the expected direction. But it was trained on only a
handful of examples. Phase 4b trains it comprehensively.

### 3.2 Context Training Data

Build context-dependent interaction examples from clinical literature.
Focus on the interaction pairs where context ACTUALLY matters:

```python
# Categories of context-dependent interactions:
CONTEXT_DEPENDENT_CATEGORIES = {
    "dose_dependent": [
        # Warfarin + acetaminophen: safe at 2g/day, risky at 4g/day chronic
        # Simvastatin + amiodarone: simvastatin > 20mg = myopathy risk
        # Methotrexate dose determines severity with NSAIDs
    ],
    "route_dependent": [
        # Ciprofloxacin + antacids: oral interaction, not IV
        # Tacrolimus + fluconazole: oral >> IV interaction magnitude
    ],
    "timing_dependent": [
        # Ciprofloxacin + calcium: separated by 2h = minimal interaction
        # Levothyroxine + calcium: separated by 4h = safe
    ],
    "patient_factor_dependent": [
        # Metformin + contrast dye: renal function determines risk
        # ACEi + potassium supplements: renal function determines risk
        # Warfarin + anything: age determines bleeding risk
    ],
}
```

Target: 200-300 context-annotated interaction examples covering the
above categories. Each example is a known interaction pair WITH context
that changes the clinical assessment.

### 3.3 Training Protocol

```python
# Stage 1: Train context encoder only (freeze everything else)
#   Teaches the gate and projection to respond to context features
#   without disrupting the base model

# Stage 2: Unfreeze all, fine-tune with context
#   Allows the oscillator and Hopfield to adapt to context-modulated
#   initial states
```

### 3.4 Validation

```python
class ContextValidation:
    def test_dose_modulation(self, engine):
        """Higher dose → higher severity for dose-dependent interactions."""
        low_dose = engine.check("warfarin", "acetaminophen",
                                 context={"dose_b_normalized": 0.2})
        high_dose = engine.check("warfarin", "acetaminophen",
                                  context={"dose_b_normalized": 0.9})
        assert SEVERITY_ORDER[high_dose.severity] >= SEVERITY_ORDER[low_dose.severity]

    def test_no_context_unchanged(self, engine):
        """Without context, results match Phase 4a exactly."""
        with_ctx = engine.check("fluoxetine", "tramadol", context=None)
        # Should match Phase 4a checkpoint output
```

---

## Step 4: Pharmacogenomic Context

### 4.1 Scope

Don't try to model all of pharmacogenomics. Focus on the polymorphisms
that have the biggest clinical impact on drug interactions:

- **CYP2D6 metabolizer status:** poor / intermediate / extensive / ultra-rapid
  (Affects ~25% of all drugs. Poor metabolizers accumulate CYP2D6 substrates;
  ultra-rapid metabolizers may not achieve therapeutic levels.)
- **CYP2C19 metabolizer status:** similar impact for clopidogrel, PPIs, etc.
- **CYP2C9 + VKORC1:** warfarin sensitivity
- **HLA-B*5701:** abacavir hypersensitivity
- **HLA-B*1502:** carbamazepine/phenytoin SJS/TEN risk

### 4.2 Implementation

Pharmacogenomic status goes into the context encoder as additional features:

```python
# Extend CONTEXT_DIM from 32 to 48
# Dims 32-35: CYP2D6 status (one-hot: poor, intermediate, extensive, ultra-rapid)
# Dims 36-39: CYP2C19 status (same encoding)
# Dims 40-43: CYP2C9 status + VKORC1
# Dims 44-47: HLA markers (B*5701, B*1502, reserved, reserved)
```

When pharmacogenomic context is provided:
```python
result = engine.check("codeine", "fluoxetine", context={
    "cyp2d6_status": "poor_metabolizer",
})
# Poor metabolizer: codeine can't convert to morphine (reduced efficacy)
# BUT fluoxetine inhibition of CYP2D6 is redundant (already poor)
# → interaction severity should be LOWER than for extensive metabolizer
```

### 4.3 Training Data

Pharmacogenomic training examples are harder to source than dose-dependent
ones. Use published pharmacogenomic guidelines (CPIC, DPWG) as the source
of truth for how metabolizer status modifies interaction severity.

Target: 50-100 pharmacogenomic-annotated interaction examples. This is a
starting point — the real value comes from Phase 5+ clinical integration
where pharmacogenomic data flows in from EHR systems.

---

## Step 5: Cross-Reference Evaluation

### 5.1 External Databases

Systematically compare PharmLoop predictions against:

- **Lexicomp:** Industry standard, used in most US hospitals
- **Micromedex:** Alternative industry standard
- **Clinical Pharmacology (Elsevier):** Another major reference

Where PharmLoop disagrees with these references, categorize the disagreement:

```python
@dataclass
class DisagreementRecord:
    drug_a: str
    drug_b: str
    pharmloop_severity: str
    reference_severity: str
    reference_source: str
    disagreement_type: str  # "pharmloop_more_severe", "pharmloop_less_severe",
                            # "mechanism_differs", "false_positive", "false_negative"
    clinical_significance: str  # pharmacist assessment of who's right
```

### 5.2 Disagreement Analysis

Disagreements fall into categories:

1. **PharmLoop more conservative (ours is more severe):** Usually acceptable.
   The DO NO HARM principle means we err on the side of caution. Flag for
   review but not a failure.

2. **PharmLoop less severe than reference:** Potentially dangerous. Every
   case must be investigated. If the reference is right, we have a data
   gap or model error to fix.

3. **Mechanism differs:** We say "CYP inhibition," reference says "protein
   binding displacement." Often both are partially right (multi-mechanism
   interaction). Not alarming unless our mechanism is clinically misleading.

4. **PharmLoop says "unknown," reference has a rating:** Data gap. The
   interaction exists in the reference but not in our training data. Add it.

5. **PharmLoop has a rating, reference says "no interaction":** False positive.
   Less dangerous than false negative but annoying. Investigate whether our
   data source is wrong or we're over-generalizing from similar drug pairs.

---

## Step 6: Production Hardening

### 6.1 API Improvements

- **Request caching:** Cache interaction results by (drug_a, drug_b, context_hash).
  Most interaction checks are repeated. LRU cache with 10K entries covers
  the most common queries.
- **Request validation:** Fuzzy drug name matching ("fluoxetine" vs "Fluoxetine"
  vs "FLUOXETINE" vs "Prozac"). Map brand names to generic names.
- **Rate limiting:** Prevent abuse. 100 requests/minute per client.
- **Structured logging:** Log every interaction check with timing, result,
  confidence. This data is gold for monitoring model performance in practice.
- **Error handling:** Graceful degradation when model fails. Never crash on
  unexpected input.

### 6.2 Brand Name Resolution

Pharmacists often use brand names. Build a brand → generic mapping:

```python
BRAND_TO_GENERIC = {
    "prozac": "fluoxetine",
    "zoloft": "sertraline",
    "coumadin": "warfarin",
    "lipitor": "atorvastatin",
    "norvasc": "amlodipine",
    # ... ~500 entries covering top brand names
}

def resolve_drug_name(name: str) -> str:
    """Resolve brand names, normalize case, handle common typos."""
    normalized = name.strip().lower()
    if normalized in drug_registry:
        return normalized
    if normalized in BRAND_TO_GENERIC:
        return BRAND_TO_GENERIC[normalized]
    # Fuzzy match for typos
    close = difflib.get_close_matches(normalized, drug_registry.keys(), n=1, cutoff=0.85)
    if close:
        return close[0]
    return None  # unknown drug
```

### 6.3 Monitoring

```python
# Track in production:
# - Request latency (p50, p95, p99)
# - Severity distribution of results (detect drift)
# - Unknown drug rate (are people asking about drugs we don't have?)
# - Confidence distribution (is the model getting less confident over time?)
# - Polypharmacy alert rate (are alerts too noisy?)
# - Most queried drug pairs (focus optimization effort)
```

---

## Implementation Files

### New Files
```
data/
  raw/
    full_drug_list.txt             ← 500-600 drug target list
    context_training_data.json     ← 200-300 context-annotated interactions
    pharmacogenomic_examples.json  ← 50-100 PGx-annotated interactions
    brand_names.json               ← Brand → generic mapping
  processed/
    drugs_v3.json                  ← 500-600 drugs
    interactions_v3.json           ← 3000+ interactions

pharmloop/
    polypharmacy_full.py           ← FullPolypharmacyAnalyzer
    drug_resolver.py               ← Brand name + fuzzy matching

api/
    cache.py                       ← LRU result caching
    monitoring.py                  ← Request logging + metrics

training/
    train_context.py               ← Context encoder training on expanded data
    train_pgx.py                   ← Pharmacogenomic context training
    cross_reference.py             ← Evaluation against external databases

tests/
    test_polypharmacy_full.py
    test_drug_resolver.py
    test_context_expanded.py
    test_pgx.py
    test_cross_reference.py
```

### Modified Files
```
pharmloop/context.py              ← Expand to 48 dims for PGx features
pharmloop/inference.py            ← Drug name resolution, FullPolypharmacyAnalyzer
api/server.py                     ← Caching, monitoring, brand name support
training/data_loader.py           ← v3 data format, context examples
```

---

## Implementation Sequence

**Priority order should be adjusted based on Phase 4a pharmacist feedback.**
The sequence below assumes feedback confirms the predicted priorities.

### 8.1 Drug expansion (do first if feedback says "you're missing drug X")
1. Expand drug list to 500-600
2. Rerun pipeline
3. Quality report + manual review
4. Retrain on expanded data

### 8.2 Full polypharmacy (do first if feedback says "the alerts miss things")
5. Implement cascade detection
6. Implement renal risk chains
7. Implement metabolic saturation detection
8. Test against known multi-drug scenarios

### 8.3 Context training (do first if feedback says "dose matters and you ignore it")
9. Build context training dataset (200-300 examples)
10. Train context encoder end-to-end
11. Validate dose/route/timing modulation

### 8.4 Brand names + production (do first if feedback says "can't find my drug")
12. Build brand → generic mapping
13. Implement fuzzy matching
14. Add caching layer
15. Add monitoring

### 8.5 Pharmacogenomics (likely lower priority unless feedback demands it)
16. Extend context to 48 dims
17. Build PGx training examples from CPIC/DPWG
18. Train PGx context
19. Validate metabolizer status modulation

### 8.6 Cross-reference evaluation (do in parallel with everything else)
20. Build cross-reference dataset from available databases
21. Run systematic comparison
22. Categorize and investigate all disagreements
23. Fix data gaps identified by cross-reference

---

## Validation Criteria

### Data
- [ ] >= 500 drugs with valid features
- [ ] >= 2500 interactions
- [ ] Brand name resolution covers top 200 brand names
- [ ] Original 50 + Phase 4a 300 drugs all present

### Model
- [ ] Severity accuracy >= 85% on expanded test set
- [ ] Zero false negatives on severe/contraindicated
- [ ] Mechanism accuracy >= 55% on expanded test set
- [ ] Confidence calibration error < 0.12
- [ ] Context encoder: dose modulation moves severity in correct direction
- [ ] Param budget < 10M

### Polypharmacy
- [ ] All Phase 4a patterns still detected
- [ ] Cascade interaction detected for at least one known case
- [ ] Renal cascade detected for nephrotoxic + renally-cleared combination
- [ ] Metabolic saturation detected for multi-substrate + inhibitor case

### Cross-Reference
- [ ] >= 500 interaction pairs compared with external reference
- [ ] All "PharmLoop less severe" disagreements investigated
- [ ] Disagreement report generated for pharmacist review

### Production
- [ ] Brand name resolution functional
- [ ] API caching reduces repeat-query latency by >= 10x
- [ ] Monitoring dashboards operational
- [ ] 20-drug polypharmacy completes in < 1 second

### Regression
- [ ] All Phase 4a validation criteria still pass
- [ ] Three-way separation test passes
- [ ] Original 50-drug test set accuracy maintained

---

## What Phase 4b Enables (Looking Ahead)

With a production-quality system over 500+ drugs, the path to clinical
deployment becomes concrete:

- **Phase 5: Clinical Integration** — FHIR-compatible output format,
  EHR integration, real-time alerting in prescription workflows,
  HL7 messaging
- **Phase 6: Regulatory Pathway** — If targeting US market, FDA 510(k)
  or De Novo classification for clinical decision support software
  (Class II medical device). Requires clinical validation study.
- **Phase 7: Temporal Dynamics** — Model drug accumulation, loading doses,
  steady-state interactions, washout periods. Needs time-series context.
- **Phase 8: Continuous Learning** — Integrate FAERS adverse event reports
  as new Hopfield patterns. The system grows from real-world evidence
  without retraining.

But don't plan these in detail until Phase 4b feedback is in. Each phase
should be shaped by the feedback from the previous one.
