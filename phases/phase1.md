# Phase 1 — Foundation + Core Architecture + Separation Test

## Goal
Get the full pipeline standing and prove the core dynamics work.
By the end of this phase, we need ONE test passing:

**The three-way separation test:**
- Known severe interaction (fluoxetine + tramadol) → converges fast, high severity
- Known safe pair (metformin + lisinopril) → converges to "none"
- Fabricated drug (QZ-7734 + aspirin) → fails to converge, outputs UNKNOWN

If the oscillator dynamics separate those three cases, the architecture works.
Everything else is scaling.

---

## Step 1: Drug Data (data pipeline)

### 1.1 Define the 50-Drug Set
Pick 50 drugs that give us good coverage of interaction types:
- SSRIs (fluoxetine, sertraline, paroxetine)
- Opioids (tramadol, codeine, fentanyl)
- Anticoagulants (warfarin, rivaroxaban)
- Antihypertensives (lisinopril, amlodipine, metoprolol)
- Antidiabetics (metformin, glipizide)
- Antiarrhythmics (amiodarone)
- Antibiotics (ciprofloxacin, clarithromycin, metronidazole)
- Antifungals (fluconazole, ketoconazole)
- PPIs (omeprazole)
- Statins (simvastatin, atorvastatin)
- NSAIDs (ibuprofen, naproxen, aspirin)
- Benzodiazepines (diazepam, alprazolam)
- Anticonvulsants (carbamazepine, phenytoin, valproic acid)
- Antipsychotics (quetiapine, haloperidol)
- CNS (lithium, gabapentin)
- Misc (digoxin, theophylline, cyclosporine, methotrexate)
- Fill to 50 with whatever gives the best interaction graph coverage

The exact list is flexible — use your pharmacological judgment. Prioritize drugs
that create DENSE interaction networks (many known interactions with each other).

### 1.2 Define the 64-Dim Feature Vector
Each drug gets a 64-dimensional structured feature vector. These are NOT learned —
they're assembled from pharmacological reference data.

Suggested feature allocation (adjust if needed):

```
Dims 0-9:    CYP metabolism profile (10 dims)
             - CYP1A2, CYP2B6, CYP2C8, CYP2C9, CYP2C19, CYP2D6, CYP3A4
               each gets substrate/inhibitor/inducer flags
             - Encode as: 0=none, 0.5=weak, 1.0=strong for each role

Dims 10-19:  Receptor binding profile (10 dims)
             - Serotonin (5-HT), Dopamine (D2), GABA-A, Mu-opioid, 
               Alpha-1, Beta-1, Histamine (H1), Muscarinic (M1),
               NMDA, Sodium channel
             - Values: 0=no affinity, 0.5=moderate, 1.0=high affinity

Dims 20-29:  Pharmacokinetic parameters (10 dims)
             - Half-life (normalized log scale)
             - Protein binding (fraction)
             - Volume of distribution (normalized log)
             - Bioavailability (fraction)
             - Renal elimination fraction
             - Hepatic elimination fraction
             - Active metabolites flag
             - Prodrug flag
             - Narrow therapeutic index flag
             - Food effect magnitude

Dims 30-39:  Drug class one-hot or embedding (10 dims)
             - Could be one-hot for major classes, or a mini-embedding
             - Classes: SSRI, opioid, anticoagulant, antihypertensive, etc.

Dims 40-49:  Interaction mechanism flags (10 dims)
             - Serotonergic risk
             - QT prolongation risk
             - Bleeding risk
             - CNS depression risk
             - Nephrotoxicity risk
             - Hepatotoxicity risk
             - Hypotension risk
             - Hyperkalemia risk
             - Seizure threshold lowering
             - Immunosuppression interaction risk

Dims 50-59:  Transporter profile (10 dims)
             - P-glycoprotein (substrate/inhibitor/inducer)
             - OATP1B1, OATP1B3, OCT2, OAT1, OAT3
             - BCRP, MRP2
             - Encode same as CYP: role + strength

Dims 60-63:  Physical/chemical (4 dims)
             - Molecular weight (normalized)
             - LogP (lipophilicity)
             - pKa-based charge at pH 7.4
             - Solubility class
```

### 1.3 Interaction Pair Dataset
For the 50-drug set, compile known interaction pairs:

```python
# Each interaction record:
{
    "drug_a": "fluoxetine",
    "drug_b": "tramadol",
    "severity": "severe",           # none/mild/moderate/severe/contraindicated
    "mechanisms": ["serotonergic"],  # from mechanism vocabulary
    "flags": ["monitor_serotonin_syndrome"],
    "source": "DrugBank / FDA / clinical guideline",
    "notes": "Both increase serotonergic activity"
}
```

Include:
- **Known interactions** with severity labels (at least 100 pairs across severity levels)
- **Known safe pairs** explicitly labeled as "none" (at least 50 pairs — drugs commonly co-prescribed with no interaction)
- **Leave gaps** — some drug pairs deliberately have no data (these test UNKNOWN behavior)

**Data format:** JSON files in `data/processed/`. One for drugs, one for interactions.

**Important:** Do NOT fabricate pharmacological data. Use real interaction data from
established references (DrugBank, FDA labels, Lexicomp / Micromedex equivalents).
Where exact feature values are uncertain, use reasonable approximations and document them.
Pharmacological accuracy matters — this is a medical system.

---

## Step 2: Core Architecture

Implement in this order. Each file should be independently testable.

### 2.1 `pharmloop/hopfield.py` — PharmHopfield

```python
class PharmHopfield(nn.Module):
    """
    Modern continuous Hopfield network for drug interaction patterns.
    
    Phase 0: initialized from raw 64-dim feature vectors (no learned projections)
    Phase 2: rebuilt with learned projections in 512-dim space
    
    Key: stored patterns are nn.Buffers, not Parameters.
    They don't get gradients. They're a database, not weights.
    """
```

Requirements:
- `store(patterns: Tensor)` — add patterns to memory bank
- `retrieve(query: Tensor, beta: float) -> Tensor` — softmax retrieval
- Beta controls sharpness: high beta → nearest neighbor, low beta → blended
- Start with capacity for 5000 patterns (most will be empty initially)
- Must work in both 64-dim (Phase 0) and 512-dim (Phase 2) modes

### 2.2 `pharmloop/encoder.py` — DrugEncoder

```python
class DrugEncoder(nn.Module):
    """
    Encodes a single drug into 512-dim pharmacological state.
    
    Two pathways fused:
      1. Learned identity embedding (captures drug-specific quirks)
      2. Structured features (the 64-dim vector — explicit pharmacology)
    
    The structured features are the ANCHOR. The identity embedding learns
    whatever the features don't capture.
    """
```

Requirements:
- Identity embedding: `nn.Embedding(num_drugs + PADDING, 256)`
  - +PADDING for unknown/fabricated drug IDs (these get random init, never trained)
- Feature pathway: `Linear(64, 256)`
- Fusion: `concat(identity_256, feature_256) → Linear(512, 512) → LayerNorm → GELU → Linear(512, 512)`
- Forward takes drug_id (int) and features (64-dim tensor)
- Must handle unknown drug IDs gracefully (return the untrained embedding — this is what makes fabricated drugs fail to converge)

### 2.3 `pharmloop/oscillator.py` — OscillatorCell + ReasoningLoop

Implement per the spec in CLAUDE.md. Key points:
- OscillatorCell: single step of the damped driven oscillator
- ReasoningLoop: runs OscillatorCell until convergence or max_steps
- Track full trajectory (positions, velocities, gray zones per step)
- Gray zone modulates Hopfield beta via `hopfield_beta_mod`
- Return trajectory dict for loss computation and visualization

### 2.4 `pharmloop/output.py` — OutputHead

```python
class OutputHead(nn.Module):
    """
    Maps converged state → structured predictions.
    
    Severity: 6-class (none/mild/moderate/severe/contraindicated/unknown)
    Mechanism: multi-label (serotonergic, CYP_inhibition, QT, bleeding, etc.)
    Flags: binary clinical monitoring flags
    
    Confidence is NOT predicted by this head. It comes from convergence dynamics.
    """
```

Requirements:
- Severity: `Linear(512, 6)` → softmax
- Mechanism: `Linear(512, num_mechanisms)` → sigmoid (multi-label)
- Flags: `Linear(512, num_flags)` → sigmoid
- Confidence computation: separate function that takes trajectory dict → confidence score
  - `confidence = f(final_gray_zone, steps_to_converge, trajectory_smoothness)`
  - This is a formula, not a neural network

### 2.5 `pharmloop/model.py` — PharmLoopModel

Full pipeline:
```
(drug_a_id, drug_a_features, drug_b_id, drug_b_features) 
  → encode both → combine pair → reasoning loop → output head
  → {severity, mechanisms, flags, confidence, converged, trajectory}
```

Pair combination: try `concat → Linear` or `element-wise product + sum`, your call.
The pair encoding becomes `initial_state` for the oscillator.

---

## Step 3: Training Infrastructure

### 3.1 `training/loss.py`

Implement the multi-objective loss from the spec. Key requirements:
- L_answer: cross-entropy (severity, mechanism) + BCE (flags)
- L_convergence: reward fast convergence on known, non-convergence on unknown
- L_smoothness: second derivative penalty on gray zone trajectory
- L_do_no_harm: 10x penalty for false-none on severe, **50x on contraindicated**
- Return component losses separately (for logging) plus total

### 3.2 `training/data_loader.py`

- Load drug features + interaction pairs from JSON
- Create batches of (drug_a, drug_b, target) triples
- Include negative sampling: randomly pair drugs with no known interaction
  - Some get labeled "none" (if we're confident they don't interact)
  - Some get labeled "unknown" (if we genuinely don't know)
- Fabricated drug injection: occasionally include a fake drug ID with random features
  - Target for these: always "unknown" / non-convergence

### 3.3 `training/train.py`

Phase 1 training loop:
1. Initialize Hopfield from raw 64-dim features (Phase 0 bootstrap)
2. Freeze Hopfield
3. Train encoder + oscillator + output head
4. Log: loss components, convergence rates, gray zone trajectories
5. Save checkpoints

---

## Step 4: The Separation Test

### `tests/test_separation.py`

After training, run the three-way test:

```python
def test_three_way_separation(model):
    """
    THE test. If this passes, the architecture works.
    
    1. fluoxetine + tramadol → should converge, severity >= severe
    2. metformin + lisinopril → should converge, severity = none
    3. QZ-7734 (fabricated) + aspirin → should NOT converge, output unknown
    
    Pass criteria:
    - Case 1 converges in <= 12 steps with severity in {severe, contraindicated}
    - Case 2 converges in <= 12 steps with severity = none
    - Case 3 does NOT converge (hits max_steps) OR outputs severity = unknown
    - Case 1 final gray zone < Case 3 final gray zone (known is more certain than unknown)
    """
```

Also test:
- **Trajectory shape**: Case 1 should show oscillation then damping. Case 3 should show sustained oscillation.
- **Gray zone dynamics**: plot gz over steps for all three cases. They should look visually different.

---

## Validation Criteria (must pass before Phase 2)

- [ ] 50 drugs with 64-dim feature vectors (documented sources)
- [ ] At least 100 known interaction pairs + 50 known safe pairs
- [ ] All architecture components instantiate and forward-pass without error
- [ ] Training runs for at least 50 epochs without NaN/explosion
- [ ] The three-way separation test shows clear qualitative separation
- [ ] Gray zone trajectories for the three cases are visually distinguishable
- [ ] Model total params < 6M (learned < 3M)

---

## What NOT to Do in Phase 1

- Don't implement the template engine yet (Phase 3)
- Don't implement the context encoder (dose/route/timing) yet (Phase 3+)
- Don't worry about Hopfield rebuild in learned space (that's Phase 2)
- Don't optimize for speed — clarity and correctness first
- Don't try to get high accuracy on all interaction pairs — we just need the DYNAMICS to separate the three cases
