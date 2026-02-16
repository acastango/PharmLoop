# Phase 2 — Hopfield Rebuild in Learned Space + End-to-End Fine-Tuning

## Prerequisites
Phase 1 (with fixes) must be validated:
- [ ] Three-way separation test passes
- [ ] Training runs 50+ epochs without NaN
- [ ] Gray zone trajectories visually distinguish the three cases
- [ ] Parameter budget holds (learned < 3M)

**Do not start Phase 2 until Phase 1 is validated.**

---

## What Phase 2 Accomplishes

Phase 1 proved the dynamics work against a crude Phase 0 Hopfield — raw
64-dim features with no learned projections, accessed through a bottleneck
projection layer. The oscillator had to do all the heavy lifting.

Phase 2 gives the oscillator what it actually needs: a **Hopfield memory bank
in the same 512-dim learned space as the oscillator state**, storing
interaction-level patterns (not just individual drug features). This means:

- Retrieval is direct — no dimension projection bottleneck
- Similar interactions retrieve each other by learned similarity, not just
  raw feature overlap
- The Hopfield becomes a proper "knowledge base" of known interaction patterns
- The oscillator can focus on reasoning about novel combinations rather than
  compensating for bad retrieval

Expected improvements over Phase 1:
- Faster convergence on known interactions (retrieval pulls harder toward right answer)
- Better mechanism attribution (Hopfield retrieves mechanism-specific patterns)
- Sharper separation between known/unknown (unknown queries retrieve noise → more oscillation)
- Higher severity accuracy on the full dataset (not just the well-documented pairs)

---

## Architecture Changes

### Overview of What Changes

```
Phase 1:
  Encoder(512) → projection(512→64) → Hopfield(64-dim, frozen) → projection(64→512) → Oscillator

Phase 2:
  Encoder(512) → Hopfield(512-dim, learned projections) → Oscillator
                     ↑
              Stores PAIR patterns from trained encoder
              Query/key projections unfrozen and trained
```

The bottleneck is gone. The oscillator talks to the Hopfield in its native
512-dim space.

---

## Step 1: Rebuild the Hopfield Memory Bank

### 1.1 What Gets Stored

In Phase 0, we stored individual drug feature vectors (50 patterns × 64-dim).
In Phase 2, we store **interaction pair patterns** in 512-dim learned space.

For each known interaction pair in the training data:

```python
# Using the Phase 1 trained encoder:
enc_a = encoder(drug_a_id, drug_a_features)    # (512,)
enc_b = encoder(drug_b_id, drug_b_features)    # (512,)
pair_state = pair_combine(enc_a, enc_b)         # (512,) — the initial oscillator state

# Store the pair state as a Hopfield pattern
hopfield.store(pair_state)
```

**Store the INITIAL pair state, not the converged state.** Reasoning:
- The initial pair state is deterministic (no oscillator dynamics involved)
- It represents "what this drug pair looks like in learned space"
- When the oscillator queries the Hopfield, it retrieves patterns of pairs
  that LOOK SIMILAR → gets pulled toward the interaction profile of similar pairs
- If we stored converged states, we'd be teaching the Hopfield "answers" rather
  than "what similar drug pairs look like" — that would short-circuit the
  oscillatory reasoning and make the model a glorified lookup table

Each stored pattern should also carry metadata (not stored in the tensor, kept
in a parallel registry):
- Which drug pair it represents
- The known severity
- The known mechanisms
- Source/provenance

### 1.2 Pattern Deduplication and Weighting

The interaction dataset has some pairs listed in both directions (A→B and B→A).
Since the pair_combine is order-invariant, these produce identical patterns.
Deduplicate before storing.

Also: not all patterns are equally valuable to retrieve. A contraindicated
interaction is more important to surface than a mild one. Consider storing
severe/contraindicated patterns with slight perturbations (2-3 copies with
small noise) so they have more Hopfield "mass" and are retrieved more
readily. This is a soft way to implement DO NO HARM at the retrieval level.

```python
# Severity-weighted storage
for pair in interaction_pairs:
    pattern = compute_pair_state(pair)
    hopfield.store(pattern.unsqueeze(0))

    # Amplify dangerous interactions in memory
    if pair.severity in ("severe", "contraindicated"):
        for _ in range(2):  # store 2 extra noisy copies
            noisy = pattern + torch.randn_like(pattern) * 0.01
            hopfield.store(noisy.unsqueeze(0))
```

### 1.3 Capacity Planning

With 209 interactions (deduplicated to ~160 unique pairs after removing
bidirectional duplicates), plus severity amplification (~50 extra for
severe/contraindicated), you'll have roughly 200-250 stored patterns.
Well within the 5000 capacity.

---

## Step 2: Rebuild PharmHopfield for Learned Space

### 2.1 New Hopfield Configuration

Create a new `PharmHopfield` instance in 512-dim space with learned projections:

```python
hopfield_v2 = PharmHopfield(
    input_dim=512,       # now in learned space
    hidden_dim=512,      # projection dim matches (can experiment with 256)
    phase0=False,        # learned projections enabled
)
```

The `query_proj` and `key_proj` are now `nn.Linear(512, 512)` — these get
trained during end-to-end fine-tuning so the Hopfield learns which aspects
of a pair state matter for retrieval.

### 2.2 Remove Dimension Projection Layers from OscillatorCell

In Phase 1, the `OscillatorCell` had `hopfield_query_proj` (512→64) and
`hopfield_value_proj` (64→512) to bridge the dimension gap. With the
Hopfield now in 512-dim, these are unnecessary.

In `OscillatorCell.__init__`:
```python
if hopfield is not None and hopfield.input_dim != state_dim:
    self.hopfield_query_proj = nn.Linear(state_dim, hopfield.input_dim)
    self.hopfield_value_proj = nn.Linear(hopfield.input_dim, state_dim)
else:
    self.hopfield_query_proj = None   # ← Phase 2 hits this branch
    self.hopfield_value_proj = None
```

This should already work via the existing dimension check — just verify
that when `hopfield.input_dim == state_dim == 512`, the projection layers
are `None` and retrieval uses the state directly.

### 2.3 Initialize Hopfield Projections Wisely

Don't initialize `query_proj` and `key_proj` randomly for Phase 2. The
stored patterns are already meaningful 512-dim vectors. Random projections
would destroy that structure.

Initialize both as near-identity:
```python
# After creating hopfield_v2:
with torch.no_grad():
    nn.init.eye_(hopfield_v2.query_proj.weight)
    nn.init.zeros_(hopfield_v2.query_proj.bias)
    nn.init.eye_(hopfield_v2.key_proj.weight)
    nn.init.zeros_(hopfield_v2.key_proj.bias)
```

This way, at the start of Phase 2 training, retrieval behaves like
direct cosine similarity in learned space (which is already meaningful).
The projections then learn to refine this during fine-tuning.

---

## Step 3: Model Reconstruction

### 3.1 Build the Phase 2 Model

```python
def build_phase2_model(
    phase1_checkpoint_path: str,
    drugs_path: str,
    interactions_path: str,
    device: torch.device,
) -> PharmLoopModel:
    """
    Build Phase 2 model:
    1. Load Phase 1 trained encoder + oscillator + output head
    2. Build new 512-dim Hopfield from trained encoder
    3. Remove dimension projection bottleneck
    4. Initialize Hopfield projections as near-identity
    """

    # Load Phase 1 model
    phase1_model = load_phase1_model(phase1_checkpoint_path, drugs_path)
    phase1_model.eval()

    # Compute pair patterns for all known interactions
    patterns = compute_all_pair_patterns(phase1_model, drugs_path, interactions_path)
    # patterns: Tensor of shape (N_unique_pairs, 512)

    # Build Phase 2 Hopfield
    hopfield_v2 = PharmHopfield(input_dim=512, hidden_dim=512, phase0=False)

    # Identity-init projections
    with torch.no_grad():
        nn.init.eye_(hopfield_v2.query_proj.weight)
        nn.init.zeros_(hopfield_v2.query_proj.bias)
        nn.init.eye_(hopfield_v2.key_proj.weight)
        nn.init.zeros_(hopfield_v2.key_proj.bias)

    # Store patterns (with severity amplification)
    hopfield_v2.store(patterns)

    # Build new model with Phase 2 Hopfield
    model = PharmLoopModel(
        num_drugs=phase1_model.num_drugs,
        hopfield=hopfield_v2,
    )

    # Transfer Phase 1 weights for encoder, pair_combine, oscillator, output head
    transfer_phase1_weights(model, phase1_model)

    return model.to(device)
```

### 3.2 Weight Transfer

Transfer ALL learned weights from Phase 1 EXCEPT:
- The old Hopfield (replaced entirely)
- The old `hopfield_query_proj` and `hopfield_value_proj` in OscillatorCell (removed)

Weights to transfer:
- `encoder.identity_embedding`
- `encoder.feature_proj`
- `encoder.fusion`
- `pair_combine`
- `cell.raw_decay`, `cell.raw_dt`, `cell.raw_spring`, `cell.raw_threshold`
- `cell.evidence_transform`
- `cell.noise_gate`
- `cell.beta_mod`
- `reasoning_loop.initial_v_proj`
- `output_head.severity_head`, `output_head.mechanism_head`, `output_head.flags_head`

Use `strict=False` in `load_state_dict` and verify that only the expected
keys are missing (Hopfield and projection layers).

### 3.3 What's Frozen vs. Trainable

In Phase 2, **everything is trainable**, including the Hopfield projections.
The stored patterns (buffers) remain frozen — they're reference data.

```python
# Freeze stored patterns (already buffers, but be explicit)
for name, buf in model.named_buffers():
    if "stored_keys" in name or "stored_values" in name:
        pass  # buffers don't get gradients anyway

# Everything else trains
trainable_params = [p for p in model.parameters() if p.requires_grad]
```

---

## Step 4: End-to-End Fine-Tuning

### 4.1 Training Configuration

Phase 2 training is fine-tuning, not training from scratch. Key differences
from Phase 1:

```python
# Lower learning rate (don't destroy Phase 1 knowledge)
lr = 1e-4  # 10x lower than Phase 1

# Shorter training (we're refining, not learning from scratch)
epochs = 30

# Same loss function, same DO NO HARM penalties
# But consider increasing convergence_weight slightly since retrieval
# is now better and convergence should be faster
criterion = PharmLoopLoss(
    convergence_weight=0.7,    # up from 0.5 — lean into convergence now
    smoothness_weight=0.1,
)

# Warmup for the Hopfield projections (they start as identity)
# Use a separate param group with higher LR so they catch up
optimizer = Adam([
    {"params": hopfield_params, "lr": 5e-4},     # Hopfield projections learn faster
    {"params": other_params, "lr": 1e-4},         # everything else fine-tunes slowly
])
```

### 4.2 Training Loop

Same structure as Phase 1, with these additions:

**Log Hopfield retrieval quality.** After each epoch, for a fixed set of
test pairs, log the cosine similarity between the query and retrieved pattern.
This tells you whether the Hopfield projections are learning useful
transformations or just staying near-identity.

```python
# After each epoch, log retrieval diagnostics:
with torch.no_grad():
    for test_pair in diagnostic_pairs:
        query = model.compute_pair_state(test_pair)
        retrieved = model.cell.hopfield.retrieve(query.unsqueeze(0), beta=1.0)
        cosine = F.cosine_similarity(query.unsqueeze(0), retrieved).item()
        logger.info(f"  Hopfield retrieval {test_pair}: cosine={cosine:.3f}")
```

**Track convergence speed improvement.** Phase 2 should converge FASTER
than Phase 1 on known interactions because retrieval is now direct.
Log average steps-to-converge per epoch and compare with Phase 1 baseline.

**Monitor for collapse.** If the Hopfield projections collapse (all queries
retrieve the same pattern), training has gone wrong. Check that retrieval
entropy stays above a minimum:

```python
# Retrieval entropy check
scores = beta * (q @ keys.T)
weights = softmax(scores)
entropy = -(weights * weights.log()).sum(dim=-1).mean()
# If entropy drops near 0, retrieval has collapsed to nearest-neighbor-always
# If entropy is at log(N), retrieval is uniform (useless)
# Good range: 0.5 to log(N)-1
```

### 4.3 Annealing Cycle (Optional but Recommended)

After initial Phase 2 fine-tuning, the encoder has shifted slightly from
Phase 1. This means the stored Hopfield patterns (computed from the Phase 1
encoder) are now slightly stale. For maximum performance:

```
Cycle 1: Rebuild Hopfield (Phase 1 encoder) → Fine-tune 30 epochs
Cycle 2: Rebuild Hopfield (Cycle 1 encoder) → Fine-tune 15 epochs
Cycle 3: Rebuild Hopfield (Cycle 2 encoder) → Fine-tune 10 epochs
```

Each cycle:
1. Freeze the model
2. Re-encode all interaction pairs with the current encoder
3. Replace the Hopfield patterns with fresh encodings
4. Unfreeze and fine-tune

Diminishing returns after 2-3 cycles. Stop when the Hopfield patterns
stop changing significantly between cycles (measure as average L2 distance
between old and new patterns).

```python
def should_stop_annealing(old_patterns: Tensor, new_patterns: Tensor, threshold: float = 0.01) -> bool:
    """Stop annealing when patterns stabilize."""
    drift = (old_patterns - new_patterns).norm(dim=-1).mean().item()
    logger.info(f"  Hopfield pattern drift: {drift:.4f}")
    return drift < threshold
```

---

## Step 5: Hopfield Growth Protocol

### 5.1 Adding New Interactions Without Retraining

This is one of the key architectural advantages. Once Phase 2 is trained,
new verified interactions can be added to the Hopfield without any gradient
descent:

```python
def add_interaction(
    model: PharmLoopModel,
    drug_a_id: int, drug_a_features: Tensor,
    drug_b_id: int, drug_b_features: Tensor,
    severity: str, mechanisms: list[str],
) -> None:
    """
    Add a new verified interaction to the Hopfield memory bank.
    No retraining required.
    """
    model.eval()
    with torch.no_grad():
        # Encode the pair
        enc_a = model.encoder(
            torch.tensor([drug_a_id]),
            drug_a_features.unsqueeze(0)
        )
        enc_b = model.encoder(
            torch.tensor([drug_b_id]),
            drug_b_features.unsqueeze(0)
        )
        pair_forward = torch.cat([enc_a, enc_b], dim=-1)
        pair_reverse = torch.cat([enc_b, enc_a], dim=-1)
        pair_state = (model.pair_combine(pair_forward) + model.pair_combine(pair_reverse)) / 2.0

        # Store in Hopfield
        model.cell.hopfield.store(pair_state)

        # Severity amplification for dangerous interactions
        if severity in ("severe", "contraindicated"):
            for _ in range(2):
                noisy = pair_state + torch.randn_like(pair_state) * 0.01
                model.cell.hopfield.store(noisy)
```

### 5.2 Adding New Drugs

New drugs require a feature vector but no retraining:

```python
def add_drug(
    model: PharmLoopModel,
    drug_name: str,
    features: Tensor,       # 64-dim pharmacological features
    known_interactions: list[dict],  # verified interactions with existing drugs
) -> int:
    """
    Add a new drug to the system.

    The drug gets an ID in the padding zone of the embedding table.
    Its identity embedding will be untrained (random init), which means
    the model will rely more heavily on the structured features for this drug.
    This is intentional: we trust the pharmacological features more than
    a random embedding.

    Returns the assigned drug ID.
    """
    # Assign next available padding ID
    new_id = model.num_drugs + next_padding_slot()

    # Add each known interaction to Hopfield
    for interaction in known_interactions:
        add_interaction(model, new_id, features, ...)

    return new_id
```

**Limitation:** The identity embedding for new drugs is random/untrained.
The model will rely entirely on structured features for new drugs. This is
actually fine for pharmacology — the 64-dim feature vector captures the
pharmacologically relevant properties. The identity embedding is for drug-specific
quirks that features miss, which matter less for a new drug with limited data.

### 5.3 Growth Validation

After adding new interactions, run a quick validation:

```python
def validate_growth(model, new_pair, expected_severity):
    """Check that the new interaction is retrievable and influences predictions."""
    model.eval()
    with torch.no_grad():
        output = model(new_pair.drug_a_id, new_pair.drug_a_features,
                       new_pair.drug_b_id, new_pair.drug_b_features)

    # The model should now have SOME signal about this pair
    # (retrieved pattern should influence convergence)
    assert output["converged"].item(), "New pair should converge after Hopfield addition"

    severity_pred = SEVERITY_NAMES[output["severity_logits"].argmax().item()]
    logger.info(f"New pair prediction: {severity_pred} (expected: {expected_severity})")
```

---

## Step 6: Validation

### 6.1 Three-Way Separation (Stricter Criteria)

Same three cases as Phase 1, but with tighter requirements:

```python
class TestPhase2Separation:
    """Phase 2 should show FASTER convergence and SHARPER separation."""

    def test_severe_interaction(self, model):
        """fluoxetine + tramadol: faster convergence than Phase 1."""
        output = run_pair(model, "fluoxetine", "tramadol")
        assert output["converged"]
        assert output["steps"] <= 8  # tighter: was 12 in Phase 1
        assert severity_pred in (SEVERITY_SEVERE, SEVERITY_CONTRAINDICATED)
        assert output["confidence"] > 0.8  # tighter: expect high confidence now

    def test_safe_pair(self, model):
        """metformin + lisinopril: fast convergence to none."""
        output = run_pair(model, "metformin", "lisinopril")
        assert output["converged"]
        assert output["steps"] <= 8
        assert severity_pred == SEVERITY_NONE

    def test_fabricated_drug(self, model):
        """QZ-7734 + aspirin: still fails to converge."""
        output = run_pair_fabricated(model, "aspirin")
        assert not output["converged"] or severity_pred == SEVERITY_UNKNOWN
        assert output["confidence"] < 0.1

    def test_gray_zone_gap_widened(self, model):
        """Phase 2 should have WIDER gap between known and unknown GZ."""
        known_gz = run_pair(model, "fluoxetine", "tramadol")["gray_zones"][-1]
        unknown_gz = run_pair_fabricated(model, "aspirin")["gray_zones"][-1]
        gap = unknown_gz - known_gz
        assert gap > 0.2, f"GZ gap ({gap:.3f}) should be wider than Phase 1"
```

### 6.2 Broad Accuracy Metrics

Phase 1 only needed the dynamics to separate three cases. Phase 2 should
show real accuracy improvements across the dataset:

```python
class TestPhase2Accuracy:
    """Broader accuracy metrics across the full dataset."""

    def test_severity_accuracy(self, model, test_data):
        """Severity classification accuracy on held-out pairs."""
        correct = 0
        total = 0
        for pair in test_data:
            pred = run_pair(model, pair.drug_a, pair.drug_b)
            if pred["severity"] == pair.true_severity:
                correct += 1
            total += 1
        accuracy = correct / total
        assert accuracy >= 0.70, f"Severity accuracy {accuracy:.1%} < 70% target"
        print(f"Severity accuracy: {accuracy:.1%}")

    def test_no_false_negatives_on_severe(self, model, test_data):
        """Zero tolerance for predicting 'none' on severe/contraindicated pairs."""
        severe_pairs = [p for p in test_data if p.severity in ("severe", "contraindicated")]
        false_negatives = 0
        for pair in severe_pairs:
            pred = run_pair(model, pair.drug_a, pair.drug_b)
            if pred["severity"] == "none":
                false_negatives += 1
                print(f"  FALSE NEGATIVE: {pair.drug_a} + {pair.drug_b} "
                      f"(true: {pair.severity}, pred: none)")
        assert false_negatives == 0, (
            f"{false_negatives} false negatives on severe/contraindicated pairs!"
        )

    def test_mechanism_accuracy(self, model, test_data):
        """Mechanism attribution accuracy (at least one correct mechanism)."""
        correct = 0
        applicable = 0
        for pair in test_data:
            if not pair.mechanisms:
                continue
            applicable += 1
            pred = run_pair(model, pair.drug_a, pair.drug_b)
            pred_mechs = set(pred["mechanisms"])
            true_mechs = set(pair.mechanisms)
            if pred_mechs & true_mechs:  # at least one overlap
                correct += 1
        accuracy = correct / max(applicable, 1)
        assert accuracy >= 0.60, f"Mechanism accuracy {accuracy:.1%} < 60% target"

    def test_convergence_rate(self, model, test_data):
        """Known pairs should converge at high rate; unknowns should not."""
        known_converged = 0
        known_total = 0
        for pair in test_data:
            if pair.severity != "unknown":
                pred = run_pair(model, pair.drug_a, pair.drug_b)
                if pred["converged"]:
                    known_converged += 1
                known_total += 1
        known_rate = known_converged / max(known_total, 1)
        assert known_rate >= 0.85, f"Known convergence rate {known_rate:.1%} < 85%"
```

### 6.3 Hopfield Retrieval Quality

```python
class TestPhase2Hopfield:
    """Verify the Hopfield is actually helping."""

    def test_similar_pairs_retrieve_each_other(self, model):
        """SSRI+opioid pairs should retrieve other SSRI+opioid patterns."""
        # fluoxetine+tramadol should retrieve patterns similar to
        # sertraline+codeine (same mechanism class)
        query = compute_pair_state(model, "fluoxetine", "tramadol")
        retrieved = model.cell.hopfield.retrieve(query.unsqueeze(0), beta=5.0)

        # Also encode sertraline+codeine
        reference = compute_pair_state(model, "sertraline", "codeine")

        # Cosine similarity should be high (both serotonergic + opioid)
        cosine = F.cosine_similarity(retrieved, reference.unsqueeze(0))
        assert cosine > 0.5, f"Similar pair retrieval cosine {cosine:.3f} < 0.5"

    def test_dissimilar_pairs_dont_retrieve(self, model):
        """SSRI+opioid should NOT strongly retrieve antihypertensive pairs."""
        query = compute_pair_state(model, "fluoxetine", "tramadol")
        reference = compute_pair_state(model, "metformin", "lisinopril")
        retrieved = model.cell.hopfield.retrieve(query.unsqueeze(0), beta=5.0)

        cosine = F.cosine_similarity(retrieved, reference.unsqueeze(0))
        assert cosine < 0.5, f"Dissimilar pair retrieval cosine {cosine:.3f} should be < 0.5"

    def test_retrieval_entropy_healthy(self, model):
        """Retrieval weights should not be collapsed or uniform."""
        queries = [compute_pair_state(model, p.drug_a, p.drug_b) for p in sample_pairs]
        queries = torch.stack(queries)

        q = model.cell.hopfield.query_proj(queries)
        keys = model.cell.hopfield.stored_keys[:model.cell.hopfield.count]
        scores = q @ keys.T
        weights = torch.softmax(scores, dim=-1)
        entropy = -(weights * (weights + 1e-10).log()).sum(dim=-1).mean()

        n = model.cell.hopfield.count
        max_entropy = torch.log(torch.tensor(float(n)))
        normalized_entropy = entropy / max_entropy

        # Should be between 0.1 and 0.9 — not collapsed, not uniform
        assert 0.05 < normalized_entropy < 0.95, (
            f"Retrieval entropy {normalized_entropy:.3f} is unhealthy"
        )
```

### 6.4 Convergence Speed Comparison

```python
def test_phase2_faster_than_phase1(phase1_model, phase2_model, test_pairs):
    """Phase 2 should converge faster on known interactions."""
    p1_steps = []
    p2_steps = []
    for pair in test_pairs:
        out1 = run_pair(phase1_model, pair.drug_a, pair.drug_b)
        out2 = run_pair(phase2_model, pair.drug_a, pair.drug_b)
        if out1["converged"] and out2["converged"]:
            p1_steps.append(out1["steps"])
            p2_steps.append(out2["steps"])

    avg_p1 = sum(p1_steps) / len(p1_steps)
    avg_p2 = sum(p2_steps) / len(p2_steps)
    print(f"Average steps — Phase 1: {avg_p1:.1f}, Phase 2: {avg_p2:.1f}")
    assert avg_p2 < avg_p1, "Phase 2 should converge faster than Phase 1"
```

---

## Step 7: Data Split

Phase 1 trained on all 209 interactions. Phase 2 needs proper evaluation,
so split the data:

```
Train: 80% of interaction pairs (~167 pairs)
Val:   10% (~21 pairs)
Test:  10% (~21 pairs)
```

**Splitting rules:**
- Stratify by severity (each split has proportional severe/moderate/mild/none)
- Ensure both drugs in a test pair appear in at least one training pair
  (we're testing generalization of interactions, not cold-start on unseen drugs)
- Keep fluoxetine+tramadol and metformin+lisinopril in the TEST set
  (for three-way comparison continuity)
- The fabricated drug test doesn't use the dataset — it's always available

Create this split deterministically (fixed seed) and save it:

```python
# data/processed/split.json
{
    "train_indices": [...],
    "val_indices": [...],
    "test_indices": [...],
    "seed": 42,
    "strategy": "stratified_by_severity"
}
```

Update `data_loader.py` to accept a split parameter.

---

## Implementation Files

### New Files
```
training/
    build_phase2.py       ← Phase 2 model construction + Hopfield rebuild
    train_phase2.py       ← Phase 2 training loop with annealing
    split_data.py         ← Create train/val/test split
tests/
    test_phase2.py        ← All Phase 2 validation tests
    test_hopfield_v2.py   ← Hopfield retrieval quality tests
```

### Modified Files
```
pharmloop/hopfield.py     ← Verify phase0=False path works correctly
pharmloop/model.py        ← Add compute_pair_state() helper method
training/data_loader.py   ← Support split parameter
```

### Unchanged Files
```
pharmloop/encoder.py      ← Weights transfer from Phase 1
pharmloop/oscillator.py   ← Architecture unchanged (projections auto-remove)
pharmloop/output.py       ← Architecture unchanged
training/loss.py          ← Same loss function
```

---

## Step 8: Implementation Sequence

Do these in order:

### 8.1 Data Split (do first — everything else needs it)
- Implement `training/split_data.py`
- Generate `data/processed/split.json`
- Update `training/data_loader.py` to filter by split
- Verify train/val/test proportions

### 8.2 Hopfield Rebuild Infrastructure
- Add `compute_pair_state()` method to `PharmLoopModel`
- Implement `training/build_phase2.py`:
  - Load Phase 1 checkpoint
  - Compute pair patterns for all training interactions
  - Deduplicate and severity-amplify
  - Build new 512-dim Hopfield with identity-init projections
  - Store patterns
  - Transfer Phase 1 weights
  - Save Phase 2 initial checkpoint

### 8.3 Phase 2 Training Loop
- Implement `training/train_phase2.py`:
  - Dual learning rate param groups
  - Retrieval quality logging
  - Retrieval entropy monitoring
  - Convergence speed tracking vs Phase 1 baseline
  - Annealing cycle support (rebuild → fine-tune → rebuild)
  - Validation loss after each epoch
  - Save best model by validation loss

### 8.4 Tests
- Implement `tests/test_phase2.py` with all validation tests above
- Implement `tests/test_hopfield_v2.py` for retrieval quality
- Run the full test suite

---

## Validation Criteria (must pass before Phase 3)

- [ ] Phase 2 Hopfield contains ~200-250 pair patterns in 512-dim space
- [ ] Dimension projection layers are removed (OscillatorCell talks directly to 512-dim Hopfield)
- [ ] Three-way separation passes with STRICTER criteria (convergence in ≤8 steps)
- [ ] Severity accuracy ≥ 70% on held-out test set
- [ ] ZERO false negatives on severe/contraindicated pairs in test set
- [ ] Mechanism accuracy ≥ 60% (at least one correct mechanism) on test set
- [ ] Known pair convergence rate ≥ 85%
- [ ] Phase 2 converges faster (fewer avg steps) than Phase 1 on same test pairs
- [ ] Hopfield retrieval entropy is healthy (not collapsed, not uniform)
- [ ] At least one annealing cycle completed (rebuild with updated encoder)
- [ ] Model total params still < 6M
- [ ] Growth protocol works: can add a new interaction and it influences predictions

---

## What NOT to Do in Phase 2

- Don't implement the template engine (Phase 3)
- Don't implement the context encoder (Phase 3+)
- Don't expand the drug set beyond 50 (save for Phase 4)
- Don't try to optimize inference speed — correctness first
- Don't skip the annealing cycle — it matters more than extra training epochs
- Don't store converged states in Hopfield (store initial pair states — see reasoning in Step 1.1)
