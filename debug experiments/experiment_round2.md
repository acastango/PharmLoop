# Experiment Round 2: Testing the Dynamics Directly

## Context

Round 1 tested whether the output head's predictions improve with depth.
They don't — the output head learned to read the encoder's first
position and ignore the oscillator entirely.

But the oscillator IS producing differentiated behavior:
- 254/512 dims oscillating for fabricated vs 60/512 for known
- Velocity separation: 0.35 (fabricated) vs 0.10 (known)
- Structured partial convergence

Round 2 asks: **is the oscillator doing real work that the output head
ignores?** These experiments evaluate the DYNAMICS, not the classifier.

---

## Experiment 4: Trajectory Information Content

### Question
Does the oscillator trajectory contain information about the interaction
that the output head doesn't use? If we train a FRESH classifier on
trajectory features instead of final position, does it outperform one
trained on encoder output alone?

### Procedure

Extract trajectory features for every test example, then compare
classifiers trained on different feature sets.

```python
"""
experiment_4_trajectory_information.py

Compare classification accuracy using three different feature sets:
  A) Encoder output only (position at step 0) — what the output head uses
  B) Trajectory statistics — what the oscillator produces
  C) Full trajectory — everything

If B or C outperform A, the oscillator contains information
that the output head is ignoring.
"""

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

def extract_features(model, dataloader, max_steps=16):
    """Run model and extract all three feature sets."""
    
    features_A = []  # encoder output (step 0 position)
    features_B = []  # trajectory statistics
    features_C = []  # full trajectory flattened
    labels_severity = []
    labels_mechanism = []
    
    for batch in dataloader:
        with torch.no_grad():
            output = model(batch, return_trajectory=True)
        
        traj = output['trajectory']
        positions = traj['positions']    # list of [batch, 512] tensors
        velocities = traj['velocities']  # list of [batch, 512] tensors
        
        for i in range(len(batch)):
            # A: Just encoder output
            step0_position = positions[0][i].cpu().numpy()
            features_A.append(step0_position)
            
            # B: Trajectory statistics (what the oscillator uniquely produces)
            pos_stack = torch.stack([p[i] for p in positions]).cpu().numpy()
            vel_stack = torch.stack([v[i] for v in velocities]).cpu().numpy()
            
            traj_features = np.concatenate([
                # Final position (512)
                pos_stack[-1],
                
                # Final velocity (512) — the uncertainty signal
                vel_stack[-1],
                
                # Per-dimension convergence step: first step where
                # |v_dim| < threshold (512)
                compute_convergence_step(vel_stack, threshold=0.05),
                
                # Velocity trajectory statistics per dimension
                vel_stack.mean(axis=0),    # mean velocity (512)
                vel_stack.std(axis=0),     # velocity variance (512)
                np.abs(vel_stack).max(axis=0),  # peak velocity (512)
                
                # Position trajectory statistics
                pos_stack.std(axis=0),     # position variance (512)
                                           # high = oscillated a lot
                
                # Direction changes per dimension (512)
                count_direction_changes(vel_stack),
                
                # Global scalars
                np.array([
                    np.linalg.norm(vel_stack[-1]),     # final |v|
                    np.linalg.norm(vel_stack[0]),      # initial |v|
                    (np.abs(vel_stack[-1]) < 0.05).sum() / 512,  # frac converged
                    count_total_flips(vel_stack),       # total direction changes
                    compute_settling_time(vel_stack),   # steps to 90% convergence
                ]),
            ])
            features_B.append(traj_features)
            
            # C: Full trajectory flattened (expensive but complete)
            # Subsample to keep manageable: steps 0, 2, 4, 8, 12, 16
            sample_steps = [0, 2, 4, 8, 12, min(15, len(positions)-1)]
            full_traj = np.concatenate([
                np.concatenate([pos_stack[s], vel_stack[s]])
                for s in sample_steps if s < len(pos_stack)
            ])
            features_C.append(full_traj)
            
            labels_severity.append(batch['severity'][i].item())
            labels_mechanism.append(batch['mechanism'][i].item())
    
    return (np.array(features_A), np.array(features_B), 
            np.array(features_C), np.array(labels_severity),
            np.array(labels_mechanism))


def compute_convergence_step(vel_stack, threshold=0.05):
    """Per-dimension: first step where |v| drops below threshold."""
    n_steps, n_dims = vel_stack.shape
    result = np.full(n_dims, n_steps, dtype=float)  # default: never converged
    for step in range(n_steps):
        newly_converged = (np.abs(vel_stack[step]) < threshold) & (result == n_steps)
        result[newly_converged] = step
    return result / n_steps  # normalize to [0, 1]


def count_direction_changes(vel_stack):
    """Per-dimension: how many times velocity changes sign."""
    signs = np.sign(vel_stack)
    changes = np.abs(np.diff(signs, axis=0))
    return (changes > 0).sum(axis=0).astype(float)


def count_total_flips(vel_stack):
    """Total direction changes across all dimensions."""
    signs = np.sign(vel_stack)
    return float(np.abs(np.diff(signs, axis=0)).sum())


def compute_settling_time(vel_stack, target_frac=0.9, threshold=0.05):
    """Step at which 90% of dimensions have converged."""
    n_steps, n_dims = vel_stack.shape
    for step in range(n_steps):
        frac = (np.abs(vel_stack[step]) < threshold).sum() / n_dims
        if frac >= target_frac:
            return step / n_steps
    return 1.0


# --- Run comparison ---
features_A, features_B, features_C, labels_sev, labels_mech = \
    extract_features(model, test_loader)

# Train fresh logistic regression on each feature set
# Use 5-fold cross-validation
for name, X in [("A: Encoder only", features_A),
                ("B: Trajectory stats", features_B),
                ("C: Full trajectory", features_C)]:
    
    sev_scores = cross_val_score(
        LogisticRegression(max_iter=1000, C=1.0),
        X, labels_sev, cv=5, scoring='accuracy'
    )
    mech_scores = cross_val_score(
        LogisticRegression(max_iter=1000, C=1.0),
        X, labels_mech, cv=5, scoring='accuracy'
    )
    
    print(f"{name}:")
    print(f"  Severity:  {sev_scores.mean():.1%} +/- {sev_scores.std():.1%}")
    print(f"  Mechanism: {mech_scores.mean():.1%} +/- {mech_scores.std():.1%}")
```

### What to Report

1. Severity accuracy for A vs B vs C
2. Mechanism accuracy for A vs B vs C
3. If B > A: **the oscillator trajectory contains information the
   output head ignores.** The dynamics are doing real work.
4. If B = A: the oscillator genuinely adds nothing. The encoder
   solved everything.
5. If C > B: the full trajectory contains structure beyond summary
   statistics — the specific path matters, not just its properties.

### Why This Matters

If a fresh linear probe on trajectory features outperforms one on
encoder output, then the oscillator is producing useful computation
that the current output head bypasses. The fix is architectural
(remove the shortcut), not fundamental (rebuild the oscillator).

---

## Experiment 5: Hopfield Retrieval Analysis

### Question
Is the Hopfield bank contributing to the dynamics, or is the
encoder output already so close to the right attractor that
retrieval is a no-op?

### Procedure

Measure the distance between encoder output and Hopfield retrieval
at each step. If retrieval returns something meaningfully different
from the current state, the Hopfield bank is providing new
information. If it returns near-identical vectors, it's an echo.

```python
"""
experiment_5_hopfield_analysis.py

For each test example, measure:
  1. Cosine similarity between position and Hopfield retrieval at each step
  2. Force magnitude (how hard the retrieval pulls the state)
  3. Whether different patterns are retrieved at different steps
"""

def analyze_hopfield_contribution(model, dataloader, max_steps=16):
    """Measure what the Hopfield bank actually contributes."""
    
    results = []
    
    for batch in dataloader:
        with torch.no_grad():
            # Need to instrument the reasoning loop to capture
            # hopfield queries and retrievals at each step
            
            # For each step, record:
            #   - current position x(t)
            #   - hopfield retrieval h(t)
            #   - cosine similarity cos(x(t), h(t))
            #   - force magnitude |spring * evidence_transform(cat(x, h))|
            #   - nearest pattern ID in hopfield bank
            
            # Hook into reasoning loop
            step_data = []
            
            x = initial_state  # from encoder
            v = torch.zeros_like(x)
            
            for step in range(max_steps):
                # Query hopfield
                h = model.hopfield.retrieve(x)
                
                # Cosine similarity
                cos_sim = torch.nn.functional.cosine_similarity(
                    x, h, dim=-1
                ).mean().item()
                
                # Force magnitude
                mismatch = torch.cat([x, h], dim=-1)
                force = model.reasoning_loop.cell.spring * \
                        model.reasoning_loop.cell.evidence_transform(mismatch)
                force_mag = force.norm(dim=-1).mean().item()
                
                # Nearest pattern in bank
                # Compute cosine sim with all stored patterns
                patterns = model.hopfield.memory  # [N, 512]
                sims = torch.nn.functional.cosine_similarity(
                    x.unsqueeze(1), patterns.unsqueeze(0), dim=-1
                )
                nearest_pattern_id = sims.argmax(dim=-1)[0].item()
                nearest_sim = sims.max(dim=-1)[0].mean().item()
                
                step_data.append({
                    'step': step,
                    'cos_sim_state_retrieval': cos_sim,
                    'force_magnitude': force_mag,
                    'nearest_pattern_id': nearest_pattern_id,
                    'nearest_pattern_sim': nearest_sim,
                })
                
                # Oscillator update
                x, v, gz = model.reasoning_loop.cell(x, v, h, step)
            
            results.append({
                'steps': step_data,
                'pattern_ids_visited': [s['nearest_pattern_id'] for s in step_data],
                'unique_patterns': len(set(s['nearest_pattern_id'] for s in step_data)),
                'cos_sim_trajectory': [s['cos_sim_state_retrieval'] for s in step_data],
                'force_trajectory': [s['force_magnitude'] for s in step_data],
            })
    
    return results

results = analyze_hopfield_contribution(model, test_loader)

# --- Analysis ---
unique_patterns = [r['unique_patterns'] for r in results]
print(f"Unique patterns visited per example:")
print(f"  Mean: {np.mean(unique_patterns):.1f}")
print(f"  Just 1 pattern: {unique_patterns.count(1)}/{len(unique_patterns)}")
print(f"  2+ patterns: {sum(1 for u in unique_patterns if u >= 2)}/{len(unique_patterns)}")

cos_sims_step0 = [r['cos_sim_trajectory'][0] for r in results]
cos_sims_final = [r['cos_sim_trajectory'][-1] for r in results]
print(f"\nCosine sim (state vs retrieval):")
print(f"  Step 0: {np.mean(cos_sims_step0):.3f}")
print(f"  Final:  {np.mean(cos_sims_final):.3f}")

force_step0 = [r['force_trajectory'][0] for r in results]
force_final = [r['force_trajectory'][-1] for r in results]
print(f"\nForce magnitude:")
print(f"  Step 0: {np.mean(force_step0):.4f}")
print(f"  Final:  {np.mean(force_final):.4f}")
```

### What to Report

1. **Unique patterns visited.** If every example visits only 1 pattern
   across all 16 steps, the state never leaves the first basin.
   The oscillator is just refining toward the nearest attractor.
   If examples visit 2+ patterns, the oscillator is exploring
   multiple basins — genuine multi-hypothesis reasoning.

2. **Cosine similarity at step 0.** If the encoder output is already
   0.95+ cosine similar to the Hopfield retrieval, the encoder has
   already solved the problem. The retrieval adds nothing. If it's
   0.6-0.8, the Hopfield bank is providing genuinely new information.

3. **Force magnitude over time.** If force drops to near-zero after
   step 1, the Hopfield bank stops contributing immediately. If force
   stays nonzero across multiple steps, there's ongoing tension between
   state and memory — that's real oscillatory dynamics.

4. **Force trajectory shape.** Does it monotonically decay (settling)?
   Oscillate (exploring)? Stay flat (constant tension, no resolution)?

---

## Experiment 6: Bypass Test

### Question
If we delete the output head entirely and classify based on
the nearest Hopfield pattern at convergence, how does accuracy
compare? This tests the original architectural intent: the answer
is the pattern you converge to, not a classification head's reading
of the final state.

### Procedure

```python
"""
experiment_6_bypass_test.py

Remove the output head. Classify by:
  1. Find nearest Hopfield pattern to final position
  2. Look up that pattern's severity and mechanism labels
  3. Compare to ground truth

This is how the system was SUPPOSED to work according to the
original spec: the answer is which attractor you converge to.
"""

def classify_by_hopfield_match(model, dataloader, max_steps=16):
    """Classify by nearest Hopfield pattern instead of output head."""
    
    # Build pattern → label lookup
    # Each stored Hopfield pattern corresponds to a specific drug pair
    # interaction with known severity and mechanism
    pattern_labels = build_pattern_label_map(model, training_data)
    
    results = {
        'hopfield_severity_correct': 0,
        'hopfield_mechanism_correct': 0,
        'head_severity_correct': 0,
        'head_mechanism_correct': 0,
        'agreement': 0,  # hopfield and head agree
        'total': 0,
    }
    
    for batch in dataloader:
        with torch.no_grad():
            output = model(batch, return_trajectory=True)
        
        final_position = output['trajectory']['positions'][-1]
        
        # Output head prediction (what we currently use)
        head_severity = output['severity'].argmax(dim=-1)
        head_mechanism = output['mechanism'].argmax(dim=-1)
        
        # Hopfield match prediction (the original intent)
        patterns = model.hopfield.memory  # [N, 512]
        sims = torch.nn.functional.cosine_similarity(
            final_position.unsqueeze(1),
            patterns.unsqueeze(0),
            dim=-1
        )  # [batch, N]
        
        nearest_ids = sims.argmax(dim=-1)  # [batch]
        nearest_sims = sims.max(dim=-1)[0]  # [batch]
        
        for i in range(len(batch)):
            true_sev = batch['severity'][i].item()
            true_mech = batch['mechanism'][i].item()
            
            pattern_id = nearest_ids[i].item()
            match_sim = nearest_sims[i].item()
            
            hopfield_sev, hopfield_mech = pattern_labels.get(
                pattern_id, (None, None)
            )
            
            if hopfield_sev is not None:
                if hopfield_sev == true_sev:
                    results['hopfield_severity_correct'] += 1
                if hopfield_mech == true_mech:
                    results['hopfield_mechanism_correct'] += 1
            
            if head_severity[i].item() == true_sev:
                results['head_severity_correct'] += 1
            if head_mechanism[i].item() == true_mech:
                results['head_mechanism_correct'] += 1
            
            if hopfield_sev == head_severity[i].item():
                results['agreement'] += 1
            
            results['total'] += 1
    
    return results


# Also run at multiple depths
for max_steps in [2, 4, 8, 16, 32]:
    model.reasoning_loop.max_steps = max_steps
    r = classify_by_hopfield_match(model, test_loader, max_steps)
    
    h_sev = r['hopfield_severity_correct'] / r['total']
    h_mech = r['hopfield_mechanism_correct'] / r['total']
    o_sev = r['head_severity_correct'] / r['total']
    o_mech = r['head_mechanism_correct'] / r['total']
    agree = r['agreement'] / r['total']
    
    print(f"Steps={max_steps:>3}: "
          f"Hopfield sev={h_sev:.1%} mech={h_mech:.1%} | "
          f"Head sev={o_sev:.1%} mech={o_mech:.1%} | "
          f"Agree={agree:.1%}")
```

### What to Report

1. **Hopfield-match accuracy vs output-head accuracy.** If they're
   close, the system is doing what the spec intended — the answer
   IS the attractor. If the head is much better, the encoder learned
   representations that the Hopfield bank doesn't align with.

2. **CRITICAL: Does Hopfield-match accuracy improve with depth?**
   This is the real depth ablation. The output head saturates at
   step 2. Does Hopfield-matching improve with more steps?
   If yes, the oscillator IS converging to better attractors over
   time — the dynamics are computational — but the output head
   bypasses this work.

3. **Agreement rate.** How often do the head and Hopfield match
   agree? If they rarely agree, the encoder has drifted away from
   the Hopfield landscape entirely — two parallel systems.

4. **Hopfield-match on fabricated drugs.** Run the fabricated pairs
   from Experiment 2. What patterns do they match? How similar?
   Low similarity = no attractor (good). High similarity to a
   specific interaction = hallucination.

---

## Experiment 7: Dimensional Semantics

### Question
The 254 oscillating dimensions on fabricated pairs — are they
random or structured? Do the same dimensions oscillate for the
same drug classes?

This is the test for whether the convergence stream has semantic
content that could function as a communication medium.

### Procedure

```python
"""
experiment_7_dimensional_semantics.py

For every test pair (known and fabricated), record which dimensions
converge and which oscillate. Then check:
  1. Are the oscillating dimensions consistent within drug classes?
  2. Do specific dimension ranges map to specific pharmacological
     properties?
  3. Can you predict drug class from convergence pattern alone?
"""

def get_convergence_mask(model, drug_a_id, drug_b_id, max_steps=16,
                          threshold=0.05):
    """Binary mask: 1 = converged, 0 = still oscillating."""
    with torch.no_grad():
        output = model(drug_a_id, drug_b_id, return_trajectory=True)
    
    final_velocity = output['trajectory']['velocities'][-1]
    mask = (final_velocity.abs() < threshold).squeeze().cpu().numpy()
    return mask  # [512] boolean

# --- Collect masks for all groups ---
# Known interacting pairs, grouped by drug class
class_masks = {}  # class_pair → list of masks
for drug_a, drug_b, class_a, class_b in test_pairs_with_classes:
    mask = get_convergence_mask(model, drug_a, drug_b)
    key = tuple(sorted([class_a, class_b]))
    class_masks.setdefault(key, []).append(mask)

# Fabricated pairs
fabricated_masks = [
    get_convergence_mask(model, a, b)
    for a, b in fabricated_pairs
]

# --- Analysis 1: Within-class consistency ---
for class_pair, masks in class_masks.items():
    masks_array = np.array(masks)  # [N, 512]
    
    # Per-dimension convergence frequency
    convergence_freq = masks_array.mean(axis=0)  # [512]
    
    # Dimensions that ALWAYS converge for this class (>95%)
    always_converged = (convergence_freq > 0.95).sum()
    # Dimensions that NEVER converge for this class (<5%)
    never_converged = (convergence_freq < 0.05).sum()
    # Dimensions that are VARIABLE (between 5% and 95%)
    variable = 512 - always_converged - never_converged
    
    print(f"{class_pair}: always={always_converged}, "
          f"never={never_converged}, variable={variable}")

# --- Analysis 2: Cross-class comparison ---
# Do different drug class pairs have different convergence signatures?
# Compute Jaccard similarity between convergence masks of different classes
from itertools import combinations
class_signatures = {}
for class_pair, masks in class_masks.items():
    # Average mask for this class
    class_signatures[class_pair] = np.array(masks).mean(axis=0) > 0.5

print("\nCross-class Jaccard similarity:")
for (c1, c2) in combinations(class_signatures.keys(), 2):
    m1, m2 = class_signatures[c1], class_signatures[c2]
    intersection = (m1 & m2).sum()
    union = (m1 | m2).sum()
    jaccard = intersection / max(union, 1)
    print(f"  {c1} vs {c2}: {jaccard:.2f}")

# --- Analysis 3: Can you predict class from mask? ---
# Train a simple classifier on convergence masks → drug class pair
all_masks = []
all_labels = []
for class_pair, masks in class_masks.items():
    for mask in masks:
        all_masks.append(mask.astype(float))
        all_labels.append(str(class_pair))

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

rf_scores = cross_val_score(
    RandomForestClassifier(n_estimators=100),
    np.array(all_masks), all_labels, cv=5, scoring='accuracy'
)
print(f"\nClass prediction from convergence mask: {rf_scores.mean():.1%}")

# --- Analysis 4: Fabricated pair mask structure ---
fab_masks_array = np.array(fabricated_masks)
fab_convergence_freq = fab_masks_array.mean(axis=0)

# Which dimensions converge even for fabricated drugs?
# These are dimensions that don't depend on knowing the drug pair
always_converge_on_fake = (fab_convergence_freq > 0.95)
never_converge_on_fake = (fab_convergence_freq < 0.05)

print(f"\nFabricated pairs:")
print(f"  Dims that converge even on fakes: {always_converge_on_fake.sum()}")
print(f"  Dims that never converge on fakes: {never_converge_on_fake.sum()}")
print(f"  Variable dims: {512 - always_converge_on_fake.sum() - never_converge_on_fake.sum()}")

# Map converging-on-fake dims to feature ranges
# (need the feature dimension mapping from the encoder)
print(f"\n  Indices of always-converge-on-fake dims: "
      f"{np.where(always_converge_on_fake)[0].tolist()[:20]}...")
print(f"  Indices of never-converge-on-fake dims: "
      f"{np.where(never_converge_on_fake)[0].tolist()[:20]}...")
```

### What to Report

1. **Within-class consistency.** For each drug class pair, how many
   dimensions always/never/variably converge? High consistency =
   structured convergence. Low consistency = noise.

2. **Cross-class differentiation.** Low Jaccard similarity between
   different class pairs means different classes produce different
   convergence signatures. The convergence pattern encodes drug class.

3. **Class prediction accuracy from mask alone.** If you can predict
   drug class pair from just the binary convergence mask at >50%
   accuracy, the mask carries semantic information. At >80%, the
   convergence pattern is a structured representation.

4. **Fabricated pair structure.** Which dims converge even on fakes?
   These are dimensions that encode something about the architecture
   itself (maybe structural priors) rather than anything about the
   input drugs. They're the "syntax" of the convergence stream.
   The dims that never converge on fakes are the "content" dims —
   they require real drug knowledge.

---

## Experiment 8: Output Head Dependency Test

### Question
Can we force the output head to depend on late-step dynamics
by giving it only late-step state? If accuracy improves when
the output head can ONLY see steps 8-16 vs steps 0-2, the
late dynamics contain useful information.

### Procedure

```python
"""
experiment_8_dependency_test.py

Retrain a fresh output head (freeze everything else) on:
  A) Position at step 0 only
  B) Position at step 8 only
  C) Position at step 16 only
  D) Mean position over steps 8-16
  E) Concatenation of position + velocity at step 16

Compare accuracy. If C or D outperform A, late dynamics carry
information. If A wins, the encoder did everything.
"""

import torch
import torch.nn as nn

class ProbeHead(nn.Module):
    """Simple linear probe for severity classification."""
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.linear(x)

# Extract features at different steps
# Freeze the full model, collect intermediate states
features_by_step = {step: [] for step in [0, 2, 4, 8, 12, 16]}
late_mean_features = []
pos_vel_features = []
labels = []

for batch in train_loader:
    with torch.no_grad():
        output = model(batch, return_trajectory=True)
    
    positions = output['trajectory']['positions']
    velocities = output['trajectory']['velocities']
    
    for step in features_by_step:
        if step < len(positions):
            features_by_step[step].append(positions[step].cpu())
    
    # Mean of steps 8-16
    late_pos = torch.stack(positions[8:]).mean(dim=0)
    late_mean_features.append(late_pos.cpu())
    
    # Position + velocity at final step
    pv = torch.cat([positions[-1], velocities[-1]], dim=-1)
    pos_vel_features.append(pv.cpu())
    
    labels.append(batch['severity'].cpu())

# Train fresh probes
probes = {
    "Step 0": (torch.cat(features_by_step[0]), 512),
    "Step 2": (torch.cat(features_by_step[2]), 512),
    "Step 4": (torch.cat(features_by_step[4]), 512),
    "Step 8": (torch.cat(features_by_step[8]), 512),
    "Step 16": (torch.cat(features_by_step[16]), 512),
    "Late mean (8-16)": (torch.cat(late_mean_features), 512),
    "Pos+Vel final": (torch.cat(pos_vel_features), 1024),
}

all_labels = torch.cat(labels)

for name, (feats, dim) in probes.items():
    probe = ProbeHead(dim, num_severity_classes)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
    
    # Train for 50 epochs
    dataset = torch.utils.data.TensorDataset(feats, all_labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    
    for epoch in range(50):
        for X, y in loader:
            loss = nn.CrossEntropyLoss()(probe(X), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Evaluate on test set
    # (collect test features the same way, run through trained probe)
    test_acc = evaluate_probe(probe, test_features[name], test_labels)
    print(f"{name:>20}: severity accuracy = {test_acc:.1%}")
```

### What to Report

1. **Accuracy at each step.** If step 0 and step 16 are the same,
   the oscillator adds nothing. If step 16 > step 0, the dynamics
   improve the representation.

2. **Late mean vs any single step.** If averaging steps 8-16
   outperforms any single step, the trajectory contains information
   not available at any one point — the dynamics are genuinely
   computational.

3. **Position+Velocity vs position alone.** If adding velocity at
   step 16 helps, then uncertainty information improves classification.
   The system knows something it can only express through velocity.

4. **The gap between step 0 and step 16 for MECHANISM accuracy
   specifically.** Severity might be solvable from encoder output.
   Mechanism is harder. If late-step states improve mechanism
   accuracy, the oscillator is doing the harder reasoning work
   even if severity is trivially solved.

---

## Directory Structure

```
experiments/
    experiment_4_trajectory_information.py
    experiment_5_hopfield_analysis.py
    experiment_6_bypass_test.py
    experiment_7_dimensional_semantics.py
    experiment_8_dependency_test.py
    round2_results/
        trajectory_information.json
        hopfield_analysis.json
        bypass_test.json
        dimensional_semantics.json
        dependency_test.json
```

---

## Interpretation Guide

### The dynamics are real and useful if:
- Experiment 4: Trajectory features (B) outperform encoder-only (A)
- Experiment 5: Multiple Hopfield patterns visited per example
- Experiment 6: Hopfield-match accuracy improves with depth
- Experiment 7: Convergence mask predicts drug class at >50%
- Experiment 8: Late-step probes outperform step-0 probes

### The output head is the problem if:
- Experiments 4-7 show the dynamics carry information
- Experiment 8 shows late states are more informative
- But Experiment 1 (from Round 1) showed the output head doesn't benefit

This combination would confirm: the oscillator does real work,
the output head bypasses it. The fix is architectural.

### The oscillator is genuinely decorative if:
- Experiment 4: Encoder-only matches trajectory features
- Experiment 5: Same pattern retrieved every step
- Experiment 6: Hopfield-match doesn't improve with depth
- Experiment 7: Convergence mask is random, no class prediction
- Experiment 8: Step 0 = Step 16

This would mean the encoder truly solved the problem alone.
The oscillator adds nothing. Revisit the entire architecture.

### The most likely outcome (prediction):
The dynamics carry information (Exp 4-7 positive) but the output
head doesn't use it (Exp 8 shows small or no improvement at late
steps). This confirms the "output head shortcut" hypothesis and
points to the architectural fix: remove the head, classify by
attractor identity.
