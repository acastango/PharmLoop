# Experiment: Is PharmLoop a Resolver or a Classifier?

## Context

Phase 4b metrics are excellent (87.6% severity, 83.4% mechanism, 0% FNR,
100% convergence). But 100% convergence is a red flag. If EVERYTHING
converges — including inputs the system has never seen — then the safety
property ("structurally cannot hallucinate") may not hold. The system
may be converging onto the nearest attractor for inputs it should reject.

These three experiments determine what kind of system we actually built.
Run them in order. Each one's results inform how to interpret the next.

---

## Experiment 1: Depth Ablation

### Question
Does accuracy improve with more recurrence steps, or does it saturate?

- If it saturates at ~6 steps → the recurrence is iterative refinement
  (lookup with extra steps)
- If it keeps climbing → the recurrence is genuinely computational
  (each step adds representational power)

### Procedure

Load the best Phase 4b checkpoint. Run the full test set at each
depth setting. Do NOT retrain — just change `max_steps` at inference.

```python
"""
experiment_1_depth_ablation.py

Load Phase 4b best checkpoint.
Run test set at max_steps = 2, 4, 6, 8, 12, 16, 24, 32, 48, 64.
Record: severity_accuracy, mechanism_accuracy, FNR, convergence_rate.
"""

import torch
import json
from pathlib import Path

# Load model and test data
# Use the same test split from Phase 4b training

DEPTH_VALUES = [2, 4, 6, 8, 12, 16, 24, 32, 48, 64]

results = []

for max_steps in DEPTH_VALUES:
    # Override model.reasoning_loop.max_steps
    model.reasoning_loop.max_steps = max_steps
    
    # Run full test set
    severity_correct = 0
    mechanism_correct = 0
    false_negatives = 0  # severe/contra predicted as none
    converged = 0
    total = 0
    severe_total = 0
    
    for batch in test_loader:
        with torch.no_grad():
            output = model(batch)
        
        # Accumulate metrics
        # ... (use same metric computation as training)
    
    results.append({
        "max_steps": max_steps,
        "severity_accuracy": severity_correct / total,
        "mechanism_accuracy": mechanism_correct / total,
        "fnr": false_negatives / max(severe_total, 1),
        "convergence_rate": converged / total,
    })

# Save results
with open("experiments/depth_ablation_results.json", "w") as f:
    json.dump(results, f, indent=2)

# Print table
print(f"{'Steps':>6} | {'Sev Acc':>8} | {'Mech Acc':>8} | {'FNR':>6} | {'Conv':>6}")
print("-" * 50)
for r in results:
    print(f"{r['max_steps']:>6} | {r['severity_accuracy']:>7.1%} | "
          f"{r['mechanism_accuracy']:>7.1%} | {r['fnr']:>5.1%} | "
          f"{r['convergence_rate']:>5.1%}")
```

### What to Report

1. The full table of metrics vs depth
2. At what depth does severity accuracy reach within 1% of its maximum?
3. At what depth does mechanism accuracy reach within 1% of its maximum?
4. Does convergence rate decrease at lower depths? (It should — fewer
   steps means less time to settle)
5. Does accuracy IMPROVE beyond the default max_steps=16? If going to
   32 or 64 helps, the recurrence is doing real work.

---

## Experiment 2: Velocity Distribution at Timeout

### Question
When inputs don't converge (or barely converge), what does velocity
look like?

- High velocity at timeout → no attractor exists, genuine uncertainty
- Low velocity just above threshold → attractor exists, threshold is
  the only safety mechanism

### Procedure

Create 100 fabricated drug pairs using random IDs that are NOT in the
drug registry. Also select 100 known test pairs (50 with known
interactions, 50 known safe). Run all 300 inputs through the model
at max_steps=16 and record the full velocity state at every step.

```python
"""
experiment_2_velocity_distribution.py

Three groups:
  A) 100 fabricated drug pairs (random IDs not in registry)
  B) 50 known interacting pairs from test set
  C) 50 known safe pairs from test set

For each: record velocity magnitude at every step.
"""

import torch
import numpy as np

# --- Generate fabricated pairs ---
# Use drug IDs that are definitely NOT in the model's vocabulary
# e.g., IDs starting from max_drug_id + 1000
max_known_id = max(drug_registry.values())
fabricated_pairs = []
for i in range(100):
    fake_a = max_known_id + 1000 + (i * 2)
    fake_b = max_known_id + 1000 + (i * 2) + 1
    fabricated_pairs.append((fake_a, fake_b))

# Also try: known drugs in UNKNOWN combinations
# (both drugs exist but this specific pair was never in training)
# This is a separate and important test case
novel_pairs = []
# Find drug pairs where both drugs are in registry but the pair
# is not in interactions_v3.json
# Collect ~50 of these

# --- Run with trajectory recording ---
def run_with_trajectory(model, drug_a_id, drug_b_id, max_steps=16):
    """Run model and record full trajectory."""
    # Need to hook into the reasoning loop to capture per-step state
    # Record at each step:
    #   - position (x)
    #   - velocity (v)
    #   - velocity magnitude (|v|, scalar)
    #   - per-dimension velocity magnitude
    #   - gray zone scalar
    #   - predicted severity class
    #   - predicted mechanism class
    
    velocities = []
    positions = []
    predictions = []
    
    # Hook the reasoning loop's per-step output
    # Implementation depends on model internals — may need to
    # modify ReasoningLoop.forward() to return full trajectory
    # (it may already do this via trajectory dict)
    
    with torch.no_grad():
        output = model(drug_a_id, drug_b_id, return_trajectory=True)
    
    trajectory = output['trajectory']
    
    return {
        'velocity_magnitudes': [v.norm().item() for v in trajectory['velocities']],
        'per_dim_velocity': [v.abs().cpu().numpy() for v in trajectory['velocities']],
        'gray_zones': [gz.item() for gz in trajectory['gray_zones']],
        'final_velocity': trajectory['velocities'][-1].norm().item(),
        'converged': output['converged'],
        'predicted_severity': output['severity'].argmax().item(),
        'confidence': output['confidence'],
    }

# --- Run all three groups ---
fabricated_results = [run_with_trajectory(model, a, b) for a, b in fabricated_pairs]
known_interact_results = [run_with_trajectory(model, a, b) for a, b in known_interacting]
known_safe_results = [run_with_trajectory(model, a, b) for a, b in known_safe]
novel_pair_results = [run_with_trajectory(model, a, b) for a, b in novel_pairs]

# --- Analysis ---
# 1. Histogram of final velocity magnitude for each group
# 2. Where does the convergence threshold sit relative to these distributions?
# 3. Do the distributions overlap? (bad — means threshold is doing all the work)
#    Or are they cleanly separated? (good — means dynamics are doing the work)

# Save everything
results = {
    'fabricated': fabricated_results,
    'known_interacting': known_interact_results,
    'known_safe': known_safe_results,
    'novel_pairs': novel_pair_results,
    'convergence_threshold': model.reasoning_loop.convergence_threshold.item(),
}

with open("experiments/velocity_distribution_results.json", "w") as f:
    json.dump(results, f, indent=2)
```

### What to Report

1. **Histogram of final |v| for each group.** Four distributions on
   one plot: fabricated, known-interacting, known-safe, novel-pairs.
2. **Where is the convergence threshold?** Draw a vertical line on
   the histogram. Are fabricated drugs above it? Are known drugs below?
3. **Separation.** If fabricated and known distributions DON'T overlap,
   the dynamics themselves separate known from unknown. If they DO
   overlap, the threshold is doing all the safety work.
4. **Novel pairs** (known drugs, unknown combination). This is the
   hardest case. Does the model converge on these? With high or low
   confidence? This tests whether the system can say "I know both
   drugs but I don't know what they do together."
5. **Mean and std of final |v|** for each group.
6. **How many fabricated pairs converge?** If >10% converge, we have
   a problem. Report what they converge TO (what severity, what
   mechanism).

---

## Experiment 3: Output Class Trajectory

### Question
For non-converging or slowly-converging inputs, does the predicted
class flip across steps, or does it stabilize early while internal
state keeps churning?

- Class flipping → semantic instability (exploring different basins)
- Class stable, state unstable → sub-symbolic instability (refining
  within a basin, or meaningless churn)
- Class stable, specific dimensions oscillating → structured partial
  convergence (the best case — each dimension carries meaning)

### Procedure

For the fabricated pairs and novel pairs from Experiment 2, record
the predicted severity AND mechanism class at EVERY reasoning step,
plus the per-dimension convergence status.

```python
"""
experiment_3_class_trajectory.py

For each fabricated and novel pair:
  - Record predicted severity class at every step
  - Record predicted mechanism class at every step
  - Record per-dimension convergence (|v_dim| < threshold) at every step
  - Identify which dimensions converge first, last, never
"""

def run_with_class_trajectory(model, drug_a_id, drug_b_id, max_steps=16):
    """Run model, record output class at every reasoning step."""
    
    # This requires modifying the forward pass to compute output
    # head predictions at EVERY step, not just the final one.
    # 
    # For each step t:
    #   1. Take position x(t) from the oscillator
    #   2. Run it through the output head
    #   3. Record argmax severity, argmax mechanism
    #   4. Record per-dimension |v(t)| < dim_threshold
    
    step_predictions = []
    per_dim_convergence = []
    
    # Threshold for per-dimension convergence
    # Use a fraction of the learned threshold
    dim_threshold = model.reasoning_loop.convergence_threshold.item() / np.sqrt(512)
    
    for step in range(max_steps):
        # Get state at this step from trajectory
        x_t = trajectory['positions'][step]
        v_t = trajectory['velocities'][step]
        
        # Run output head on intermediate state
        with torch.no_grad():
            severity_logits = model.output_head.severity_head(x_t)
            mechanism_logits = model.output_head.mechanism_head(x_t)
        
        step_predictions.append({
            'step': step,
            'severity_class': severity_logits.argmax().item(),
            'severity_probs': torch.softmax(severity_logits, -1).cpu().tolist(),
            'mechanism_class': mechanism_logits.argmax().item(),
            'mechanism_probs': torch.softmax(mechanism_logits, -1).cpu().tolist(),
        })
        
        # Per-dimension convergence
        dim_converged = (v_t.abs() < dim_threshold).cpu().numpy()
        per_dim_convergence.append(dim_converged)
    
    # Analysis: classify the trajectory
    severity_classes = [p['severity_class'] for p in step_predictions]
    mechanism_classes = [p['mechanism_class'] for p in step_predictions]
    
    # Count class flips
    severity_flips = sum(1 for i in range(1, len(severity_classes))
                         if severity_classes[i] != severity_classes[i-1])
    mechanism_flips = sum(1 for i in range(1, len(mechanism_classes))
                          if mechanism_classes[i] != mechanism_classes[i-1])
    
    # Find dimensions that converge vs oscillate
    final_dim_convergence = per_dim_convergence[-1]  # at last step
    dims_converged = np.where(final_dim_convergence)[0].tolist()
    dims_oscillating = np.where(~final_dim_convergence)[0].tolist()
    
    # Check if oscillating dims are random or structured
    # (clustered in specific ranges that map to known feature groups)
    
    return {
        'step_predictions': step_predictions,
        'per_dim_convergence': [d.tolist() for d in per_dim_convergence],
        'severity_flips': severity_flips,
        'mechanism_flips': mechanism_flips,
        'final_severity': severity_classes[-1],
        'final_mechanism': mechanism_classes[-1],
        'n_dims_converged': int(final_dim_convergence.sum()),
        'n_dims_oscillating': int((~final_dim_convergence).sum()),
        'dims_oscillating': dims_oscillating,
    }

# Run on fabricated pairs
fabricated_trajectories = [
    run_with_class_trajectory(model, a, b)
    for a, b in fabricated_pairs
]

# Run on novel pairs (known drugs, unknown combination)
novel_trajectories = [
    run_with_class_trajectory(model, a, b)
    for a, b in novel_pairs
]

# Run on known pairs for comparison
known_trajectories = [
    run_with_class_trajectory(model, a, b)
    for a, b in known_interacting[:50]
]

# --- Summary Statistics ---
print("=== FABRICATED PAIRS ===")
fab_sev_flips = [t['severity_flips'] for t in fabricated_trajectories]
fab_mech_flips = [t['mechanism_flips'] for t in fabricated_trajectories]
print(f"Severity flips:  mean={np.mean(fab_sev_flips):.1f}, "
      f"max={max(fab_sev_flips)}, zero_flips={fab_sev_flips.count(0)}/100")
print(f"Mechanism flips: mean={np.mean(fab_mech_flips):.1f}, "
      f"max={max(fab_mech_flips)}, zero_flips={fab_mech_flips.count(0)}/100")
print(f"Dims oscillating at step 16: "
      f"mean={np.mean([t['n_dims_oscillating'] for t in fabricated_trajectories]):.0f}/512")

print("\n=== KNOWN PAIRS ===")
known_sev_flips = [t['severity_flips'] for t in known_trajectories]
known_mech_flips = [t['mechanism_flips'] for t in known_trajectories]
print(f"Severity flips:  mean={np.mean(known_sev_flips):.1f}, "
      f"max={max(known_sev_flips)}, zero_flips={known_sev_flips.count(0)}/50")
print(f"Mechanism flips: mean={np.mean(known_mech_flips):.1f}, "
      f"max={max(known_mech_flips)}, zero_flips={known_mech_flips.count(0)}/50")
print(f"Dims oscillating at step 16: "
      f"mean={np.mean([t['n_dims_oscillating'] for t in known_trajectories]):.0f}/512")

# Save full results
with open("experiments/class_trajectory_results.json", "w") as f:
    json.dump({
        'fabricated': fabricated_trajectories,
        'novel': novel_trajectories,
        'known': known_trajectories,
    }, f, indent=2)
```

### What to Report

1. **Severity flip count** for fabricated vs known pairs. If fabricated
   pairs show many flips (>3) and known pairs show zero, we have
   semantic instability for unknowns. That's the best case.

2. **Mechanism flip count** same comparison. Mechanism should be
   noisier than severity (harder task), so more flips expected even
   for known pairs.

3. **Dims oscillating at step 16.** For known pairs this should be
   near 0. For fabricated pairs:
   - If ~512 dims oscillating → total non-convergence (good)
   - If ~50-200 dims oscillating → partial convergence (interesting —
     which dimensions converge on fabricated drugs?)
   - If ~0 dims oscillating → full convergence on fabricated drugs
     (bad — system is hallucinating)

4. **CRITICAL: Which dimensions oscillate for fabricated pairs?**
   Are they random (different dims for each pair) or structured
   (same dims always oscillate)? If structured, map them back to
   the feature dimensions. This tells us whether the partial
   convergence has semantic meaning.

5. **What do fabricated pairs converge TO?** If they converge, what
   severity and mechanism? If they all converge to "none" — that's
   actually reasonable (unknown → no known interaction). If they
   converge to specific severity/mechanism classes — the model is
   fabricating.

6. **Novel pairs** (known drugs, unknown combination) are the most
   important test case for clinical use. Report everything above
   for this group separately.

---

## Directory Structure

```
experiments/
    experiment_1_depth_ablation.py
    experiment_2_velocity_distribution.py
    experiment_3_class_trajectory.py
    depth_ablation_results.json
    velocity_distribution_results.json
    class_trajectory_results.json
    README.md                         ← this file
```

---

## Interpretation Guide

### If Experiment 1 shows saturation at ~6 steps:
The recurrence is refinement. Still useful as a drug interaction
checker, but the binary convergence stream is low-information.
Resolver concept needs revision.

### If Experiment 1 shows continued improvement to 32+:
The recurrence is computational. Each step adds representational
power. Proceed with resolver concept as designed.

### If Experiment 2 shows fabricated drugs at HIGH velocity:
The dynamics genuinely separate known from unknown. Safety property
is architectural. Strong evidence for resolver thesis.

### If Experiment 2 shows fabricated drugs at LOW velocity above threshold:
The threshold is the safety mechanism, not the dynamics. System will
converge on wrong answers given enough steps. Reconsider the
"structurally cannot hallucinate" claim. May need architectural
changes (e.g., entropy-based rejection instead of velocity threshold).

### If Experiment 3 shows class flipping for unknowns:
Semantic instability — the system is genuinely exploring incompatible
hypotheses. The convergence stream carries high-information content.
Strongest evidence that the resolver is a different kind of
computation, not just a classifier with a confidence threshold.

### If Experiment 3 shows class stability with dimensional oscillation:
Check WHICH dimensions. If structured (same dims for similar drug
classes), the partial convergence has semantic meaning — the system
knows WHAT it knows and WHAT it doesn't, per dimension. This is the
best case for the convergence-stream-as-language thesis.

### If Experiment 3 shows full convergence for fabricated drugs:
The system is hallucinating. The safety property does not hold.
This doesn't kill the project — PharmLoop is still a good drug
interaction checker — but the resolver concept document's claims
about structural honesty need revision, and the architecture needs
changes before scaling to multi-agent communication.

---

## Priority

Run these BEFORE Phase 5. The entire resolver architecture thesis
depends on these results. If the experiments show we built a
classifier, that's fine — it's a very good classifier. But the
multi-agent convergence-stream communication concept only works
if the recurrence is genuinely computational and the convergence
dynamics carry semantic structure.

These experiments take ~1 hour to implement and ~10 minutes to run.
They are the highest-value work remaining on the project.
