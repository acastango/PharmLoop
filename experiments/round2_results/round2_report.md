# PharmLoop Experiment Round 2: Testing the Dynamics Directly

**Date:** 2026-02-17 18:49:44
**Checkpoint:** checkpoints/best_model_phase4b.pt
**Forced depth:** 16 steps (no early stopping)
**Convergence threshold:** 0.150100

**Context:** Round 1 showed the output head saturates at depth 2, reading answers from the encoder. But the oscillator produces differentiated dynamics (254/512 dims oscillating for fabricated vs 60/512 for known). Round 2 tests whether those dynamics carry real information.

---

## Experiment 4: Trajectory Information Content

| Feature Set | Severity Acc | Mechanism Acc | Dims |
|-------------|-------------:|--------------:|-----:|
| A: Encoder only (step 0) | 81.4% +/- 2.7% | 95.2% | 512 |
| B: Trajectory statistics | 77.1% +/- 4.9% | 94.9% | 4100 |
| C: Full trajectory | 76.2% +/- 2.5% | 94.6% | 3072 |
| D: Delta (traj - encoder) | 72.8% +/- 5.6% | 94.7% | 3588 |

**Encoder (81.4%) dominates trajectory (77.1%).** But delta features alone achieve 72.8% — dynamics carry independent signal.

---

## Experiment 5: Hopfield Retrieval Analysis

- Unique patterns visited: mean=1.0, single=197/200
- Cosine(state, retrieval): step0=0.4908 -> final=0.4876
- Force: step0=0.0270 -> final=0.0045

```
Force trajectory:
  Step  0: 0.0270 #####
  Step  1: 0.0235 ####
  Step  2: 0.0201 ####
  Step  3: 0.0168 ###
  Step  4: 0.0140 ##
  Step  5: 0.0116 ##
  Step  6: 0.0098 #
  Step  7: 0.0083 #
  Step  8: 0.0072 #
  Step  9: 0.0064 #
  Step 10: 0.0059 #
  Step 11: 0.0054 #
  Step 12: 0.0051 #
  Step 13: 0.0049 
  Step 14: 0.0047 
  Step 15: 0.0045 
```

**Oscillator stays near one attractor — refinement, not exploration.**

---

## Experiment 6: Bypass Test (NN-Attractor)

| Step | NN Sev Acc | NN Mech Acc | Per-Mech | FNR | Sim |
|-----:|-----------:|------------:|---------:|----:|----:|
| 0 | 84.2% | 73.3% | 96.6% | 3.8% | 0.9914 |
| 2 | 84.4% | 73.6% | 96.6% | 3.2% | 0.9915 |
| 4 | 84.4% | 73.6% | 96.6% | 3.2% | 0.9915 |
| 8 | 84.4% | 73.6% | 96.6% | 3.2% | 0.9915 |
| 12 | 84.4% | 73.6% | 96.6% | 3.2% | 0.9915 |
| 16 | 84.4% | 73.6% | 96.6% | 3.2% | 0.9915 |

**NN flat: 84.2% -> 84.4%.** Oscillator doesn't change attractor proximity.

---

## Experiment 7: Dimensional Semantics

- Class prediction from timing: **41.9%** (chance=3.6%)

| Class Pair | N | Mean Step | Std |
|------------|--:|----------:|----:|
| antibiotic + cardiac | 111 | 0.00 | 0.01 |
| antibiotic + immunosuppressant | 83 | 0.01 | 0.02 |
| cardiac + cns_psych | 80 | 0.01 | 0.02 |
| antibiotic + antidiabetic | 58 | 0.00 | 0.01 |
| cns_psych + ssri_snri | 56 | 0.02 | 0.03 |
| cns_psych + cns_psych | 56 | 0.04 | 0.06 |
| antibiotic + cns_psych | 51 | 0.02 | 0.05 |
| antibiotic + anticoagulant | 50 | 0.01 | 0.03 |
| cns_psych + opioid | 50 | 0.03 | 0.05 |
| antibiotic + antihypertensive | 46 | 0.01 | 0.02 |

**Strong semantic structure in convergence timing.**

---

## Experiment 8: Output Head Dependency Test

| Feature Source | Sev Acc | Mech Acc | Dims |
|---------------|--------:|---------:|-----:|
| Step 0 (encoder) | 85.6% | 96.6% | 512 |
| Step 2 | 85.6% | 96.6% | 512 |
| Step 4 | 86.0% | 96.6% | 512 |
| Step 8 | 86.0% | 96.6% | 512 |
| Step 12 | 86.0% | 96.6% | 512 |
| Step 16 (final) | 86.0% | 96.6% | 512 |
| Late mean (8-16) | 86.0% | 96.6% | 512 |
| Pos+Vel final | 83.7% | 96.3% | 1024 |
| Pos+Vel step 0 | 83.7% | 96.4% | 1024 |
| Delta (16 - 0) | 83.4% | 96.5% | 512 |

**Severity:** step0=85.6% -> step16=86.0% (+0.3%)
**Mechanism:** step0=96.6% -> step16=96.6% (+0.0%)

---

## Overall Verdict

### Evidence dynamics carry real information:
- Exp 4: Delta features achieve 72.8% alone — independent signal
- Exp 7: Strong class prediction from timing (41.9%)

### Evidence encoder solved everything:
- Exp 4: Encoder dominates (81.4% vs 77.1%)
- Exp 5: Single pattern (1.0 avg)
- Exp 6: NN flat (84.2% -> 84.4%)
- Exp 8: Late probes don't improve (sev: 85.6%->86.0%)

### Assessment: **MIXED — PARTIAL SHORTCUT**

Some dynamic information exists but it's unclear if the oscillator adds enough to justify its cost. Consider architectural experiments.
