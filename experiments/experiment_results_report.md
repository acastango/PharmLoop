# PharmLoop Experiment Results: Resolver or Classifier?

**Date:** 2026-02-17 15:05:32
**Checkpoint:** checkpoints/best_model_phase4b.pt
**Convergence Threshold:** 0.150100

---

## Experiment 1: Depth Ablation

**Question:** Does accuracy improve with more recurrence steps, or saturate?

| Depth | Sev Acc | Mech Acc | FNR | Conv Rate | Avg Steps |
|------:|--------:|---------:|----:|----------:|----------:|
|     2 |  88.3% |   73.0% | 0.0% |    50.6% |       2.0 |
|     4 |  88.3% |   73.0% | 0.0% |    96.5% |       3.0 |
|     6 |  88.3% |   73.0% | 0.0% |    96.5% |       3.1 |
|     8 |  88.3% |   73.0% | 0.0% |    96.5% |       3.2 |
|    12 |  88.3% |   73.0% | 0.0% |    96.5% |       3.5 |
|    16 |  88.3% |   73.0% | 0.0% |    96.5% |       3.6 |
|    24 |  88.3% |   73.0% | 0.0% |    96.5% |       4.0 |
|    32 |  88.3% |   73.0% | 0.0% |    96.5% |       4.5 |
|    48 |  88.3% |   73.0% | 0.0% |    96.5% |       5.2 |
|    64 |  88.3% |   73.0% | 0.0% |    96.5% |       6.0 |

### Analysis

- **Peak severity accuracy:** 88.3% (within 1% at depth 2)
- **Peak mechanism accuracy:** 73.0% (within 1% at depth 2)
- **Accuracy at depth 16:** 88.3%, **at 32:** 88.3%, **at 64:** 88.3%

**Verdict:** Accuracy **saturates early (depth ~2)** — the recurrence is **iterative refinement**, not genuinely computational. The oscillator is essentially a lookup with extra steps.

---

## Experiment 2: Velocity Distribution

**Question:** Do fabricated drugs maintain high velocity (genuine uncertainty) or converge to wrong answers?

| Group | N | Final |v| Mean | Final |v| Std | Converged | Confidence Mean |
|-------|--:|---------------:|--------------:|----------:|----------------:|
| FABRICATED | 100 | 0.347914 | 0.199369 | 14/100 (14%) | 0.131 |
| KNOWN INTERACTING | 50 | 0.096434 | 0.017989 | 50/50 (100%) | 0.540 |
| KNOWN SAFE | 50 | 0.111198 | 0.021963 | 50/50 (100%) | 0.472 |
| NOVEL PAIRS | 50 | 0.091582 | 0.023818 | 50/50 (100%) | 0.556 |

### Severity Distribution by Group

- **FABRICATED:** {'moderate': 22, 'contraindicated': 41, 'none': 22, 'severe': 10, 'unknown': 4, 'mild': 1}
- **KNOWN INTERACTING:** {'severe': 25, 'contraindicated': 1, 'moderate': 24}
- **KNOWN SAFE:** {'none': 33, 'moderate': 14, 'severe': 3}
- **NOVEL PAIRS:** {'none': 18, 'mild': 7, 'moderate': 13, 'severe': 12}

### Separation Analysis

- Convergence threshold: **0.150100**
- Fabricated final |v| mean: **0.347914** (above threshold: **True**)
- Fabricated drugs that converge: **14/100** (14%)

**Verdict:** Some fabricated drugs converge (14/100). The dynamics provide **partial** separation, but the threshold is doing significant safety work.

### Novel Pairs (Known Drugs, Unknown Combination)

- Converged: **50/50** (100%)
- Mean confidence: **0.556**
- Severity distribution: {'none': 18, 'mild': 7, 'moderate': 13, 'severe': 12}

Novel pairs (known drugs, unknown combination) **mostly converge**. The system is producing answers for unseen combinations — check whether these are reasonable generalizations or hallucinations.

---

## Experiment 3: Output Class Trajectory

**Question:** Do predicted classes flip during oscillation (semantic instability) or stabilize early (classifier behavior)?

| Group | Sev Flips (mean) | Sev Flips (max) | Zero Flips | Mech Flips (mean) | Dims Oscillating |
|-------|------------------:|----------------:|-----------:|------------------:|-----------------:|
| FABRICATED | 0.1 | 2 | 47/50 | 0.4 | 254/512 |
| NOVEL PAIRS | 0.0 | 0 | 50/50 | 0.0 | 69/512 |
| KNOWN INTERACTING | 0.0 | 0 | 50/50 | 0.0 | 60/512 |

### Convergence by Group

- **FABRICATED:** converged 5/50, final severity: {'moderate': 11, 'contraindicated': 24, 'none': 7, 'unknown': 2, 'mild': 2, 'severe': 4}
- **NOVEL PAIRS:** converged 50/50, final severity: {'moderate': 14, 'none': 27, 'mild': 1, 'severe': 8}
- **KNOWN INTERACTING:** converged 50/50, final severity: {'severe': 25, 'contraindicated': 1, 'moderate': 24}

### Interpretation

Fabricated pairs show **more class flipping** (mean 0.1) than known pairs (mean 0.0), but the difference is moderate. There is some semantic instability for unknowns.

**Dimensional analysis:** Fabricated pairs have ~254/512 dimensions oscillating — **partial convergence**. Some aspects settle while others remain uncertain. This is interesting: the system may know WHAT it knows and what it doesn't, per dimension.

---

## Overall Verdict

**Resolver Score: 2/6**

- Recurrence saturates early (depth ~2) — refinement, not computation
- Most fabricated drugs fail to converge — partial architectural safety
- Significant dimensional non-convergence for unknowns

### Assessment: **CLASSIFIER**

PharmLoop behaves primarily as a classifier with a confidence threshold. The recurrence is refinement rather than computation, and safety relies on the threshold rather than dynamics. This is still a very good drug interaction checker, but the resolver concept document's claims about structural honesty need revision before scaling to multi-agent communication.
