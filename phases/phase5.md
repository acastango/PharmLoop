# Phase 5 — Pharmacist Validation Loop + Continuous Learning

## Why This Comes Before FHIR/EHR

Phase 4b's teasers listed Phase 5 as "Clinical Integration — FHIR, EHR, HL7."
That's wrong. FHIR formatting and EHR adapters are plumbing — they're
important but they don't teach you anything about whether the system
actually works. Phase 5 should be the phase where real pharmacists
evaluate real interactions and PharmLoop learns from their corrections.

The reason this matters architecturally: PharmLoop has a property that
most drug interaction checkers don't. The Hopfield memory can grow
without retraining the neural network. A validated new interaction
pattern gets stored in the Hopfield bank, and the oscillator immediately
starts using it. The pharmacist validation process isn't just QA — it's
the production training pipeline.

This is the phase where PharmLoop becomes a system that gets better from
use rather than a static model that degrades from data drift.

---

## Prerequisites

Phase 4b validated:
- [ ] >= 500 drugs with real DrugBank features (not synthetic)
- [ ] Severity accuracy >= 85%
- [ ] Zero false negatives on severe/contraindicated
- [ ] Full polypharmacy analyzer (cascade, renal, metabolic saturation)
- [ ] Cross-reference evaluation against Lexicomp/Micromedex completed
- [ ] Brand name resolution functional
- [ ] API production-hardened (caching, monitoring, error handling)
- [ ] At least one pharmacist has seen Phase 4a/4b output and given feedback

---

## What Phase 5 Accomplishes

- **Pharmacist review interface** — web UI where pharmacists check
  interactions, see convergence dynamics, and provide structured feedback
- **Feedback-to-pattern pipeline** — validated corrections become new
  Hopfield patterns without retraining
- **Disagreement triage system** — systematic investigation when model
  and pharmacist disagree
- **Coverage expansion from usage** — pharmacist queries reveal which
  drugs and interactions are most requested but missing
- **Confidence recalibration** — as Hopfield grows, confidence thresholds
  adjust automatically
- **Convergence visualization** — pharmacists can see WHY the system is
  uncertain, not just THAT it's uncertain

Phase 5 is the transition from "developer project" to "clinical tool
under evaluation." The system is not deployed for patient care yet —
it's deployed for pharmacist evaluation with structured feedback capture.

---

## Step 1: Pharmacist Review Interface

### 1.1 Core Interaction Check View

The primary screen. Pharmacist enters two drugs (or a medication list),
gets the interaction result, and can provide feedback.

```
┌─────────────────────────────────────────────────────────────────┐
│  PharmLoop Interaction Check                                    │
│                                                                 │
│  Drug A: [fluoxetine     ▼]    Drug B: [tramadol         ▼]    │
│  [Check Interaction]  [Check Full Med List]                     │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ⚠ SEVERE INTERACTION                                          │
│                                                                 │
│  Fluoxetine and tramadol both increase serotonergic activity.   │
│  Combined use significantly increases the risk of serotonin     │
│  syndrome. Fluoxetine inhibits CYP2D6, which metabolizes        │
│  tramadol to its active metabolite.                             │
│                                                                 │
│  Mechanisms: serotonergic, cyp_inhibition                       │
│  Flags: monitor_serotonin_syndrome, avoid_combination           │
│  Confidence: 91% | Converged in 5 steps                        │
│                                                                 │
│  [▸ Show Convergence Dynamics]                                  │
│  [▸ Show Partial Convergence Detail]                            │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  Pharmacist Feedback                                            │
│                                                                 │
│  Severity assessment:  ○ Agree  ○ Too high  ○ Too low           │
│  Mechanism assessment: ○ Agree  ○ Incomplete  ○ Wrong           │
│  If wrong/incomplete, correct mechanism: [____________]         │
│  Clinical notes: [                                           ]  │
│  [Submit Feedback]                                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Convergence Dynamics View

This is PharmLoop's differentiator. When the pharmacist clicks
"Show Convergence Dynamics," they see a visualization of how the
oscillator reasoned:

```
Gray Zone (velocity magnitude) over reasoning steps:

  1.0 │ ●
      │  ╲
      │   ╲
  0.5 │    ╲    ●
      │     ╲  ╱ ╲
      │      ●╱   ╲
  0.1 │            ╲●───●───●    ← converged here (step 5)
      │
  0.0 └──1──2──3──4──5──6──7──8──→ steps

  ● Fast convergence — strong evidence for this interaction.
  The model "recognized" this drug pair quickly.
```

For non-converging (unknown) cases:

```
  1.0 │ ●
      │  ╲    ●         ●
      │   ╲  ╱ ╲       ╱ ╲    ●
  0.5 │    ●╱   ╲     ╱   ╲  ╱ ╲
      │         ╲   ●╱     ●╱   ╲
  0.1 │          ● ╱              ╲●
      │
  0.0 └──1──2──3──4──5──6──7──...──16──→ steps

  ● Still oscillating at timeout — the model cannot determine
  this interaction with confidence. This is a genuine "I don't know."
```

For partial convergence:

```
  Per-dimension convergence breakdown:
  ┌──────────────────────────────────┐
  │ Severity:     ████████████  98%  │  ← settled
  │ CYP pathway:  ████████░░░░  67%  │  ← mostly settled
  │ Serotonergic: ████████████  95%  │  ← settled
  │ Bleeding:     ██░░░░░░░░░░  18%  │  ← uncertain
  │ Renal:        ████░░░░░░░░  35%  │  ← uncertain
  └──────────────────────────────────┘

  The model is confident about severity and serotonergic mechanism,
  but uncertain about bleeding and renal involvement.
```

This visualization is clinically meaningful because it tells the
pharmacist exactly what the model knows and doesn't know. A pharmacist
can look at this and say "the model is right that there's a serotonergic
risk, but it missed that tramadol also has seizure-lowering threshold
effects" — that's actionable feedback.

### 1.3 Medication List View

Full polypharmacy analysis with all pairwise results and alerts:

```
┌─────────────────────────────────────────────────────────────────┐
│  Medication List Analysis (7 drugs, 21 pairs checked)           │
│                                                                 │
│  ⚠ 2 MULTI-DRUG ALERTS                                        │
│                                                                 │
│  ❶ Additive Serotonergic Risk (3 pairs)                        │
│     Fluoxetine, tramadol, trazodone all increase serotonergic   │
│     activity. Combined risk exceeds individual pair risks.      │
│     [▸ Show involved pairs]                                     │
│                                                                 │
│  ❷ CYP3A4 Metabolic Saturation                                 │
│     Clarithromycin inhibits CYP3A4. Simvastatin and amlodipine  │
│     both require CYP3A4 for metabolism. Competition for reduced │
│     enzyme capacity may elevate both drug levels.               │
│     [▸ Show involved pairs]                                     │
│                                                                 │
│  Pairwise Results (ranked by severity):                         │
│  ┌────────────────────────────┬──────────┬───────────┐          │
│  │ Pair                       │ Severity │ Conf.     │          │
│  ├────────────────────────────┼──────────┼───────────┤          │
│  │ fluoxetine + tramadol      │ SEVERE   │ 91%       │          │
│  │ clarithromycin + simvast.  │ SEVERE   │ 84%       │          │
│  │ fluoxetine + trazodone     │ MODERATE │ 78%       │          │
│  │ tramadol + trazodone       │ MODERATE │ 72%       │          │
│  │ lisinopril + metformin     │ NONE     │ 88%       │          │
│  │ ...                        │          │           │          │
│  └────────────────────────────┴──────────┴───────────┘          │
│                                                                 │
│  [▸ Expand all pairs]  [▸ Export report]                        │
└─────────────────────────────────────────────────────────────────┘
```

### 1.4 Implementation

The review interface is a web app, not a CLI. It talks to the
existing REST API from Phase 4a/4b.

```
Frontend:  React or plain HTML/JS (lightweight)
Backend:   Phase 4b FastAPI server + feedback endpoints
Storage:   SQLite for feedback data (no heavy infrastructure)
Auth:      Simple token-based (this is internal validation, not public)
```

New API endpoints for Phase 5:

```
POST /feedback                → submit pharmacist feedback on a result
GET  /feedback/pending        → list interactions awaiting review
GET  /feedback/stats          → dashboard data (agreement rate, coverage)
POST /hopfield/ingest         → trigger pattern ingestion from validated feedback
GET  /coverage/gaps           → drugs/pairs most requested but unknown
```

**File: `api/feedback.py`**

```python
@dataclass
class PharmacistFeedback:
    """Structured feedback from pharmacist review."""
    interaction_id: str          # hash of (drug_a, drug_b, timestamp)
    pharmacist_id: str           # anonymous identifier
    
    # Model's prediction
    model_severity: str
    model_mechanisms: list[str]
    model_confidence: float
    model_converged: bool
    
    # Pharmacist's assessment
    severity_agreement: str      # "agree", "too_high", "too_low"
    mechanism_agreement: str     # "agree", "incomplete", "wrong"
    correct_severity: str | None
    correct_mechanisms: list[str] | None
    missing_mechanisms: list[str] | None
    
    # Clinical context
    clinical_notes: str
    references_cited: list[str]  # pharmacist can cite sources
    
    # Metadata
    timestamp: str
    review_time_seconds: float   # how long the pharmacist spent
```

---

## Step 2: Feedback-to-Pattern Pipeline

This is the architectural payoff. When a pharmacist corrects PharmLoop,
the correction can become a new Hopfield pattern.

### 2.1 The Pipeline

```
Pharmacist submits correction
    ↓
Correction stored in feedback database
    ↓
Correction queued for validation (requires N=2 pharmacist agreement)
    ↓
Validated correction → compute pair state with current encoder
    ↓
New pattern stored in Hopfield memory (no retraining needed)
    ↓
Next time this drug pair (or similar pair) is queried,
the oscillator retrieves the new pattern and converges differently
```

### 2.2 Validation Requirements

Not every pharmacist correction should become a pattern. Requirements:

1. **Agreement threshold:** At least 2 pharmacists must agree on the
   correction for the same drug pair. This prevents individual bias.
   (Initially with a small pharmacist pool, N=1 may be necessary —
   but flag these as "single-reviewer" patterns.)

2. **Reference requirement:** Corrections should cite a source
   (Lexicomp, Micromedex, UpToDate, package insert, clinical study).
   Uncited corrections are stored but not ingested.

3. **Severity direction check:** Corrections that LOWER severity from
   severe/contraindicated require higher scrutiny (N=3 agreement or
   primary literature citation). The DO NO HARM principle applies
   to the learning loop too.

4. **Consistency check:** Before ingestion, verify the new pattern
   doesn't create contradictions with existing patterns. If drug A+B
   is being corrected from "severe" to "moderate," but A+B already has
   a stored "severe" pattern, the old pattern must be removed first.

### 2.3 Pattern Ingestion

```python
class HopfieldPatternIngester:
    """
    Converts validated pharmacist corrections into Hopfield patterns.
    
    Does NOT retrain the neural network. Instead:
    1. Encodes the drug pair using the existing (frozen) encoder
    2. Stores the resulting pair state in the Hopfield memory
    3. Optionally stores augmented variants (noise-injected copies
       for robustness, severity-amplified copies for dangerous pairs)
    """
    
    def ingest(self, correction: ValidatedCorrection) -> bool:
        """
        Add a validated correction as a new Hopfield pattern.
        
        Returns True if the pattern was stored, False if rejected
        (e.g., capacity limit, consistency violation).
        """
        # 1. Encode the drug pair
        pair_state = self.model.compute_pair_state(
            drug_a_id, drug_a_features,
            drug_b_id, drug_b_features,
        )
        
        # 2. Store in appropriate Hopfield banks
        drug_classes = (
            self.drug_class_map.get(drug_a_id, "other"),
            self.drug_class_map.get(drug_b_id, "other"),
        )
        self.hopfield.store(pair_state, drug_classes=[drug_classes])
        
        # 3. Severity amplification for dangerous interactions
        if correction.severity in ("severe", "contraindicated"):
            for _ in range(2):
                noisy = pair_state + torch.randn_like(pair_state) * 0.01
                self.hopfield.store(noisy, drug_classes=[drug_classes])
        
        # 4. Log the ingestion
        self.ingestion_log.append({
            "drug_a": correction.drug_a,
            "drug_b": correction.drug_b,
            "severity": correction.severity,
            "mechanisms": correction.mechanisms,
            "source": "pharmacist_validation",
            "pharmacist_count": correction.agreement_count,
            "references": correction.references,
            "timestamp": correction.timestamp,
        })
        
        return True
```

### 2.4 What This Means

After Phase 5, the system has two knowledge sources:

1. **Training data** — the original DrugBank interactions that the
   neural network was trained on. Encoded in weights and Hopfield.
2. **Pharmacist corrections** — new patterns added to Hopfield
   post-training. No weight changes, just memory growth.

The second source is provenance-tracked: every pharmacist-added pattern
has a citation, reviewer IDs, and timestamp. If a pattern is later found
to be wrong, it can be specifically removed from the Hopfield bank
without affecting anything else.

This is the Hopfield architecture paying off. In a transformer-based
system, you'd need to retrain or fine-tune to incorporate new knowledge.
Here, you just store a new pattern and the oscillator starts using it
immediately.

---

## Step 3: Disagreement Triage

When the model and pharmacist disagree, that's signal. Phase 5 builds
a systematic process for investigating disagreements.

### 3.1 Disagreement Categories

From the Phase 4b cross-reference evaluation, plus pharmacist feedback:

| Category | Model Says | Pharmacist Says | Action |
|----------|-----------|----------------|--------|
| **False safe** | none/mild | moderate/severe | CRITICAL — data gap. Add interaction immediately. |
| **Over-alert** | severe | moderate/mild | Investigate. Is the model over-generalizing from drug class? |
| **Mechanism mismatch** | cyp_inhibition | serotonergic | Both may be partially right (multi-mechanism). Expand labels. |
| **Missing drug** | unknown | has an opinion | Drug not in registry. Add to drug list for next pipeline run. |
| **Confidence mismatch** | 90% confidence | pharmacist disagrees | Model is confidently wrong. Highest priority for investigation. |
| **Agreement** | matches | matches | Good. Strengthens existing pattern. |

### 3.2 Priority Queue

Disagreements are prioritized for investigation:

1. **P0:** False safe on severe/contraindicated (safety critical)
2. **P1:** Model confident + pharmacist disagrees (calibration problem)
3. **P2:** Mechanism mismatch (data quality problem)
4. **P3:** Over-alerting (usability problem — alert fatigue)
5. **P4:** Missing drug (coverage problem)

### 3.3 Disagreement Dashboard

```
Disagreement Summary (last 30 days):
─────────────────────────────────────
Total checks:        847
Agreement rate:      82.3%
False safe (P0):     3  ← investigate immediately
Confident+wrong:     12 ← investigate this week
Mechanism mismatch:  28 ← batch review monthly
Over-alerting:       42 ← tune thresholds
Missing drugs:       67 ← next pipeline run

Top requested missing drugs:
  1. semaglutide (41 queries)
  2. tirzepatide (28 queries)
  3. naltrexone (19 queries)
  4. buprenorphine (17 queries)
```

This dashboard drives prioritization for the next development cycle.
If pharmacists keep asking about GLP-1 agonists and PharmLoop doesn't
have them, that's the next drug list expansion.

---

## Step 4: Confidence Recalibration

As the Hopfield memory grows from pharmacist feedback, the model's
confidence characteristics change. More stored patterns means the
oscillator has more attractors to converge to, which generally means
faster convergence and higher confidence. But it also means the
convergence threshold may need adjustment.

### 4.1 Calibration Monitoring

Track the relationship between model confidence and pharmacist
agreement rate:

```
Confidence bin  | Model predictions | Pharmacist agrees | Calibration
90-100%         | 234               | 218 (93.2%)       | well-calibrated
80-90%          | 178               | 142 (79.8%)       | well-calibrated
70-80%          | 156               | 109 (69.9%)       | slightly over-confident
60-70%          | 98                | 54 (55.1%)        | over-confident ← flag
50-60%          | 67                | 31 (46.3%)        | slightly under-confident
<50%            | 114               | 22 (19.3%)        | well-calibrated (low conf = uncertain)
```

If the 60-70% bin has only 55% agreement, the model is over-confident
in that range. Two options:

1. **Threshold adjustment:** Shift the convergence threshold so that
   interactions currently rated 60-70% get pushed to 50-60% instead.
   This is a single parameter change.

2. **Hopfield annealing:** Rebuild the Hopfield memory from scratch
   (including pharmacist-validated patterns) to recalibrate the energy
   landscape. This is the nuclear option but may be needed after
   significant pattern ingestion.

### 4.2 Automatic Recalibration

```python
class ConfidenceCalibrator:
    """
    Monitors and adjusts confidence calibration based on pharmacist feedback.
    
    Uses isotonic regression on (model_confidence, pharmacist_agreement)
    pairs to produce a calibration function that maps raw model confidence
    to calibrated confidence.
    """
    
    def calibrate(self, raw_confidence: float) -> float:
        """Map raw model confidence to calibrated confidence."""
        return self.isotonic_model.predict([[raw_confidence]])[0]
    
    def update(self, feedbacks: list[PharmacistFeedback]) -> None:
        """Refit calibration from accumulated feedback."""
        # Binary: did pharmacist agree with severity?
        X = [f.model_confidence for f in feedbacks]
        y = [1.0 if f.severity_agreement == "agree" else 0.0 for f in feedbacks]
        self.isotonic_model.fit(X, y)
```

---

## Step 5: Coverage Expansion from Usage

### 5.1 Unknown Drug Tracking

Every time PharmLoop returns "unknown" for a drug, log it. After 30 days,
the most-requested unknown drugs become the expansion list for the next
pipeline run.

```python
class CoverageTracker:
    """Track what pharmacists are asking about that we can't answer."""
    
    def log_unknown(self, drug_name: str, query_context: str) -> None:
        """Log an unknown drug query."""
        self.unknown_queries.append({
            "drug": drug_name,
            "context": query_context,  # what was it paired with?
            "timestamp": now(),
        })
    
    def get_expansion_candidates(self, min_queries: int = 5) -> list[dict]:
        """Get drugs requested >= min_queries times."""
        counts = Counter(q["drug"] for q in self.unknown_queries)
        return [
            {"drug": drug, "query_count": count,
             "common_pairs": self._common_pairs(drug)}
            for drug, count in counts.most_common()
            if count >= min_queries
        ]
```

### 5.2 Interaction Gap Detection

Also track which KNOWN drug pairs are queried most often but have
low confidence:

```
Low-confidence but frequently queried interactions:
  1. metformin + empagliflozin  (37 queries, 52% confidence)
  2. apixaban + clopidogrel     (29 queries, 48% confidence)
  3. duloxetine + gabapentin    (24 queries, 61% confidence)
```

These are interactions where the model has the drugs but isn't
confident about the interaction. Prioritize these for pharmacist
review and potential Hopfield pattern ingestion.

---

## Step 6: Lightweight Temporal Context

Full temporal dynamics (drug accumulation, steady-state modeling,
washout periods) is Phase 7. But pharmacists will immediately ask
one temporal question that Phase 5 should handle:

**"Is this a new prescription or has the patient been on it?"**

New prescriptions are higher risk than chronic stable therapy. If a
patient has been on fluoxetine for 2 years and you're adding tramadol,
that's different from starting both simultaneously.

### 6.1 Temporal Context Extension

Add two fields to the context encoder:

```python
# Extend context from 32 dims to 36 dims
# New dims 28-31:
vec[0, 28] = float(context.get("drug_a_is_new", True))      # new Rx
vec[0, 29] = context.get("drug_a_duration_days", 0) / 365.0  # how long on it
vec[0, 30] = float(context.get("drug_b_is_new", True))
vec[0, 31] = context.get("drug_b_duration_days", 0) / 365.0
```

### 6.2 Clinical Rationale

Why this matters before full temporal dynamics:

- **New SSRI + existing opioid:** Higher serotonergic risk during
  the first 2 weeks as SSRI reaches steady state. The interaction is
  more dangerous during dose titration.
- **Chronic warfarin + new antibiotic:** The antibiotic will disrupt
  the patient's established INR. More dangerous than starting both new
  because the patient's warfarin dose was calibrated without the
  antibiotic's CYP effect.
- **Both chronic:** Lower acute risk (patient has been tolerating the
  combination), but monitoring is still needed.

The template engine can incorporate this:

```
"Fluoxetine is being newly added to a regimen that already includes
tramadol. Serotonergic risk is elevated during the first 2-4 weeks
as fluoxetine reaches steady state. Monitor closely during this period."
```

vs.

```
"Patient has been on both fluoxetine and tramadol. The combination
carries ongoing serotonergic risk. Continue monitoring for symptoms
of serotonin syndrome."
```

---

## Implementation Files

### New Files
```
api/
    feedback.py              ← Feedback endpoints + data models
    
pharmloop/
    feedback_store.py        ← SQLite feedback storage
    pattern_ingester.py      ← Validated correction → Hopfield pattern
    coverage_tracker.py      ← Unknown drug + low-confidence tracking
    confidence_calibrator.py ← Isotonic regression calibration
    
frontend/
    index.html               ← Single-page pharmacist review interface
    app.js                   ← Interaction check + feedback submission
    dynamics.js              ← Convergence visualization (canvas/SVG)
    styles.css               ← Minimal clinical styling
    
evaluation/
    disagreement_report.py   ← Disagreement triage + dashboard data
    calibration_monitor.py   ← Confidence calibration tracking
    
tests/
    test_feedback.py         ← Feedback API + storage
    test_pattern_ingester.py ← Correction → pattern pipeline
    test_calibration.py      ← Confidence recalibration
    test_coverage.py         ← Coverage gap detection
    test_temporal_context.py ← New Rx vs chronic therapy
```

### Modified Files
```
pharmloop/context.py         ← Extend to 36 dims (temporal fields)
pharmloop/templates.py       ← New/chronic therapy template variants
pharmloop/inference.py       ← Coverage tracking hooks
api/server.py                ← Feedback endpoints, coverage endpoint
```

---

## Implementation Sequence

### Week 1-2: Feedback Infrastructure
1. Design PharmacistFeedback data model
2. Build SQLite feedback store
3. Add feedback API endpoints
4. Build minimal web interface (interaction check + feedback form)

### Week 3-4: Pattern Ingestion
5. Implement HopfieldPatternIngester
6. Build validation requirements (N=2 agreement, reference citation)
7. Implement consistency checks
8. Test: submit correction → verify Hopfield pattern stored →
   verify next query on same pair gives corrected result

### Week 5-6: Disagreement + Coverage
9. Build disagreement triage system
10. Implement coverage tracker (unknown drugs + low-confidence pairs)
11. Build dashboard data endpoints
12. Build disagreement dashboard in web interface

### Week 7-8: Calibration + Temporal
13. Implement confidence calibrator
14. Extend context encoder for temporal fields
15. Update template engine for new/chronic therapy
16. End-to-end testing with simulated pharmacist review workflow

---

## Validation Criteria

### Feedback Loop
- [ ] Pharmacist can submit structured feedback via web interface
- [ ] Feedback stored with full provenance (pharmacist ID, timestamp, references)
- [ ] Validated correction becomes Hopfield pattern within 1 minute
- [ ] After pattern ingestion, querying the same drug pair returns
      the corrected severity (not the old prediction)
- [ ] Pattern removal works: deleting a correction removes the pattern

### Safety
- [ ] Severity-lowering corrections require N >= 2 agreement
      or primary literature citation
- [ ] DO NO HARM: cannot bulk-ingest patterns that lower severity
      on contraindicated pairs without manual review
- [ ] Consistency check prevents contradictory patterns in same bank
- [ ] All ingested patterns have full provenance trail

### Calibration
- [ ] Confidence calibration error < 0.10 after 200+ feedback samples
- [ ] Over-confidence detected and flagged when calibration drifts
- [ ] Recalibration adjusts without retraining

### Coverage
- [ ] Top 10 most-requested unknown drugs identified within 30 days
- [ ] Low-confidence frequently-queried pairs surfaced for review
- [ ] Coverage tracker data feeds next pipeline run drug list

### Temporal
- [ ] New Rx vs chronic therapy changes template output
- [ ] Context encoder produces measurably different outputs for
      new vs chronic therapy context

### Regression
- [ ] All Phase 4b validation criteria still pass
- [ ] Hopfield growth from ingested patterns does not degrade
      accuracy on original test set
- [ ] Pattern ingestion does not exceed Hopfield capacity limits

---

## What Phase 5 Enables

After Phase 5, PharmLoop is no longer a static model. It's a system
that improves from pharmacist interaction:

- Every pharmacist check is a data point
- Every correction is a potential new pattern
- Every "unknown" response reveals a coverage gap
- Every confidence mismatch improves calibration

This creates a flywheel: better system → more pharmacist trust → more
usage → more feedback → better system.

Phase 6 (FHIR/EHR integration) now makes sense because you have a
system that pharmacists have validated and that continues to improve.
You're integrating something that works, not hoping it works.

Phase 7 (temporal dynamics) builds on the lightweight temporal context
from Step 6, extending it to full pharmacokinetic modeling.

Phase 8 (regulatory) can begin in parallel with Phase 6, using the
pharmacist validation data from Phase 5 as part of the clinical
evidence package for FDA submission.

---

## A Note on Anthony's Role

Anthony is a pharmacy technician, not a pharmacist. He cannot serve
as the clinical validator for Phase 5. He can:

- Build the infrastructure (code, interface, pipeline)
- Recruit pharmacist reviewers from his professional network
- Manage the disagreement triage process
- Analyze coverage gaps and calibration data

But the actual "pharmacist says agree/disagree" feedback must come
from licensed pharmacists. Phase 5 is the phase where PharmLoop
needs to expand beyond a solo developer project into something with
clinical collaborators.

This is also the phase where Anthony's pharmacy technician background
becomes directly valuable — he understands pharmacy workflows,
medication verification processes, and the practical constraints
pharmacists face. He can design the interface and feedback workflow
to match how pharmacists actually think about drug interactions,
rather than how a developer imagines they do.
