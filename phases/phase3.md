# Phase 3 — Clinical Output, Context Encoder, and Inference Pipeline

## Prerequisites
Phase 2 validated:
- [x] Severity accuracy >= 70% (achieved: 92%)
- [x] Zero false negatives on severe/contraindicated
- [x] Known pair convergence rate >= 85% (achieved: 100%)
- [x] GZ separation gap > 0.1 (achieved: 3.12)
- [x] Hopfield retrieval cosine > 0.3 (achieved: 0.82)
- [x] Annealing cycles completed (3 cycles)

**Phase 2 stretch targets carried forward:**
- [ ] Mechanism accuracy >= 60% (current: 52.9% — addressed in this phase)
- [ ] Convergence in <= 8 steps (current: 16 — addressed in this phase)

---

## What Phase 3 Accomplishes

Phases 1-2 proved the architecture works and the Hopfield retrieval is sound.
Phase 3 turns PharmLoop into something a pharmacist could actually use:

1. **Mechanism accuracy improvement** — fix the 52.9% gap with better readout architecture
2. **Faster convergence** — step-weighted convergence reward
3. **Partial convergence detection** — per-dimension analysis of what settled vs what didn't
4. **Template engine** — zero-param clinical natural language output
5. **Context encoder** — dose, route, timing, patient factors
6. **Inference pipeline** — drug names in, clinical narrative out, single function call

After Phase 3, the full pipeline is:
```
("fluoxetine", "tramadol", context={dose_a: "20mg", dose_b: "50mg", route: "oral"})
  →
  "Serious interaction between fluoxetine and tramadol. Both fluoxetine and
   tramadol increase serotonergic activity. Combined use increases the risk
   of serotonin syndrome, characterized by agitation, hyperthermia, and tremor.
   Watch for symptoms of serotonin syndrome: agitation, hyperthermia, clonus,
   tremor. Consider an alternative analgesic without serotonergic activity."
   [Confidence: 94% | Converged in 7 steps | Severity: settled | Mechanism: settled]
```

---

## Step 1: Mechanism Accuracy Improvement

### 1.1 The Problem

The current mechanism head is `Linear(512, 15)` — a single linear layer reading
the final oscillator position. This has to disentangle severity, mechanism, and
flag information from a single 512-dim vector with no nonlinearity.

Severity accuracy is 92% because severity is a coarse signal (6 classes, well-separated
in learned space). Mechanism is 15-class multi-label with many labels appearing in
only a handful of training pairs — a linear layer doesn't have enough capacity
to find the decision boundaries.

### 1.2 Fix: Trajectory-Aware Mechanism Head

Replace the single-linear mechanism head with a small MLP that reads from the
**trajectory**, not just the final position. The intuition: mechanism information
emerges during oscillation as the system explores different attractors. The path
the oscillator takes contains signal about *which* mechanisms are at play, not
just the endpoint.

```python
class TrajectoryMechanismHead(nn.Module):
    """
    Reads mechanism signal from the oscillation trajectory, not just
    the final position.

    Takes the last K positions from the trajectory, pools them, and
    runs through a small MLP. This captures mechanism information that
    emerges during oscillation but may not survive to the final position.

    Zero additional oscillator parameters — this only adds readout capacity.
    """

    def __init__(self, state_dim: int = 512, num_mechanisms: int = 15,
                 trajectory_window: int = 4, hidden_dim: int = 256) -> None:
        super().__init__()
        self.trajectory_window = trajectory_window

        # Attention pool over trajectory window
        # Learn which steps matter most for mechanism attribution
        self.step_attention = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # MLP for mechanism classification
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_mechanisms),
        )

    def forward(self, positions: list[Tensor]) -> Tensor:
        """
        Args:
            positions: List of (batch, state_dim) tensors from trajectory.

        Returns:
            (batch, num_mechanisms) mechanism logits.
        """
        # Take last K positions (or all if fewer than K)
        window = positions[-self.trajectory_window:]
        stacked = torch.stack(window, dim=1)  # (batch, K, state_dim)

        # Attention-weighted pool
        attn_scores = self.step_attention(stacked).squeeze(-1)  # (batch, K)
        attn_weights = torch.softmax(attn_scores, dim=1)         # (batch, K)
        pooled = (stacked * attn_weights.unsqueeze(-1)).sum(dim=1)  # (batch, state_dim)

        return self.mlp(pooled)
```

### 1.3 Integration

In `OutputHead`, replace:
```python
self.mechanism_head = nn.Linear(state_dim, num_mechanisms)
```

With:
```python
self.mechanism_head = TrajectoryMechanismHead(state_dim, num_mechanisms)
```

The `OutputHead.forward` signature changes to accept the trajectory:
```python
def forward(self, state: Tensor, positions: list[Tensor] | None = None) -> dict:
    severity_logits = self.severity_head(state)
    flag_logits = self.flags_head(state)

    if positions is not None:
        mechanism_logits = self.mechanism_head(positions)
    else:
        # Fallback: use final state only (for backward compatibility)
        mechanism_logits = self.mechanism_head([state])

    return {
        "severity_logits": severity_logits,
        "mechanism_logits": mechanism_logits,
        "flag_logits": flag_logits,
    }
```

Update `model.py` to pass trajectory positions to the output head:
```python
predictions = self.output_head(trajectory["final_x"], trajectory["positions"])
```

### 1.4 Parameter Impact

```
TrajectoryMechanismHead:
  step_attention:  512*64 + 64 + 64*1 + 1 = ~33K
  mlp:             512*256 + 256 + 256*15 + 15 = ~135K
  Total:           ~168K

Replaces:          Linear(512, 15) = ~8K
Net increase:      ~160K

New learned total: ~3.06M (still well under 10M budget)
```

### 1.5 Training

Fine-tune the new mechanism head while keeping everything else frozen for
5-10 epochs, then unfreeze all and fine-tune end-to-end for another 10-15
epochs. This prevents the new head from destabilizing the already-working
severity and convergence dynamics.

```python
# Stage 1: Train only new mechanism head (5-10 epochs)
for name, param in model.named_parameters():
    if "mechanism_head" not in name:
        param.requires_grad = False

train(model, epochs=10, lr=5e-4)

# Stage 2: Unfreeze all, fine-tune (10-15 epochs)
for param in model.parameters():
    param.requires_grad = True

# Lower LR for the rest of the model
optimizer = Adam([
    {"params": mechanism_params, "lr": 3e-4},
    {"params": other_params, "lr": 5e-5},
])
train(model, epochs=15)
```

---

## Step 2: Faster Convergence

### 2.1 The Problem

Training always runs all 16 steps for gradient consistency. The convergence
loss penalizes high final |v| but doesn't reward *early* velocity decay.
The oscillator learns to converge by step 16 but has no incentive to
converge by step 8.

### 2.2 Fix: Step-Weighted Convergence Bonus

Add a cumulative gray zone penalty that weights earlier steps more heavily:

```python
# In PharmLoopLoss.forward, add to L_convergence for known pairs:

# Step-weighted convergence: reward early settling
# Weight each step's |v| with exponential decay — early high-|v| costs more
if known_mask.sum() > 0:
    gz_steps = [vel.norm(dim=-1) for vel in velocities]  # list of (batch,)
    step_weights = torch.tensor(
        [0.95 ** (len(gz_steps) - 1 - i) for i in range(len(gz_steps))],
        device=device,
    )  # earlier steps have higher weight
    gz_stack = torch.stack(gz_steps, dim=0)  # (steps, batch)
    weighted_gz = (gz_stack * step_weights.unsqueeze(1)).mean(dim=0)  # (batch,)
    l_early_convergence = (weighted_gz * known_mask).sum() / (known_mask.sum() + 1e-8)
else:
    l_early_convergence = torch.tensor(0.0, device=device)
```

Add `l_early_convergence` to the total loss with weight ~0.3.

This creates pressure to drop |v| early: a pair that converges by step 4
accumulates less weighted gray zone than one that converges at step 16,
even if both have the same final |v|.

### 2.3 Target

After training with this loss, known severe interactions should converge
in 6-10 steps. Safe pairs might converge even faster (4-8 steps). The
exact numbers depend on training, but the gap between known and unknown
should widen — unknown pairs accumulate high weighted GZ throughout.

---

## Step 3: Partial Convergence Detection

### 3.1 Concept

Per-dimension oscillator parameters (from Phase 1 fixes) mean different
dimensions of velocity decay at different rates. Some dimensions may settle
by step 6 while others are still oscillating at step 16. This encodes
clinically meaningful partial certainty.

We need to detect and report this. The output head dimensions map to
interpretable categories (severity uses some dimensions, mechanism uses
others, flags use others). By tracking which dimension clusters converge,
we can report "severity is confident but mechanism is uncertain."

### 3.2 Implementation

```python
class PartialConvergenceAnalyzer:
    """
    Analyzes per-dimension velocity to determine which aspects of the
    prediction have settled and which are still uncertain.

    Maps oscillator dimensions to semantic categories using the output
    head weights as a guide: dimensions that the severity head weights
    heavily are "severity dimensions," etc.

    Zero parameters — this is pure analysis of trained model state.
    """

    def __init__(self, output_head: OutputHead, convergence_threshold: float = 0.05):
        self.threshold = convergence_threshold

        # Extract which dimensions each head relies on most
        # by looking at the L1 norm of each head's weight rows
        self.severity_dims = self._important_dims(output_head.severity_head)
        self.mechanism_dims = self._important_dims(output_head.mechanism_head)
        self.flag_dims = self._important_dims(output_head.flags_head)

    def _important_dims(self, head) -> Tensor:
        """
        Get dimension importance mask from a linear layer's weights.
        Returns indices of the top-K most important input dimensions.
        """
        if isinstance(head, nn.Linear):
            weight = head.weight.data
        elif hasattr(head, 'mlp'):
            # TrajectoryMechanismHead: use the first layer of the MLP
            weight = head.mlp[0].weight.data
        else:
            return torch.arange(512)

        importance = weight.abs().sum(dim=0)  # (state_dim,)
        # Top 30% of dimensions for each head
        k = max(1, int(0.3 * importance.shape[0]))
        _, top_indices = importance.topk(k)
        return top_indices

    def analyze(self, final_v: Tensor) -> dict:
        """
        Analyze partial convergence from final velocity.

        Args:
            final_v: (batch, state_dim) final velocity tensor.

        Returns:
            Dict with per-aspect convergence info:
              - "severity_settled": bool
              - "mechanism_settled": bool
              - "flags_settled": bool
              - "settled_aspects": list of settled aspect names
              - "unsettled_aspects": list of unsettled aspect names
              - "partial_convergence": bool (some settled, some not)
        """
        per_dim_gz = final_v.abs().mean(dim=0)  # (state_dim,)

        severity_gz = per_dim_gz[self.severity_dims].mean().item()
        mechanism_gz = per_dim_gz[self.mechanism_dims].mean().item()
        flag_gz = per_dim_gz[self.flag_dims].mean().item()

        severity_settled = severity_gz < self.threshold
        mechanism_settled = mechanism_gz < self.threshold
        flags_settled = flag_gz < self.threshold

        settled = []
        unsettled = []
        for name, is_settled in [("severity", severity_settled),
                                  ("mechanism", mechanism_settled),
                                  ("clinical flags", flags_settled)]:
            (settled if is_settled else unsettled).append(name)

        return {
            "severity_settled": severity_settled,
            "mechanism_settled": mechanism_settled,
            "flags_settled": flags_settled,
            "settled_aspects": settled,
            "unsettled_aspects": unsettled,
            "partial_convergence": bool(settled) and bool(unsettled),
            "severity_gz": severity_gz,
            "mechanism_gz": mechanism_gz,
            "flags_gz": flag_gz,
        }
```

This is built at inference time from the trained model's weights — no training
needed. It's a diagnostic tool that reads the oscillator's learned structure.

---

## Step 4: Template Engine (ClinicalNarrator)

### 4.1 Design Principles

- **Zero learned parameters.** Every sentence comes from a verified template.
- **Deterministic.** Same structured output → same clinical narrative, always.
- **Auditable.** A pharmacist can read the templates and verify they're correct
  independently of the model.
- **Conservative.** When in doubt, recommend caution. The templates never
  downplay risk.
- **Composable.** Severity template + mechanism explanation + flag-specific
  monitoring + confidence qualifier + partial convergence report.

### 4.2 Implementation

**File: `pharmloop/templates.py`**

```python
class ClinicalNarrator:
    """
    Maps structured model output → clinical English.
    Zero learned parameters. Pure lookup + composition.
    """

    # ── Severity templates ──
    # Each template has slots for {A}, {B}, {mechanism}, {monitoring}, etc.
    # Slots that don't get filled are omitted (not left as "{monitoring}")
    SEVERITY_TEMPLATES = {
        "none": (
            "No clinically significant interaction identified between "
            "{A} and {B}."
        ),
        "mild": (
            "Minor interaction between {A} and {B}. {mechanism} "
            "Generally safe with routine monitoring. {monitoring}"
        ),
        "moderate": (
            "Moderate interaction between {A} and {B}. {mechanism} "
            "{action} {monitoring}"
        ),
        "severe": (
            "Serious interaction between {A} and {B}. {mechanism} "
            "{action} {monitoring}"
        ),
        "contraindicated": (
            "CONTRAINDICATED: {A} and {B} should not be used together. "
            "{mechanism} {action}"
        ),
        "unknown": (
            "Insufficient data to assess interaction between {A} and {B}. "
            "The interaction profile could not be determined with confidence. "
            "Consult a pharmacist or prescriber for guidance."
        ),
    }

    # ── Mechanism explanations ──
    # Keyed by mechanism name from MECHANISM_NAMES in output.py
    MECHANISM_EXPLANATIONS = {
        "serotonergic": (
            "Both {A} and {B} increase serotonergic activity. "
            "Combined use increases the risk of serotonin syndrome, "
            "characterized by agitation, hyperthermia, clonus, and tremor."
        ),
        "cyp_inhibition": (
            "{A} inhibits cytochrome P450 enzymes involved in the metabolism "
            "of {B}. This may increase {B} plasma concentrations and the risk "
            "of dose-related adverse effects."
        ),
        "cyp_induction": (
            "{A} induces cytochrome P450 enzymes involved in the metabolism "
            "of {B}. This may decrease {B} plasma concentrations and reduce "
            "therapeutic efficacy."
        ),
        "qt_prolongation": (
            "Both {A} and {B} are associated with QT interval prolongation. "
            "Combined use increases the risk of serious cardiac arrhythmias "
            "including torsades de pointes."
        ),
        "bleeding_risk": (
            "Both {A} and {B} affect hemostasis. Combined use may "
            "significantly increase the risk of bleeding."
        ),
        "cns_depression": (
            "Both {A} and {B} have central nervous system depressant effects. "
            "Combined use may cause excessive sedation, respiratory depression, "
            "and impaired cognitive and motor function."
        ),
        "nephrotoxicity": (
            "Both {A} and {B} may adversely affect renal function. "
            "Combined use increases the risk of nephrotoxicity."
        ),
        "hepatotoxicity": (
            "Both {A} and {B} may adversely affect hepatic function. "
            "Combined use increases the risk of hepatotoxicity."
        ),
        "hypotension": (
            "Both {A} and {B} can lower blood pressure. "
            "Combined use increases the risk of hypotension, "
            "particularly orthostatic hypotension."
        ),
        "hyperkalemia": (
            "Both {A} and {B} can increase serum potassium levels. "
            "Combined use increases the risk of hyperkalemia."
        ),
        "seizure_risk": (
            "Both {A} and {B} may lower the seizure threshold. "
            "Combined use increases the risk of seizures."
        ),
        "immunosuppression": (
            "{A} may alter the metabolism or effect of {B}, affecting "
            "immunosuppressive drug levels and efficacy."
        ),
        "absorption_altered": (
            "{A} may alter the gastrointestinal absorption of {B}, "
            "potentially affecting its bioavailability and therapeutic effect."
        ),
        "protein_binding_displacement": (
            "{A} and {B} compete for plasma protein binding sites. "
            "Displacement may transiently increase free drug concentrations "
            "and the risk of adverse effects."
        ),
        "electrolyte_imbalance": (
            "The combination of {A} and {B} may cause electrolyte "
            "disturbances. Monitor serum electrolytes."
        ),
    }

    # ── Clinical flag → monitoring recommendation ──
    # Keyed by flag name from FLAG_NAMES in output.py
    FLAG_RECOMMENDATIONS = {
        "monitor_serotonin_syndrome": (
            "Watch for symptoms of serotonin syndrome: agitation, "
            "hyperthermia, clonus, diaphoresis, tremor, and hyperreflexia."
        ),
        "monitor_inr": (
            "Monitor INR and adjust anticoagulant dosage as needed."
        ),
        "monitor_qt_interval": (
            "Obtain baseline ECG and monitor QTc interval."
        ),
        "monitor_renal_function": (
            "Monitor serum creatinine, BUN, and urine output."
        ),
        "monitor_hepatic_function": (
            "Monitor liver function tests (ALT, AST, bilirubin)."
        ),
        "monitor_blood_pressure": (
            "Monitor blood pressure regularly, particularly with "
            "position changes."
        ),
        "monitor_blood_glucose": (
            "Monitor blood glucose levels more frequently."
        ),
        "monitor_electrolytes": (
            "Monitor serum electrolytes including potassium, sodium, "
            "and magnesium."
        ),
        "monitor_drug_levels": (
            "Monitor serum drug levels and adjust dosage as needed."
        ),
        "monitor_cns_depression": (
            "Monitor for excessive sedation, respiratory depression, "
            "and impaired cognitive function."
        ),
        "avoid_combination": (
            "Consider avoiding this combination if possible. "
            "If co-administration is necessary, use with extreme caution."
        ),
        "monitor_bleeding": (
            "Monitor for signs of bleeding: bruising, petechiae, "
            "gastrointestinal or gum bleeding, dark stools."
        ),
        "monitor_digoxin_levels": (
            "Monitor serum digoxin levels and watch for signs of "
            "digoxin toxicity (nausea, visual disturbances, arrhythmias)."
        ),
        "monitor_lithium_levels": (
            "Monitor serum lithium levels and watch for signs of "
            "lithium toxicity (tremor, nausea, confusion, ataxia)."
        ),
        "monitor_cyclosporine_levels": (
            "Monitor serum cyclosporine levels and adjust dosage "
            "to maintain therapeutic range."
        ),
        "monitor_theophylline_levels": (
            "Monitor serum theophylline levels and adjust dosage "
            "as needed to avoid toxicity."
        ),
        "reduce_statin_dose": (
            "Consider reducing the statin dose. The interaction may "
            "increase statin plasma levels and the risk of myopathy "
            "or rhabdomyolysis."
        ),
        "separate_administration": (
            "Separate administration times by at least 2 hours "
            "to minimize the interaction."
        ),
    }

    # ── Severity-specific action recommendations ──
    ACTIONS = {
        "none": "",
        "mild": "No dose adjustment typically required.",
        "moderate": (
            "Use with caution. Dose adjustment may be required. "
            "Weigh benefits against risks."
        ),
        "severe": (
            "Use only if benefit clearly outweighs risk. "
            "Consider therapeutic alternatives."
        ),
        "contraindicated": (
            "Do not co-administer. Select an alternative agent."
        ),
        "unknown": "",
    }

    def narrate(
        self,
        drug_a_name: str,
        drug_b_name: str,
        severity: str,
        mechanisms: list[str],
        flags: list[str],
        confidence: float,
        converged: bool,
        steps: int,
        partial_convergence: dict | None = None,
    ) -> str:
        """
        Compose clinical narrative from structured model output.

        All inputs are post-processed model outputs (strings and floats),
        not raw tensors.

        Returns:
            Multi-paragraph clinical narrative string.
        """
        A = drug_a_name.capitalize()
        B = drug_b_name.capitalize()
        sections = []

        # ── Primary assessment ──
        mechanism_text = self._compose_mechanisms(mechanisms, A, B)
        monitoring_text = self._compose_monitoring(flags)
        action_text = self.ACTIONS.get(severity, "")

        template = self.SEVERITY_TEMPLATES.get(severity, self.SEVERITY_TEMPLATES["unknown"])
        primary = template.format(
            A=A, B=B,
            mechanism=mechanism_text,
            monitoring=monitoring_text,
            action=action_text,
        )
        # Clean up double spaces and trailing whitespace from empty slots
        primary = " ".join(primary.split())
        sections.append(primary)

        # ── Monitoring recommendations (detailed) ──
        if flags and severity not in ("none", "unknown"):
            recs = [self.FLAG_RECOMMENDATIONS[f] for f in flags
                    if f in self.FLAG_RECOMMENDATIONS]
            if recs:
                sections.append("Monitoring: " + " ".join(recs))

        # ── Confidence qualifier ──
        if confidence < 0.5 and severity != "unknown":
            sections.append(
                f"Note: This assessment has limited confidence ({confidence:.0%}). "
                f"The interaction profile for this combination is not well-characterized. "
                f"Clinical judgment should be applied."
            )

        # ── Partial convergence report ──
        if partial_convergence and partial_convergence.get("partial_convergence"):
            settled = partial_convergence["settled_aspects"]
            unsettled = partial_convergence["unsettled_aspects"]
            if unsettled:
                sections.append(
                    f"The assessment of {', '.join(settled)} is confident, "
                    f"but {', '.join(unsettled)} could not be fully determined."
                )

        # ── Metadata line ──
        status = f"Converged in {steps} steps" if converged else f"Did not converge ({steps}/{16} steps)"
        sections.append(f"[Confidence: {confidence:.0%} | {status}]")

        return "\n\n".join(sections)

    def _compose_mechanisms(self, mechanisms: list[str], A: str, B: str) -> str:
        """Compose mechanism explanation from list of active mechanisms."""
        if not mechanisms:
            return ""
        explanations = []
        for mech in mechanisms:
            template = self.MECHANISM_EXPLANATIONS.get(mech)
            if template:
                explanations.append(template.format(A=A, B=B))
        return " ".join(explanations)

    def _compose_monitoring(self, flags: list[str]) -> str:
        """Compose brief monitoring summary from flags."""
        if not flags:
            return ""
        brief = []
        for flag in flags[:3]:  # limit to top 3 for the summary line
            if "serotonin" in flag:
                brief.append("serotonin syndrome symptoms")
            elif "inr" in flag:
                brief.append("INR")
            elif "qt" in flag:
                brief.append("QTc interval")
            elif "renal" in flag:
                brief.append("renal function")
            elif "hepatic" in flag:
                brief.append("liver function")
            elif "blood_pressure" in flag:
                brief.append("blood pressure")
            elif "glucose" in flag:
                brief.append("blood glucose")
            elif "electrolyte" in flag:
                brief.append("electrolytes")
            elif "bleeding" in flag:
                brief.append("signs of bleeding")
            elif "cns" in flag:
                brief.append("CNS depression")
            elif "drug_levels" in flag or "digoxin" in flag or "lithium" in flag \
                 or "cyclosporine" in flag or "theophylline" in flag:
                brief.append("drug levels")
        if brief:
            return f"Monitor {', '.join(brief)}."
        return ""
```

### 4.3 Template Verification

Every template must be clinically reviewed. Create a test file that renders
every possible template path and outputs it for human review:

```python
# tests/test_templates.py

class TestTemplateCompleteness:
    """Verify every model output maps to a valid template."""

    def test_all_severities_have_templates(self):
        narrator = ClinicalNarrator()
        for severity in SEVERITY_NAMES:
            assert severity in narrator.SEVERITY_TEMPLATES

    def test_all_mechanisms_have_explanations(self):
        narrator = ClinicalNarrator()
        for mechanism in MECHANISM_NAMES:
            assert mechanism in narrator.MECHANISM_EXPLANATIONS, (
                f"No template for mechanism: {mechanism}"
            )

    def test_all_flags_have_recommendations(self):
        narrator = ClinicalNarrator()
        for flag in FLAG_NAMES:
            assert flag in narrator.FLAG_RECOMMENDATIONS, (
                f"No template for flag: {flag}"
            )

    def test_no_unresolved_format_slots(self):
        """No output should contain {A} or {B} or other unresolved slots."""
        narrator = ClinicalNarrator()
        for severity in SEVERITY_NAMES:
            for mechanism in MECHANISM_NAMES:
                output = narrator.narrate(
                    "fluoxetine", "tramadol", severity,
                    [mechanism], ["monitor_serotonin_syndrome"],
                    confidence=0.85, converged=True, steps=8,
                )
                assert "{" not in output, f"Unresolved slot in: {output[:100]}"

    def test_unknown_severity_ignores_mechanisms(self):
        """Unknown severity should not mention mechanisms."""
        narrator = ClinicalNarrator()
        output = narrator.narrate(
            "QZ-7734", "aspirin", "unknown", [], [],
            confidence=0.08, converged=False, steps=16,
        )
        assert "serotonin" not in output.lower()
        assert "insufficient data" in output.lower()

    def test_render_all_paths(self):
        """Render every severity × mechanism combination for human review."""
        narrator = ClinicalNarrator()
        for severity in SEVERITY_NAMES:
            for mechanism in MECHANISM_NAMES:
                output = narrator.narrate(
                    "drug_A", "drug_B", severity,
                    [mechanism], [],
                    confidence=0.75, converged=True, steps=10,
                )
                # Print for human review (run with pytest -s)
                print(f"\n{'='*60}")
                print(f"Severity: {severity} | Mechanism: {mechanism}")
                print(f"{'='*60}")
                print(output)
```

---

## Step 5: Context Encoder

### 5.1 What Context Adds

Some interactions are dose-dependent, route-dependent, or timing-dependent:
- Warfarin + acetaminophen: safe at low doses, risky at high chronic doses
- Ciprofloxacin + antacids: interaction blocked if separated by 2 hours
- CYP interactions are dose-proportional: higher dose → stronger inhibition
- Renal impairment changes elimination-dependent interactions

The context encoder injects this information into the initial pair state
before the oscillator runs.

### 5.2 Context Feature Vector

Define a structured context vector (~32 dims):

```python
CONTEXT_DIM = 32

# Context feature layout:
# Dims 0-3:   Drug A dosing (dose_normalized, frequency, duration_days, is_loading_dose)
# Dims 4-7:   Drug B dosing (same layout)
# Dims 8-11:  Route flags (both_oral, any_iv, any_topical, any_inhaled)
# Dims 12-15: Timing (simultaneous, separated_hours_norm, a_before_b, b_before_a)
# Dims 16-23: Patient factors (age_norm, weight_norm, renal_gfr_norm,
#             hepatic_child_pugh_norm, pregnancy, pediatric, geriatric, genetic_pm)
# Dims 24-27: Comedication burden (total_drugs_norm, cyp_inhibitor_count,
#             cyp_inducer_count, protein_bound_count)
# Dims 28-31: Reserved
```

### 5.3 Context Encoder Architecture

```python
class ContextEncoder(nn.Module):
    """
    Encodes contextual factors (dose, route, timing, patient) into a
    modulation signal for the pair state.

    Does NOT replace the pair state — it MODULATES it. The base interaction
    profile comes from the drug pair; context adjusts it.

    ~200K params.
    """

    def __init__(self, context_dim: int = 32, state_dim: int = 512):
        super().__init__()
        self.context_dim = context_dim

        # Project context to state_dim
        self.context_proj = nn.Sequential(
            nn.Linear(context_dim, 128),
            nn.GELU(),
            nn.Linear(128, state_dim),
        )

        # Gating: context_gate ∈ [0, 1] per-dimension
        # Controls how much context modifies the base pair state
        # Initialized near-zero so the model starts ~= Phase 2 behavior
        self.gate = nn.Sequential(
            nn.Linear(context_dim, 128),
            nn.GELU(),
            nn.Linear(128, state_dim),
            nn.Sigmoid(),
        )

    def forward(self, pair_state: Tensor, context: Tensor) -> Tensor:
        """
        Modulate pair state with context.

        Args:
            pair_state: (batch, state_dim) from drug pair encoding.
            context: (batch, context_dim) structured context features.

        Returns:
            (batch, state_dim) context-modulated pair state.
        """
        ctx_signal = self.context_proj(context)   # (batch, state_dim)
        gate = self.gate(context)                  # (batch, state_dim)

        # Gated additive modulation:
        # pair_state + gate * ctx_signal
        # When gate ≈ 0 (no context or irrelevant context), output ≈ pair_state
        # When gate > 0, context shifts the starting point of oscillation
        return pair_state + gate * ctx_signal
```

### 5.4 Integration into PharmLoopModel

Context is optional. When not provided, the model behaves identically to
Phase 2 (backward compatible).

```python
class PharmLoopModel(nn.Module):
    def __init__(self, ..., use_context: bool = False):
        ...
        self.context_encoder = ContextEncoder() if use_context else None

    def forward(self, drug_a_id, drug_a_features, drug_b_id, drug_b_features,
                context: Tensor | None = None):
        ...
        initial_state = self.pair_combine(pair_forward) + self.pair_combine(pair_reverse)
        initial_state = initial_state / 2.0

        # Context modulation (optional)
        if self.context_encoder is not None and context is not None:
            initial_state = self.context_encoder(initial_state, context)

        trajectory = self.reasoning_loop(initial_state, training=self.training)
        ...
```

### 5.5 Context Data

For Phase 3, create a SMALL set of context-dependent interaction examples
to validate the context encoder works:

```python
# data/processed/context_examples.json
[
    {
        "drug_a": "warfarin",
        "drug_b": "acetaminophen",
        "context": {"dose_b": "low", "duration_b": "acute"},
        "severity_without_context": "moderate",
        "severity_with_context": "mild",
        "note": "Low-dose short-term acetaminophen has minimal warfarin interaction"
    },
    {
        "drug_a": "warfarin",
        "drug_b": "acetaminophen",
        "context": {"dose_b": "high", "duration_b": "chronic"},
        "severity_without_context": "moderate",
        "severity_with_context": "severe",
        "note": "Chronic high-dose acetaminophen significantly increases INR"
    },
    {
        "drug_a": "ciprofloxacin",
        "drug_b": "omeprazole",
        "context": {"timing": "simultaneous"},
        "severity_without_context": "moderate",
        "severity_with_context": "moderate"
    },
    {
        "drug_a": "ciprofloxacin",
        "drug_b": "omeprazole",
        "context": {"timing": "separated_2h"},
        "severity_without_context": "moderate",
        "severity_with_context": "mild",
        "note": "Separation reduces absorption interaction"
    }
]
```

Don't try to train the context encoder comprehensively in Phase 3 — just
validate that context modulation CHANGES the output in the expected direction.
Comprehensive context training is Phase 4+ with a larger dataset.

### 5.6 Gate Initialization

Initialize the gate bias to -2.0 so the sigmoid outputs ~0.12 at init.
This ensures the model starts very close to Phase 2 behavior and context
has to earn its influence through training:

```python
# After creating context_encoder:
with torch.no_grad():
    self.context_encoder.gate[-2].bias.fill_(-2.0)
```

---

## Step 6: Inference Pipeline

### 6.1 Top-Level API

**File: `pharmloop/inference.py`**

This is the function a pharmacist-facing application calls. Drug names in,
clinical narrative out.

```python
class PharmLoopInference:
    """
    Complete inference pipeline: drug names → clinical narrative.

    Usage:
        engine = PharmLoopInference.load("checkpoints/phase3_best.pt")
        result = engine.check("fluoxetine", "tramadol")
        print(result.narrative)
        print(result.severity, result.confidence)
    """

    def __init__(self, model: PharmLoopModel, drug_registry: dict,
                 narrator: ClinicalNarrator,
                 convergence_analyzer: PartialConvergenceAnalyzer):
        self.model = model
        self.drug_registry = drug_registry
        self.narrator = narrator
        self.analyzer = convergence_analyzer
        self.model.eval()

    @classmethod
    def load(cls, checkpoint_path: str, data_dir: str = "data/processed"):
        """Load a trained model and build the inference engine."""
        ...  # load checkpoint, build model, load drug registry,
             # create narrator and analyzer

    def check(
        self,
        drug_a_name: str,
        drug_b_name: str,
        context: dict | None = None,
    ) -> "InteractionResult":
        """
        Check interaction between two drugs.

        Args:
            drug_a_name: Drug name (must be in registry or returns unknown).
            drug_b_name: Drug name.
            context: Optional dict with dose, route, timing, patient info.

        Returns:
            InteractionResult with all predictions and clinical narrative.
        """
        # Look up drugs
        drug_a = self.drug_registry.get(drug_a_name.lower())
        drug_b = self.drug_registry.get(drug_b_name.lower())

        # Unknown drug → fabricated input (will fail to converge → UNKNOWN)
        if drug_a is None or drug_b is None:
            return self._handle_unknown(drug_a_name, drug_b_name, drug_a, drug_b)

        # Prepare tensors
        a_id = torch.tensor([drug_a["id"]], dtype=torch.long)
        a_feat = torch.tensor([drug_a["features"]], dtype=torch.float32)
        b_id = torch.tensor([drug_b["id"]], dtype=torch.long)
        b_feat = torch.tensor([drug_b["features"]], dtype=torch.float32)

        # Context encoding (if provided)
        ctx_tensor = self._encode_context(context) if context else None

        # Forward pass
        with torch.no_grad():
            output = self.model(a_id, a_feat, b_id, b_feat, context=ctx_tensor)

        # Post-process
        severity_idx = output["severity_logits"].argmax(dim=-1).item()
        severity = SEVERITY_NAMES[severity_idx]
        confidence = output["confidence"].item()
        converged = output["converged"].item()
        steps = output["trajectory"]["steps"]

        # Mechanisms above threshold
        mech_probs = torch.sigmoid(output["mechanism_logits"]).squeeze()
        mechanisms = [MECHANISM_NAMES[i] for i, p in enumerate(mech_probs)
                      if p.item() > 0.5]

        # Flags above threshold
        flag_probs = torch.sigmoid(output["flag_logits"]).squeeze()
        flags = [FLAG_NAMES[i] for i, p in enumerate(flag_probs)
                 if p.item() > 0.5]

        # Partial convergence analysis
        partial = self.analyzer.analyze(output["trajectory"]["velocities"][-1])

        # Generate narrative
        narrative = self.narrator.narrate(
            drug_a_name, drug_b_name, severity,
            mechanisms, flags, confidence, converged, steps, partial,
        )

        return InteractionResult(
            drug_a=drug_a_name,
            drug_b=drug_b_name,
            severity=severity,
            mechanisms=mechanisms,
            flags=flags,
            confidence=confidence,
            converged=converged,
            steps=steps,
            partial_convergence=partial,
            narrative=narrative,
            gray_zone_trajectory=[gz.item() for gz in output["trajectory"]["gray_zones"]],
        )

    def _handle_unknown(self, drug_a_name, drug_b_name, drug_a, drug_b):
        """Handle case where one or both drugs are not in registry."""
        unknown_drugs = []
        if drug_a is None:
            unknown_drugs.append(drug_a_name)
        if drug_b is None:
            unknown_drugs.append(drug_b_name)

        narrative = self.narrator.narrate(
            drug_a_name, drug_b_name, "unknown", [], [],
            confidence=0.0, converged=False, steps=0,
        )

        return InteractionResult(
            drug_a=drug_a_name, drug_b=drug_b_name,
            severity="unknown", mechanisms=[], flags=[],
            confidence=0.0, converged=False, steps=0,
            partial_convergence=None, narrative=narrative,
            gray_zone_trajectory=[],
            unknown_drugs=unknown_drugs,
        )

    def _encode_context(self, context: dict) -> Tensor:
        """Convert context dict to 32-dim feature tensor."""
        vec = torch.zeros(1, 32)
        # Map context dict keys to feature dims
        # ... (dose normalization, route flags, timing, patient factors)
        return vec


@dataclass
class InteractionResult:
    """Complete result of a drug interaction check."""
    drug_a: str
    drug_b: str
    severity: str
    mechanisms: list[str]
    flags: list[str]
    confidence: float
    converged: bool
    steps: int
    partial_convergence: dict | None
    narrative: str
    gray_zone_trajectory: list[float]
    unknown_drugs: list[str] | None = None
```

---

## Implementation Files

### New Files
```
pharmloop/
    templates.py              ← ClinicalNarrator (zero-param template engine)
    context.py                ← ContextEncoder (~200K params)
    partial_convergence.py    ← PartialConvergenceAnalyzer (zero params)
    inference.py              ← PharmLoopInference + InteractionResult
tests/
    test_templates.py         ← Template completeness and rendering
    test_context.py           ← Context modulation validation
    test_inference.py         ← End-to-end inference pipeline
    test_partial_conv.py      ← Partial convergence detection
data/processed/
    context_examples.json     ← Small set of context-dependent interactions
```

### Modified Files
```
pharmloop/output.py           ← TrajectoryMechanismHead replaces linear mech head
pharmloop/model.py            ← Pass positions to output head, optional context
training/loss.py              ← Add l_early_convergence term
training/train_phase3.py      ← New: staged training for mechanism head + context
```

### Unchanged Files
```
pharmloop/encoder.py
pharmloop/hopfield.py
pharmloop/oscillator.py       ← No changes to the core dynamics
```

---

## Step 7: Implementation Sequence

### 7.1 Mechanism head upgrade + faster convergence loss (do first — needs retraining)
1. Implement `TrajectoryMechanismHead` in `output.py`
2. Update `model.py` to pass positions to output head
3. Add `l_early_convergence` to `loss.py`
4. Implement `training/train_phase3.py` with staged training
5. Train: freeze-all-except-mechanism (10 epochs) → unfreeze-all (15 epochs)
6. Verify mechanism accuracy >= 60% and convergence speed improved

### 7.2 Partial convergence analyzer (no training needed)
7. Implement `pharmloop/partial_convergence.py`
8. Implement `tests/test_partial_conv.py`
9. Run on test pairs — verify partial convergence detected on ambiguous pairs

### 7.3 Template engine (no training needed)
10. Implement `pharmloop/templates.py`
11. Implement `tests/test_templates.py`
12. Run `test_render_all_paths` and review output for clinical accuracy
13. Manual review of narrative output for the three test cases

### 7.4 Context encoder (needs training)
14. Create `data/processed/context_examples.json`
15. Implement `pharmloop/context.py`
16. Integrate into `model.py`
17. Implement `tests/test_context.py`
18. Train with context examples (light — just validate modulation works)

### 7.5 Inference pipeline (integration, no training)
19. Implement `pharmloop/inference.py`
20. Implement `tests/test_inference.py`
21. End-to-end test: drug names in, clinical narrative out

---

## Validation Criteria (must pass before Phase 4)

### Architecture
- [ ] TrajectoryMechanismHead replaces linear mechanism head
- [ ] Context encoder integrated with gate initialization
- [ ] Partial convergence analyzer built from trained model weights
- [ ] Template engine covers all 6 severities × 15 mechanisms × 18 flags
- [ ] Inference pipeline works: string drug names → InteractionResult

### Accuracy
- [ ] Mechanism accuracy >= 60% (target: 65%+)
- [ ] Severity accuracy maintained >= 90% (no regression from mechanism change)
- [ ] Zero false negatives on severe/contraindicated (maintained)
- [ ] Average convergence steps on known pairs <= 10 (improved from 16)

### Template Quality
- [ ] All template paths render without unresolved {slots}
- [ ] Unknown severity produces only "insufficient data" language
- [ ] Confidence qualifiers appear only when confidence < 50%
- [ ] Partial convergence reports appear only when truly partial
- [ ] Clinical narrative for fluoxetine+tramadol is pharmacologically accurate
- [ ] Clinical narrative for fabricated drug produces clear "unknown" output

### Context
- [ ] Context-free inference produces identical results to Phase 2
- [ ] High-dose warfarin+acetaminophen predicts higher severity than low-dose
- [ ] Context modulation gate starts near-zero (backward compatible)
- [ ] Model with context disabled matches Phase 2 checkpoints

### Integration
- [ ] `PharmLoopInference.check("fluoxetine", "tramadol")` returns complete result
- [ ] `PharmLoopInference.check("QZ-7734", "aspirin")` returns unknown result
- [ ] Result dataclass contains narrative, severity, confidence, trajectory
- [ ] Param budget still < 10M

---

## What NOT to Do in Phase 3

- Don't expand the drug set (Phase 4)
- Don't build a GUI or REST API (Phase 4+)
- Don't try to comprehensively train the context encoder — just validate the mechanism works with a few examples
- Don't modify the oscillator core or Hopfield — those are proven
- Don't add a learned text decoder — templates are the right choice at this scale
- Don't over-engineer the context feature vector — 32 dims is plenty, most will be zero initially
