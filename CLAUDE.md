# CLAUDE.md — D2C PharmLoop

## What This Is

D2C PharmLoop is a **recurrent oscillatory reasoning engine** for drug interaction checking. It is NOT an LLM. It thinks by oscillating a damped dynamical system until beliefs converge — or explicitly fail to converge and output "UNKNOWN."

**Core invariant: the model cannot confidently fabricate.** Confident output requires convergence. Unknown inputs don't converge. This is architectural, not behavioral.

Target: offline drug interaction reference for pharmacists. **DO NO HARM over helpfulness, always.**

---

## Architecture Overview

```
Drug A ─→ ┐                              ┌─→ Severity (6-class)
           ├─→ Drug Encoder ─→ Pair State ─→ Oscillatory Core ─→ Output Head ─→ Mechanism (multi-label)
Drug B ─→ ┘        ↑                ↕                                          └─→ Clinical Flags
                    │          Hopfield Memory
              64-dim features   (verified patterns)
```

### Components

1. **Drug Encoder** (~1.1M params)
   - Per-drug identity embedding (learnable, 256-dim)
   - 64-dim structured pharmacological feature vector (CYP profile, receptors, binding, elimination)
   - Fusion: concat → Linear → LayerNorm → GELU → Linear → 512-dim

2. **Oscillatory Reasoning Core** (~609K params) — THE HEART
   - State = (position `x`, velocity `v`) in 512-dim space
   - `x` = current belief about the drug pair interaction
   - `v` = rate/direction of belief change
   - **Gray zone = |v|** — not a side computation, IS the uncertainty
   - Update rule (damped driven oscillator):
     ```
     force = spring * evidence_transform(cat(x, hopfield_retrieved))
     noise = randn * noise_gate(|v|) * 0.1
     v(t+1) = clamp(decay, 0.5, 0.99) * v(t) + force + noise
     x(t+1) = x(t) + clamp(dt, 0.01, 0.5) * v(t+1)
     ```
   - Convergence: |v| drops below learned threshold → stop, output answer
   - Non-convergence: max_steps (16) reached with |v| still high → output UNKNOWN
   - Hopfield retrieval beta modulated by gray zone (high uncertainty → broader retrieval)

3. **Hopfield Memory Bank** (~524K learned + buffer for stored patterns)
   - Modern continuous Hopfield network (exponential energy)
   - Stores verified drug interaction patterns
   - Query/key projections are learned; stored patterns are buffers
   - Can grow: new verified interactions added without retraining the core
   - Retrieval: softmax(beta * query @ keys.T) @ values

4. **Output Head** (~412K params)
   - Severity classifier: 6 classes (none / mild / moderate / severe / contraindicated / unknown)
   - Mechanism classifier: multi-label (CYP inhibition, CYP induction, serotonergic, QT prolongation, etc.)
   - Clinical flags: binary flags for monitoring recommendations
   - Confidence: derived directly from convergence dynamics (final |v|, steps to converge), NOT a separate learned head

5. **Template Engine** (0 params)
   - Structured output → deterministic natural language
   - Pure lookup/format — no generation, no hallucination possible
   - Severity-keyed templates with mechanism/flag slot fills

### Parameter Budget

```
TOTAL LEARNED:       ~2.9M params
TOTAL WITH BUFFERS:  ~5.4M params
HARD BUDGET:         10M params
```

---

## Training Strategy

### Phase 0: Feature-Space Hopfield (no training)
- Build Hopfield bank directly from 64-dim pharmacological feature vectors
- This gives the oscillator something to retrieve against BEFORE the encoder exists

### Phase 1: Train Encoder Against Fixed Hopfield
- Freeze Hopfield, train encoder + oscillator + output head
- The oscillator learns its dynamics against stable retrieval targets

### Phase 2: Rebuild Hopfield in Learned Space
- Encode all drugs with trained encoder
- Rebuild Hopfield bank in 512-dim learned space
- Fine-tune end-to-end

### Loss Function (multi-objective)
```
L_total = L_answer + L_convergence + L_smoothness + L_do_no_harm

L_answer:       cross-entropy on severity + mechanism + flags
L_convergence:  reward fast convergence on known, non-convergence on unknown
L_smoothness:   penalize chaotic oscillation (second derivative of GZ trajectory)
L_do_no_harm:   10x penalty for false-none on severe, 50x on contraindicated
```

---

## Project Structure

```
d2c-pharmloop/
├── CLAUDE.md              ← you are here
├── phases/                ← phase-by-phase implementation specs
│   ├── phase1.md          ← current phase instructions
│   ├── phase2.md          ← unlocked after phase 1 validated
│   └── ...
├── data/
│   ├── raw/               ← DrugBank exports, interaction databases
│   ├── processed/         ← 64-dim feature vectors, interaction pairs
│   └── drugs.json         ← master drug registry (50 drugs initially)
├── pharmloop/
│   ├── __init__.py
│   ├── encoder.py         ← DrugEncoder
│   ├── oscillator.py      ← OscillatorCell + ReasoningLoop
│   ├── hopfield.py        ← PharmHopfield (modern continuous)
│   ├── output.py          ← OutputHead + confidence derivation
│   ├── model.py           ← PharmLoopModel (full pipeline)
│   ├── templates.py       ← TemplateEngine (zero-param NL output)
│   └── context.py         ← ContextEncoder (dose/route/timing — later phase)
├── training/
│   ├── loss.py            ← multi-objective loss with DO NO HARM
│   ├── train.py           ← training loop with phase switching
│   └── data_loader.py     ← dataset + batching
├── tests/
│   ├── test_oscillator.py ← does it actually oscillate? convergence tests
│   ├── test_hopfield.py   ← retrieval accuracy, capacity
│   ├── test_separation.py ← THE BIG TEST: 3-way separation
│   └── test_templates.py  ← NL output correctness
├── notebooks/
│   └── trajectory_viz.ipynb  ← visualize oscillation trajectories
├── requirements.txt
└── pyproject.toml
```

---

## Coding Conventions

- **Python 3.10+**, PyTorch 2.x
- Type hints everywhere: `def forward(self, x: Tensor, v: Tensor) -> tuple[Tensor, Tensor, float]:`
- Docstrings on every class and public method — this is a medical system, clarity is non-negotiable
- No magic numbers without named constants or comments explaining them
- Assertions on tensor shapes in forward passes during development (`assert x.shape == (batch, 512)`)
- Use `torch.no_grad()` explicitly where needed — don't rely on inference mode context
- Tests use `pytest`; every component gets unit tests before integration

---

## Development Workflow

1. Read the current phase file in `phases/`
2. Implement what it asks for
3. Run the tests it specifies
4. Report results (especially: does it pass the phase's validation criteria?)
5. Anthony reviews, we iterate, then unlock the next phase

**Do not jump ahead.** Each phase builds on validated output from the previous one. If a phase's tests fail, we fix before moving on.

---

## Key Design Decisions (Non-Negotiable)

1. **Gray zone IS |v|.** Never compute uncertainty separately. If you're tempted to add an uncertainty head, stop. The velocity IS the uncertainty.

2. **Hopfield memory is explicit.** Patterns have provenance. You can point to WHY the model thinks two drugs interact. This is not a black box.

3. **Template engine has zero learned parameters.** Natural language output is deterministic lookup from structured predictions. No generation.

4. **Convergence drives confidence.** Confidence = f(steps_to_converge, final_|v|, trajectory_smoothness). Not a learned scalar.

5. **Unknown > Wrong.** The system must fail to "I don't know" rather than fabricate. This is enforced by architecture (non-convergence → UNKNOWN) and by loss (heavy penalty for false-none on dangerous interactions).

6. **DO NO HARM asymmetry.** False negatives on severe/contraindicated interactions are catastrophically worse than false positives. The loss function, the output thresholds, and the template language all reflect this.

---

## Current Phase

**Read `phases/phase1.md` to begin.**
