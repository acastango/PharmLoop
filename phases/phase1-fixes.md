# Phase 1 Fixes — Architectural Corrections

## Context
Phase 1 implementation is structurally sound but has 5 deviations from spec
that will affect whether the dynamics actually separate the three test cases.
Fix these in order before running training.

---

## Fix 1: Flag Vocabulary Alignment
**Priority: HIGH — silent data loss**
**File: `pharmloop/output.py`**

The interaction data contains 8 flag types not in `FLAG_NAMES`. This silently
drops ~102 flag occurrences during data loading, including `avoid_combination`
(the most frequent flag in the dataset at 40 occurrences).

### What to do
Expand `FLAG_NAMES` to include the missing flags:

```python
FLAG_NAMES = [
    "monitor_serotonin_syndrome",
    "monitor_inr",
    "monitor_qt_interval",
    "monitor_renal_function",
    "monitor_hepatic_function",
    "monitor_blood_pressure",
    "monitor_blood_glucose",
    "monitor_electrolytes",
    "monitor_drug_levels",
    "monitor_cns_depression",
    # --- previously missing ---
    "avoid_combination",
    "monitor_bleeding",
    "monitor_digoxin_levels",
    "monitor_lithium_levels",
    "monitor_cyclosporine_levels",
    "monitor_theophylline_levels",
    "reduce_statin_dose",
    "separate_administration",
]
NUM_FLAGS = len(FLAG_NAMES)  # now 18
```

Update `OutputHead.__init__` — it already uses `num_flags` param, so this
should propagate automatically. Verify `flags_head` output dim matches.

### Verification
Run this check after fixing:
```python
import json
from pharmloop.output import FLAG_NAMES

with open("data/processed/interactions.json") as f:
    data = json.load(f)

all_flags = set()
for inter in data["interactions"]:
    for flag in inter.get("flags", []):
        all_flags.add(flag)

missing = all_flags - set(FLAG_NAMES)
assert len(missing) == 0, f"Still missing flags: {missing}"
print(f"All {len(all_flags)} data flags covered by {len(FLAG_NAMES)} vocab entries")
```

---

## Fix 2: Per-Dimension Oscillator Parameters
**Priority: HIGH — affects partial convergence behavior**
**File: `pharmloop/oscillator.py`, class `OscillatorCell`**

The spec calls for per-dimension (512-dim) decay, spring, and dt so that
different aspects of the belief state can settle at different rates. The
current implementation uses scalar parameters, meaning all 512 dimensions
oscillate identically — this prevents partial convergence.

### What to change

In `OscillatorCell.__init__`:

```python
# BEFORE (scalar):
self.raw_decay = nn.Parameter(torch.tensor(0.9))
self.raw_dt = nn.Parameter(torch.tensor(0.1))
self.raw_spring = nn.Parameter(torch.tensor(0.5))

# AFTER (per-dimension):
self.raw_decay = nn.Parameter(torch.ones(state_dim) * 0.9)
self.raw_dt = nn.Parameter(torch.ones(state_dim) * 0.1)
self.raw_spring = nn.Parameter(torch.ones(state_dim) * 0.5)
```

The properties that clamp these values need to work per-dimension too —
`torch.clamp` already supports tensor inputs, so `self.raw_decay.clamp(0.5, 0.99)`
will work without changes. Verify the shapes broadcast correctly in the forward pass.

The `threshold` can stay scalar — it operates on the norm of v, not per-dimension.

### Verification
After fixing, run:
```python
cell = OscillatorCell(state_dim=512)
assert cell.decay.shape == (512,), f"decay shape: {cell.decay.shape}"
assert cell.dt.shape == (512,), f"dt shape: {cell.dt.shape}"
assert cell.spring.shape == (512,), f"spring shape: {cell.spring.shape}"

# Forward pass still works
x = torch.randn(2, 512)
v = torch.randn(2, 512)
x_new, v_new, gz = cell(x, v, training=False)
assert x_new.shape == (2, 512)
```

---

## Fix 3: Per-Dimension Noise Gating
**Priority: MEDIUM — follows naturally from Fix 2**
**File: `pharmloop/oscillator.py`, class `OscillatorCell`**

Currently the noise gate takes the scalar L2 norm of velocity and outputs
uniform noise scaling. The spec has noise gated by per-dimension |v| so that
dimensions with high uncertainty get more exploration while settled dimensions
stay stable.

### What to change

In `__init__`, change the noise gate architecture:

```python
# BEFORE (scalar in, scalar out):
self.noise_gate = nn.Sequential(
    nn.Linear(1, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
    nn.Sigmoid(),
)

# AFTER (per-dimension in, per-dimension out):
self.noise_gate = nn.Sequential(
    nn.Linear(state_dim, 128),
    nn.GELU(),
    nn.Linear(128, state_dim),
    nn.Sigmoid(),
)
```

In `forward`, change the noise computation:

```python
# BEFORE:
gray_zone_per_sample = v.norm(dim=-1, keepdim=True)  # (batch, 1)
# ...
noise_scale = self.noise_gate(gray_zone_per_sample)   # (batch, 1)
noise = torch.randn_like(v) * noise_scale * NOISE_SCALE

# AFTER:
gz_per_dim = torch.abs(v)                             # (batch, state_dim)
gz_scalar = v.norm(dim=-1, keepdim=True)               # (batch, 1) — keep for beta_mod
# ...
noise_scale = self.noise_gate(gz_per_dim)              # (batch, state_dim)
noise = torch.randn_like(v) * noise_scale * NOISE_SCALE
```

Note: the scalar gray zone (`gz_scalar`) is still needed for:
- The beta_mod (Hopfield retrieval sharpness) — this should stay scalar/per-sample
- The returned gray_zone_scalar for trajectory tracking
- The convergence threshold check

So you'll compute BOTH `gz_per_dim` (for noise) and `gz_scalar` (for everything else).

### Verification
```python
cell = OscillatorCell(state_dim=64)
x = torch.randn(2, 64)
v_high = torch.ones(2, 64) * 5.0   # high velocity everywhere
v_mixed = torch.cat([torch.ones(2, 32) * 5.0, torch.zeros(2, 32)], dim=-1)  # half settled

# With per-dim gating, mixed should get less total noise than high
# (because 32 dims are near-zero velocity → near-zero noise)
torch.manual_seed(42)
_, v_out_high, _ = cell(x.clone(), v_high.clone(), training=True)
torch.manual_seed(42)
_, v_out_mixed, _ = cell(x.clone(), v_mixed.clone(), training=True)
# Not a strict test (force differs too), but check shapes work
assert v_out_high.shape == (2, 64)
assert v_out_mixed.shape == (2, 64)
```

---

## Fix 4: Hopfield Phase 0 Bootstrap
**Priority: MEDIUM — affects early training stability**
**Files: `pharmloop/hopfield.py`, `training/train.py`**

In Phase 0, the Hopfield bank should store raw 64-dim feature vectors and
retrieve in raw feature space — so that pharmacologically similar drugs
retrieve each other's patterns based on actual structural similarity.

Currently, `store()` projects patterns through the randomly-initialized
`key_proj` before storing. This means the Phase 0 Hopfield retrieves against
random projections of features, not the features themselves. The encoder will
learn to compensate, but early training will be noisier than necessary.

### What to change

**Option A (recommended — simplest):** Add a `phase0` mode to PharmHopfield.

In `PharmHopfield.__init__`, add a `phase0` flag:

```python
def __init__(self, input_dim: int, hidden_dim: int = 512,
             max_capacity: int = MAX_CAPACITY, phase0: bool = False) -> None:
    super().__init__()
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim if not phase0 else input_dim
    self.max_capacity = max_capacity
    self.phase0 = phase0

    if phase0:
        # Phase 0: no learned projections, operate in raw feature space
        self.query_proj = nn.Identity()
        self.key_proj = nn.Identity()
        # Stored keys are in input_dim space
        self.register_buffer("stored_keys", torch.zeros(max_capacity, input_dim))
    else:
        # Phase 2: learned projections in hidden_dim space
        self.query_proj = nn.Linear(input_dim, self.hidden_dim)
        self.key_proj = nn.Linear(input_dim, self.hidden_dim)
        self.register_buffer("stored_keys", torch.zeros(max_capacity, self.hidden_dim))

    self.register_buffer("stored_values", torch.zeros(max_capacity, input_dim))
    self.register_buffer("num_stored", torch.tensor(0, dtype=torch.long))
```

In `train.py`, build the Phase 0 Hopfield with the flag:

```python
hopfield = PharmHopfield(input_dim=64, hidden_dim=512, phase0=True)
```

The `retrieve` method should work without changes since it uses `self.query_proj`
and `self.key_proj` which are now `Identity` in Phase 0.

### Important
The `OscillatorCell` already handles the dimension mismatch between Hopfield
(64-dim in Phase 0) and state (512-dim) via `hopfield_query_proj` and
`hopfield_value_proj`. This should continue to work — just verify that the
Hopfield's `hidden_dim` is now `input_dim` (64) in Phase 0 so the dimension
check `hopfield.input_dim != state_dim` still triggers the projection layers.

### Verification
```python
hopfield = PharmHopfield(input_dim=64, phase0=True)
patterns = torch.randn(10, 64)
hopfield.store(patterns)

# Retrieve: a query close to pattern[0] should retrieve something close to pattern[0]
query = patterns[0:1] + torch.randn(1, 64) * 0.01  # slight noise
result = hopfield.retrieve(query, beta=10.0)
cosine_sim = torch.cosine_similarity(result, patterns[0:1])
assert cosine_sim > 0.8, f"Phase 0 retrieval failed: cosine sim = {cosine_sim.item():.3f}"
```

---

## Fix 5: Per-Sample Confidence and Gray Zone Tracking
**Priority: LOW-MEDIUM — won't affect 3-way test (batch=1) but needed for training**
**Files: `pharmloop/oscillator.py`, `pharmloop/output.py`**

Currently, gray zones in the trajectory are batch-averaged scalars:
```python
gray_zones: list[float] = [v.norm(dim=-1).mean().item()]
```

And `compute_confidence` uses these scalars, producing identical confidence
for all samples in a batch (except the converged/not-converged clamp).

### What to change

In `ReasoningLoop.forward`, track per-sample gray zones:

```python
# BEFORE:
gray_zones: list[float] = [v.norm(dim=-1).mean().item()]
# ...
gray_zones.append(gz)  # gz is a float

# AFTER:
gray_zones: list[Tensor] = [v.norm(dim=-1)]  # (batch,) tensor
# ...
# In OscillatorCell.forward, return gz as tensor not scalar:
gz_per_sample = v.norm(dim=-1)  # (batch,)
# ...
gray_zones.append(gz_per_sample)  # (batch,) tensor
```

The scalar gray zone for logging can be computed at the call site:
`gz_mean = gz_per_sample.mean().item()`

Then rewrite `compute_confidence` to work with per-sample tensors:

```python
def compute_confidence(
    gray_zones: list[Tensor],   # list of (batch,) tensors
    converged: Tensor,           # (batch,)
    max_steps: int = 16,
) -> Tensor:
    """Per-sample confidence from convergence dynamics."""
    batch = converged.shape[0]
    device = converged.device

    final_gz = gray_zones[-1]  # (batch,)
    gz_confidence = (1.0 - final_gz * 5.0).clamp(min=0.0)

    steps_taken = len(gray_zones) - 1
    speed_confidence = max(0.0, 1.0 - steps_taken / max_steps)
    speed_confidence = torch.full((batch,), speed_confidence, device=device)

    # Per-sample smoothness
    if len(gray_zones) >= 3:
        gz_stack = torch.stack(gray_zones, dim=0)          # (steps, batch)
        first_d = gz_stack[1:] - gz_stack[:-1]             # (steps-1, batch)
        second_d = first_d[1:] - first_d[:-1]              # (steps-2, batch)
        roughness = second_d.abs().mean(dim=0)              # (batch,)
        smoothness_confidence = (1.0 - roughness * 10.0).clamp(min=0.0)
    else:
        smoothness_confidence = torch.full((batch,), 0.5, device=device)

    raw = 0.4 * gz_confidence + 0.3 * speed_confidence + 0.3 * smoothness_confidence
    raw[~converged] = raw[~converged].clamp(max=0.1)
    return raw
```

### Impact on other files
- `OscillatorCell.forward` return type changes: `gray_zone_scalar` → `gz_per_sample` tensor
- `ReasoningLoop.forward` return type: `gray_zones` becomes `list[Tensor]` not `list[float]`
- `training/train.py` logging: use `.mean().item()` on gray zone tensors for epoch logging
- `tests/test_separation.py`: gray zone printouts need `.item()` or `.mean().item()`
- `training/loss.py`: the smoothness loss already uses velocity tensors directly,
  so no changes needed there

### Verification
```python
cell = OscillatorCell(state_dim=32)
loop = ReasoningLoop(cell, max_steps=4)
initial = torch.randn(4, 32)
result = loop(initial, training=False)

# Gray zones should be tensors now
assert isinstance(result["gray_zones"][0], torch.Tensor)
assert result["gray_zones"][0].shape == (4,)  # per-sample

from pharmloop.output import compute_confidence
conf = compute_confidence(result["gray_zones"], result["converged"])
assert conf.shape == (4,)  # per-sample confidence
```

---

## After All Fixes: Run the Validation

1. **Unit tests pass:**
   ```bash
   pytest tests/test_hopfield.py tests/test_oscillator.py -v
   ```

2. **Training runs without NaN for 50 epochs:**
   ```bash
   python -m training.train
   ```

3. **Three-way separation test:**
   ```bash
   pytest tests/test_separation.py -v -s
   ```

4. **Parameter budget check:** learned < 3M, total < 6M

## Order of Implementation

Do them in this order because of dependencies:

1. Fix 1 (flag vocab) — standalone, no dependencies
2. Fix 2 (per-dim params) — standalone
3. Fix 3 (per-dim noise) — depends on Fix 2
4. Fix 4 (Hopfield phase0) — standalone
5. Fix 5 (per-sample tracking) — touches multiple files, do last

Fixes 1-4 can be done independently. Fix 5 is the most invasive
(touches oscillator, output, train, tests) so save it for last.

Run unit tests after each fix to catch regressions.
