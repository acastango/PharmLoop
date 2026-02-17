#!/usr/bin/env python3
"""
PharmLoop Experiments: Is PharmLoop a Resolver or a Classifier?

Three experiments to determine the nature of the oscillatory reasoning:
  1. Depth Ablation — does accuracy improve with more recurrence steps?
  2. Velocity Distribution — do fabricated drugs stay at high velocity?
  3. Output Class Trajectory — do predicted classes flip during oscillation?

Run from project root:
    python experiments/run_all_experiments.py
"""

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pharmloop.inference import PharmLoopInference
from pharmloop.output import (
    SEVERITY_NAMES, MECHANISM_NAMES, FLAG_NAMES,
    NUM_SEVERITY_CLASSES, NUM_MECHANISMS, NUM_FLAGS,
)
from training.data_loader import DrugInteractionDataset, create_dataloader

# ─── Configuration ────────────────────────────────────────────────────────────

CHECKPOINT = "checkpoints/best_model_phase4b.pt"
DATA_DIR = "data/processed"
DRUGS_PATH = "data/processed/drugs_v3.json"
INTERACTIONS_PATH = "data/processed/interactions_v3.json"
SPLIT_PATH = "data/processed/split_v3.json"
OUTPUT_DIR = Path("experiments")
BATCH_SIZE = 64
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def load_model_and_data():
    """Load the Phase 4b model and test data."""
    print("=" * 70)
    print("LOADING MODEL AND DATA")
    print("=" * 70)

    # Load model manually to match Phase 4b checkpoint capacity (2000/15000)
    from pharmloop.hierarchical_hopfield import HierarchicalHopfield, DRUG_CLASSES
    from pharmloop.model import PharmLoopModel

    with open(DRUGS_PATH) as f:
        drugs_raw = json.load(f)

    num_drugs = drugs_raw.get("num_drugs", len(drugs_raw["drugs"]))
    drug_class_map: dict[int, str] = {}
    for name, info in drugs_raw["drugs"].items():
        dc = info.get("class", "other")
        drug_class_map[info["id"]] = dc if dc in DRUG_CLASSES else "other"

    hopfield = HierarchicalHopfield(
        input_dim=512, class_names=DRUG_CLASSES,
        class_capacity=2000, global_capacity=15000,
    )

    ck = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    has_context = any(k.startswith("context_encoder.") for k in ck["model_state_dict"])

    model = PharmLoopModel(
        num_drugs=num_drugs, hopfield=hopfield,
        drug_class_map=drug_class_map, use_context=has_context,
    )
    model.load_state_dict(ck["model_state_dict"], strict=False)
    model.eval()

    with open(SPLIT_PATH) as f:
        split = json.load(f)

    # Clean test loader (no fabricated pairs)
    test_dataset = DrugInteractionDataset(
        DRUGS_PATH, INTERACTIONS_PATH,
        fabricated_ratio=0.0,
        split_indices=split["test_indices"],
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
    )

    # Load drug registry for experiment 2
    with open(DRUGS_PATH) as f:
        drugs_data = json.load(f)

    drug_registry = {}
    for name, info in drugs_data["drugs"].items():
        drug_registry[name] = {
            "id": info["id"],
            "features": info["features"],
            "class": info.get("class", "other"),
        }

    # Load interactions for novel pair detection
    with open(INTERACTIONS_PATH) as f:
        interactions_data = json.load(f)

    known_pairs = set()
    for inter in interactions_data["interactions"]:
        pair = tuple(sorted([inter["drug_a"], inter["drug_b"]]))
        known_pairs.add(pair)

    print(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"  Test set: {len(test_dataset)} samples")
    print(f"  Drug registry: {len(drug_registry)} drugs")
    print(f"  Known interaction pairs: {len(known_pairs)}")

    threshold = model.reasoning_loop.cell.threshold.item()
    print(f"  Convergence threshold: {threshold:.6f}")
    print()

    return model, test_loader, drug_registry, known_pairs


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: DEPTH ABLATION
# ═══════════════════════════════════════════════════════════════════════════════

def run_experiment_1(model, test_loader):
    """Test accuracy at different oscillation depths."""
    print("=" * 70)
    print("EXPERIMENT 1: DEPTH ABLATION")
    print("=" * 70)
    print("Question: Does accuracy improve with more recurrence steps,")
    print("          or does it saturate early?")
    print()

    DEPTH_VALUES = [2, 4, 6, 8, 12, 16, 24, 32, 48, 64]
    original_max = model.reasoning_loop.max_steps
    results = []

    for max_steps in DEPTH_VALUES:
        model.reasoning_loop.max_steps = max_steps
        # Force all steps by using train mode for step count but no_grad
        # Actually, we want eval mode behavior (no noise) but without early stopping
        # We'll run in eval mode — early stopping only triggers if mean gz < threshold
        # At low depths it won't have time to converge, at high depths it may stop early

        severity_correct = 0
        mechanism_correct = 0
        false_negatives = 0  # severe/contra predicted as none
        severe_total = 0
        converged_count = 0
        total = 0
        total_steps_taken = 0

        for batch in test_loader:
            with torch.no_grad():
                output = model(
                    batch["drug_a_id"],
                    batch["drug_a_features"],
                    batch["drug_b_id"],
                    batch["drug_b_features"],
                )

            pred_severity = output["severity_logits"].argmax(dim=-1)
            true_severity = batch["target_severity"]
            batch_size = pred_severity.shape[0]

            # Severity accuracy
            severity_correct += (pred_severity == true_severity).sum().item()

            # Mechanism accuracy (multi-label: threshold at 0.5)
            pred_mechs = (torch.sigmoid(output["mechanism_logits"]) > 0.5).float()
            true_mechs = batch["target_mechanisms"]
            # Per-sample: all mechanism predictions match
            mech_match = (pred_mechs == true_mechs).all(dim=-1).sum().item()
            mechanism_correct += mech_match

            # False negative rate on severe/contraindicated
            for i in range(batch_size):
                true_sev = true_severity[i].item()
                pred_sev = pred_severity[i].item()
                if true_sev in (3, 4):  # severe or contraindicated
                    severe_total += 1
                    if pred_sev == 0:  # predicted none
                        false_negatives += 1

            converged_count += output["converged"].sum().item()
            total_steps_taken += output["trajectory"]["steps"]
            total += batch_size

        sev_acc = severity_correct / total
        mech_acc = mechanism_correct / total
        fnr = false_negatives / max(severe_total, 1)
        conv_rate = converged_count / total

        result = {
            "max_steps": max_steps,
            "severity_accuracy": round(sev_acc, 4),
            "mechanism_accuracy": round(mech_acc, 4),
            "fnr": round(fnr, 4),
            "convergence_rate": round(conv_rate, 4),
            "avg_steps_taken": round(total_steps_taken / len(test_loader), 1),
            "total_samples": total,
            "severe_total": severe_total,
        }
        results.append(result)
        print(f"  depth={max_steps:>3}: sev_acc={sev_acc:6.1%}  mech_acc={mech_acc:6.1%}  "
              f"FNR={fnr:5.1%}  conv={conv_rate:5.1%}  steps_taken={result['avg_steps_taken']}")

    # Restore original
    model.reasoning_loop.max_steps = original_max

    # Analysis
    print()
    print("--- Analysis ---")
    max_sev = max(r["severity_accuracy"] for r in results)
    max_mech = max(r["mechanism_accuracy"] for r in results)

    sev_saturate = None
    mech_saturate = None
    for r in results:
        if sev_saturate is None and r["severity_accuracy"] >= max_sev - 0.01:
            sev_saturate = r["max_steps"]
        if mech_saturate is None and r["mechanism_accuracy"] >= max_mech - 0.01:
            mech_saturate = r["max_steps"]

    print(f"  Peak severity accuracy: {max_sev:.1%} (reached within 1% at depth {sev_saturate})")
    print(f"  Peak mechanism accuracy: {max_mech:.1%} (reached within 1% at depth {mech_saturate})")

    # Does accuracy improve beyond default 16?
    acc_at_16 = next(r for r in results if r["max_steps"] == 16)["severity_accuracy"]
    acc_at_32 = next(r for r in results if r["max_steps"] == 32)["severity_accuracy"]
    acc_at_64 = next(r for r in results if r["max_steps"] == 64)["severity_accuracy"]
    beyond_16 = acc_at_32 > acc_at_16 + 0.005 or acc_at_64 > acc_at_16 + 0.005
    print(f"  Accuracy at 16: {acc_at_16:.1%}, at 32: {acc_at_32:.1%}, at 64: {acc_at_64:.1%}")
    if beyond_16:
        print("  → IMPROVEMENT beyond depth 16: recurrence is genuinely computational")
    else:
        print("  → NO improvement beyond depth 16: recurrence is iterative refinement")
    print()

    # Save
    with open(OUTPUT_DIR / "depth_ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: VELOCITY DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════════

def run_experiment_2(model, test_loader, drug_registry, known_pairs):
    """Analyze velocity distributions for fabricated vs known pairs."""
    print("=" * 70)
    print("EXPERIMENT 2: VELOCITY DISTRIBUTION")
    print("=" * 70)
    print("Question: When inputs don't converge, what does velocity look like?")
    print()

    model.reasoning_loop.max_steps = 16
    threshold = model.reasoning_loop.cell.threshold.item()
    drug_names = list(drug_registry.keys())
    max_known_id = max(d["id"] for d in drug_registry.values())

    # --- Group A: 100 fabricated drug pairs (random IDs NOT in registry) ---
    print("  Generating fabricated drug pairs (unknown IDs)...")
    fabricated_pairs = []
    for i in range(100):
        fake_a_id = max_known_id + 1000 + (i * 2)
        fake_b_id = max_known_id + 1000 + (i * 2) + 1
        # Random features (uniform 0-0.5 to be clearly unphysiological)
        fake_a_feat = [random.uniform(0, 0.5) for _ in range(64)]
        fake_b_feat = [random.uniform(0, 0.5) for _ in range(64)]
        fabricated_pairs.append({
            "a_id": fake_a_id, "b_id": fake_b_id,
            "a_feat": fake_a_feat, "b_feat": fake_b_feat,
            "label": f"fabricated_{i}_a + fabricated_{i}_b",
        })

    # --- Group B: 50 known interacting pairs from test set ---
    print("  Selecting known interacting pairs from test set...")
    known_interacting = []
    for batch in test_loader:
        for i in range(batch["drug_a_id"].shape[0]):
            sev = batch["target_severity"][i].item()
            if sev != 0 and len(known_interacting) < 50:  # not "none"
                known_interacting.append({
                    "a_id": batch["drug_a_id"][i].item(),
                    "b_id": batch["drug_b_id"][i].item(),
                    "a_feat": batch["drug_a_features"][i].tolist(),
                    "b_feat": batch["drug_b_features"][i].tolist(),
                    "severity": SEVERITY_NAMES[sev],
                })
        if len(known_interacting) >= 50:
            break

    # --- Group C: 50 known safe pairs from test set ---
    print("  Selecting known safe pairs from test set...")
    known_safe = []
    for batch in test_loader:
        for i in range(batch["drug_a_id"].shape[0]):
            sev = batch["target_severity"][i].item()
            if sev == 0 and len(known_safe) < 50:  # "none"
                known_safe.append({
                    "a_id": batch["drug_a_id"][i].item(),
                    "b_id": batch["drug_b_id"][i].item(),
                    "a_feat": batch["drug_a_features"][i].tolist(),
                    "b_feat": batch["drug_b_features"][i].tolist(),
                })
        if len(known_safe) >= 50:
            break

    # --- Group D: ~50 novel pairs (known drugs, unknown combination) ---
    print("  Finding novel pairs (known drugs, unknown combination)...")
    novel_pairs = []
    drug_list = list(drug_registry.items())
    attempts = 0
    while len(novel_pairs) < 50 and attempts < 5000:
        da_name, da_info = random.choice(drug_list)
        db_name, db_info = random.choice(drug_list)
        if da_name == db_name:
            attempts += 1
            continue
        pair_key = tuple(sorted([da_name, db_name]))
        if pair_key not in known_pairs:
            novel_pairs.append({
                "a_id": da_info["id"],
                "b_id": db_info["id"],
                "a_feat": da_info["features"],
                "b_feat": db_info["features"],
                "label": f"{da_name} + {db_name}",
            })
            known_pairs.add(pair_key)  # avoid duplicates in our sample
        attempts += 1

    print(f"  Found {len(novel_pairs)} novel pairs")

    def run_group(pairs_list, group_name):
        """Run model on a group of pairs and collect velocity trajectories."""
        results = []
        for idx, pair in enumerate(pairs_list):
            a_id = torch.tensor([pair["a_id"]], dtype=torch.long)
            a_feat = torch.tensor([pair["a_feat"]], dtype=torch.float32)
            b_id = torch.tensor([pair["b_id"]], dtype=torch.long)
            b_feat = torch.tensor([pair["b_feat"]], dtype=torch.float32)

            with torch.no_grad():
                output = model(a_id, a_feat, b_id, b_feat)

            traj = output["trajectory"]
            vel_mags = [gz[0].item() for gz in traj["gray_zones"]]
            final_v = traj["velocities"][-1][0]  # (512,) tensor

            results.append({
                "velocity_magnitudes": vel_mags,
                "final_velocity_mag": vel_mags[-1],
                "final_velocity_per_dim_stats": {
                    "mean": final_v.abs().mean().item(),
                    "std": final_v.abs().std().item(),
                    "max": final_v.abs().max().item(),
                    "min": final_v.abs().min().item(),
                },
                "converged": output["converged"][0].item(),
                "confidence": output["confidence"][0].item(),
                "predicted_severity": SEVERITY_NAMES[output["severity_logits"].argmax(-1).item()],
                "steps": traj["steps"],
            })
        return results

    print("\n  Running fabricated pairs...")
    fabricated_results = run_group(fabricated_pairs, "fabricated")
    print("  Running known interacting pairs...")
    known_interact_results = run_group(known_interacting, "known_interacting")
    print("  Running known safe pairs...")
    known_safe_results = run_group(known_safe, "known_safe")
    print("  Running novel pairs...")
    novel_results = run_group(novel_pairs, "novel")

    # --- Analysis ---
    print()
    print("--- Results ---")
    print(f"  Convergence threshold: {threshold:.6f}")
    print()

    def summarize_group(name, results_list):
        final_vs = [r["final_velocity_mag"] for r in results_list]
        conv_count = sum(1 for r in results_list if r["converged"])
        confidences = [r["confidence"] for r in results_list]
        sev_dist = {}
        for r in results_list:
            s = r["predicted_severity"]
            sev_dist[s] = sev_dist.get(s, 0) + 1

        print(f"  {name} ({len(results_list)} samples):")
        print(f"    Final |v|:    mean={np.mean(final_vs):.6f}  std={np.std(final_vs):.6f}  "
              f"min={np.min(final_vs):.6f}  max={np.max(final_vs):.6f}")
        print(f"    Converged:    {conv_count}/{len(results_list)} ({conv_count/len(results_list):.0%})")
        print(f"    Confidence:   mean={np.mean(confidences):.3f}  std={np.std(confidences):.3f}")
        print(f"    Severity:     {sev_dist}")
        return {
            "name": name,
            "count": len(results_list),
            "final_v_mean": float(np.mean(final_vs)),
            "final_v_std": float(np.std(final_vs)),
            "final_v_min": float(np.min(final_vs)),
            "final_v_max": float(np.max(final_vs)),
            "converged_count": conv_count,
            "converged_rate": conv_count / len(results_list),
            "confidence_mean": float(np.mean(confidences)),
            "confidence_std": float(np.std(confidences)),
            "severity_distribution": sev_dist,
        }

    fab_summary = summarize_group("FABRICATED", fabricated_results)
    print()
    interact_summary = summarize_group("KNOWN INTERACTING", known_interact_results)
    print()
    safe_summary = summarize_group("KNOWN SAFE", known_safe_results)
    print()
    novel_summary = summarize_group("NOVEL PAIRS", novel_results)

    # Separation analysis
    print()
    fab_vs = [r["final_velocity_mag"] for r in fabricated_results]
    known_vs = [r["final_velocity_mag"] for r in known_interact_results + known_safe_results]
    novel_vs = [r["final_velocity_mag"] for r in novel_results]

    fab_below = sum(1 for v in fab_vs if v < threshold)
    known_above = sum(1 for v in known_vs if v >= threshold)
    novel_below = sum(1 for v in novel_vs if v < threshold)

    print("--- Separation Analysis ---")
    print(f"  Fabricated below threshold (converging — BAD): {fab_below}/{len(fab_vs)}")
    print(f"  Known above threshold (not converging — BAD):  {known_above}/{len(known_vs)}")
    print(f"  Novel below threshold (converging):            {novel_below}/{len(novel_vs)}")

    if fab_below <= 10:
        print("  → Fabricated drugs mostly do NOT converge — dynamics separate known from unknown")
    else:
        print(f"  → WARNING: {fab_below}% of fabricated drugs converge — threshold doing safety work")

    # Save
    all_results = {
        "threshold": threshold,
        "fabricated": fabricated_results,
        "known_interacting": known_interact_results,
        "known_safe": known_safe_results,
        "novel": novel_results,
        "summaries": {
            "fabricated": fab_summary,
            "known_interacting": interact_summary,
            "known_safe": safe_summary,
            "novel": novel_summary,
        },
    }
    with open(OUTPUT_DIR / "velocity_distribution_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: OUTPUT CLASS TRAJECTORY
# ═══════════════════════════════════════════════════════════════════════════════

def run_experiment_3(model, drug_registry, known_pairs):
    """Track how predicted classes change across oscillation steps."""
    print("=" * 70)
    print("EXPERIMENT 3: OUTPUT CLASS TRAJECTORY")
    print("=" * 70)
    print("Question: For non-converging inputs, does the predicted class flip,")
    print("          or does it stabilize early while internal state churns?")
    print()

    model.reasoning_loop.max_steps = 16
    threshold = model.reasoning_loop.cell.threshold.item()
    drug_names = list(drug_registry.keys())
    max_known_id = max(d["id"] for d in drug_registry.values())

    # Per-dimension convergence threshold
    dim_threshold = threshold / np.sqrt(512)

    def run_with_class_trajectory(a_id, a_feat, b_id, b_feat, max_steps=16):
        """Run model, decode output class at every step, track per-dim convergence."""
        original_max = model.reasoning_loop.max_steps
        model.reasoning_loop.max_steps = max_steps

        a_id_t = torch.tensor([a_id], dtype=torch.long)
        a_feat_t = torch.tensor([a_feat], dtype=torch.float32)
        b_id_t = torch.tensor([b_id], dtype=torch.long)
        b_feat_t = torch.tensor([b_feat], dtype=torch.float32)

        with torch.no_grad():
            output = model(a_id_t, a_feat_t, b_id_t, b_feat_t)

        traj = output["trajectory"]
        positions = traj["positions"]
        velocities = traj["velocities"]
        steps_taken = traj["steps"]

        step_predictions = []
        per_dim_convergence = []

        for step_idx in range(len(positions)):
            x_t = positions[step_idx]   # (1, 512)
            v_t = velocities[step_idx]  # (1, 512)

            # Run output head on this intermediate state
            with torch.no_grad():
                sev_logits = model.output_head.severity_head(x_t)  # (1, 6)
                # Mechanism head needs trajectory — use positions up to this step
                positions_so_far = positions[:step_idx + 1]
                mech_logits = model.output_head.mechanism_head(positions_so_far)  # (1, 15)

            sev_probs = torch.softmax(sev_logits, -1).squeeze().tolist()
            mech_probs = torch.sigmoid(mech_logits).squeeze().tolist()

            step_predictions.append({
                "step": step_idx,
                "severity_class": int(sev_logits.argmax(-1).item()),
                "severity_name": SEVERITY_NAMES[int(sev_logits.argmax(-1).item())],
                "severity_probs": [round(p, 4) for p in sev_probs],
                "mechanism_active": [
                    MECHANISM_NAMES[i] for i, p in enumerate(mech_probs) if p > 0.5
                ],
                "mechanism_probs": [round(p, 4) for p in mech_probs],
            })

            # Per-dimension convergence
            dim_converged = (v_t[0].abs() < dim_threshold).cpu().numpy()
            per_dim_convergence.append(dim_converged)

        # Count class flips
        severity_classes = [p["severity_class"] for p in step_predictions]
        severity_flips = sum(
            1 for i in range(1, len(severity_classes))
            if severity_classes[i] != severity_classes[i - 1]
        )

        # Mechanism flips: count changes in the set of active mechanisms
        mech_sets = [frozenset(p["mechanism_active"]) for p in step_predictions]
        mechanism_flips = sum(
            1 for i in range(1, len(mech_sets))
            if mech_sets[i] != mech_sets[i - 1]
        )

        # Dimensional convergence analysis
        final_dim_conv = per_dim_convergence[-1] if per_dim_convergence else np.zeros(512, dtype=bool)
        n_dims_converged = int(final_dim_conv.sum())
        n_dims_oscillating = 512 - n_dims_converged
        dims_oscillating = np.where(~final_dim_conv)[0].tolist()

        # Track convergence progression
        dims_converged_per_step = [int(dc.sum()) for dc in per_dim_convergence]

        model.reasoning_loop.max_steps = original_max

        return {
            "step_predictions": step_predictions,
            "severity_flips": severity_flips,
            "mechanism_flips": mechanism_flips,
            "final_severity": SEVERITY_NAMES[severity_classes[-1]],
            "final_mechanisms": list(mech_sets[-1]) if mech_sets else [],
            "n_dims_converged": n_dims_converged,
            "n_dims_oscillating": n_dims_oscillating,
            "dims_converged_per_step": dims_converged_per_step,
            "converged": output["converged"][0].item(),
            "confidence": output["confidence"][0].item(),
            "steps_taken": steps_taken,
        }

    # --- Generate groups ---
    # Fabricated pairs (50 — enough for statistics)
    print("  Running class trajectories for fabricated pairs (50)...")
    fabricated_trajectories = []
    for i in range(50):
        fake_a_id = max_known_id + 2000 + (i * 2)
        fake_b_id = max_known_id + 2000 + (i * 2) + 1
        fake_a_feat = [random.uniform(0, 0.5) for _ in range(64)]
        fake_b_feat = [random.uniform(0, 0.5) for _ in range(64)]
        result = run_with_class_trajectory(fake_a_id, fake_a_feat, fake_b_id, fake_b_feat)
        fabricated_trajectories.append(result)

    # Novel pairs (50 — known drugs, unknown combination)
    print("  Running class trajectories for novel pairs (50)...")
    novel_trajectories = []
    drug_list = list(drug_registry.items())
    used_novel = set()
    attempts = 0
    while len(novel_trajectories) < 50 and attempts < 5000:
        da_name, da_info = random.choice(drug_list)
        db_name, db_info = random.choice(drug_list)
        if da_name == db_name:
            attempts += 1
            continue
        pair_key = tuple(sorted([da_name, db_name]))
        if pair_key not in known_pairs and pair_key not in used_novel:
            used_novel.add(pair_key)
            result = run_with_class_trajectory(
                da_info["id"], da_info["features"],
                db_info["id"], db_info["features"],
            )
            result["pair_label"] = f"{da_name} + {db_name}"
            novel_trajectories.append(result)
        attempts += 1

    # Known interacting pairs (50 — for comparison baseline)
    print("  Running class trajectories for known interacting pairs (50)...")
    known_trajectories = []

    with open(INTERACTIONS_PATH) as f:
        interactions = json.load(f)["interactions"]

    with open(SPLIT_PATH) as f:
        split = json.load(f)

    test_indices = split["test_indices"]
    for idx in test_indices:
        if len(known_trajectories) >= 50:
            break
        inter = interactions[idx]
        if inter["severity"] == "none":
            continue
        da_name = inter["drug_a"]
        db_name = inter["drug_b"]
        if da_name not in drug_registry or db_name not in drug_registry:
            continue

        da = drug_registry[da_name]
        db = drug_registry[db_name]
        result = run_with_class_trajectory(
            da["id"], da["features"], db["id"], db["features"],
        )
        result["pair_label"] = f"{da_name} + {db_name}"
        result["true_severity"] = inter["severity"]
        known_trajectories.append(result)

    # --- Analysis ---
    print()
    print("--- Results ---")

    def analyze_group(name, trajectories):
        sev_flips = [t["severity_flips"] for t in trajectories]
        mech_flips = [t["mechanism_flips"] for t in trajectories]
        dims_osc = [t["n_dims_oscillating"] for t in trajectories]
        dims_conv = [t["n_dims_converged"] for t in trajectories]

        print(f"  {name} ({len(trajectories)} samples):")
        print(f"    Severity flips:    mean={np.mean(sev_flips):.1f}  max={max(sev_flips)}  "
              f"zero_flips={sev_flips.count(0)}/{len(sev_flips)}")
        print(f"    Mechanism flips:   mean={np.mean(mech_flips):.1f}  max={max(mech_flips)}  "
              f"zero_flips={mech_flips.count(0)}/{len(mech_flips)}")
        print(f"    Dims oscillating:  mean={np.mean(dims_osc):.0f}/512  "
              f"min={min(dims_osc)}  max={max(dims_osc)}")
        print(f"    Dims converged:    mean={np.mean(dims_conv):.0f}/512")

        # What do they converge to?
        sev_dist = {}
        for t in trajectories:
            s = t["final_severity"]
            sev_dist[s] = sev_dist.get(s, 0) + 1
        print(f"    Final severity:    {sev_dist}")
        conv_count = sum(1 for t in trajectories if t["converged"])
        print(f"    Converged:         {conv_count}/{len(trajectories)}")

        return {
            "name": name,
            "count": len(trajectories),
            "severity_flips_mean": float(np.mean(sev_flips)),
            "severity_flips_max": max(sev_flips),
            "severity_zero_flips": sev_flips.count(0),
            "mechanism_flips_mean": float(np.mean(mech_flips)),
            "mechanism_flips_max": max(mech_flips),
            "mechanism_zero_flips": mech_flips.count(0),
            "dims_oscillating_mean": float(np.mean(dims_osc)),
            "dims_converged_mean": float(np.mean(dims_conv)),
            "final_severity_distribution": sev_dist,
            "converged_count": conv_count,
        }

    fab_summary = analyze_group("FABRICATED", fabricated_trajectories)
    print()
    novel_summary = analyze_group("NOVEL PAIRS", novel_trajectories)
    print()
    known_summary = analyze_group("KNOWN INTERACTING", known_trajectories)

    # Dimension structure analysis for fabricated pairs
    print()
    print("--- Dimensional Structure Analysis (Fabricated) ---")
    if fabricated_trajectories:
        # Which dimensions oscillate most consistently across fabricated pairs?
        all_osc_dims = np.zeros(512)
        for t in fabricated_trajectories:
            osc_mask = np.ones(512)
            osc_mask[list(set(range(512)) - set(range(512 - t["n_dims_oscillating"], 512)))] = 0
            # Actually use the dims_converged_per_step info
            # We need the actual oscillating dim indices — reconstruct from n_dims info
            # Since we don't store the exact indices (to save memory), we count

        # Instead, analyze convergence progression (pad to same length for averaging)
        max_len = max(len(t["dims_converged_per_step"]) for t in fabricated_trajectories)
        padded = []
        for t in fabricated_trajectories:
            prog = t["dims_converged_per_step"]
            padded.append(prog + [prog[-1]] * (max_len - len(prog)))
        mean_prog = np.mean(padded, axis=0)
        print(f"  Mean dims converged at each step (fabricated):")
        for step_idx, mc in enumerate(mean_prog):
            print(f"    Step {step_idx:2d}: {mc:5.0f}/512 dims converged")

    print()
    print("--- Dimensional Structure Analysis (Known) ---")
    if known_trajectories:
        max_len = max(len(t["dims_converged_per_step"]) for t in known_trajectories)
        padded = []
        for t in known_trajectories:
            prog = t["dims_converged_per_step"]
            padded.append(prog + [prog[-1]] * (max_len - len(prog)))
        mean_prog = np.mean(padded, axis=0)
        print(f"  Mean dims converged at each step (known):")
        for step_idx, mc in enumerate(mean_prog):
            print(f"    Step {step_idx:2d}: {mc:5.0f}/512 dims converged")

    # Save
    all_results = {
        "dim_threshold": dim_threshold,
        "convergence_threshold": threshold,
        "fabricated": fabricated_trajectories,
        "novel": novel_trajectories,
        "known": known_trajectories,
        "summaries": {
            "fabricated": fab_summary,
            "novel": novel_summary,
            "known": known_summary,
        },
    }

    # Strip large per-step data for JSON (keep summaries)
    save_results = {
        "dim_threshold": dim_threshold,
        "convergence_threshold": threshold,
        "summaries": all_results["summaries"],
        "fabricated_sample": fabricated_trajectories[:5],  # first 5 full trajectories
        "novel_sample": novel_trajectories[:5],
        "known_sample": known_trajectories[:5],
    }
    with open(OUTPUT_DIR / "class_trajectory_results.json", "w") as f:
        json.dump(save_results, f, indent=2)

    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN — Run all experiments and generate report
# ═══════════════════════════════════════════════════════════════════════════════

def generate_report(exp1_results, exp2_results, exp3_results, threshold):
    """Generate a comprehensive markdown report."""
    lines = []
    lines.append("# PharmLoop Experiment Results: Resolver or Classifier?")
    lines.append("")
    lines.append(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Checkpoint:** {CHECKPOINT}")
    lines.append(f"**Convergence Threshold:** {threshold:.6f}")
    lines.append("")

    # ─── Experiment 1 ─────────────────────────────────────────────────────
    lines.append("---")
    lines.append("")
    lines.append("## Experiment 1: Depth Ablation")
    lines.append("")
    lines.append("**Question:** Does accuracy improve with more recurrence steps, or saturate?")
    lines.append("")
    lines.append("| Depth | Sev Acc | Mech Acc | FNR | Conv Rate | Avg Steps |")
    lines.append("|------:|--------:|---------:|----:|----------:|----------:|")
    for r in exp1_results:
        lines.append(
            f"| {r['max_steps']:>5} | {r['severity_accuracy']:>6.1%} | "
            f"{r['mechanism_accuracy']:>7.1%} | {r['fnr']:>4.1%} | "
            f"{r['convergence_rate']:>8.1%} | {r['avg_steps_taken']:>9} |"
        )
    lines.append("")

    max_sev = max(r["severity_accuracy"] for r in exp1_results)
    max_mech = max(r["mechanism_accuracy"] for r in exp1_results)
    sev_sat = next(r["max_steps"] for r in exp1_results if r["severity_accuracy"] >= max_sev - 0.01)
    mech_sat = next(r["max_steps"] for r in exp1_results if r["mechanism_accuracy"] >= max_mech - 0.01)

    acc_16 = next(r for r in exp1_results if r["max_steps"] == 16)["severity_accuracy"]
    acc_32 = next(r for r in exp1_results if r["max_steps"] == 32)["severity_accuracy"]
    acc_64 = next(r for r in exp1_results if r["max_steps"] == 64)["severity_accuracy"]

    lines.append("### Analysis")
    lines.append("")
    lines.append(f"- **Peak severity accuracy:** {max_sev:.1%} (within 1% at depth {sev_sat})")
    lines.append(f"- **Peak mechanism accuracy:** {max_mech:.1%} (within 1% at depth {mech_sat})")
    lines.append(f"- **Accuracy at depth 16:** {acc_16:.1%}, **at 32:** {acc_32:.1%}, **at 64:** {acc_64:.1%}")
    lines.append("")

    if acc_32 > acc_16 + 0.005 or acc_64 > acc_16 + 0.005:
        lines.append("**Verdict:** Accuracy **improves beyond depth 16** — the recurrence is **genuinely computational**. "
                      "Each additional step adds representational power.")
    elif sev_sat <= 6:
        lines.append("**Verdict:** Accuracy **saturates early (depth ~{})** — the recurrence is **iterative refinement**, "
                      "not genuinely computational. The oscillator is essentially a lookup with extra steps.".format(sev_sat))
    else:
        lines.append(f"**Verdict:** Accuracy saturates at depth ~{sev_sat}. The recurrence provides moderate "
                      "computational benefit but levels off before the default max_steps=16.")
    lines.append("")

    # ─── Experiment 2 ─────────────────────────────────────────────────────
    lines.append("---")
    lines.append("")
    lines.append("## Experiment 2: Velocity Distribution")
    lines.append("")
    lines.append("**Question:** Do fabricated drugs maintain high velocity (genuine uncertainty) "
                 "or converge to wrong answers?")
    lines.append("")

    summaries = exp2_results["summaries"]
    lines.append("| Group | N | Final |v| Mean | Final |v| Std | Converged | Confidence Mean |")
    lines.append("|-------|--:|---------------:|--------------:|----------:|----------------:|")
    for key in ["fabricated", "known_interacting", "known_safe", "novel"]:
        s = summaries[key]
        lines.append(
            f"| {s['name']} | {s['count']} | {s['final_v_mean']:.6f} | "
            f"{s['final_v_std']:.6f} | {s['converged_count']}/{s['count']} "
            f"({s['converged_rate']:.0%}) | {s['confidence_mean']:.3f} |"
        )
    lines.append("")

    lines.append("### Severity Distribution by Group")
    lines.append("")
    for key in ["fabricated", "known_interacting", "known_safe", "novel"]:
        s = summaries[key]
        lines.append(f"- **{s['name']}:** {s['severity_distribution']}")
    lines.append("")

    fab_s = summaries["fabricated"]
    lines.append("### Separation Analysis")
    lines.append("")
    lines.append(f"- Convergence threshold: **{threshold:.6f}**")
    lines.append(f"- Fabricated final |v| mean: **{fab_s['final_v_mean']:.6f}** "
                 f"(above threshold: **{fab_s['final_v_mean'] > threshold}**)")

    fab_conv = fab_s["converged_count"]
    lines.append(f"- Fabricated drugs that converge: **{fab_conv}/{fab_s['count']}** ({fab_conv/fab_s['count']:.0%})")
    lines.append("")

    if fab_conv <= 10:
        lines.append("**Verdict:** Fabricated drugs **mostly do NOT converge** — the dynamics themselves "
                      "separate known from unknown. The safety property is **architectural**.")
    elif fab_conv <= 30:
        lines.append("**Verdict:** Some fabricated drugs converge ({}/{}). The dynamics provide **partial** "
                      "separation, but the threshold is doing significant safety work.".format(fab_conv, fab_s['count']))
    else:
        lines.append("**Verdict:** Most fabricated drugs converge ({}/{}). The **threshold is the primary "
                      "safety mechanism**, not the dynamics. The 'structurally cannot hallucinate' "
                      "claim needs revision.".format(fab_conv, fab_s['count']))
    lines.append("")

    novel_s = summaries["novel"]
    lines.append(f"### Novel Pairs (Known Drugs, Unknown Combination)")
    lines.append("")
    lines.append(f"- Converged: **{novel_s['converged_count']}/{novel_s['count']}** ({novel_s['converged_rate']:.0%})")
    lines.append(f"- Mean confidence: **{novel_s['confidence_mean']:.3f}**")
    lines.append(f"- Severity distribution: {novel_s['severity_distribution']}")
    lines.append("")

    if novel_s["converged_rate"] > 0.8:
        lines.append("Novel pairs (known drugs, unknown combination) **mostly converge**. The system "
                      "is producing answers for unseen combinations — check whether these are reasonable "
                      "generalizations or hallucinations.")
    else:
        lines.append("Novel pairs show **mixed convergence** — the system is partially uncertain about "
                      "unseen drug combinations, which is appropriate behavior.")
    lines.append("")

    # ─── Experiment 3 ─────────────────────────────────────────────────────
    lines.append("---")
    lines.append("")
    lines.append("## Experiment 3: Output Class Trajectory")
    lines.append("")
    lines.append("**Question:** Do predicted classes flip during oscillation (semantic instability) "
                 "or stabilize early (classifier behavior)?")
    lines.append("")

    exp3_summaries = exp3_results["summaries"]
    lines.append("| Group | Sev Flips (mean) | Sev Flips (max) | Zero Flips | Mech Flips (mean) | Dims Oscillating |")
    lines.append("|-------|------------------:|----------------:|-----------:|------------------:|-----------------:|")
    for key in ["fabricated", "novel", "known"]:
        s = exp3_summaries[key]
        lines.append(
            f"| {s['name']} | {s['severity_flips_mean']:.1f} | {s['severity_flips_max']} | "
            f"{s['severity_zero_flips']}/{s['count']} | {s['mechanism_flips_mean']:.1f} | "
            f"{s['dims_oscillating_mean']:.0f}/512 |"
        )
    lines.append("")

    lines.append("### Convergence by Group")
    lines.append("")
    for key in ["fabricated", "novel", "known"]:
        s = exp3_summaries[key]
        lines.append(f"- **{s['name']}:** converged {s['converged_count']}/{s['count']}, "
                      f"final severity: {s['final_severity_distribution']}")
    lines.append("")

    fab_s3 = exp3_summaries["fabricated"]
    known_s3 = exp3_summaries["known"]

    lines.append("### Interpretation")
    lines.append("")

    if fab_s3["severity_flips_mean"] > 2 and known_s3["severity_flips_mean"] < 1:
        lines.append("**Semantic instability for unknowns:** Fabricated pairs show **significant class flipping** "
                      f"(mean {fab_s3['severity_flips_mean']:.1f} flips) while known pairs are **stable** "
                      f"(mean {known_s3['severity_flips_mean']:.1f} flips). The system is genuinely exploring "
                      "incompatible hypotheses for unknown inputs. This is the **strongest evidence** "
                      "that the resolver is a different kind of computation.")
    elif fab_s3["severity_flips_mean"] > known_s3["severity_flips_mean"]:
        lines.append(f"Fabricated pairs show **more class flipping** (mean {fab_s3['severity_flips_mean']:.1f}) "
                      f"than known pairs (mean {known_s3['severity_flips_mean']:.1f}), but the difference "
                      "is moderate. There is some semantic instability for unknowns.")
    else:
        lines.append("Both fabricated and known pairs show similar class stability. The output class "
                      "stabilizes early regardless of input type — this is **classifier behavior**.")
    lines.append("")

    if fab_s3["dims_oscillating_mean"] > 400:
        lines.append(f"**Dimensional analysis:** Fabricated pairs have ~{fab_s3['dims_oscillating_mean']:.0f}/512 "
                      "dimensions still oscillating at step 16 — **total non-convergence**. The system "
                      "has no stable attractor for unknown inputs.")
    elif fab_s3["dims_oscillating_mean"] > 100:
        lines.append(f"**Dimensional analysis:** Fabricated pairs have ~{fab_s3['dims_oscillating_mean']:.0f}/512 "
                      "dimensions oscillating — **partial convergence**. Some aspects settle while "
                      "others remain uncertain. This is interesting: the system may know WHAT it "
                      "knows and what it doesn't, per dimension.")
    else:
        lines.append(f"**Dimensional analysis:** Fabricated pairs have only ~{fab_s3['dims_oscillating_mean']:.0f}/512 "
                      "dimensions oscillating — most dimensions converge even for unknown inputs. "
                      "The system is hallucinating at the dimensional level.")
    lines.append("")

    # ─── Overall Verdict ──────────────────────────────────────────────────
    lines.append("---")
    lines.append("")
    lines.append("## Overall Verdict")
    lines.append("")

    # Score the resolver thesis
    resolver_score = 0
    reasons = []

    # Exp 1: Does accuracy improve beyond 16?
    if acc_32 > acc_16 + 0.005 or acc_64 > acc_16 + 0.005:
        resolver_score += 2
        reasons.append("Recurrence is genuinely computational (accuracy improves beyond depth 16)")
    elif sev_sat > 8:
        resolver_score += 1
        reasons.append(f"Recurrence has moderate depth benefit (saturates at ~{sev_sat})")
    else:
        reasons.append(f"Recurrence saturates early (depth ~{sev_sat}) — refinement, not computation")

    # Exp 2: Do fabricated drugs stay at high velocity?
    if fab_s["converged_rate"] <= 0.1:
        resolver_score += 2
        reasons.append("Fabricated drugs overwhelmingly fail to converge — architectural safety")
    elif fab_s["converged_rate"] <= 0.3:
        resolver_score += 1
        reasons.append("Most fabricated drugs fail to converge — partial architectural safety")
    else:
        reasons.append("Fabricated drugs converge — threshold doing safety work, not dynamics")

    # Exp 3: Semantic instability for unknowns?
    if fab_s3["severity_flips_mean"] > 2 * known_s3["severity_flips_mean"] + 1:
        resolver_score += 2
        reasons.append("Strong semantic instability for unknowns (class flipping)")
    elif fab_s3["dims_oscillating_mean"] > 200:
        resolver_score += 1
        reasons.append("Significant dimensional non-convergence for unknowns")
    else:
        reasons.append("Limited semantic/dimensional distinction between known and unknown")

    lines.append(f"**Resolver Score: {resolver_score}/6**")
    lines.append("")
    for reason in reasons:
        lines.append(f"- {reason}")
    lines.append("")

    if resolver_score >= 5:
        lines.append("### Assessment: **RESOLVER**")
        lines.append("")
        lines.append("PharmLoop is a genuine oscillatory resolver. The recurrence is computational, "
                      "the dynamics separate known from unknown, and the convergence stream carries "
                      "semantic structure. The multi-agent convergence-stream communication concept "
                      "is viable.")
    elif resolver_score >= 3:
        lines.append("### Assessment: **HYBRID**")
        lines.append("")
        lines.append("PharmLoop shows elements of both resolver and classifier behavior. The recurrence "
                      "provides some genuine computation and the dynamics partially separate known from "
                      "unknown, but some safety properties rely on thresholds rather than architecture. "
                      "The system is a strong drug interaction checker with resolver-like properties "
                      "that could be strengthened with architectural changes.")
    else:
        lines.append("### Assessment: **CLASSIFIER**")
        lines.append("")
        lines.append("PharmLoop behaves primarily as a classifier with a confidence threshold. The "
                      "recurrence is refinement rather than computation, and safety relies on the "
                      "threshold rather than dynamics. This is still a very good drug interaction "
                      "checker, but the resolver concept document's claims about structural honesty "
                      "need revision before scaling to multi-agent communication.")
    lines.append("")

    report = "\n".join(lines)

    with open(OUTPUT_DIR / "experiment_results_report.md", "w") as f:
        f.write(report)

    return report


def main():
    start_time = time.time()

    model, test_loader, drug_registry, known_pairs = load_model_and_data()
    threshold = model.reasoning_loop.cell.threshold.item()

    print()
    exp1_results = run_experiment_1(model, test_loader)

    print()
    exp2_results = run_experiment_2(model, test_loader, drug_registry, known_pairs)

    print()
    exp3_results = run_experiment_3(model, drug_registry, known_pairs)

    print()
    print("=" * 70)
    print("GENERATING REPORT")
    print("=" * 70)
    report = generate_report(exp1_results, exp2_results, exp3_results, threshold)

    elapsed = time.time() - start_time
    print(f"\nAll experiments completed in {elapsed:.1f}s")
    print(f"Results saved to {OUTPUT_DIR}/")
    print(f"  - depth_ablation_results.json")
    print(f"  - velocity_distribution_results.json")
    print(f"  - class_trajectory_results.json")
    print(f"  - experiment_results_report.md")
    print()
    print(report)


if __name__ == "__main__":
    main()
