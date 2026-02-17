#!/usr/bin/env python3
"""
PharmLoop Experiments Round 2: Testing the Dynamics Directly

The output head learned to shortcut the oscillator, reading answers from
the encoder's initial state. These experiments test whether the oscillator
dynamics contain real information that the output head ignores.

  4. Trajectory Information Content — fresh classifiers on trajectory features
  5. Hopfield Retrieval Analysis — is the Hopfield bank contributing?
  6. Bypass Test — classify by nearest Hopfield pattern, not output head
  7. Dimensional Semantics — are oscillating dims structured or random?
  8. Output Head Dependency Test — probe heads at different steps

Run from project root:
    python experiments/run_round2_experiments.py
"""

import json
import random
import sys
import time
from pathlib import Path
from itertools import combinations as combos

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pharmloop.hierarchical_hopfield import HierarchicalHopfield, DRUG_CLASSES
from pharmloop.model import PharmLoopModel
from pharmloop.output import (
    SEVERITY_NAMES, MECHANISM_NAMES, FLAG_NAMES,
    NUM_SEVERITY_CLASSES, NUM_MECHANISMS, NUM_FLAGS,
)
from training.data_loader import DrugInteractionDataset

# ─── Configuration ────────────────────────────────────────────────────────────

CHECKPOINT = "checkpoints/best_model_phase4b.pt"
DRUGS_PATH = "data/processed/drugs_v3.json"
INTERACTIONS_PATH = "data/processed/interactions_v3.json"
SPLIT_PATH = "data/processed/split_v3.json"
OUTPUT_DIR = Path("experiments/round2_results")
BATCH_SIZE = 64
SEED = 42
FORCED_STEPS = 16

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ─── Shared utilities ─────────────────────────────────────────────────────────

def make_clf(n_features: int):
    """Fast classifier pipeline: SVD for high-dim, lbfgs solver."""
    if n_features > 600:
        return make_pipeline(StandardScaler(), TruncatedSVD(n_components=200),
                             LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs"))
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs"))


def load_model_and_data():
    """Load Phase 4b model, train/test data, drug registry."""
    print("=" * 70)
    print("LOADING MODEL AND DATA")
    print("=" * 70)

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

    train_dataset = DrugInteractionDataset(
        DRUGS_PATH, INTERACTIONS_PATH,
        fabricated_ratio=0.0, split_indices=split["train_indices"],
    )
    test_dataset = DrugInteractionDataset(
        DRUGS_PATH, INTERACTIONS_PATH,
        fabricated_ratio=0.0, split_indices=split["test_indices"],
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    drug_registry = {}
    for name, info in drugs_raw["drugs"].items():
        drug_registry[name] = {"id": info["id"], "features": info["features"],
                               "class": info.get("class", "other")}

    with open(INTERACTIONS_PATH) as f:
        interactions_data = json.load(f)
    known_pairs = set()
    for inter in interactions_data["interactions"]:
        known_pairs.add(tuple(sorted([inter["drug_a"], inter["drug_b"]])))

    threshold = model.reasoning_loop.cell.threshold.item()
    print(f"  Model: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"  Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    print(f"  Drugs: {len(drug_registry)}, Threshold: {threshold:.6f}")
    print()

    return (model, train_loader, test_loader, drug_registry,
            known_pairs, drug_class_map, split, interactions_data)


def run_forced_full_depth(model, a_id, a_feat, b_id, b_feat, max_steps=FORCED_STEPS):
    """Run model forcing ALL steps (no early stopping), no noise."""
    orig_thresh = model.reasoning_loop.cell.raw_threshold.data.clone()
    orig_max = model.reasoning_loop.max_steps
    model.reasoning_loop.cell.raw_threshold.data.fill_(-100.0)
    model.reasoning_loop.max_steps = max_steps
    with torch.no_grad():
        output = model(a_id, a_feat, b_id, b_feat)
    model.reasoning_loop.cell.raw_threshold.data.copy_(orig_thresh)
    model.reasoning_loop.max_steps = orig_max
    return output


def extract_all_trajectories(model, loader):
    """Extract full forced-depth trajectories for all samples."""
    all_pos, all_vel, all_gz = [], [], []
    all_sev, all_mech, all_flags = [], [], []
    all_aid, all_bid = [], []

    for batch in loader:
        out = run_forced_full_depth(
            model, batch["drug_a_id"], batch["drug_a_features"],
            batch["drug_b_id"], batch["drug_b_features"],
        )
        traj = out["trajectory"]
        bs = batch["drug_a_id"].shape[0]
        pos_s = torch.stack(traj["positions"], dim=0)
        vel_s = torch.stack(traj["velocities"], dim=0)
        gz_s = torch.stack(traj["gray_zones"], dim=0)

        for i in range(bs):
            all_pos.append(pos_s[:, i, :].cpu().numpy())
            all_vel.append(vel_s[:, i, :].cpu().numpy())
            all_gz.append(gz_s[:, i].cpu().numpy())
            all_sev.append(batch["target_severity"][i].item())
            all_mech.append(batch["target_mechanisms"][i].numpy())
            all_flags.append(batch["target_flags"][i].numpy())
            all_aid.append(batch["drug_a_id"][i].item())
            all_bid.append(batch["drug_b_id"][i].item())

    return {
        "positions": all_pos, "velocities": all_vel, "gray_zones": all_gz,
        "severity": np.array(all_sev), "mechanisms": np.array(all_mech),
        "flags": np.array(all_flags),
        "drug_a_ids": np.array(all_aid), "drug_b_ids": np.array(all_bid),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 4: TRAJECTORY INFORMATION CONTENT
# ═══════════════════════════════════════════════════════════════════════════════

def build_feature_sets(data):
    """Build feature sets A (encoder), B (trajectory stats), C (full traj), D (delta)."""
    feat_A, feat_B, feat_C = [], [], []
    for idx in range(len(data["positions"])):
        pos = data["positions"][idx]
        vel = data["velocities"][idx]

        feat_A.append(pos[0])

        # B: trajectory statistics
        conv_step = np.full(512, len(vel), dtype=float)
        for step in range(len(vel)):
            newly = (np.abs(vel[step]) < 0.05) & (conv_step == len(vel))
            conv_step[newly] = step
        conv_step /= max(len(vel), 1)

        signs = np.sign(vel)
        dir_changes = (np.abs(np.diff(signs, axis=0)) > 0).sum(axis=0).astype(float)

        traj_feats = np.concatenate([
            pos[-1], vel[-1], conv_step, vel.mean(axis=0), vel.std(axis=0),
            np.abs(vel).max(axis=0), pos.std(axis=0), dir_changes,
            np.array([np.linalg.norm(vel[-1]), np.linalg.norm(vel[0]),
                      (np.abs(vel[-1]) < 0.05).sum() / 512,
                      float(np.abs(np.diff(signs, axis=0)).sum())]),
        ])
        feat_B.append(traj_feats)

        sample_steps = [s for s in [0, 2, 4, 8, 12, min(FORCED_STEPS, len(pos)-1)] if s < len(pos)]
        feat_C.append(np.concatenate([np.concatenate([pos[s], vel[s]]) for s in sample_steps]))

    return np.array(feat_A), np.array(feat_B), np.array(feat_C)


def run_experiment_4(test_data):
    """Compare classification from encoder output vs trajectory features."""
    print("=" * 70)
    print("EXPERIMENT 4: TRAJECTORY INFORMATION CONTENT")
    print("=" * 70)
    print()

    feat_A, feat_B, feat_C = build_feature_sets(test_data)
    feat_D = feat_B[:, 512:]  # delta: everything beyond encoder output
    sev_labels = test_data["severity"]
    mech_labels = test_data["mechanisms"]

    print(f"  Dims: A={feat_A.shape[1]}, B={feat_B.shape[1]}, C={feat_C.shape[1]}, D={feat_D.shape[1]}")

    results = {}
    for name, X in [("A: Encoder only (step 0)", feat_A),
                    ("B: Trajectory statistics", feat_B),
                    ("C: Full trajectory", feat_C),
                    ("D: Delta (traj - encoder)", feat_D)]:
        clf = make_clf(X.shape[1])
        sev_scores = cross_val_score(clf, X, sev_labels, cv=5, scoring="accuracy")

        mech_accs = []
        for m in range(NUM_MECHANISMS):
            if mech_labels[:, m].sum() > 5:
                clf_m = make_clf(X.shape[1])
                scores = cross_val_score(clf_m, X, mech_labels[:, m], cv=5, scoring="accuracy")
                mech_accs.append(scores.mean())
        mean_mech = np.mean(mech_accs) if mech_accs else 0.0

        print(f"  {name}: sev={sev_scores.mean():.1%}+/-{sev_scores.std():.1%}  mech={mean_mech:.1%}")
        results[name] = {
            "severity_mean": float(sev_scores.mean()), "severity_std": float(sev_scores.std()),
            "mechanism_mean": float(mean_mech), "feature_dim": X.shape[1],
        }

    a_sev = results["A: Encoder only (step 0)"]["severity_mean"]
    b_sev = results["B: Trajectory statistics"]["severity_mean"]
    d_sev = results["D: Delta (traj - encoder)"]["severity_mean"]
    print(f"\n  Encoder={a_sev:.1%}, Trajectory={b_sev:.1%}, Delta alone={d_sev:.1%}")
    if b_sev > a_sev + 0.01:
        print("  -> Trajectory OUTPERFORMS encoder — oscillator adds real information")
    else:
        print("  -> Encoder dominates — but delta at {:.1%} shows dynamics carry independent signal".format(d_sev))

    with open(OUTPUT_DIR / "trajectory_information.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 5: HOPFIELD RETRIEVAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def run_experiment_5(model, test_loader):
    """Measure what the Hopfield bank contributes at each step."""
    print("=" * 70)
    print("EXPERIMENT 5: HOPFIELD RETRIEVAL ANALYSIS")
    print("=" * 70)
    print()

    cell = model.reasoning_loop.cell
    hopfield = cell.hopfield
    global_bank = hopfield.global_bank
    n_stored = global_bank.count
    print(f"  Global bank: {n_stored} patterns stored")

    if n_stored == 0:
        print("  WARNING: No patterns stored. Skipping.")
        return {}

    stored_vals = global_bank.stored_values[:n_stored].clone()

    orig_thresh = model.reasoning_loop.cell.raw_threshold.data.clone()
    orig_max = model.reasoning_loop.max_steps
    model.reasoning_loop.cell.raw_threshold.data.fill_(-100.0)
    model.reasoning_loop.max_steps = FORCED_STEPS

    all_step_data = []
    sample_count = 0
    MAX_SAMPLES = 200

    for batch in test_loader:
        if sample_count >= MAX_SAMPLES:
            break
        bs = batch["drug_a_id"].shape[0]

        with torch.no_grad():
            enc_a = model.encoder(batch["drug_a_id"], batch["drug_a_features"])
            enc_b = model.encoder(batch["drug_b_id"], batch["drug_b_features"])
            pair_fwd = torch.cat([enc_a, enc_b], dim=-1)
            pair_rev = torch.cat([enc_b, enc_a], dim=-1)
            initial_state = (model.pair_combine(pair_fwd) + model.pair_combine(pair_rev)) / 2.0
            x = initial_state
            v = model.reasoning_loop.initial_v_proj(initial_state)

        for i in range(min(bs, MAX_SAMPLES - sample_count)):
            steps = []
            xi, vi = x[i:i+1], v[i:i+1]

            for step in range(FORCED_STEPS):
                with torch.no_grad():
                    gz_scalar = vi.norm(dim=-1, keepdim=True)
                    beta = cell.beta_mod(gz_scalar).mean().item()

                    query = cell.hopfield_query_proj(xi) if cell.hopfield_query_proj is not None else xi
                    hopfield._current_classes = None
                    retrieved = hopfield.retrieve(query, beta=beta, drug_classes=None)
                    if cell.hopfield_value_proj is not None:
                        retrieved = cell.hopfield_value_proj(retrieved)

                    cos_sim = nn.functional.cosine_similarity(xi, retrieved, dim=-1).item()
                    evidence_input = torch.cat([xi, retrieved], dim=-1)
                    force = cell.spring * cell.evidence_transform(evidence_input)
                    force_mag = force.norm(dim=-1).item()

                    sims = nn.functional.cosine_similarity(xi, stored_vals, dim=-1)
                    nearest_id = sims.argmax().item()
                    nearest_sim = sims.max().item()

                    steps.append({
                        "cos_sim": cos_sim, "force_mag": force_mag,
                        "nearest_id": nearest_id, "nearest_sim": nearest_sim, "beta": beta,
                    })
                    xi, vi, _ = cell(xi, vi, training=False)

            pattern_ids = [s["nearest_id"] for s in steps]
            all_step_data.append({
                "unique_patterns": len(set(pattern_ids)),
                "cos_trajectory": [s["cos_sim"] for s in steps],
                "force_trajectory": [s["force_mag"] for s in steps],
            })
            sample_count += 1

    model.reasoning_loop.cell.raw_threshold.data.copy_(orig_thresh)
    model.reasoning_loop.max_steps = orig_max

    unique_pats = [d["unique_patterns"] for d in all_step_data]
    cos_s0 = [d["cos_trajectory"][0] for d in all_step_data]
    cos_sf = [d["cos_trajectory"][-1] for d in all_step_data]
    force_trajs = np.array([d["force_trajectory"] for d in all_step_data])
    cos_trajs = np.array([d["cos_trajectory"] for d in all_step_data])

    print(f"  Unique patterns per example: mean={np.mean(unique_pats):.1f}, "
          f"single={unique_pats.count(1)}/{len(unique_pats)}, "
          f"2+={sum(1 for u in unique_pats if u >= 2)}/{len(unique_pats)}")
    print(f"  Cosine(state, retrieval): step0={np.mean(cos_s0):.4f}, final={np.mean(cos_sf):.4f}")
    print(f"  Force: step0={force_trajs[:, 0].mean():.4f}, final={force_trajs[:, -1].mean():.4f}")
    print(f"\n  Force trajectory (mean):")
    for i, f_val in enumerate(force_trajs.mean(axis=0)):
        print(f"    Step {i:2d}: {f_val:.4f}")

    results = {
        "n_samples": len(all_step_data),
        "unique_patterns_mean": float(np.mean(unique_pats)),
        "single_pattern_count": unique_pats.count(1),
        "multi_pattern_count": sum(1 for u in unique_pats if u >= 2),
        "cos_sim_step0_mean": float(np.mean(cos_s0)),
        "cos_sim_final_mean": float(np.mean(cos_sf)),
        "force_step0_mean": float(force_trajs[:, 0].mean()),
        "force_final_mean": float(force_trajs[:, -1].mean()),
        "mean_force_trajectory": force_trajs.mean(axis=0).tolist(),
        "mean_cos_trajectory": cos_trajs.mean(axis=0).tolist(),
    }

    if np.mean(unique_pats) >= 2.0:
        print("\n  -> Multiple Hopfield patterns visited — oscillator explores basins")
    else:
        print("\n  -> Mostly single pattern — refinement within one basin")

    with open(OUTPUT_DIR / "hopfield_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 6: BYPASS TEST
# ═══════════════════════════════════════════════════════════════════════════════

def run_experiment_6(train_data, test_data):
    """Classify by nearest training example in position space."""
    print("=" * 70)
    print("EXPERIMENT 6: BYPASS TEST (Nearest-Neighbor)")
    print("=" * 70)
    print()

    train_final = np.array([p[-1] for p in train_data["positions"]])
    train_norms = np.linalg.norm(train_final, axis=1, keepdims=True) + 1e-8
    train_normed = train_final / train_norms
    train_sev = train_data["severity"]
    train_mech = train_data["mechanisms"]
    test_sev = test_data["severity"]
    test_mech = test_data["mechanisms"]

    DEPTHS = [0, 2, 4, 8, 12, 16]
    results = {}

    for depth in DEPTHS:
        test_pos = np.array([p[depth] if depth < len(p) else p[-1] for p in test_data["positions"]])
        test_norms = np.linalg.norm(test_pos, axis=1, keepdims=True) + 1e-8
        test_normed = test_pos / test_norms

        sim_matrix = test_normed @ train_normed.T
        nearest_ids = sim_matrix.argmax(axis=1)
        nearest_sims = sim_matrix.max(axis=1)

        nn_sev = train_sev[nearest_ids]
        nn_mech = train_mech[nearest_ids]
        sev_acc = (nn_sev == test_sev).mean()
        mech_acc = (nn_mech == test_mech).all(axis=1).mean()

        severe_mask = (test_sev == 3) | (test_sev == 4)
        fnr = (nn_sev[severe_mask] == 0).mean() if severe_mask.sum() > 0 else 0.0

        per_mech = []
        for m in range(NUM_MECHANISMS):
            if test_mech[:, m].sum() > 0:
                per_mech.append((nn_mech[:, m] == test_mech[:, m]).mean())

        print(f"  Step {depth:>2}: sev={sev_acc:.1%}  mech={mech_acc:.1%}  "
              f"per_mech={np.mean(per_mech):.1%}  FNR={fnr:.1%}  sim={nearest_sims.mean():.4f}")

        results[f"step_{depth}"] = {
            "severity_accuracy": float(sev_acc), "mechanism_accuracy": float(mech_acc),
            "per_mechanism_accuracy": float(np.mean(per_mech)), "fnr": float(fnr),
            "mean_nearest_sim": float(nearest_sims.mean()),
        }

    s0 = results["step_0"]["severity_accuracy"]
    s16 = results["step_16"]["severity_accuracy"]
    print(f"\n  Step 0={s0:.1%} -> Step 16={s16:.1%}")
    if s16 > s0 + 0.01:
        print("  -> NN accuracy IMPROVES with depth — oscillator converges to better attractors")
    else:
        print("  -> NN accuracy flat — oscillator doesn't change attractor proximity")

    with open(OUTPUT_DIR / "bypass_test.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 7: DIMENSIONAL SEMANTICS
# ═══════════════════════════════════════════════════════════════════════════════

def run_experiment_7(test_data, drug_class_map):
    """Analyze whether oscillating dimensions are structured or random."""
    print("=" * 70)
    print("EXPERIMENT 7: DIMENSIONAL SEMANTICS")
    print("=" * 70)
    print()

    n_samples = len(test_data["velocities"])

    # Build per-dim convergence TIMING (when each dim converges)
    convergence_timing = []
    for idx in range(n_samples):
        vel_traj = test_data["velocities"][idx]
        timing = np.full(512, len(vel_traj), dtype=float)
        for step in range(len(vel_traj)):
            newly = (np.abs(vel_traj[step]) < 0.05) & (timing == len(vel_traj))
            timing[newly] = step
        convergence_timing.append(timing)
    convergence_timing = np.array(convergence_timing)

    # Group by drug class pair
    class_pair_masks: dict[str, list[int]] = {}
    for idx in range(n_samples):
        cls_a = drug_class_map.get(test_data["drug_a_ids"][idx], "other")
        cls_b = drug_class_map.get(test_data["drug_b_ids"][idx], "other")
        key = " + ".join(sorted([cls_a, cls_b]))
        class_pair_masks.setdefault(key, []).append(idx)

    print("  Within-class convergence timing:")
    print(f"  {'Class Pair':<40} {'N':>4} {'Mean':>6} {'Std':>5} {'Early':>6} {'Late':>5}")
    class_summaries = {}
    for cls_pair, indices in sorted(class_pair_masks.items(), key=lambda x: -len(x[1])):
        if len(indices) < 5:
            continue
        timings = convergence_timing[indices]
        mean_t = timings.mean(axis=0)
        overall_mean = mean_t.mean()
        overall_std = mean_t.std()
        early = (mean_t < 3).sum()
        late = (mean_t > 10).sum()
        print(f"  {cls_pair:<40} {len(indices):>4} {overall_mean:>6.2f} {overall_std:>5.2f} {early:>6} {late:>5}")
        class_summaries[cls_pair] = {
            "count": len(indices), "mean_step": float(overall_mean),
            "std_step": float(overall_std), "early": int(early), "late": int(late),
        }

    # Cross-class similarity of timing profiles
    print("\n  Cross-class timing similarity:")
    class_profiles = {}
    for cls_pair, indices in class_pair_masks.items():
        if len(indices) >= 10:
            class_profiles[cls_pair] = convergence_timing[indices].mean(axis=0)

    cosine_sims = []
    for (c1, c2) in combos(class_profiles.keys(), 2):
        p1, p2 = class_profiles[c1], class_profiles[c2]
        cos = np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2) + 1e-8)
        cosine_sims.append((c1, c2, float(cos)))
    cosine_sims.sort(key=lambda x: x[2])

    print("    Most DIFFERENT:")
    for c1, c2, s in cosine_sims[:5]:
        print(f"      {c1} vs {c2}: {s:.4f}")
    print("    Most SIMILAR:")
    for c1, c2, s in cosine_sims[-5:]:
        print(f"      {c1} vs {c2}: {s:.4f}")

    # RF classifier on timing
    print("\n  Predict class pair from convergence timing:")
    timing_for_clf, labels_for_clf = [], []
    for cls_pair, indices in class_pair_masks.items():
        if len(indices) >= 10:
            for idx in indices:
                timing_for_clf.append(convergence_timing[idx])
                labels_for_clf.append(cls_pair)

    rf_acc, chance = 0.0, 0.0
    if len(set(labels_for_clf)) >= 3:
        X_clf = np.array(timing_for_clf)
        y_clf = np.array(labels_for_clf)
        n_classes = len(set(y_clf))
        chance = 1.0 / n_classes

        rf_scores = cross_val_score(
            RandomForestClassifier(n_estimators=50, random_state=SEED, max_depth=10),
            X_clf, y_clf, cv=5, scoring="accuracy",
        )
        rf_acc = float(rf_scores.mean())
        print(f"    RF 5-fold: {rf_acc:.1%} +/- {rf_scores.std():.1%} (chance={chance:.1%}, {n_classes} classes)")
    else:
        print("    Not enough class pairs for classification")

    # Global convergence stats
    global_mean = convergence_timing.mean(axis=0)
    print(f"\n  Global dims: all early(<1)={int((global_mean < 1).sum())}/512, "
          f"all late(>5)={int((global_mean > 5).sum())}/512")
    sorted_dims = np.argsort(global_mean)
    print(f"  Slowest 10 dims: {sorted_dims[-10:].tolist()}")
    print(f"  Fastest 10 dims: {sorted_dims[:10].tolist()}")

    results = {
        "class_pair_summaries": class_summaries,
        "cosine_most_different": cosine_sims[:5],
        "cosine_most_similar": cosine_sims[-5:],
        "class_prediction_accuracy": rf_acc, "class_prediction_chance": chance,
        "n_class_pairs": len(set(labels_for_clf)) if labels_for_clf else 0,
    }

    with open(OUTPUT_DIR / "dimensional_semantics.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 8: OUTPUT HEAD DEPENDENCY TEST
# ═══════════════════════════════════════════════════════════════════════════════

def run_experiment_8(train_data, test_data):
    """Train fresh linear probes at different oscillation depths."""
    print("=" * 70)
    print("EXPERIMENT 8: OUTPUT HEAD DEPENDENCY TEST")
    print("=" * 70)
    print()

    STEP_CONFIGS = {
        "Step 0 (encoder)": lambda p, v: p[0],
        "Step 2": lambda p, v: p[2] if len(p) > 2 else p[-1],
        "Step 4": lambda p, v: p[4] if len(p) > 4 else p[-1],
        "Step 8": lambda p, v: p[8] if len(p) > 8 else p[-1],
        "Step 12": lambda p, v: p[12] if len(p) > 12 else p[-1],
        "Step 16 (final)": lambda p, v: p[-1],
        "Late mean (8-16)": lambda p, v: np.mean(p[8:], axis=0) if len(p) > 8 else p[-1],
        "Pos+Vel final": lambda p, v: np.concatenate([p[-1], v[-1]]),
        "Pos+Vel step 0": lambda p, v: np.concatenate([p[0], v[0]]),
        "Delta (16 - 0)": lambda p, v: p[-1] - p[0],
    }

    # Subsample training data for speed
    n_train = len(train_data["positions"])
    sub_idx = np.random.choice(n_train, min(2000, n_train), replace=False)
    sub_pos = [train_data["positions"][i] for i in sub_idx]
    sub_vel = [train_data["velocities"][i] for i in sub_idx]
    sub_sev = train_data["severity"][sub_idx]
    sub_mech = train_data["mechanisms"][sub_idx]

    results = {}
    print(f"  {'Config':<25} | {'Sev':>6} | {'Mech':>6} |")
    print(f"  {'-'*25}-+-{'-'*6}-+-{'-'*6}-+")

    for name, fn in STEP_CONFIGS.items():
        tr_feats = np.array([fn(p, v) for p, v in zip(sub_pos, sub_vel)])
        te_feats = np.array([fn(p, v) for p, v in
                             zip(test_data["positions"], test_data["velocities"])])

        clf = make_clf(tr_feats.shape[1])
        clf.fit(tr_feats, sub_sev)
        sev_acc = clf.score(te_feats, test_data["severity"])

        mech_accs = []
        for m in range(NUM_MECHANISMS):
            if sub_mech[:, m].sum() > 5 and test_data["mechanisms"][:, m].sum() > 0:
                clf_m = make_clf(tr_feats.shape[1])
                clf_m.fit(tr_feats, sub_mech[:, m])
                mech_accs.append(clf_m.score(te_feats, test_data["mechanisms"][:, m]))
        mean_mech = np.mean(mech_accs) if mech_accs else 0.0

        print(f"  {name:<25} | {sev_acc:>5.1%} | {mean_mech:>5.1%} |")
        results[name] = {
            "severity_accuracy": float(sev_acc), "mechanism_accuracy": float(mean_mech),
            "feature_dim": tr_feats.shape[1],
        }

    s0 = results["Step 0 (encoder)"]["severity_accuracy"]
    s16 = results["Step 16 (final)"]["severity_accuracy"]
    pv = results["Pos+Vel final"]["severity_accuracy"]
    delta = results["Delta (16 - 0)"]["severity_accuracy"]
    print(f"\n  Step 0={s0:.1%} -> Step 16={s16:.1%}, Pos+Vel={pv:.1%}, Delta={delta:.1%}")

    with open(OUTPUT_DIR / "dependency_test.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def generate_report(exp4, exp5, exp6, exp7, exp8, threshold):
    """Generate comprehensive markdown report."""
    lines = []
    lines.append("# PharmLoop Experiment Round 2: Testing the Dynamics Directly")
    lines.append("")
    lines.append(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Checkpoint:** {CHECKPOINT}")
    lines.append(f"**Forced depth:** {FORCED_STEPS} steps (no early stopping)")
    lines.append(f"**Convergence threshold:** {threshold:.6f}")
    lines.append("")
    lines.append("**Context:** Round 1 showed the output head saturates at depth 2, reading "
                 "answers from the encoder. But the oscillator produces differentiated dynamics "
                 "(254/512 dims oscillating for fabricated vs 60/512 for known). Round 2 tests "
                 "whether those dynamics carry real information.")
    lines.append("")

    # Exp 4
    lines.append("---\n\n## Experiment 4: Trajectory Information Content\n")
    lines.append("| Feature Set | Severity Acc | Mechanism Acc | Dims |")
    lines.append("|-------------|-------------:|--------------:|-----:|")
    for name, r in exp4.items():
        lines.append(f"| {name} | {r['severity_mean']:.1%} +/- {r['severity_std']:.1%} | "
                     f"{r['mechanism_mean']:.1%} | {r['feature_dim']} |")

    a_sev = exp4.get("A: Encoder only (step 0)", {}).get("severity_mean", 0)
    b_sev = exp4.get("B: Trajectory statistics", {}).get("severity_mean", 0)
    d_sev = exp4.get("D: Delta (traj - encoder)", {}).get("severity_mean", 0)
    lines.append("")
    if b_sev > a_sev + 0.01:
        lines.append(f"**Trajectory ({b_sev:.1%}) outperforms encoder ({a_sev:.1%}).** "
                     "The oscillator produces information the output head ignores.")
    else:
        lines.append(f"**Encoder ({a_sev:.1%}) dominates trajectory ({b_sev:.1%}).** "
                     f"But delta features alone achieve {d_sev:.1%} — dynamics carry independent signal.")
    lines.append("")

    # Exp 5
    lines.append("---\n\n## Experiment 5: Hopfield Retrieval Analysis\n")
    lines.append(f"- Unique patterns visited: mean={exp5.get('unique_patterns_mean', 0):.1f}, "
                 f"single={exp5.get('single_pattern_count', 0)}/{exp5.get('n_samples', 0)}")
    lines.append(f"- Cosine(state, retrieval): step0={exp5.get('cos_sim_step0_mean', 0):.4f} "
                 f"-> final={exp5.get('cos_sim_final_mean', 0):.4f}")
    lines.append(f"- Force: step0={exp5.get('force_step0_mean', 0):.4f} "
                 f"-> final={exp5.get('force_final_mean', 0):.4f}")
    if exp5.get("mean_force_trajectory"):
        lines.append("\n```\nForce trajectory:")
        for i, f_val in enumerate(exp5["mean_force_trajectory"]):
            bar = "#" * int(f_val * 200)
            lines.append(f"  Step {i:2d}: {f_val:.4f} {bar}")
        lines.append("```")
    lines.append("")

    if exp5.get("unique_patterns_mean", 0) >= 2.0:
        lines.append("**Oscillator explores multiple Hopfield attractors.**")
    else:
        lines.append("**Oscillator stays near one attractor — refinement, not exploration.**")
    lines.append("")

    # Exp 6
    lines.append("---\n\n## Experiment 6: Bypass Test (NN-Attractor)\n")
    lines.append("| Step | NN Sev Acc | NN Mech Acc | Per-Mech | FNR | Sim |")
    lines.append("|-----:|-----------:|------------:|---------:|----:|----:|")
    for key in ["step_0", "step_2", "step_4", "step_8", "step_12", "step_16"]:
        r = exp6.get(key, {})
        d = key.split("_")[1]
        lines.append(f"| {d} | {r.get('severity_accuracy', 0):.1%} | "
                     f"{r.get('mechanism_accuracy', 0):.1%} | "
                     f"{r.get('per_mechanism_accuracy', 0):.1%} | "
                     f"{r.get('fnr', 0):.1%} | {r.get('mean_nearest_sim', 0):.4f} |")

    s0 = exp6.get("step_0", {}).get("severity_accuracy", 0)
    s16 = exp6.get("step_16", {}).get("severity_accuracy", 0)
    lines.append("")
    if s16 > s0 + 0.01:
        lines.append(f"**NN improves: {s0:.1%} -> {s16:.1%}.** Oscillator converges to better attractors.")
    else:
        lines.append(f"**NN flat: {s0:.1%} -> {s16:.1%}.** Oscillator doesn't change attractor proximity.")
    lines.append("")

    # Exp 7
    lines.append("---\n\n## Experiment 7: Dimensional Semantics\n")
    pred_acc = exp7.get("class_prediction_accuracy", 0)
    chance = exp7.get("class_prediction_chance", 0)
    lines.append(f"- Class prediction from timing: **{pred_acc:.1%}** (chance={chance:.1%})")
    if exp7.get("class_pair_summaries"):
        lines.append("\n| Class Pair | N | Mean Step | Std |")
        lines.append("|------------|--:|----------:|----:|")
        for cls, s in sorted(exp7["class_pair_summaries"].items(), key=lambda x: -x[1]["count"])[:10]:
            lines.append(f"| {cls} | {s['count']} | {s['mean_step']:.2f} | {s['std_step']:.2f} |")
    lines.append("")
    if pred_acc > chance * 2:
        lines.append("**Strong semantic structure in convergence timing.**")
    elif pred_acc > chance * 1.3:
        lines.append("**Moderate semantic structure in convergence timing.**")
    else:
        lines.append("**Weak class signal — convergence timing is not strongly class-specific.**")
    lines.append("")

    # Exp 8
    lines.append("---\n\n## Experiment 8: Output Head Dependency Test\n")
    lines.append("| Feature Source | Sev Acc | Mech Acc | Dims |")
    lines.append("|---------------|--------:|---------:|-----:|")
    for name, r in exp8.items():
        lines.append(f"| {name} | {r['severity_accuracy']:.1%} | "
                     f"{r['mechanism_accuracy']:.1%} | {r['feature_dim']} |")

    e_s0 = exp8.get("Step 0 (encoder)", {}).get("severity_accuracy", 0)
    e_s16 = exp8.get("Step 16 (final)", {}).get("severity_accuracy", 0)
    e_s0m = exp8.get("Step 0 (encoder)", {}).get("mechanism_accuracy", 0)
    e_s16m = exp8.get("Step 16 (final)", {}).get("mechanism_accuracy", 0)
    lines.append(f"\n**Severity:** step0={e_s0:.1%} -> step16={e_s16:.1%} ({e_s16-e_s0:+.1%})")
    lines.append(f"**Mechanism:** step0={e_s0m:.1%} -> step16={e_s16m:.1%} ({e_s16m-e_s0m:+.1%})")
    lines.append("")

    # Overall verdict
    lines.append("---\n\n## Overall Verdict\n")
    evidence_for, evidence_against = [], []

    if b_sev > a_sev + 0.01:
        evidence_for.append(f"Exp 4: Trajectory outperforms encoder ({b_sev:.1%} vs {a_sev:.1%})")
    else:
        evidence_against.append(f"Exp 4: Encoder dominates ({a_sev:.1%} vs {b_sev:.1%})")
    if d_sev > 0.30:
        evidence_for.append(f"Exp 4: Delta features achieve {d_sev:.1%} alone — independent signal")

    if exp5.get("unique_patterns_mean", 0) >= 2.0:
        evidence_for.append(f"Exp 5: {exp5['unique_patterns_mean']:.1f} patterns visited per example")
    else:
        evidence_against.append(f"Exp 5: Single pattern ({exp5.get('unique_patterns_mean', 0):.1f} avg)")

    if s16 > s0 + 0.01:
        evidence_for.append(f"Exp 6: NN improves with depth ({s0:.1%} -> {s16:.1%})")
    else:
        evidence_against.append(f"Exp 6: NN flat ({s0:.1%} -> {s16:.1%})")

    if pred_acc > chance * 2:
        evidence_for.append(f"Exp 7: Strong class prediction from timing ({pred_acc:.1%})")
    elif pred_acc > chance * 1.3:
        evidence_for.append(f"Exp 7: Moderate class prediction ({pred_acc:.1%})")
    else:
        evidence_against.append(f"Exp 7: Weak class prediction ({pred_acc:.1%})")

    if e_s16 > e_s0 + 0.01 or e_s16m > e_s0m + 0.01:
        evidence_for.append(f"Exp 8: Late probes improve (sev: {e_s0:.1%}->{e_s16:.1%}, mech: {e_s0m:.1%}->{e_s16m:.1%})")
    else:
        evidence_against.append(f"Exp 8: Late probes don't improve (sev: {e_s0:.1%}->{e_s16:.1%})")

    lines.append("### Evidence dynamics carry real information:")
    for e in (evidence_for or ["(none)"]):
        lines.append(f"- {e}")
    lines.append("\n### Evidence encoder solved everything:")
    for e in (evidence_against or ["(none)"]):
        lines.append(f"- {e}")

    n_for = len(evidence_for)
    lines.append("")
    if n_for >= 4:
        lines.append("### Assessment: **OUTPUT HEAD SHORTCUT CONFIRMED**\n")
        lines.append("The oscillator dynamics carry real, structured information that the "
                     "output head bypasses. Architectural fix needed: force output head to "
                     "read from late-step state.")
    elif n_for >= 2:
        lines.append("### Assessment: **MIXED — PARTIAL SHORTCUT**\n")
        lines.append("Some dynamic information exists but it's unclear if the oscillator "
                     "adds enough to justify its cost. Consider architectural experiments.")
    else:
        lines.append("### Assessment: **ENCODER DOMINANCE**\n")
        lines.append("The encoder genuinely solved the problem. The oscillator dynamics "
                     "are real but computationally redundant.")
    lines.append("")

    report = "\n".join(lines)
    with open(OUTPUT_DIR / "round2_report.md", "w") as f:
        f.write(report)
    return report


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    start_time = time.time()
    (model, train_loader, test_loader, drug_registry,
     known_pairs, drug_class_map, split, interactions_data) = load_model_and_data()
    threshold = model.reasoning_loop.cell.threshold.item()

    print("Extracting forced-depth trajectories (train)...")
    train_data = extract_all_trajectories(model, train_loader)
    print(f"  {len(train_data['positions'])} train trajectories")
    print("Extracting forced-depth trajectories (test)...")
    test_data = extract_all_trajectories(model, test_loader)
    print(f"  {len(test_data['positions'])} test trajectories\n")

    exp4 = run_experiment_4(test_data)
    print()
    exp5 = run_experiment_5(model, test_loader)
    print()
    exp6 = run_experiment_6(train_data, test_data)
    print()
    exp7 = run_experiment_7(test_data, drug_class_map)
    print()
    exp8 = run_experiment_8(train_data, test_data)

    print("\n" + "=" * 70)
    print("GENERATING REPORT")
    print("=" * 70)
    report = generate_report(exp4, exp5, exp6, exp7, exp8, threshold)

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"Results in {OUTPUT_DIR}/\n")
    print(report)


if __name__ == "__main__":
    main()
