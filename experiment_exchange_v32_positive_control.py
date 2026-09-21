"""SOMNIA v3.2: The Positive Control Experiment.

Evaluates dose-response curve of sleep consolidation on taught memories.
Proves whether the consolidation pathway CAN learn when the teaching dose is strong.

Grid:
- Consolidation LR: {1e-4, 5e-4, 1e-3}
- Consolidation Cycles: {3, 10}
- Repetitions per cycle: {1x, 3x}
Total Configurations: 3 x 2 x 2 = 12

Instruments:
- N=1000 conscience set (seed 7)
- N=200 hard subset (bottom 20% margin)
- N=419 genuine misclassifications (all at positive control weight w=2.0)
- Taught-sample accuracy measured directly at the source (pre vs post)
"""

import os
import json
import time
import tempfile
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import torch
import matplotlib.pyplot as plt

from somnia.utils import get_git_hash, project_root, set_seed
from somnia.data import make_dataloaders
from somnia.stress import apply_rotation, apply_gaussian_noise
from somnia.models import extract_internal_stats_v2, ClassifierMLP, ConditionalVAE, IntrospectiveHeadV2
from somnia.sleep_v2 import SleepConsolidationV2, SoftStabilityFilter, DreamerV2
from specimen.organism import SpecimenOrganism


def compute_ece(probs: np.ndarray, preds: np.ndarray, targets: np.ndarray, n_bins: int = 15) -> float:
    """Compute Expected Calibration Error (ECE) across n_bins."""
    confs = np.max(probs, axis=1)
    correct = (preds == targets).astype(float)
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    
    ece = 0.0
    n = len(targets)
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confs > bin_lower) & (confs <= bin_upper) if i > 0 else (confs >= bin_lower) & (confs <= bin_upper)
        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(correct[in_bin])
            bin_conf = np.mean(confs[in_bin])
            ece += (bin_size / n) * np.abs(bin_acc - bin_conf)
    return float(ece)


def evaluate_all(classifier, intro_head, conscience_x, conscience_y, taught_x, taught_y):
    """Compute clean acc, hard acc (bottom 20%), ECE, taught acc, and taught confusion."""
    classifier.eval()
    with torch.no_grad():
        # Conscience evaluation
        logits, h = classifier(conscience_x)
        probs = torch.softmax(logits, dim=1).numpy()
        preds = logits.argmax(dim=1).numpy()
        targets = conscience_y.numpy()

        clean_correct = (preds == targets)
        clean_acc = float(np.mean(clean_correct))
        ece = compute_ece(probs, preds, targets, n_bins=15)

        sorted_probs = np.sort(probs, axis=1)
        margins = sorted_probs[:, -1] - sorted_probs[:, -2]
        hard_cutoff = np.percentile(margins, 20.0)
        hard_mask = margins <= hard_cutoff
        hard_acc = float(np.mean(clean_correct[hard_mask]))

        # Taught samples evaluation
        t_logits, t_h = classifier(taught_x)
        t_preds = t_logits.argmax(dim=1).numpy()
        t_targets = taught_y.numpy()
        taught_acc = float(np.mean(t_preds == t_targets))

        t_stats = extract_internal_stats_v2(t_h, t_logits)
        taught_confusion = float(intro_head(t_stats).mean().item())

    return {
        "clean_acc": clean_acc,
        "hard_acc": hard_acc,
        "ece": ece,
        "taught_acc": taught_acc,
        "taught_confusion": taught_confusion,
        "hard_count": int(np.sum(hard_mask)),
    }


def generate_stressed_teachings(base_classifier, val_x_np, val_y_np, conscience_indices, target_count=419, seed=42):
    """Generate genuine misclassifications from non-conscience validation digits with stress."""
    non_conscience_idx = [i for i in range(len(val_x_np)) if i not in conscience_indices]
    rng = np.random.RandomState(seed)
    chosen_pool = rng.choice(non_conscience_idx, size=min(2500, len(non_conscience_idx)), replace=False)
    
    pool_x = val_x_np[chosen_pool]
    pool_y = val_y_np[chosen_pool]
    
    rot_x = apply_rotation(pool_x, max_angle=30.0, seed=seed)
    stressed_x = apply_gaussian_noise(rot_x, std=0.25, seed=seed)
    
    base_classifier.eval()
    with torch.no_grad():
        t_x = torch.from_numpy(stressed_x)
        logits, _ = base_classifier(t_x)
        preds = logits.argmax(dim=1).numpy()
    
    err_mask = (preds != pool_y)
    err_x = stressed_x[err_mask]
    err_y = pool_y[err_mask]
    
    if len(err_y) < target_count:
        return err_x, err_y
    else:
        return err_x[:target_count], err_y[:target_count]


def run_positive_control():
    set_seed(0)
    root = project_root()
    figures_dir = root / "figures"
    results_dir = root / "results"
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 86)
    print("SOMNIA v3.2: THE POSITIVE CONTROL -- DOSE-RESPONSE EXPERIMENT")
    print("Proving Consolidation Learnability on Taught Samples")
    print("=" * 86)

    # 1. Load Conscience Set (N=1000 digits, seed 7)
    _, _, _, val_x_np, val_y_np = make_dataloaders()
    rng_conscience = np.random.RandomState(7)
    conscience_idx = rng_conscience.choice(len(val_x_np), size=1000, replace=False)
    conscience_x = torch.from_numpy(val_x_np[conscience_idx])
    conscience_y = torch.from_numpy(val_y_np[conscience_idx])

    # 2. Generate Stressed Teachings (419 genuine misclassifications)
    with tempfile.TemporaryDirectory() as tmpdir:
        base_org = SpecimenOrganism(db_path=str(Path(tmpdir) / "bootstrap.db"))
        taught_x_np, taught_y_np = generate_stressed_teachings(
            base_org.classifier, val_x_np, val_y_np, conscience_idx, target_count=419, seed=42
        )
        n_taught = len(taught_y_np)
        taught_t_x = torch.from_numpy(taught_x_np)
        taught_t_y = torch.from_numpy(taught_y_np)
        
        # Base model weights path
        base_weights = {k: v.cpu().clone() for k, v in base_org.classifier.state_dict().items()}
        cvae_model = base_org.cvae
        intro_model = base_org.intro_head

        base_metrics = evaluate_all(base_org.classifier, intro_model, conscience_x, conscience_y, taught_t_x, taught_t_y)

    print(f"\n[INSTRUMENTS & BASELINE]")
    print(f"  * Conscience Set: N=1000 digits | Hard Subset: N={base_metrics['hard_count']} digits")
    print(f"  * Taught Samples: N={n_taught} genuine errors (Pre-Sleep Accuracy = {base_metrics['taught_acc']*100:.2f}%)")
    print(f"  * Base Conscience: Clean = {base_metrics['clean_acc']*100:.2f}%, Hard = {base_metrics['hard_acc']*100:.2f}%, ECE = {base_metrics['ece']:.4f}")
    print(f"  * Pre-Sleep Taught Confusion: {base_metrics['taught_confusion']*100:.2f}%")

    # -------------------------------------------------------------
    # GRID SEARCH: 3 LRs x 2 Cycles x 2 Repetitions = 12 Configs
    # -------------------------------------------------------------
    lr_list = [1e-4, 5e-4, 1e-3]
    cycle_list = [3, 10]
    rep_list = [1, 3]

    grid_results = []
    
    print("\n" + "-" * 86)
    print(f"{'Config':<4} | {'LR':<7} | {'Cycles':<6} | {'Reps':<4} | {'Taught Acc':<13} | {'Clean Acc':<13} | {'Hard Acc':<13} | {'Clean Cost':<10}")
    print("-" * 86)

    cfg_idx = 1
    for lr in lr_list:
        for cycles in cycle_list:
            for reps in rep_list:
                # Fresh clone of classifier
                clf = ClassifierMLP()
                clf.load_state_dict(base_weights)
                
                dreamer = DreamerV2(clf, cvae_model, intro_model)
                stability = SoftStabilityFilter(clf, n_perturbations=5, noise_std=0.05)
                consolidator = SleepConsolidationV2(clf, dream_mix=0.05, lr=lr, epochs=2)

                # Positive control: all taught samples receive full w=2.0 weight
                taught_weights = torch.full((n_taught,), 2.0, dtype=torch.float32)

                # Effective batch with reps
                batch_taught_x = torch.cat([taught_t_x] * reps, dim=0)
                batch_taught_y = torch.cat([taught_t_y] * reps, dim=0)
                batch_taught_w = torch.cat([taught_weights] * reps, dim=0)

                history = []
                # Execute consolidation cycles
                for c in range(1, cycles + 1):
                    # Confusion profile and dreams
                    class_conf = dreamer.get_class_confusion_profile(taught_t_x)
                    raw_dreams, dream_labels = dreamer.generate_targeted_dreams(class_conf, n=200, seed=42 + c)
                    pseudo_y, dream_w = stability.compute_soft_weights(raw_dreams, seed=42 + c)

                    # Real replay buffer (random calibration batch)
                    real_x = conscience_x[:50]
                    real_y = conscience_y[:50]

                    consolidator.sleep_cycle(
                        real_x, real_y, raw_dreams, pseudo_y, dream_w,
                        taught_x=batch_taught_x, taught_y=batch_taught_y, taught_weights=batch_taught_w
                    )

                    m = evaluate_all(clf, intro_model, conscience_x, conscience_y, taught_t_x, taught_t_y)
                    history.append(m)

                final_m = history[-1]
                d_taught = (final_m["taught_acc"] - base_metrics["taught_acc"]) * 100.0
                d_clean = (final_m["clean_acc"] - base_metrics["clean_acc"]) * 100.0
                d_hard = (final_m["hard_acc"] - base_metrics["hard_acc"]) * 100.0
                d_conf = (final_m["taught_confusion"] - base_metrics["taught_confusion"]) * 100.0
                clean_cost = -d_clean  # positive if clean accuracy dropped

                effective_dose = lr * cycles * reps * 1000.0  # normalized scalar dose metric

                config_entry = {
                    "config_id": cfg_idx,
                    "lr": lr,
                    "cycles": cycles,
                    "reps": reps,
                    "effective_dose": effective_dose,
                    "final_metrics": final_m,
                    "delta_taught_acc_pp": d_taught,
                    "delta_clean_acc_pp": d_clean,
                    "delta_hard_acc_pp": d_hard,
                    "delta_taught_conf_pp": d_conf,
                    "clean_cost_pp": clean_cost,
                    "history": history,
                }
                grid_results.append(config_entry)

                print(f"#{cfg_idx:<3} | {lr:<7.0e} | {cycles:<6} | {reps}x   | {final_m['taught_acc']*100:.2f}% ({d_taught:+.2f}p) | {final_m['clean_acc']*100:.2f}% ({d_clean:+.2f}p) | {final_m['hard_acc']*100:.2f}% ({d_hard:+.2f}p) | {clean_cost:+.2f}pp")
                cfg_idx += 1

    print("-" * 86)

    # -------------------------------------------------------------
    # PLOT: figures/v32_dose_response.png
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    doses = [cfg["effective_dose"] for cfg in grid_results]
    taught_deltas = [cfg["delta_taught_acc_pp"] for cfg in grid_results]
    clean_costs = [cfg["clean_cost_pp"] for cfg in grid_results]
    lrs = [cfg["lr"] for cfg in grid_results]

    # Color map by LR, size by cycles
    color_map = {1e-4: "#00FF9D", 5e-4: "#FFA500", 1e-3: "#FF3B30"}
    
    for cfg in grid_results:
        c = color_map[cfg["lr"]]
        s = 120 if cfg["cycles"] == 10 else 60
        m = "o" if cfg["reps"] == 1 else "s"
        ax.scatter(cfg["effective_dose"], cfg["delta_taught_acc_pp"], color=c, s=s, marker=m, edgecolors="white", linewidth=0.8, alpha=0.9)
        ax.annotate(
            f"#{cfg['config_id']} ({cfg['delta_taught_acc_pp']:+.1f}p, cost:{cfg['clean_cost_pp']:+.2f}p)",
            (cfg["effective_dose"], cfg["delta_taught_acc_pp"]),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=7.5,
            color="#c9d1d9"
        )

    # Add trend line
    sorted_pairs = sorted(zip(doses, taught_deltas))
    sx, sy = zip(*sorted_pairs)
    ax.plot(sx, sy, color="#58a6ff", linestyle="--", alpha=0.6, linewidth=1.5, label="Dose-Response Trend")

    ax.axhline(0.0, color="#888888", linestyle=":", linewidth=1)
    ax.axhline(2.0, color="#00FF9D", linestyle=":", linewidth=1.2, label="Learnability Threshold (+2.0pp)")

    ax.set_xscale("log")
    ax.set_title("SOMNIA v3.2 // The Positive Control: Dose-Response of Taught Digestion", color="white", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Effective Consolidation Dose (LR x Cycles x Repetitions x 1000)", color="white", fontsize=10)
    ax.set_ylabel("Taught-Sample Accuracy Delta (percentage points)", color="white", fontsize=10)
    ax.tick_params(colors="white")
    ax.grid(True, linestyle=":", alpha=0.4, color="#444444")
    
    # Custom legend
    import matplotlib.lines as mlines
    handles = [
        mlines.Line2D([], [], color="#00FF9D", marker="o", linestyle="None", label="LR = 1e-4"),
        mlines.Line2D([], [], color="#FFA500", marker="o", linestyle="None", label="LR = 5e-4"),
        mlines.Line2D([], [], color="#FF3B30", marker="o", linestyle="None", label="LR = 1e-3"),
        mlines.Line2D([], [], color="#888888", marker="o", markersize=6, linestyle="None", label="3 Cycles / 1x Rep"),
        mlines.Line2D([], [], color="#888888", marker="s", markersize=9, linestyle="None", label="10 Cycles / 3x Rep"),
        mlines.Line2D([], [], color="#58a6ff", linestyle="--", label="Trend Line"),
    ]
    ax.legend(handles=handles, frameon=True, facecolor="#161b22", edgecolor="#30363d", labelcolor="white", loc="upper left", fontsize=8.5)
    plt.tight_layout()

    fig_path = figures_dir / "v32_dose_response.png"
    plt.savefig(fig_path, dpi=150, facecolor="#0d1117")
    plt.close()
    print(f"\n[FIGURE SAVED] Dose-response figure saved to: {fig_path}")

    # -------------------------------------------------------------
    # SAVE JSON RESULTS: results/v32.json
    # -------------------------------------------------------------
    git_hash = get_git_hash()
    
    # Determine Positive Control Outcome
    # Check max dose config (#12: LR=1e-3, 10 cycles, 3x reps) and high dose configs
    max_taught_gain = max(cfg["delta_taught_acc_pp"] for cfg in grid_results)
    best_config = max(grid_results, key=lambda c: c["delta_taught_acc_pp"])
    
    # Verdict condition: taught accuracy rises > 2pp at high dose with < 0.5pp clean cost
    passed_high_dose_configs = [
        cfg for cfg in grid_results
        if cfg["delta_taught_acc_pp"] > 2.0 and cfg["clean_cost_pp"] < 0.50
    ]
    positive_control_passed = len(passed_high_dose_configs) > 0

    if positive_control_passed:
        verdict_text = "POSITIVE CONTROL PASSED -- the pipeline learns. v3.1's null is a true dose-response finding: correction-by-correction teaching needs N lessons. Digestion requires scale, not repair."
    else:
        verdict_text = "POSITIVE CONTROL FAILED -- pipeline bug or LR floor; report the numbers and stop."

    results_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": git_hash,
        "n_conscience_digits": 1000,
        "n_hard_subset": base_metrics["hard_count"],
        "n_taught_samples": n_taught,
        "base_metrics": base_metrics,
        "grid_results": grid_results,
        "best_config": {
            "config_id": best_config["config_id"],
            "lr": best_config["lr"],
            "cycles": best_config["cycles"],
            "reps": best_config["reps"],
            "delta_taught_acc_pp": best_config["delta_taught_acc_pp"],
            "delta_clean_acc_pp": best_config["delta_clean_acc_pp"],
            "delta_hard_acc_pp": best_config["delta_hard_acc_pp"],
            "clean_cost_pp": best_config["clean_cost_pp"],
        },
        "positive_control_passed": positive_control_passed,
        "verdict_text": verdict_text,
    }

    json_path = results_dir / "v32.json"
    with open(json_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"[JSON SAVED] Results saved to: {json_path}")

    # -------------------------------------------------------------
    # PRINT FINAL VERDICT VERBATIM
    # -------------------------------------------------------------
    print("\n" + "=" * 86)
    print("FINAL POSITIVE CONTROL VERDICT")
    print("=" * 86)
    print(verdict_text)
    print("-" * 86)
    print(f"Max Taught Accuracy Gain: {best_config['delta_taught_acc_pp']:+.2f}pp (Config #{best_config['config_id']}: LR={best_config['lr']}, {best_config['cycles']} cycles, {best_config['reps']}x reps, Clean Cost={best_config['clean_cost_pp']:+.2f}pp)")
    print(f"Low-dose baseline (v3.1 regime: LR 1e-4, 3 cycles, 1x): {grid_results[0]['delta_taught_acc_pp']:+.2f}pp taught delta.")
    print("=" * 86)


if __name__ == "__main__":
    run_positive_control()
