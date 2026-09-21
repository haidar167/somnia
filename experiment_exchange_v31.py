"""SOMNIA v3.1 Phase 2: The Digestion Experiment.

Evaluates sleep consolidation with Second Opinion Generative Vouching under:
1. No Teaching (Baseline dream sleep only)
2. Honest Visitor Teaching (419 genuine misclassifications, 100% true labels)
3. 30% Liar Visitor Teaching (126 corrupted labels, 293 true labels)

Instruments:
- N=1000 conscience set (seed 7)
- N=200 hard subset (bottom 20% margin, top1 - top2)
- N=419 stressed misclassified digits (seed 42)
- Second Opinion cVAE Vouching (tolerance=1.10)
- Paired 95% Binomial Confidence Intervals on all deltas
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
from somnia.models import extract_internal_stats_v2, ClassifierMLP
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


def compute_paired_ci(pre_correct: np.ndarray, post_correct: np.ndarray) -> Tuple[float, float, float, bool]:
    """Compute mean delta and paired 95% confidence interval for binary accuracy outcomes.
    
    Returns:
        delta_pp: percentage point change
        ci_lower_pp: lower bound of 95% CI (in pp)
        ci_upper_pp: upper bound of 95% CI (in pp)
        is_significant: whether 0 is outside the 95% CI
    """
    diffs = post_correct.astype(float) - pre_correct.astype(float)
    n = len(diffs)
    mean_diff = float(np.mean(diffs))
    if n <= 1:
        return mean_diff * 100.0, mean_diff * 100.0, mean_diff * 100.0, False
    
    se = float(np.std(diffs, ddof=1) / np.sqrt(n))
    margin = 1.96 * se
    ci_lower = mean_diff - margin
    ci_upper = mean_diff + margin
    is_significant = bool((ci_lower > 0) or (ci_upper < 0))
    return mean_diff * 100.0, ci_lower * 100.0, ci_upper * 100.0, is_significant


def evaluate_metrics(classifier, intro_head, test_x: torch.Tensor, test_y: torch.Tensor, taught_x: torch.Tensor = None):
    """Compute clean acc, hard-subset acc (bottom 20% margin), ECE, and taught confusion."""
    classifier.eval()
    with torch.no_grad():
        logits, h = classifier(test_x)
        probs = torch.softmax(logits, dim=1).numpy()
        preds = logits.argmax(dim=1).numpy()
        targets = test_y.numpy()

    clean_correct = (preds == targets)
    clean_acc = float(np.mean(clean_correct))
    ece = compute_ece(probs, preds, targets, n_bins=15)

    # Hard subset: lowest 20% margin (top1 - top2)
    sorted_probs = np.sort(probs, axis=1)
    margins = sorted_probs[:, -1] - sorted_probs[:, -2]
    hard_cutoff = np.percentile(margins, 20.0)
    hard_mask = margins <= hard_cutoff
    hard_correct = clean_correct[hard_mask]
    hard_acc = float(np.mean(hard_correct))

    # Mean P(error) on taught memories if available
    taught_confusion = 0.0
    taught_acc = 0.0
    taught_correct = None
    if taught_x is not None and taught_x.shape[0] > 0:
        with torch.no_grad():
            t_logits, t_h = classifier(taught_x)
            t_stats = extract_internal_stats_v2(t_h, t_logits)
            taught_confusion = float(intro_head(t_stats).mean().item())

    return {
        "clean_acc": clean_acc,
        "clean_correct": clean_correct,
        "hard_acc": hard_acc,
        "hard_correct": hard_correct,
        "hard_mask": hard_mask,
        "ece": ece,
        "taught_confusion": taught_confusion,
        "hard_count": int(np.sum(hard_mask)),
        "n_samples": len(targets),
    }


def generate_stressed_teachings(base_classifier, val_x_np, val_y_np, conscience_indices, target_count=419, seed=42):
    """Generate genuine misclassifications from non-conscience validation digits with stress."""
    non_conscience_idx = [i for i in range(len(val_x_np)) if i not in conscience_indices]
    rng = np.random.RandomState(seed)
    chosen_pool = rng.choice(non_conscience_idx, size=min(2500, len(non_conscience_idx)), replace=False)
    
    pool_x = val_x_np[chosen_pool]
    pool_y = val_y_np[chosen_pool]
    
    # Apply rotation and noise stress
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
        print(f"[TEACHING GEN] Found {len(err_y)} errors, taking all.")
        return err_x, err_y
    else:
        return err_x[:target_count], err_y[:target_count]


def run_exchange_v31_experiment():
    set_seed(0)
    root = project_root()
    figures_dir = root / "figures"
    results_dir = root / "results"
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 84)
    print("SOMNIA v3.1: THE EXCHANGE -- FROM DEFENSE TO DIGESTION")
    print("Scalable Conscience (N=1000) & Generative Second Opinion Vouching")
    print("=" * 84)

    # 1. Load Conscience Set (1000 MNIST digits, seed 7)
    _, _, _, val_x_np, val_y_np = make_dataloaders()
    rng_conscience = np.random.RandomState(7)
    conscience_idx = rng_conscience.choice(len(val_x_np), size=1000, replace=False)
    conscience_x = torch.from_numpy(val_x_np[conscience_idx])
    conscience_y = torch.from_numpy(val_y_np[conscience_idx])

    # 2. Build Stressed Teachings (419 genuine misclassifications)
    with tempfile.TemporaryDirectory() as tmpdir:
        base_org = SpecimenOrganism(db_path=str(Path(tmpdir) / "bootstrap.db"))
        taught_x_np, taught_y_np = generate_stressed_teachings(
            base_org.classifier, val_x_np, val_y_np, conscience_idx, target_count=419, seed=42
        )
        n_taught = len(taught_y_np)
        taught_t_x = torch.from_numpy(taught_x_np)
        taught_t_y = torch.from_numpy(taught_y_np)
        
        # Initial voucher diagnostic on teachings
        voucher_diag = base_org.voucher.evaluate_taught_batch(taught_t_x, taught_t_y, base_org.classifier)
        v_weights = voucher_diag["weights"]
        n_w2 = int((v_weights == 2.0).sum().item())
        n_w1 = int((v_weights == 1.0).sum().item())
        n_w01 = int((v_weights == 0.1).sum().item())
        
        base_metrics = evaluate_metrics(base_org.classifier, base_org.intro_head, conscience_x, conscience_y, taught_t_x)

    print(f"\n[INSTRUMENT SPECIFICATIONS]")
    print(f"  * Conscience Set: N=1000 digits (1 sample = 0.10pp)")
    print(f"  * Hard Subset: N={base_metrics['hard_count']} digits (lowest 20% margin, 1 sample = 0.50pp)")
    print(f"  * Stressed Teachings: N={n_taught} genuine misclassifications")
    print(f"  * Initial Voucher Weighting: {n_w2} stable (w=2.0), {n_w1} rescued plausible (w=1.0), {n_w01} implausible (w=0.1)")
    print(f"  * Pre-Sleep Baseline: Clean = {base_metrics['clean_acc']*100:.2f}%, Hard = {base_metrics['hard_acc']*100:.2f}%, ECE = {base_metrics['ece']:.4f}, Taught Conf = {base_metrics['taught_confusion']*100:.2f}%")

    print("\n" + "=" * 84)
    print("PRE-REGISTERED HYPOTHESES")
    print("=" * 84)
    print("  [H1] Digestion & Healing: Honest teaching yields Hard-Subset Delta >= 0 AND delta Taught Conf < 0.")
    print("  [H2] Immune Robustness: 30% Liar attack damage <= 0.50pp on clean and hard-subset accuracy.")
    print("  [H3] Generative Rescue: Plausible-correction bucket (w=1.0) is non-empty (>0 rescued samples).")
    print("=" * 84)

    # -------------------------------------------------------------
    # CONDITION 0: NO TEACHING (Baseline Dream Sleep Only)
    # -------------------------------------------------------------
    print("\n" + "-" * 60)
    print("CONDITION 0: NO TEACHING (Baseline Targeted Dreams)")
    print("-" * 60)
    with tempfile.TemporaryDirectory() as tmpdir:
        org_noteach = SpecimenOrganism(db_path=str(Path(tmpdir) / "noteach.db"))
        m0 = evaluate_metrics(org_noteach.classifier, org_noteach.intro_head, conscience_x, conscience_y, taught_t_x)
        noteach_history = [m0]
        print(f"Cycle 0 (Pre-Sleep): Hard Acc = {m0['hard_acc']*100:.2f}%, Clean Acc = {m0['clean_acc']*100:.2f}%, Taught Conf = {m0['taught_confusion']*100:.2f}%")

        for cycle in range(1, 4):
            sleep_res = org_noteach.sleep_cycle()
            m = evaluate_metrics(org_noteach.classifier, org_noteach.intro_head, conscience_x, conscience_y, taught_t_x)
            noteach_history.append(m)
            d_hard = (m['hard_acc'] - m0['hard_acc']) * 100.0
            print(f"Cycle {cycle} (GEN {sleep_res['generation']}): Hard Acc = {m['hard_acc']*100:.2f}% (delta: {d_hard:+.2f}pp), Clean Acc = {m['clean_acc']*100:.2f}%, Taught Conf = {m['taught_confusion']*100:.2f}%")

    # -------------------------------------------------------------
    # CONDITION 1: HONEST TEACHERS (100% True Labels)
    # -------------------------------------------------------------
    print("\n" + "-" * 60)
    print("CONDITION 1: HONEST TEACHERS (100% True Corrections)")
    print("-" * 60)
    with tempfile.TemporaryDirectory() as tmpdir:
        org_honest = SpecimenOrganism(db_path=str(Path(tmpdir) / "honest.db"))
        m0_h = evaluate_metrics(org_honest.classifier, org_honest.intro_head, conscience_x, conscience_y, taught_t_x)
        honest_history = [m0_h]
        print(f"Cycle 0 (Pre-Sleep): Hard Acc = {m0_h['hard_acc']*100:.2f}%, Clean Acc = {m0_h['clean_acc']*100:.2f}%, Taught Conf = {m0_h['taught_confusion']*100:.2f}%")

        # Teach all 419 samples into organism
        for i in range(n_taught):
            org_honest.feed(taught_x_np[i].reshape(28, 28), visitor_id=f"honest_{i}")
            org_honest.reveal(true_label=int(taught_y_np[i]), visitor_id=f"honest_{i}")

        for cycle in range(1, 4):
            sleep_res = org_honest.sleep_cycle()
            m = evaluate_metrics(org_honest.classifier, org_honest.intro_head, conscience_x, conscience_y, taught_t_x)
            honest_history.append(m)
            d_hard = (m['hard_acc'] - m0_h['hard_acc']) * 100.0
            print(f"Cycle {cycle} (GEN {sleep_res['generation']}): Hard Acc = {m['hard_acc']*100:.2f}% (delta: {d_hard:+.2f}pp), Clean Acc = {m['clean_acc']*100:.2f}%, Taught Conf = {m['taught_confusion']*100:.2f}%")

    # -------------------------------------------------------------
    # CONDITION 2: 30% LIAR ADVERSARIES (30% Corrupted Labels)
    # -------------------------------------------------------------
    print("\n" + "-" * 60)
    print("CONDITION 2: 30% LIAR ADVERSARIES (30% Corrupted Labels)")
    print("-" * 60)
    with tempfile.TemporaryDirectory() as tmpdir:
        org_liar = SpecimenOrganism(db_path=str(Path(tmpdir) / "liar.db"))
        m0_l = evaluate_metrics(org_liar.classifier, org_liar.intro_head, conscience_x, conscience_y, taught_t_x)
        liar_history = [m0_l]

        rng_liar = np.random.RandomState(42)
        corrupted_y = taught_y_np.copy()
        n_liars = int(round(n_taught * 0.30))
        liar_idx = rng_liar.choice(n_taught, size=n_liars, replace=False)
        for idx in liar_idx:
            corrupted_y[idx] = (corrupted_y[idx] + rng_liar.randint(1, 10)) % 10

        print(f"Injected {n_liars}/{n_taught} (30.0%) synthetic adversary liar labels.")

        for i in range(n_taught):
            org_liar.feed(taught_x_np[i].reshape(28, 28), visitor_id=f"visitor_{i}")
            org_liar.reveal(true_label=int(corrupted_y[i]), visitor_id=f"visitor_{i}")

        for cycle in range(1, 4):
            sleep_res = org_liar.sleep_cycle()
            m = evaluate_metrics(org_liar.classifier, org_liar.intro_head, conscience_x, conscience_y, taught_t_x)
            liar_history.append(m)
            d_hard = (m['hard_acc'] - m0_l['hard_acc']) * 100.0
            print(f"Cycle {cycle} (GEN {sleep_res['generation']}): Hard Acc = {m['hard_acc']*100:.2f}% (delta: {d_hard:+.2f}pp), Clean Acc = {m['clean_acc']*100:.2f}%, Taught Conf = {m['taught_confusion']*100:.2f}%")

    # -------------------------------------------------------------
    # STATISTICAL ANALYSIS & CONFIDENCE INTERVALS
    # -------------------------------------------------------------
    # Paired CIs on final cycle (Cycle 3 vs Cycle 0)
    h_c0 = honest_history[0]
    h_c3 = honest_history[3]
    h_clean_d, h_clean_lo, h_clean_hi, h_clean_sig = compute_paired_ci(h_c0["clean_correct"], h_c3["clean_correct"])
    h_hard_d, h_hard_lo, h_hard_hi, h_hard_sig = compute_paired_ci(h_c0["hard_correct"], h_c3["hard_correct"])
    h_conf_d = (h_c3["taught_confusion"] - h_c0["taught_confusion"]) * 100.0

    l_c0 = liar_history[0]
    l_c3 = liar_history[3]
    l_clean_d, l_clean_lo, l_clean_hi, l_clean_sig = compute_paired_ci(l_c0["clean_correct"], l_c3["clean_correct"])
    l_hard_d, l_hard_lo, l_hard_hi, l_hard_sig = compute_paired_ci(l_c0["hard_correct"], l_c3["hard_correct"])
    l_conf_d = (l_c3["taught_confusion"] - l_c0["taught_confusion"]) * 100.0

    nt_c0 = noteach_history[0]
    nt_c3 = noteach_history[3]
    nt_clean_d, nt_clean_lo, nt_clean_hi, nt_clean_sig = compute_paired_ci(nt_c0["clean_correct"], nt_c3["clean_correct"])
    nt_hard_d, nt_hard_lo, nt_hard_hi, nt_hard_sig = compute_paired_ci(nt_c0["hard_correct"], nt_c3["hard_correct"])

    # -------------------------------------------------------------
    # GENERATE PLOT: figures/exchange_v31_curve.png
    # -------------------------------------------------------------
    cycles = [0, 1, 2, 3]
    noteach_hard = [m["hard_acc"] * 100 for m in noteach_history]
    honest_hard = [m["hard_acc"] * 100 for m in honest_history]
    liar_hard = [m["hard_acc"] * 100 for m in liar_history]
    baseline_ref = h_c0["hard_acc"] * 100

    plt.figure(figsize=(9, 5.5), dpi=150)
    plt.axhline(baseline_ref, color="#888888", linestyle="--", linewidth=1.5, label=f"Pre-Sleep Baseline ({baseline_ref:.1f}%)")
    plt.plot(cycles, noteach_hard, marker="^", color="#7952b3", linewidth=2.0, linestyle=":", markersize=7, label="No Teaching (Baseline Dreams)")
    plt.plot(cycles, honest_hard, marker="o", color="#00FF9D", linewidth=2.6, markersize=8, label="Honest Visitors + Second Opinion (100% True)")
    plt.plot(cycles, liar_hard, marker="s", color="#FF3B30", linewidth=2.2, markersize=7, label="30% Liar Attack + Second Opinion")

    plt.grid(True, linestyle=":", alpha=0.5, color="#444444")
    plt.legend(frameon=True, facecolor="#161b22", edgecolor="#30363d", labelcolor="white", loc="lower right", fontsize=10)
    plt.gca().set_facecolor("#0d1117")
    plt.gcf().patch.set_facecolor("#0d1117")
    plt.tick_params(colors="white")
    plt.xticks(cycles, ["Cycle 0\n(Pre-Sleep)", "Cycle 1\n(GEN 2)", "Cycle 2\n(GEN 3)", "Cycle 3\n(GEN 4)"], color="white", fontsize=10)
    plt.yticks(color="white", fontsize=10)
    plt.title("SOMNIA v3.1 // The Exchange: Hard-Subset Digestion & Healing", color="white", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Sleep Consolidation Cycles", color="white", fontsize=11)
    plt.ylabel("Hard-Subset Accuracy (%) [N=200]", color="white", fontsize=11)
    plt.tight_layout()

    fig_path = figures_dir / "exchange_v31_curve.png"
    plt.savefig(fig_path, dpi=150, facecolor="#0d1117")
    plt.close()
    print(f"\n[FIGURE SAVED] Saved digestion curve to: {fig_path}")

    # -------------------------------------------------------------
    # SAVE JSON RESULTS: results/exchange_v31.json
    # -------------------------------------------------------------
    git_hash = get_git_hash()
    
    # Check Hypotheses
    h1_passed = bool((h_hard_d >= 0.0) and (h_conf_d < 0.0))
    h2_passed = bool((abs(l_clean_d) <= 0.50) and (abs(l_hard_d) <= 0.50))
    h3_passed = bool(n_w1 > 0)

    clean_d_str = f"{h_clean_d:+.2f}pp [95% CI: {h_clean_lo:+.2f}, {h_clean_hi:+.2f}]" + ("" if h_clean_sig else " (not sig)")
    hard_d_str = f"{h_hard_d:+.2f}pp [95% CI: {h_hard_lo:+.2f}, {h_hard_hi:+.2f}]" + ("" if h_hard_sig else " (not sig)")

    results_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": git_hash,
        "instruments": {
            "n_conscience_digits": 1000,
            "n_hard_subset": base_metrics["hard_count"],
            "hard_subset_definition": "lowest 20% margin (top1 - top2)",
            "n_taught_samples": n_taught,
            "n_liar_samples": n_liars,
            "seed_model": 0,
            "seed_conscience": 7,
            "seed_stress": 42,
            "voucher_tolerance_ratio": 1.10,
        },
        "voucher_weights_initial": {
            "stable_w2": n_w2,
            "plausible_w1": n_w1,
            "implausible_w01": n_w01,
            "plausible_rescue_rate": n_w1 / n_taught,
        },
        "baseline_pre_sleep": {
            "clean_acc": base_metrics["clean_acc"],
            "hard_acc": base_metrics["hard_acc"],
            "ece": base_metrics["ece"],
            "taught_confusion": base_metrics["taught_confusion"],
        },
        "conditions": {
            "no_teaching": {
                "cycle_0": {k: v for k, v in noteach_history[0].items() if not isinstance(v, (np.ndarray, list))},
                "cycle_1": {k: v for k, v in noteach_history[1].items() if not isinstance(v, (np.ndarray, list))},
                "cycle_2": {k: v for k, v in noteach_history[2].items() if not isinstance(v, (np.ndarray, list))},
                "cycle_3": {k: v for k, v in noteach_history[3].items() if not isinstance(v, (np.ndarray, list))},
                "clean_delta_pp": nt_clean_d,
                "hard_delta_pp": nt_hard_d,
            },
            "honest_teaching": {
                "cycle_0": {k: v for k, v in honest_history[0].items() if not isinstance(v, (np.ndarray, list))},
                "cycle_1": {k: v for k, v in honest_history[1].items() if not isinstance(v, (np.ndarray, list))},
                "cycle_2": {k: v for k, v in honest_history[2].items() if not isinstance(v, (np.ndarray, list))},
                "cycle_3": {k: v for k, v in honest_history[3].items() if not isinstance(v, (np.ndarray, list))},
                "clean_delta_pp": h_clean_d,
                "clean_ci_95": [h_clean_lo, h_clean_hi],
                "clean_significant": h_clean_sig,
                "hard_delta_pp": h_hard_d,
                "hard_ci_95": [h_hard_lo, h_hard_hi],
                "hard_significant": h_hard_sig,
                "taught_conf_delta_pp": h_conf_d,
            },
            "liar_teaching": {
                "cycle_0": {k: v for k, v in liar_history[0].items() if not isinstance(v, (np.ndarray, list))},
                "cycle_1": {k: v for k, v in liar_history[1].items() if not isinstance(v, (np.ndarray, list))},
                "cycle_2": {k: v for k, v in liar_history[2].items() if not isinstance(v, (np.ndarray, list))},
                "cycle_3": {k: v for k, v in liar_history[3].items() if not isinstance(v, (np.ndarray, list))},
                "clean_delta_pp": l_clean_d,
                "clean_ci_95": [l_clean_lo, l_clean_hi],
                "clean_significant": l_clean_sig,
                "hard_delta_pp": l_hard_d,
                "hard_ci_95": [l_hard_lo, l_hard_hi],
                "hard_significant": l_hard_sig,
                "taught_conf_delta_pp": l_conf_d,
            },
        },
        "hypotheses_verdict": {
            "H1_digestion_healing": {
                "statement": "Honest teaching hard-subset delta >= 0 AND taught confusion drops",
                "result": h1_passed,
                "hard_delta_pp": h_hard_d,
                "taught_conf_delta_pp": h_conf_d,
            },
            "H2_immune_robustness": {
                "statement": "30% Liar attack damage <= 0.50pp",
                "result": h2_passed,
                "clean_damage_pp": abs(l_clean_d),
                "hard_damage_pp": abs(l_hard_d),
            },
            "H3_generative_rescue": {
                "statement": "Plausible corrections rescued by cVAE (w=1.0 count > 0)",
                "result": h3_passed,
                "rescued_count": n_w1,
                "rescue_rate": n_w1 / n_taught,
            },
        }
    }

    json_path = results_dir / "exchange_v31.json"
    with open(json_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"[JSON SAVED] Saved structured experiment results to: {json_path}")

    # -------------------------------------------------------------
    # PRINT ACCEPTANCE RESEARCH TABLE
    # -------------------------------------------------------------
    print("\n" + "=" * 90)
    print("SOMNIA v3.1: THE EXCHANGE DIGESTION SUMMARY TABLE")
    print("=" * 90)
    print(f"{'Condition':<24} | {'Clean Acc (N=1k)':<16} | {'Hard Acc (N=200)':<18} | {'ECE (15 bins)':<13} | {'Taught Conf':<13}")
    print("-" * 90)
    
    # Baseline
    print(f"{'Baseline (Pre-Sleep)':<24} | {base_metrics['clean_acc']*100:.2f}%           | {base_metrics['hard_acc']*100:.2f}%             | {base_metrics['ece']:.4f}        | {base_metrics['taught_confusion']*100:.2f}%")
    
    # No Teaching
    print(f"{'No Teaching (Dreams)':<24} | {nt_c3['clean_acc']*100:.2f}% ({nt_clean_d:+.2f}p)  | {nt_c3['hard_acc']*100:.2f}% ({nt_hard_d:+.2f}p)    | {nt_c3['ece']:.4f} ({nt_c3['ece']-nt_c0['ece']:+.4f}) | {nt_c3['taught_confusion']*100:.2f}% ({nt_c3['taught_confusion']-nt_c0['taught_confusion']:+.2f}p)")
    
    # Honest Teaching
    h_sig_tag = "" if h_hard_sig else " [ns]"
    print(f"{'Honest Teachers (100%)':<24} | {h_c3['clean_acc']*100:.2f}% ({h_clean_d:+.2f}p)  | {h_c3['hard_acc']*100:.2f}% ({h_hard_d:+.2f}p{h_sig_tag})| {h_c3['ece']:.4f} ({h_c3['ece']-h_c0['ece']:+.4f}) | {h_c3['taught_confusion']*100:.2f}% ({h_conf_d:+.2f}p)")
    
    # 30% Liars
    l_sig_tag = "" if l_hard_sig else " [ns]"
    print(f"{'30% Liar Adversaries':<24} | {l_c3['clean_acc']*100:.2f}% ({l_clean_d:+.2f}p)  | {l_c3['hard_acc']*100:.2f}% ({l_hard_d:+.2f}p{l_sig_tag})| {l_c3['ece']:.4f} ({l_c3['ece']-l_c0['ece']:+.4f}) | {l_c3['taught_confusion']*100:.2f}% ({l_conf_d:+.2f}p)")
    print("=" * 90)

    print("\nSTATISTICAL RIGOR & 95% CONFIDENCE INTERVALS:")
    print(f"  * Honest Clean Delta: {clean_d_str}")
    print(f"  * Honest Hard Delta:  {hard_d_str}")
    print(f"  * Honest Taught Confusion: {h_c0['taught_confusion']*100:.2f}% -> {h_c3['taught_confusion']*100:.2f}% ({h_conf_d:+.2f}pp)")
    print(f"  * Second Opinion Voucher Rescue: {n_w1}/{n_taught} ({n_w1/n_taught*100:.1f}%) corrections rescued at w=1.0.")

    print("\nHYPOTHESIS EVALUATION VERDICT:")
    print(f"  [H1] Digestion & Healing:  {'PASSED' if h1_passed else 'NEGATIVE RESULT'} (Hard delta: {h_hard_d:+.2f}pp, Conf delta: {h_conf_d:+.2f}pp)")
    print(f"  [H2] Immune Robustness:    {'PASSED' if h2_passed else 'FAILED'} (Max liar damage: {max(abs(l_clean_d), abs(l_hard_d)):.2f}pp <= 0.50pp)")
    print(f"  [H3] Generative Rescue:    {'PASSED' if h3_passed else 'FAILED'} ({n_w1} corrections assigned w=1.0 via cVAE plausibility)")
    print("=" * 90)


if __name__ == "__main__":
    run_exchange_v31_experiment()
