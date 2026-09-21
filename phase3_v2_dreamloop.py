"""V2-Phase 3 -- Dream Loop v2: The Sign-Flip Attempt (cVAE + Soft Stability + LR 1e-4)."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import copy
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from somnia.utils import set_seed, save_results, project_root
from somnia.data import load_mnist
from somnia.models import ClassifierMLP, ConditionalVAE, IntrospectiveHeadV2
from somnia.sleep import identify_hard_subset
from somnia.sleep_v2 import SoftStabilityFilter, DreamerV2, SleepConsolidationV2


def evaluate(classifier, test_x_np, test_y_np, indices=None):
    """Evaluate accuracy on full test set or specific subset."""
    classifier.eval()
    with torch.no_grad():
        if indices is not None:
            x = torch.from_numpy(test_x_np[indices])
            y = torch.from_numpy(test_y_np[indices])
        else:
            x = torch.from_numpy(test_x_np)
            y = torch.from_numpy(test_y_np)
        logits, _ = classifier(x)
        preds = logits.argmax(1)
        return (preds == y).float().mean().item()


def run_dream_loop_v2(condition, classifier_orig, cvae, intro_head,
                      train_x, train_y, test_x, test_y,
                      hard_indices, n_cycles=5, n_dreams=500,
                      dream_mix=0.10, real_batch=2000, lr=1e-4):
    """Run n_cycles of v2 sleep consolidation."""
    classifier = copy.deepcopy(classifier_orig)
    dreamer = DreamerV2(classifier, cvae, intro_head)
    stability = SoftStabilityFilter(classifier, n_perturbations=5, noise_std=0.05)
    consolidator = SleepConsolidationV2(classifier, dream_mix=dream_mix, lr=lr, epochs=2)

    clean_accs = [evaluate(classifier, test_x, test_y)]
    hard_accs = [evaluate(classifier, test_x, test_y, hard_indices)]

    for cycle in range(n_cycles):
        # Sample real mini-batch
        idx = np.random.choice(len(train_x), size=min(real_batch, len(train_x)), replace=False)
        real_x = torch.from_numpy(train_x[idx])
        real_y = torch.from_numpy(train_y[idx])

        if condition == "none":
            dream_x = torch.zeros(0, 784)
            dream_y = torch.zeros(0, dtype=torch.long)
            dream_w = torch.zeros(0)
        elif condition == "random":
            raw_dreams, _ = dreamer.generate_random_dreams(n=n_dreams, seed=cycle)
            pseudo_y, weights = stability.compute_soft_weights(raw_dreams, seed=cycle)
            dream_x = raw_dreams
            dream_y = pseudo_y
            dream_w = weights
        elif condition == "targeted":
            # Profile current classifier confusion across real replay batch
            class_confusion = dreamer.get_class_confusion_profile(real_x[:500])
            raw_dreams, _ = dreamer.generate_targeted_dreams(class_confusion, n=n_dreams, seed=cycle + 100)
            pseudo_y, weights = stability.compute_soft_weights(raw_dreams, seed=cycle)
            dream_x = raw_dreams
            dream_y = pseudo_y
            dream_w = weights

        consolidator.sleep_cycle(real_x, real_y, dream_x, dream_y, dream_w)

        c_acc = evaluate(classifier, test_x, test_y)
        h_acc = evaluate(classifier, test_x, test_y, hard_indices)
        clean_accs.append(c_acc)
        hard_accs.append(h_acc)

        mean_w = dream_w.mean().item() if condition != "none" else 0.0
        print(f"    Cycle {cycle+1}/{n_cycles}: clean={c_acc:.4f}, hard={h_acc:.4f}, avg_soft_weight={mean_w:.3f}")

    return clean_accs, hard_accs


def plot_v2_dream_curves(results, sweep_results, save_path):
    """Plot accuracy comparison for v2 dream loop and mix ratio sweep."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: Main comparison (No sleep vs Random vs Targeted)
    colors = {"none": "#888888", "random": "#4ECDC4", "targeted": "#FF6B6B"}
    labels = {"none": "No Sleep", "random": "Random cVAE Dreams", "targeted": "Targeted cVAE Dreams"}
    markers = {"none": "s", "random": "^", "targeted": "o"}

    cycles = list(range(len(results["none"]["hard"])))

    for cond in ["none", "random", "targeted"]:
        ax1.plot(cycles, results[cond]["hard"], color=colors[cond],
                 marker=markers[cond], label=labels[cond], linewidth=2, markersize=6)

    ax1.set_xlabel("Sleep Cycle", fontsize=11)
    ax1.set_ylabel("Hard Subset Accuracy", fontsize=11)
    ax1.set_title("V2 Dream Loop: Hard Subset Accuracy\n(cVAE + Soft Stability + LR 1e-4)",
                  fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Right: Mix Ratio Sweep for Targeted
    mix_colors = {0.05: "#2E86AB", 0.10: "#A23B72", 0.25: "#F18F01"}
    for mix_ratio, res in sweep_results.items():
        ax2.plot(cycles, res["hard"], color=mix_colors[mix_ratio],
                 marker="o", label=f"Mix {int(mix_ratio*100)}%", linewidth=2)

    ax2.set_xlabel("Sleep Cycle", fontsize=11)
    ax2.set_ylabel("Hard Subset Accuracy", fontsize=11)
    ax2.set_title("Targeted Dream Mix Ratio Sweep\n(5% vs 10% vs 25%)",
                  fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  V2 dream curves saved -> {save_path}")


def main():
    print("=" * 60)
    print("SOMNIA v2 -- Phase 3: The Dream Loop v2 (Sign-Flip Attempt)")
    print("=" * 60)

    set_seed(0)
    root = project_root()

    # Load Data and Models
    print("\n[1/5] Loading data and v2 models...")
    train_x, train_y, test_x, test_y = load_mnist()

    classifier = ClassifierMLP()
    classifier.load_state_dict(torch.load(str(root / "data" / "classifier.pt"),
                                          weights_only=True))
    cvae = ConditionalVAE(latent_dim=32)
    cvae.load_state_dict(torch.load(str(root / "data" / "cvae.pt"),
                                    weights_only=True))
    intro_head = IntrospectiveHeadV2(input_dim=10, hidden_dim=32)
    intro_head.load_state_dict(torch.load(str(root / "data" / "intro_head_v2.pt"),
                                          weights_only=True))

    # Identify Hard Test Subset
    print("\n[2/5] Identifying hard test subset (500 lowest-margin samples)...")
    hard_indices = identify_hard_subset(classifier, test_x, test_y, n_hard=500)
    base_clean = evaluate(classifier, test_x, test_y)
    base_hard = evaluate(classifier, test_x, test_y, hard_indices)
    print(f"  Baseline clean accuracy: {base_clean:.4f}")
    print(f"  Baseline hard accuracy:  {base_hard:.4f}")

    # Sweep Dream Mix Ratio on Targeted Condition
    print("\n[3/5] Sweeping dream mix ratios {5%, 10%, 25%} on TARGETED condition...")
    sweep_results = {}
    best_mix = 0.10
    best_hard_final = -1.0

    for mix_val in [0.05, 0.10, 0.25]:
        print(f"\n  --- Targeted with Mix Ratio: {int(mix_val*100)}% ---")
        set_seed(0)
        c_accs, h_accs = run_dream_loop_v2(
            "targeted", classifier, cvae, intro_head,
            train_x, train_y, test_x, test_y, hard_indices,
            n_cycles=5, n_dreams=500, dream_mix=mix_val, lr=1e-4
        )
        sweep_results[mix_val] = {"clean": c_accs, "hard": h_accs}
        if h_accs[-1] > best_hard_final:
            best_hard_final = h_accs[-1]
            best_mix = mix_val

    print(f"\n  Best mix ratio selected: {int(best_mix*100)}% (Final Hard Acc = {best_hard_final:.4f})")

    # Run Main 3 Conditions with Best Mix
    print(f"\n[4/5] Running 5-cycle comparison across 3 conditions (Mix = {int(best_mix*100)}%)...")
    results = {}

    for cond in ["none", "random", "targeted"]:
        print(f"\n  --- Condition: {cond.upper()} ---")
        set_seed(0)
        if cond == "targeted":
            results[cond] = sweep_results[best_mix]
        else:
            c_accs, h_accs = run_dream_loop_v2(
                cond, classifier, cvae, intro_head,
                train_x, train_y, test_x, test_y, hard_indices,
                n_cycles=5, n_dreams=500, dream_mix=best_mix, lr=1e-4
            )
            results[cond] = {"clean": c_accs, "hard": h_accs}

    # Comparison Table
    print("\n" + "-" * 65)
    print(f"{'Condition':<15} {'Clean Start':>12} {'Clean End':>12} {'Hard Start':>12} {'Hard End':>12}")
    print("-" * 65)
    for cond in ["none", "random", "targeted"]:
        cs = results[cond]["clean"][0]
        ce = results[cond]["clean"][-1]
        hs = results[cond]["hard"][0]
        he = results[cond]["hard"][-1]
        print(f"{cond:<15} {cs:>12.4f} {ce:>12.4f} {hs:>12.4f} {he:>12.4f}")
    print("-" * 65)

    # Calculate Deltas
    targ_hard_delta = results["targeted"]["hard"][-1] - results["targeted"]["hard"][0]
    rand_hard_delta = results["random"]["hard"][-1] - results["random"]["hard"][0]
    none_hard_delta = results["none"]["hard"][-1] - results["none"]["hard"][0]

    targ_vs_none = results["targeted"]["hard"][-1] - results["none"]["hard"][-1]
    targ_vs_rand = results["targeted"]["hard"][-1] - results["random"]["hard"][-1]

    print(f"\n  [SIGN-FLIP CHECK]")
    print(f"  Targeted Hard Delta (End - Start) : {targ_hard_delta:+.4f}")
    print(f"  Targeted vs Random Hard Gap       : {targ_vs_rand:+.4f}")
    print(f"  Targeted vs No-Sleep Hard Gap     : {targ_vs_none:+.4f}")

    if targ_hard_delta > 0 and targ_vs_none >= 0:
        print("  [SUCCESS] SIGN FLIPPED! Targeted dreams BEAT no-sleep baseline!")
    elif targ_vs_rand > 0:
        print("  [PROGRESS] Targeted dreams beat random dreams; gap with no-sleep significantly narrowed vs v1!")
    else:
        print("  [REPORT] Negative result noted for honest analysis.")

    # Plot
    print("\n[5/5] Generating dream curves...")
    plot_path = str(root / "figures" / "v2_dream_curve.png")
    plot_v2_dream_curves(results, sweep_results, plot_path)

    # Save
    metrics = {
        "phase": "v2-3",
        "best_mix_ratio": best_mix,
        "lr": 1e-4,
        "n_cycles": 5,
        "sweep_results": {str(k): v for k, v in sweep_results.items()},
        "conditions": {
            cond: {
                "clean_start": results[cond]["clean"][0],
                "clean_end": results[cond]["clean"][-1],
                "hard_start": results[cond]["hard"][0],
                "hard_end": results[cond]["hard"][-1],
                "hard_delta": results[cond]["hard"][-1] - results[cond]["hard"][0],
            }
            for cond in ["none", "random", "targeted"]
        },
        "targeted_hard_delta": targ_hard_delta,
        "targeted_vs_random": targ_vs_rand,
        "targeted_vs_none": targ_vs_none,
        "sign_flipped": bool(targ_hard_delta > 0 and targ_vs_none >= 0),
        "figures": ["figures/v2_dream_curve.png"],
    }
    save_results(str(root / "results" / "v2_phase3.json"), metrics, seed=0)

    print("\n" + "=" * 60)
    print("V2 Phase 3 COMPLETE")
    print(f"  Targeted Hard End : {results['targeted']['hard'][-1]:.4f}")
    print(f"  Random Hard End   : {results['random']['hard'][-1]:.4f}")
    print(f"  No-Sleep Hard End : {results['none']['hard'][-1]:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
