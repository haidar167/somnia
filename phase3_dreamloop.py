"""Phase 3 -- The Dream Loop: Targeted vs Random vs No Sleep."""

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
from somnia.models import ClassifierMLP, VAE, IntrospectiveHead
from somnia.dreamer import Dreamer
from somnia.sleep import StabilityFilter, SleepConsolidation, identify_hard_subset


def evaluate(classifier, test_x_np, test_y_np, indices=None):
    """Evaluate accuracy on full test set or subset."""
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


def run_dream_loop(condition, classifier_orig, vae, intro_head,
                   train_x, train_y, test_x, test_y,
                   hard_indices, n_cycles=5, n_dreams=500,
                   real_batch=2000):
    """Run n_cycles of sleep consolidation under a given condition.

    Conditions:
        'none': No dreams, only real data replay.
        'random': Random dreams (uniform latent sampling).
        'targeted': Top-confusion dreams (nightmare-targeted).
    """
    classifier = copy.deepcopy(classifier_orig)
    dreamer = Dreamer(classifier, vae, intro_head)
    stability = StabilityFilter(classifier)
    consolidator = SleepConsolidation(classifier, dream_weight=0.25, lr=5e-4, epochs=2)

    clean_accs = []
    hard_accs = []

    # Initial accuracy
    clean_accs.append(evaluate(classifier, test_x, test_y))
    hard_accs.append(evaluate(classifier, test_x, test_y, hard_indices))

    for cycle in range(n_cycles):
        # Sample real mini-batch for replay
        idx = np.random.choice(len(train_x), size=min(real_batch, len(train_x)), replace=False)
        real_x = torch.from_numpy(train_x[idx])
        real_y = torch.from_numpy(train_y[idx])

        if condition == "none":
            dream_x = torch.zeros(0, 784)
            dream_y = torch.zeros(0, dtype=torch.long)
        elif condition == "random":
            # Random latent sampling
            z = torch.randn(n_dreams, vae.latent_dim)
            with torch.no_grad():
                raw_dreams = vae.decode(z)
            filtered, labels, keep, _ = stability.filter(raw_dreams, seed=cycle)
            dream_x = filtered
            dream_y = labels
        elif condition == "targeted":
            # Confusion-targeted sampling
            dreams_all, z_all, confusion, _, _ = dreamer.dream(n=n_dreams * 2, seed=cycle + 100)
            # Select top-confusion dreams
            _, top_idx = confusion.topk(n_dreams)
            targeted = dreams_all[top_idx]
            filtered, labels, keep, _ = stability.filter(targeted, seed=cycle)
            dream_x = filtered
            dream_y = labels

        # Sleep
        consolidator.sleep_cycle(real_x, real_y, dream_x, dream_y)

        # Evaluate
        c_acc = evaluate(classifier, test_x, test_y)
        h_acc = evaluate(classifier, test_x, test_y, hard_indices)
        clean_accs.append(c_acc)
        hard_accs.append(h_acc)

        n_dreams_used = dream_x.shape[0] if condition != "none" else 0
        print(f"    Cycle {cycle+1}/{n_cycles}: clean={c_acc:.4f}, hard={h_acc:.4f}, "
              f"dreams_used={n_dreams_used}")

    return clean_accs, hard_accs


def plot_dream_curve(results, save_path):
    """Plot accuracy curves across sleep cycles for all conditions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = {"none": "#888888", "random": "#4ECDC4", "targeted": "#FF6B6B"}
    labels = {"none": "No Sleep", "random": "Random Dreams", "targeted": "Targeted Dreams"}
    markers = {"none": "s", "random": "^", "targeted": "o"}

    cycles = list(range(len(results["none"]["clean"])))

    for cond in ["none", "random", "targeted"]:
        ax1.plot(cycles, results[cond]["clean"], color=colors[cond],
                 marker=markers[cond], label=labels[cond], linewidth=2, markersize=6)
        ax2.plot(cycles, results[cond]["hard"], color=colors[cond],
                 marker=markers[cond], label=labels[cond], linewidth=2, markersize=6)

    ax1.set_xlabel("Sleep Cycle", fontsize=11)
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.set_title("Clean Test Accuracy", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Sleep Cycle", fontsize=11)
    ax2.set_ylabel("Accuracy", fontsize=11)
    ax2.set_title("Hard Subset Accuracy", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Phase 3 -- Dream Loop: Self-Healing Comparison",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Dream curve saved -> {save_path}")


def main():
    print("=" * 60)
    print("SOMNIA -- Phase 3: The Dream Loop")
    print("=" * 60)

    set_seed(0)
    root = project_root()

    # Load data
    print("\n[1/4] Loading data and models...")
    train_x, train_y, test_x, test_y = load_mnist()

    classifier = ClassifierMLP()
    classifier.load_state_dict(torch.load(str(root / "data" / "classifier.pt"),
                                          weights_only=True))
    vae = VAE()
    vae.load_state_dict(torch.load(str(root / "data" / "vae.pt"),
                                   weights_only=True))
    intro_head = IntrospectiveHead()
    intro_head.load_state_dict(torch.load(str(root / "data" / "intro_head.pt"),
                                          weights_only=True))

    # Identify hard subset
    print("\n[2/4] Identifying hard test subset (500 lowest-margin samples)...")
    hard_indices = identify_hard_subset(classifier, test_x, test_y, n_hard=500)
    baseline_hard = evaluate(classifier, test_x, test_y, hard_indices)
    baseline_clean = evaluate(classifier, test_x, test_y)
    print(f"  Baseline clean accuracy: {baseline_clean:.4f}")
    print(f"  Baseline hard accuracy:  {baseline_hard:.4f}")

    # Run dream loops
    print("\n[3/4] Running dream loops (5 cycles each)...")
    results = {}

    for cond in ["none", "random", "targeted"]:
        print(f"\n  --- Condition: {cond.upper()} ---")
        set_seed(0)  # Reset seed for fair comparison
        clean, hard = run_dream_loop(
            cond, classifier, vae, intro_head,
            train_x, train_y, test_x, test_y,
            hard_indices, n_cycles=5, n_dreams=500, real_batch=2000
        )
        results[cond] = {"clean": clean, "hard": hard}

    # Print comparison table
    print("\n" + "-" * 60)
    print(f"{'Condition':<15} {'Clean Start':>12} {'Clean End':>12} {'Hard Start':>12} {'Hard End':>12}")
    print("-" * 60)
    for cond in ["none", "random", "targeted"]:
        cs = results[cond]["clean"][0]
        ce = results[cond]["clean"][-1]
        hs = results[cond]["hard"][0]
        he = results[cond]["hard"][-1]
        print(f"{cond:<15} {cs:>12.4f} {ce:>12.4f} {hs:>12.4f} {he:>12.4f}")
    print("-" * 60)

    # Honest reporting
    targ_hard_end = results["targeted"]["hard"][-1]
    rand_hard_end = results["random"]["hard"][-1]
    none_hard_end = results["none"]["hard"][-1]

    if targ_hard_end > rand_hard_end:
        print("\n  [RESULT] Targeted dreams OUTPERFORM random dreams on hard subset!")
    elif targ_hard_end < rand_hard_end:
        print("\n  [HONEST WARNING] Random dreams outperform targeted dreams on hard subset.")
        print("  This may indicate the introspective head needs better calibration.")
    else:
        print("\n  [RESULT] Targeted and random dreams perform equally on hard subset.")

    # Plot
    print("\n[4/4] Plotting results...")
    plot_path = str(root / "figures" / "phase3_dream_curve.png")
    plot_dream_curve(results, plot_path)

    # Save
    metrics = {
        "phase": 3,
        "n_cycles": 5,
        "n_dreams_per_cycle": 500,
        "conditions": {},
    }
    for cond in ["none", "random", "targeted"]:
        metrics["conditions"][cond] = {
            "clean_acc_trajectory": results[cond]["clean"],
            "hard_acc_trajectory": results[cond]["hard"],
            "clean_acc_final": results[cond]["clean"][-1],
            "hard_acc_final": results[cond]["hard"][-1],
            "clean_acc_delta": results[cond]["clean"][-1] - results[cond]["clean"][0],
            "hard_acc_delta": results[cond]["hard"][-1] - results[cond]["hard"][0],
        }
    metrics["targeted_beats_random_hard"] = bool(targ_hard_end > rand_hard_end)
    metrics["figures"] = ["figures/phase3_dream_curve.png"]

    save_results(str(root / "results" / "phase3.json"), metrics, seed=0)

    print("\n" + "=" * 60)
    print("Phase 3 COMPLETE")
    print(f"  Targeted hard final: {targ_hard_end:.4f}")
    print(f"  Random hard final:   {rand_hard_end:.4f}")
    print(f"  No-sleep hard final: {none_hard_end:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
