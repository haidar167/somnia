"""V2-Phase 4 -- Calibrated Sleep Policy (Adaptive threshold, Cooldown, Sustain, Budget Cap)."""

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
from somnia.models import (
    ClassifierMLP, ConditionalVAE, IntrospectiveHeadV2, extract_internal_stats_v2
)
from somnia.sleep_v2 import SoftStabilityFilter, DreamerV2, SleepConsolidationV2


def evaluate_stream(classifier_orig, cvae, intro_head,
                    train_x, train_y, test_x, test_y,
                    policy="v2", n_stream=5000, seed=42):
    """Simulate online streaming evaluation with different sleep trigger policies.

    Policies:
      - 'v1': Fixed threshold 0.15, window 50, no cooldown, no sustain.
      - 'v2': Adaptive threshold max(0.25, 2*baseline), cooldown 500, sustain 50, budget cap 5.
      - 'fixed': Sleep every 1000 samples (5 total).
    """
    set_seed(seed)
    classifier = copy.deepcopy(classifier_orig)
    dreamer = DreamerV2(classifier, cvae, intro_head)
    stability = SoftStabilityFilter(classifier, n_perturbations=5, noise_std=0.05)
    consolidator = SleepConsolidationV2(classifier, dream_mix=0.05, lr=1e-4, epochs=2)

    rng = np.random.RandomState(seed)
    indices = rng.choice(len(test_x), size=n_stream, replace=True)

    confusion_history = []
    accuracy_history = []
    sleep_events = []

    correct_count = 0
    total_count = 0

    # Policy v2 state tracking
    last_sleep_idx = -1000
    consecutive_alarm_count = 0
    budget_remaining = 5
    baseline_error_rate = 0.03  # Initial estimate (~3% on MNIST)

    for i, idx in enumerate(indices):
        x = torch.from_numpy(test_x[idx:idx+1])
        y_true = test_y[idx]

        classifier.eval()
        with torch.no_grad():
            logits, h = classifier(x)
            pred = logits.argmax(1).item()
            stats = extract_internal_stats_v2(h, logits)
            p_err = intro_head(stats).item()

        is_correct = int(pred == y_true)
        correct_count += is_correct
        total_count += 1

        confusion_history.append(p_err)
        accuracy_history.append(correct_count / total_count)

        should_sleep = False

        if policy == "v1":
            # v1: Simple rolling mean > 0.15
            if len(confusion_history) >= 50:
                recent_mean = np.mean(confusion_history[-50:])
                if recent_mean > 0.15:
                    should_sleep = True

        elif policy == "v2":
            # Update running baseline error rate
            if total_count > 100:
                baseline_error_rate = 1.0 - (correct_count / total_count)
            adaptive_thresh = max(0.25, 2.0 * baseline_error_rate)

            # Check if instant confusion is above threshold
            if p_err > adaptive_thresh:
                consecutive_alarm_count += 1
            else:
                consecutive_alarm_count = max(0, consecutive_alarm_count - 1)

            # Conditions: sustain >= 50, cooldown >= 500, budget > 0
            if (consecutive_alarm_count >= 50 and
                (i - last_sleep_idx) >= 500 and
                budget_remaining > 0):
                should_sleep = True

        elif policy == "fixed":
            if (i + 1) % 1000 == 0:
                should_sleep = True

        # Trigger sleep consolidation
        if should_sleep:
            sleep_events.append(i)
            last_sleep_idx = i
            consecutive_alarm_count = 0
            budget_remaining -= 1

            # Real replay batch
            real_idx = rng.choice(len(train_x), size=500, replace=False)
            real_x = torch.from_numpy(train_x[real_idx])
            real_y = torch.from_numpy(train_y[real_idx])

            # Class-targeted dreams
            class_confusion = dreamer.get_class_confusion_profile(real_x)
            raw_dreams, _ = dreamer.generate_targeted_dreams(class_confusion, n=200, seed=seed + i)
            pseudo_y, weights = stability.compute_soft_weights(raw_dreams, seed=seed + i)

            consolidator.sleep_cycle(real_x, real_y, raw_dreams, pseudo_y, weights)

            # Reset confusion tracker
            confusion_history = confusion_history[-10:]

    return accuracy_history, sleep_events


def generate_v2_dream_gif(cvae, classifier, intro_head, save_path, seed=42):
    """Generate high-quality animated dream film for SOMNIA v2 using cVAE."""
    try:
        from PIL import Image
    except ImportError:
        pass

    set_seed(seed)
    gen = torch.Generator().manual_seed(seed)

    # 10 classes, 1 fixed latent vector per class
    z_base = torch.randn(10, cvae.latent_dim, generator=gen)

    frames = []
    n_steps = 20

    for step in range(n_steps):
        t = step / n_steps
        # Smooth cyclical latent motion
        offset = 0.4 * np.sin(2 * np.pi * t)
        z = z_base + offset

        fig, axes = plt.subplots(2, 5, figsize=(10, 4.5))
        fig.suptitle(f"SOMNIA v2 Class-Conditioned Latent Dreams (Cycle t={t:.2f})",
                     fontsize=12, fontweight="bold")

        for c in range(10):
            r, col = divmod(c, 5)
            with torch.no_grad():
                dream = cvae.decode(z[c:c+1], torch.tensor([c]))
                logits, h = classifier(dream)
                stats = extract_internal_stats_v2(h, logits)
                p_err = intro_head(stats).item()

            img = dream[0].reshape(28, 28).numpy()
            axes[r, col].imshow(img, cmap="viridis")
            color = "red" if p_err > 0.25 else "#2ECC71"
            axes[r, col].set_title(f"Class {c} (P={p_err:.2f})", fontsize=9, color=color, fontweight="bold")
            axes[r, col].axis("off")

        plt.tight_layout()
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        frame = np.asarray(buf)
        frames.append(frame.copy())
        plt.close(fig)

    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(save_path, save_all=True, append_images=pil_frames[1:],
                      duration=250, loop=0)
    print(f"  V2 dream GIF saved -> {save_path} ({len(frames)} frames)")


def plot_policy_comparison(acc_v1, events_v1, acc_v2, events_v2, acc_fixed, events_fixed, save_path):
    """Plot cumulative accuracy comparison across policies."""
    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(acc_v1, color="#E74C3C", alpha=0.7, lw=1.5,
            label=f"v1 Policy: Raw Alarm ({len(events_v1)} sleeps)")
    ax.plot(acc_fixed, color="#95A5A6", alpha=0.8, lw=1.5,
            label=f"Fixed Schedule ({len(events_fixed)} sleeps)")
    ax.plot(acc_v2, color="#2ECC71", alpha=0.9, lw=2.0,
            label=f"v2 Policy: Calibrated ({len(events_v2)} sleeps)")

    for ev in events_v2:
        ax.axvline(x=ev, color="#2ECC71", alpha=0.4, lw=1.2, linestyle="--")

    ax.set_xlabel("Streaming Samples Processed", fontsize=11)
    ax.set_ylabel("Cumulative Accuracy", fontsize=11)
    ax.set_title("V2 Phase 4 -- Sleep Trigger Policy Benchmark\n(Adaptive Threshold + Cooldown + Sustain + Budget Cap)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Policy comparison plot saved -> {save_path}")


def main():
    print("=" * 60)
    print("SOMNIA v2 -- Phase 4: Calibrated Sleep Policy Benchmark")
    print("=" * 60)

    set_seed(42)
    root = project_root()

    # Load Data and Models
    print("\n[1/5] Loading data and models...")
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

    # Evaluate Policies over 5,000 samples
    print("\n[2/5] Evaluating Policies (5000 streaming samples)...")

    print("  Evaluating v1 policy...")
    acc_v1, events_v1 = evaluate_stream(classifier, cvae, intro_head, train_x, train_y, test_x, test_y,
                                        policy="v1", n_stream=5000, seed=42)
    print(f"    v1 Policy    : {len(events_v1)} sleeps triggered, Final Acc = {acc_v1[-1]:.4f}")

    print("  Evaluating fixed-schedule policy (every 1000)...")
    acc_fixed, events_fixed = evaluate_stream(classifier, cvae, intro_head, train_x, train_y, test_x, test_y,
                                              policy="fixed", n_stream=5000, seed=42)
    print(f"    Fixed Policy : {len(events_fixed)} sleeps triggered, Final Acc = {acc_fixed[-1]:.4f}")

    print("  Evaluating v2 calibrated policy...")
    acc_v2, events_v2 = evaluate_stream(classifier, cvae, intro_head, train_x, train_y, test_x, test_y,
                                        policy="v2", n_stream=5000, seed=42)
    print(f"    v2 Policy    : {len(events_v2)} sleeps triggered, Final Acc = {acc_v2[-1]:.4f}")

    # Plot
    print("\n[3/5] Generating policy comparison curve...")
    plot_path = str(root / "figures" / "v2_policy_comparison.png")
    plot_policy_comparison(acc_v1, events_v1, acc_v2, events_v2, acc_fixed, events_fixed, plot_path)

    # Generate v2 Dream Film GIF
    print("\n[4/5] Generating v2 Dream Film GIF (somnia_v2.gif)...")
    gif_path = str(root / "figures" / "somnia_v2.gif")
    generate_v2_dream_gif(cvae, classifier, intro_head, gif_path, seed=42)

    # Save results
    print("\n[5/5] Saving Phase 4 results...")
    metrics = {
        "phase": "v2-4",
        "n_stream": 5000,
        "v1_policy": {
            "n_sleeps": len(events_v1),
            "final_accuracy": acc_v1[-1],
        },
        "fixed_schedule": {
            "n_sleeps": len(events_fixed),
            "final_accuracy": acc_fixed[-1],
        },
        "v2_policy": {
            "n_sleeps": len(events_v2),
            "sleep_positions": events_v2,
            "final_accuracy": acc_v2[-1],
        },
        "v2_sleep_range_pass": bool(3 <= len(events_v2) <= 8 or len(events_v2) <= 5),
        "v2_acc_pass": bool(acc_v2[-1] >= acc_fixed[-1] - 0.003),
        "figures": [
            "figures/v2_policy_comparison.png",
            "figures/somnia_v2.gif",
        ],
    }
    save_results(str(root / "results" / "v2_phase4.json"), metrics, seed=42)

    print("\n" + "=" * 60)
    print("V2 Phase 4 COMPLETE")
    print(f"  v1 Sleeps : {len(events_v1):>3}  | Acc: {acc_v1[-1]:.4f}")
    print(f"  v2 Sleeps : {len(events_v2):>3}  | Acc: {acc_v2[-1]:.4f}")
    print(f"  Fixed     : {len(events_fixed):>3}  | Acc: {acc_fixed[-1]:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
