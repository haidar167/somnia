"""Phase 4 -- Sleep-on-Demand policy, animated dream GIF, and final evaluation."""

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
from somnia.models import ClassifierMLP, VAE, IntrospectiveHead, extract_internal_stats
from somnia.dreamer import Dreamer
from somnia.sleep import StabilityFilter, SleepConsolidation


def rolling_alarm(confusion_history, window=50, threshold=0.15):
    """Check if rolling mean P(error) exceeds alarm threshold."""
    if len(confusion_history) < window:
        return False
    recent = confusion_history[-window:]
    return np.mean(recent) > threshold


def sleep_on_demand_eval(classifier_orig, vae, intro_head,
                         train_x, train_y, test_x, test_y,
                         n_stream=2000, window=50, threshold=0.15,
                         n_dreams=200, seed=42):
    """Evaluate sleep-on-demand policy on a streaming data scenario.

    Simulates streaming data where the model monitors its own confusion
    and triggers sleep cycles only when the alarm fires.
    """
    set_seed(seed)
    classifier = copy.deepcopy(classifier_orig)
    dreamer = Dreamer(classifier, vae, intro_head)
    stability = StabilityFilter(classifier)
    consolidator = SleepConsolidation(classifier, dream_weight=0.25, lr=5e-4, epochs=2)

    # Stream random test samples
    indices = np.random.choice(len(test_x), size=n_stream, replace=True)
    confusion_history = []
    accuracy_history = []
    sleep_events = []
    correct = 0
    total = 0

    for i, idx in enumerate(indices):
        x = torch.from_numpy(test_x[idx:idx+1])
        y_true = test_y[idx]

        classifier.eval()
        with torch.no_grad():
            logits, h = classifier(x)
            pred = logits.argmax(1).item()
            stats = extract_internal_stats(h)
            p_err = intro_head(stats).item()

        is_correct = int(pred == y_true)
        correct += is_correct
        total += 1
        confusion_history.append(p_err)
        accuracy_history.append(correct / total)

        # Check alarm
        if rolling_alarm(confusion_history, window, threshold):
            # Trigger sleep
            sleep_events.append(i)

            # Generate targeted dreams
            dreams_all, _, confusion, _, _ = dreamer.dream(n=n_dreams * 2, seed=seed + i)
            _, top_idx = confusion.topk(n_dreams)
            targeted = dreams_all[top_idx]
            filtered, labels, _, _ = stability.filter(targeted, seed=seed + i)

            # Real data mini-batch
            real_idx = np.random.choice(len(train_x), size=500, replace=False)
            real_x = torch.from_numpy(train_x[real_idx])
            real_y = torch.from_numpy(train_y[real_idx])

            consolidator.sleep_cycle(real_x, real_y, filtered, labels)

            # Reset confusion history after sleep
            confusion_history = confusion_history[-10:]

    return accuracy_history, sleep_events, confusion_history


def fixed_schedule_eval(classifier_orig, vae, intro_head,
                        train_x, train_y, test_x, test_y,
                        n_stream=2000, sleep_interval=400,
                        n_dreams=200, seed=42):
    """Evaluate fixed-schedule sleep (sleep every N samples)."""
    set_seed(seed)
    classifier = copy.deepcopy(classifier_orig)
    dreamer = Dreamer(classifier, vae, intro_head)
    stability = StabilityFilter(classifier)
    consolidator = SleepConsolidation(classifier, dream_weight=0.25, lr=5e-4, epochs=2)

    indices = np.random.choice(len(test_x), size=n_stream, replace=True)
    accuracy_history = []
    sleep_events = []
    correct = 0
    total = 0

    for i, idx in enumerate(indices):
        x = torch.from_numpy(test_x[idx:idx+1])
        y_true = test_y[idx]

        classifier.eval()
        with torch.no_grad():
            logits, _ = classifier(x)
            pred = logits.argmax(1).item()

        correct += int(pred == y_true)
        total += 1
        accuracy_history.append(correct / total)

        if (i + 1) % sleep_interval == 0:
            sleep_events.append(i)
            dreams_all, _, confusion, _, _ = dreamer.dream(n=n_dreams * 2, seed=seed + i)
            _, top_idx = confusion.topk(n_dreams)
            targeted = dreams_all[top_idx]
            filtered, labels, _, _ = stability.filter(targeted, seed=seed + i)

            real_idx = np.random.choice(len(train_x), size=500, replace=False)
            real_x = torch.from_numpy(train_x[real_idx])
            real_y = torch.from_numpy(train_y[real_idx])
            consolidator.sleep_cycle(real_x, real_y, filtered, labels)

    return accuracy_history, sleep_events


def generate_dream_gif(vae, classifier, intro_head, save_path, n_frames=16, seed=42):
    """Generate animated GIF showing dreams evolving across different latent points.

    Tracks 16 fixed latent vectors and visualizes the decoded dreams.
    """
    try:
        import imageio
    except ImportError:
        from PIL import Image
        imageio = None

    set_seed(seed)
    gen = torch.Generator().manual_seed(seed)

    # 16 fixed latent base vectors
    z_base = torch.randn(16, vae.latent_dim, generator=gen)

    frames = []
    n_interpolation_steps = 20

    for step in range(n_interpolation_steps):
        # Smoothly vary latent vectors
        t = step / n_interpolation_steps
        z_offset = 0.5 * torch.sin(torch.tensor(2 * np.pi * t)) * torch.randn_like(z_base)
        z = z_base + z_offset

        with torch.no_grad():
            dreams = vae.decode(z)

        # Create 4x4 grid
        fig, axes = plt.subplots(4, 4, figsize=(6, 6))
        fig.suptitle(f"SOMNIA Dream Cycle (t={t:.2f})", fontsize=12, fontweight="bold")

        for idx in range(16):
            r, c = divmod(idx, 4)
            img = dreams[idx].reshape(28, 28).numpy()
            # Get confusion score
            logits, h = classifier(dreams[idx:idx+1])
            stats = extract_internal_stats(h)
            p_err = intro_head(stats).item()

            axes[r, c].imshow(img, cmap="magma")
            axes[r, c].set_title(f"P={p_err:.2f}", fontsize=7,
                                color="red" if p_err > 0.2 else "green")
            axes[r, c].axis("off")

        plt.tight_layout()

        # Save frame to buffer
        fig.canvas.draw()
        # Get image from buffer
        buf = fig.canvas.buffer_rgba()
        frame = np.asarray(buf)
        frames.append(frame.copy())
        plt.close(fig)

    # Save GIF
    if imageio is not None:
        imageio.mimsave(save_path, frames, duration=0.3, loop=0)
    else:
        # PIL fallback
        pil_frames = [Image.fromarray(f) for f in frames]
        pil_frames[0].save(save_path, save_all=True, append_images=pil_frames[1:],
                          duration=300, loop=0)

    print(f"  Dream GIF saved -> {save_path} ({len(frames)} frames)")


def plot_ondemand_comparison(od_acc, od_events, fs_acc, fs_events, save_path):
    """Plot sleep-on-demand vs fixed-schedule accuracy."""
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(od_acc, color="#FF6B6B", alpha=0.8, linewidth=1.5,
            label=f"Sleep-on-Demand ({len(od_events)} sleeps)")
    ax.plot(fs_acc, color="#4ECDC4", alpha=0.8, linewidth=1.5,
            label=f"Fixed Schedule ({len(fs_events)} sleeps)")

    # Mark sleep events
    for ev in od_events:
        ax.axvline(x=ev, color="#FF6B6B", alpha=0.3, linewidth=0.5, linestyle="--")
    for ev in fs_events:
        ax.axvline(x=ev, color="#4ECDC4", alpha=0.3, linewidth=0.5, linestyle="--")

    ax.set_xlabel("Stream Position", fontsize=11)
    ax.set_ylabel("Cumulative Accuracy", fontsize=11)
    ax.set_title("Phase 4 -- Sleep-on-Demand vs Fixed Schedule",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  On-demand comparison saved -> {save_path}")


def main():
    print("=" * 60)
    print("SOMNIA -- Phase 4: Sleep-on-Demand & Dream Film")
    print("=" * 60)

    set_seed(42)
    root = project_root()

    # Load
    print("\n[1/5] Loading data and models...")
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

    # Sleep-on-demand
    print("\n[2/5] Running sleep-on-demand evaluation (2000 stream)...")
    od_acc, od_events, od_conf = sleep_on_demand_eval(
        classifier, vae, intro_head, train_x, train_y, test_x, test_y,
        n_stream=2000, window=50, threshold=0.15, seed=42
    )
    print(f"  On-demand: {len(od_events)} sleep events triggered")
    print(f"  Final accuracy: {od_acc[-1]:.4f}")

    # Fixed schedule
    print("\n[3/5] Running fixed-schedule evaluation...")
    fs_acc, fs_events = fixed_schedule_eval(
        classifier, vae, intro_head, train_x, train_y, test_x, test_y,
        n_stream=2000, sleep_interval=400, seed=42
    )
    print(f"  Fixed: {len(fs_events)} sleep events")
    print(f"  Final accuracy: {fs_acc[-1]:.4f}")

    # Plot comparison
    comp_path = str(root / "figures" / "phase4_ondemand.png")
    plot_ondemand_comparison(od_acc, od_events, fs_acc, fs_events, comp_path)

    # Generate dream GIF
    print("\n[4/5] Generating dream film GIF...")
    gif_path = str(root / "figures" / "somnia_dreams.gif")
    generate_dream_gif(vae, classifier, intro_head, gif_path, seed=42)

    # Save results
    print("\n[5/5] Saving results...")
    metrics = {
        "phase": 4,
        "sleep_on_demand": {
            "n_sleep_events": len(od_events),
            "sleep_positions": od_events[:20],
            "final_accuracy": od_acc[-1],
            "alarm_threshold": 0.15,
            "window_size": 50,
        },
        "fixed_schedule": {
            "n_sleep_events": len(fs_events),
            "sleep_positions": fs_events,
            "final_accuracy": fs_acc[-1],
            "interval": 400,
        },
        "on_demand_better": bool(od_acc[-1] >= fs_acc[-1]),
        "figures": [
            "figures/phase4_ondemand.png",
            "figures/somnia_dreams.gif",
        ],
    }
    save_results(str(root / "results" / "phase4.json"), metrics, seed=42)

    print("\n" + "=" * 60)
    print("Phase 4 COMPLETE")
    print(f"  On-demand accuracy:  {od_acc[-1]:.4f} ({len(od_events)} sleeps)")
    print(f"  Fixed-sched accuracy:{fs_acc[-1]:.4f} ({len(fs_events)} sleeps)")
    if od_acc[-1] >= fs_acc[-1]:
        print("  [RESULT] Sleep-on-demand matches or beats fixed schedule!")
    else:
        print("  [HONEST] Fixed schedule outperforms on-demand in this run.")
    print("=" * 60)


if __name__ == "__main__":
    main()
