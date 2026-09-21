"""Phase 2 -- Latent confusion mapping & nightmare collage."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from somnia.utils import set_seed, save_results, project_root
from somnia.data import make_dataloaders
from somnia.models import ClassifierMLP, VAE, IntrospectiveHead, extract_internal_stats
from somnia.dreamer import Dreamer


def plot_confusion_map(z_2d, confusion, save_path):
    """Plot 2D PCA confusion density map of latent dream space."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    sc = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=confusion, cmap="RdYlGn_r",
                    s=8, alpha=0.7, vmin=0, vmax=1)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("P(error) -- Confusion Score", fontsize=10)
    ax.set_xlabel("PCA Component 1", fontsize=10)
    ax.set_ylabel("PCA Component 2", fontsize=10)
    ax.set_title("Phase 2 -- Latent Confusion Map\n"
                 "Red = nightmares (high P(error)), Green = lucid dreams",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion map saved -> {save_path}")


def plot_dream_collage(nightmares, lucid_dreams, nightmare_scores,
                       lucid_scores, save_path, n=25):
    """Plot grid of top nightmares vs lucid dreams."""
    cols = 5
    rows = n // cols

    fig, axes = plt.subplots(2 * rows, cols, figsize=(cols * 1.8, 2 * rows * 1.8))
    fig.suptitle("Phase 2 -- Nightmare vs Lucid Dream Collage", fontsize=13,
                 fontweight="bold", y=1.02)

    # Nightmares (top half)
    for i in range(min(n, nightmares.shape[0])):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        ax.imshow(nightmares[i].reshape(28, 28).numpy(), cmap="Reds")
        ax.set_title(f"P={nightmare_scores[i]:.2f}", fontsize=7, color="red")
        ax.axis("off")
        if i == 0:
            ax.set_ylabel("NIGHTMARES", fontsize=8, color="red", fontweight="bold")

    # Lucid dreams (bottom half)
    for i in range(min(n, lucid_dreams.shape[0])):
        r, c = divmod(i, cols)
        ax = axes[rows + r, c]
        ax.imshow(lucid_dreams[i].reshape(28, 28).numpy(), cmap="Greens")
        ax.set_title(f"P={lucid_scores[i]:.2f}", fontsize=7, color="green")
        ax.axis("off")
        if i == 0:
            ax.set_ylabel("LUCID", fontsize=8, color="green", fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Dream collage saved -> {save_path}")


def compute_real_auc(classifier, intro_head, test_loader):
    """Compute introspective AUC on real test data."""
    classifier.eval()
    intro_head.eval()
    all_scores, all_errors = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            logits, h = classifier(xb)
            preds = logits.argmax(1)
            errors = (preds != yb).float()
            stats = extract_internal_stats(h)
            scores = intro_head(stats).squeeze()
            all_scores.append(scores)
            all_errors.append(errors)

    scores = torch.cat(all_scores).numpy()
    errors = torch.cat(all_errors).numpy()
    if len(np.unique(errors)) < 2:
        return 0.5
    return roc_auc_score(errors, scores)


def main():
    print("=" * 60)
    print("SOMNIA -- Phase 2: The Confusion Map")
    print("=" * 60)

    set_seed(42)
    root = project_root()

    # Load models
    print("\n[1/5] Loading trained models...")
    classifier = ClassifierMLP()
    classifier.load_state_dict(torch.load(str(root / "data" / "classifier.pt"),
                                          weights_only=True))
    vae = VAE()
    vae.load_state_dict(torch.load(str(root / "data" / "vae.pt"),
                                   weights_only=True))
    intro_head = IntrospectiveHead()
    intro_head.load_state_dict(torch.load(str(root / "data" / "intro_head.pt"),
                                          weights_only=True))
    print("  Models loaded.")

    # Create dreamer
    dreamer = Dreamer(classifier, vae, intro_head)

    # Dream
    print("\n[2/5] Generating 2000 dreams...")
    dreams, z_latent, confusion, pseudo_labels, stats = dreamer.dream(n=2000, seed=42)
    print(f"  Dreams shape: {dreams.shape}")
    print(f"  Confusion range: [{confusion.min():.4f}, {confusion.max():.4f}]")
    print(f"  Mean confusion: {confusion.mean():.4f}")
    print(f"  Pseudo-label distribution: {torch.bincount(pseudo_labels, minlength=10).tolist()}")

    # PCA confusion map
    print("\n[3/5] Fitting PCA and plotting confusion map...")
    z_2d, pca = Dreamer.fit_pca(z_latent)
    map_path = str(root / "figures" / "phase2_confusion_map.png")
    plot_confusion_map(z_2d, confusion.numpy(), map_path)

    # Nightmare vs lucid collage
    print("\n[4/5] Extracting nightmares and lucid dreams...")
    nightmares, nm_scores, nm_idx = Dreamer.select_top_k(dreams, confusion, k=25, highest=True)
    lucid, lc_scores, lc_idx = Dreamer.select_top_k(dreams, confusion, k=25, highest=False)
    print(f"  Top 25 nightmare confusion: [{nm_scores.min():.4f}, {nm_scores.max():.4f}]")
    print(f"  Top 25 lucid confusion:     [{lc_scores.min():.4f}, {lc_scores.max():.4f}]")

    collage_path = str(root / "figures" / "phase2_dream_collage.png")
    plot_dream_collage(nightmares, lucid, nm_scores, lc_scores, collage_path)

    # Real AUC comparison
    print("\n[5/5] Computing real-data AUC for comparison...")
    _, _, test_loader, _, _ = make_dataloaders()
    real_auc = compute_real_auc(classifier, intro_head, test_loader)
    print(f"  Real-data introspective AUC: {real_auc:.4f}")

    # Dream AUC proxy: correlation between confusion score and actual classifier margin
    with torch.no_grad():
        logits, _ = classifier(dreams)
        probs = torch.softmax(logits, dim=1)
        margin = probs.max(dim=1).values - probs.topk(2, dim=1).values[:, 1]
        # Low margin = likely error; high confusion should correlate with low margin
        dream_margin_corr = np.corrcoef(confusion.numpy(), margin.numpy())[0, 1]
    print(f"  Dream confusion vs margin correlation: {dream_margin_corr:.4f}")
    print(f"  (Negative = introspection correctly identifies low-margin dreams)")

    # Save results
    metrics = {
        "phase": 2,
        "n_dreams": 2000,
        "confusion_mean": float(confusion.mean()),
        "confusion_std": float(confusion.std()),
        "confusion_min": float(confusion.min()),
        "confusion_max": float(confusion.max()),
        "nightmare_scores_top25": nm_scores.tolist(),
        "lucid_scores_top25": lc_scores.tolist(),
        "real_auc": real_auc,
        "dream_margin_correlation": float(dream_margin_corr),
        "pseudo_label_distribution": torch.bincount(pseudo_labels, minlength=10).tolist(),
        "pca_explained_variance": pca.explained_variance_ratio_.tolist(),
        "figures": [
            "figures/phase2_confusion_map.png",
            "figures/phase2_dream_collage.png",
        ],
    }
    save_results(str(root / "results" / "phase2.json"), metrics, seed=42)

    print("\n" + "=" * 60)
    print("Phase 2 COMPLETE")
    print(f"  Dreams generated     : 2000")
    print(f"  Mean confusion       : {confusion.mean():.4f}")
    print(f"  Real-data AUC        : {real_auc:.4f}")
    print(f"  Margin correlation   : {dream_margin_corr:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
