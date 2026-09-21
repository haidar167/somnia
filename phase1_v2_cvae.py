"""V2-Phase 1 -- Conditional VAE: sharper, class-aware dreams."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from somnia.utils import set_seed, save_results, project_root
from somnia.data import make_dataloaders
from somnia.models import ConditionalVAE


def train_cvae(model, loader, epochs=5, lr=1e-3):
    """Train conditional VAE with class labels."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = []
    for epoch in range(epochs):
        model.train()
        total_loss, total = 0.0, 0
        for xb, yb in loader:
            optimizer.zero_grad()
            x_recon, mu, logvar = model(xb, yb)
            loss = ConditionalVAE.loss_function(x_recon, xb, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total += xb.size(0)
        avg = total_loss / total
        history.append({"epoch": epoch + 1, "loss_per_sample": avg})
        print(f"  cVAE Epoch {epoch+1}/{epochs} -- loss/sample: {avg:.2f}")
    return history


def plot_cvae_samples(cvae, save_path, n_per_class=16):
    """Plot 10x16 grid: each row is a class, 16 samples per class."""
    cvae.eval()
    fig, axes = plt.subplots(10, n_per_class, figsize=(n_per_class * 1.2, 10 * 1.2))
    fig.suptitle("V2 Phase 1 -- Conditional VAE Class-Conditioned Samples",
                 fontsize=13, fontweight="bold")

    for c in range(10):
        dreams, _ = cvae.sample_class(n_per_class, target_class=c, seed=42 + c)
        for i in range(n_per_class):
            ax = axes[c, i]
            ax.imshow(dreams[i].reshape(28, 28).numpy(), cmap="gray")
            ax.axis("off")
            if i == 0:
                ax.set_ylabel(str(c), fontsize=11, fontweight="bold", rotation=0,
                             labelpad=15)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  cVAE sample grid saved -> {save_path}")


def plot_recon_comparison(v1_vae, cvae, test_loader, save_path, n=8):
    """Side-by-side reconstruction: v1 VAE vs v2 cVAE."""
    from somnia.models import VAE
    v1_vae.eval()
    cvae.eval()
    xb, yb = next(iter(test_loader))
    xb, yb = xb[:n], yb[:n]

    with torch.no_grad():
        recon_v1, _, _ = v1_vae(xb)
        recon_v2, _, _ = cvae(xb, yb)

    fig, axes = plt.subplots(3, n, figsize=(n * 1.5, 4.5))
    row_labels = ["Original", "v1 VAE", "v2 cVAE"]
    rows = [xb, recon_v1, recon_v2]

    for row, (label, data) in enumerate(zip(row_labels, rows)):
        for i in range(n):
            axes[row, i].imshow(data[i].reshape(28, 28).numpy(), cmap="gray")
            axes[row, i].axis("off")
            if i == 0:
                axes[row, i].set_ylabel(label, fontsize=9, fontweight="bold",
                                        rotation=0, labelpad=45)

    fig.suptitle("v1 VAE vs v2 cVAE Reconstruction", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Reconstruction comparison saved -> {save_path}")


def main():
    print("=" * 60)
    print("SOMNIA v2 -- Phase 1: Conditional VAE")
    print("=" * 60)

    set_seed(0)
    root = project_root()

    # Data
    print("\n[1/4] Loading data...")
    train_loader, val_loader, test_loader, _, _ = make_dataloaders()

    # Train cVAE
    print("\n[2/4] Training ConditionalVAE (5 epochs, latent_dim=32)...")
    cvae = ConditionalVAE(latent_dim=32)
    cvae_history = train_cvae(cvae, train_loader, epochs=5)
    final_loss = cvae_history[-1]["loss_per_sample"]
    print(f"  Final cVAE loss/sample: {final_loss:.2f}")
    print(f"  v1 VAE was: 113.94")
    print(f"  Improvement: {113.94 - final_loss:.2f}")

    if final_loss < 90:
        print("  [PASS] cVAE recon loss < 90!")
    else:
        print(f"  [WARN] cVAE recon loss {final_loss:.2f} >= 90, not meeting target")

    # Plot class-conditioned samples
    print("\n[3/4] Generating class-conditioned sample grid...")
    samples_path = str(root / "figures" / "v2_cvae_samples.png")
    plot_cvae_samples(cvae, samples_path, n_per_class=16)

    # Compare with v1 VAE
    print("\n[4/4] Comparing v1 VAE vs v2 cVAE reconstructions...")
    from somnia.models import VAE
    v1_vae = VAE()
    v1_vae.load_state_dict(torch.load(str(root / "data" / "vae.pt"),
                                      weights_only=True))
    comp_path = str(root / "figures" / "v2_recon_comparison.png")
    plot_recon_comparison(v1_vae, cvae, test_loader, comp_path)

    # Save cVAE
    torch.save(cvae.state_dict(), str(root / "data" / "cvae.pt"))

    # Results
    metrics = {
        "phase": "v2-1",
        "cvae_final_loss": final_loss,
        "v1_vae_loss": 113.94,
        "improvement": 113.94 - final_loss,
        "target_met": final_loss < 90,
        "cvae_history": cvae_history,
        "latent_dim": 32,
        "figures": [
            "figures/v2_cvae_samples.png",
            "figures/v2_recon_comparison.png",
        ],
    }
    save_results(str(root / "results" / "v2_phase1.json"), metrics, seed=0)

    print("\n" + "=" * 60)
    print("V2 Phase 1 COMPLETE")
    print(f"  cVAE recon loss : {final_loss:.2f} (v1: 113.94)")
    print(f"  Target (<90)    : {'PASS' if final_loss < 90 else 'MISS'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
