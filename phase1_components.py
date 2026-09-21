"""Phase 1 — Train base classifier, VAE, and calibrate introspective head."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from somnia.utils import set_seed, save_results, project_root
from somnia.data import make_dataloaders
from somnia.models import ClassifierMLP, VAE, extract_internal_stats, IntrospectiveHead


def train_classifier(model, loader, epochs=3, lr=1e-3):
    """Train classifier and return per-epoch losses."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    history = []
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for xb, yb in loader:
            optimizer.zero_grad()
            logits, _ = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            total += xb.size(0)
        acc = correct / total
        avg_loss = total_loss / total
        history.append({"epoch": epoch + 1, "loss": avg_loss, "acc": acc})
        print(f"  Classifier Epoch {epoch+1}/{epochs} — loss: {avg_loss:.4f}, acc: {acc:.4f}")
    return history


def train_vae(model, loader, epochs=5, lr=1e-3):
    """Train VAE and return per-epoch losses."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = []
    for epoch in range(epochs):
        model.train()
        total_loss, total = 0.0, 0
        for xb, _ in loader:
            optimizer.zero_grad()
            x_recon, mu, logvar = model(xb)
            loss = VAE.loss_function(x_recon, xb, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total += xb.size(0)
        avg = total_loss / total
        history.append({"epoch": epoch + 1, "loss_per_sample": avg})
        print(f"  VAE Epoch {epoch+1}/{epochs} — loss/sample: {avg:.2f}")
    return history


def evaluate_classifier(model, loader):
    """Return accuracy on a DataLoader."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            logits, _ = model(xb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += xb.size(0)
    return correct / total


def calibrate_introspective_head(classifier, intro_head, val_x_np, val_y_np, epochs=50, lr=1e-2):
    """Train logistic introspective head on held-out validation errors."""
    classifier.eval()
    with torch.no_grad():
        x_val = torch.from_numpy(val_x_np)
        y_val = torch.from_numpy(val_y_np)
        logits, h = classifier(x_val)
        preds = logits.argmax(1)
        errors = (preds != y_val).float().unsqueeze(1)  # (N, 1)
        stats = extract_internal_stats(h)                # (N, 4)

    print(f"  Calibration set: {len(val_x_np)} samples, {errors.sum().item():.0f} errors "
          f"({errors.mean().item():.4f} error rate)")

    optimizer = optim.Adam(intro_head.parameters(), lr=lr)
    criterion = nn.BCELoss()
    for epoch in range(epochs):
        intro_head.train()
        optimizer.zero_grad()
        p_err = intro_head(stats)
        loss = criterion(p_err, errors)
        loss.backward()
        optimizer.step()

    # Evaluate AUC
    intro_head.eval()
    with torch.no_grad():
        p_err = intro_head(stats).squeeze().numpy()
        errors_np = errors.squeeze().numpy()

    if len(np.unique(errors_np)) < 2:
        auc = 0.5
        print("  WARNING: Only one class in calibration errors, AUC defaulting to 0.5")
    else:
        auc = roc_auc_score(errors_np, p_err)
    print(f"  Introspective head calibration AUC: {auc:.4f}")
    return auc


def plot_reconstructions(vae, test_loader, save_path, n=8):
    """Plot original vs VAE reconstruction side-by-side."""
    vae.eval()
    xb, _ = next(iter(test_loader))
    xb = xb[:n]
    with torch.no_grad():
        recon, _, _ = vae(xb)

    fig, axes = plt.subplots(2, n, figsize=(n * 1.5, 3))
    for i in range(n):
        axes[0, i].imshow(xb[i].reshape(28, 28).numpy(), cmap="gray")
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_title("Original", fontsize=9)
        axes[1, i].imshow(recon[i].reshape(28, 28).numpy(), cmap="gray")
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_title("Reconstructed", fontsize=9)

    fig.suptitle("Phase 1 — VAE Reconstruction", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Reconstruction plot saved -> {save_path}")


def main():
    print("=" * 60)
    print("SOMNIA — Phase 1: The Components")
    print("=" * 60)

    set_seed(0)
    root = project_root()

    # Data
    print("\n[1/4] Loading data...")
    train_loader, val_loader, test_loader, val_x_np, val_y_np = make_dataloaders()

    # Train Classifier
    print("\n[2/4] Training ClassifierMLP (3 epochs)...")
    classifier = ClassifierMLP()
    cls_history = train_classifier(classifier, train_loader, epochs=3)
    test_acc = evaluate_classifier(classifier, test_loader)
    print(f"  [OK] Test accuracy: {test_acc:.4f}")
    assert test_acc > 0.90, f"Classifier accuracy too low: {test_acc:.4f}"

    # Train VAE
    print("\n[3/4] Training VAE (5 epochs)...")
    vae = VAE()
    vae_history = train_vae(vae, train_loader, epochs=5)

    # Reconstruction plot
    recon_path = str(root / "figures" / "phase1_recon.png")
    plot_reconstructions(vae, test_loader, recon_path)

    # Calibrate Introspective Head
    print("\n[4/4] Calibrating introspective head...")
    intro_head = IntrospectiveHead()
    intro_auc = calibrate_introspective_head(classifier, intro_head, val_x_np, val_y_np)

    # Save models
    models_dir = root / "data"
    models_dir.mkdir(exist_ok=True)
    torch.save(classifier.state_dict(), str(models_dir / "classifier.pt"))
    torch.save(vae.state_dict(), str(models_dir / "vae.pt"))
    torch.save(intro_head.state_dict(), str(models_dir / "intro_head.pt"))
    print(f"\n  Models saved -> {models_dir}/")

    # Results
    metrics = {
        "phase": 1,
        "classifier_test_acc": test_acc,
        "classifier_history": cls_history,
        "vae_history": vae_history,
        "introspective_auc": intro_auc,
        "figures": ["figures/phase1_recon.png"],
    }
    save_results(str(root / "results" / "phase1.json"), metrics, seed=0)

    print("\n" + "=" * 60)
    print("Phase 1 COMPLETE")
    print(f"  Classifier accuracy : {test_acc:.4f}")
    print(f"  Introspective AUC   : {intro_auc:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
