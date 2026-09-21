"""V2-Phase 2 -- Stress-trained introspective head (pushing AUC above 0.80)."""

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
from sklearn.metrics import roc_auc_score, roc_curve

from somnia.utils import set_seed, save_results, project_root
from somnia.data import make_dataloaders, load_mnist
from somnia.models import ClassifierMLP, IntrospectiveHeadV2, extract_internal_stats_v2
from somnia.stress import build_stress_dataset


def extract_features_and_errors(classifier, x_arr, y_arr, batch_size=256):
    """Compute 10-feature representation and ground truth error indicator."""
    classifier.eval()
    all_stats = []
    all_errors = []

    with torch.no_grad():
        for i in range(0, len(x_arr), batch_size):
            xb = torch.from_numpy(x_arr[i:i+batch_size])
            yb = torch.from_numpy(y_arr[i:i+batch_size])
            logits, h = classifier(xb)
            preds = logits.argmax(1)
            err = (preds != yb).float().unsqueeze(1)
            stats = extract_internal_stats_v2(h, logits)
            all_stats.append(stats)
            all_errors.append(err)

    return torch.cat(all_stats, dim=0), torch.cat(all_errors, dim=0)


def train_head_v2(head, stats_train, errors_train, epochs=80, lr=1e-2):
    """Train MLP introspective head on stress-extracted features."""
    optimizer = optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCELoss()

    dataset = torch.utils.data.TensorDataset(stats_train, errors_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

    for epoch in range(epochs):
        head.train()
        for sb, eb in loader:
            optimizer.zero_grad()
            pred = head(sb)
            loss = criterion(pred, eb)
            loss.backward()
            optimizer.step()

    head.eval()


def evaluate_auc(head, stats, errors):
    """Compute ROC-AUC on evaluated dataset."""
    head.eval()
    with torch.no_grad():
        pred_scores = head(stats).squeeze().numpy()
        err_np = errors.squeeze().numpy()
    if len(np.unique(err_np)) < 2:
        return 0.5, pred_scores, err_np
    auc = roc_auc_score(err_np, pred_scores)
    return auc, pred_scores, err_np


def plot_roc_curves(clean_err, clean_scores, stress_err, stress_scores, save_path):
    """Plot ROC curves comparing clean test vs stressed test error detection."""
    fpr_clean, tpr_clean, _ = roc_curve(clean_err, clean_scores)
    fpr_stress, tpr_stress, _ = roc_curve(stress_err, stress_scores)
    auc_clean = roc_auc_score(clean_err, clean_scores)
    auc_stress = roc_auc_score(stress_err, stress_scores)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr_clean, tpr_clean, color="#4ECDC4", lw=2,
            label=f"Clean Test (AUC = {auc_clean:.4f})")
    ax.plot(fpr_stress, tpr_stress, color="#FF6B6B", lw=2,
            label=f"Stressed Test (AUC = {auc_stress:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Chance (AUC = 0.50)")

    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("V2 Phase 2 -- Introspective Head v2 ROC Curves\n(10-feature MLP on Stress Distribution)",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ROC curve saved -> {save_path}")


def main():
    print("=" * 60)
    print("SOMNIA v2 -- Phase 2: Stress-Trained Introspective Head")
    print("=" * 60)

    set_seed(0)
    root = project_root()

    # Load Base Classifier & Data
    print("\n[1/5] Loading data and base classifier...")
    _, _, test_loader, val_x_np, val_y_np = make_dataloaders()
    train_x, train_y, test_x, test_y = load_mnist()

    classifier = ClassifierMLP()
    classifier.load_state_dict(torch.load(str(root / "data" / "classifier.pt"),
                                          weights_only=True))

    # Build Stress Calibration Dataset
    print("\n[2/5] Synthesizing stress calibration set (5000 samples)...")
    stress_val_x, stress_val_y = build_stress_dataset(val_x_np, val_y_np, seed=42)
    stats_train, errors_train = extract_features_and_errors(classifier, stress_val_x, stress_val_y)
    stress_err_rate = errors_train.mean().item()
    print(f"  Stress calibration set: {len(stress_val_x)} samples, error rate: {stress_err_rate*100:.2f}%")

    # Train Introspective Head v2
    print("\n[3/5] Training 10-feature MLP Introspective Head v2...")
    head_v2 = IntrospectiveHeadV2(input_dim=10, hidden_dim=32)
    train_head_v2(head_v2, stats_train, errors_train, epochs=80, lr=1e-2)

    # Evaluate on Clean Test Data
    print("\n[4/5] Evaluating on Clean Test vs Stressed Test...")
    stats_clean_test, errors_clean_test = extract_features_and_errors(classifier, test_x, test_y)
    auc_clean, clean_scores, clean_err = evaluate_auc(head_v2, stats_clean_test, errors_clean_test)

    # Build Stress Test Data
    stress_test_x, stress_test_y = build_stress_dataset(test_x, test_y, seed=100)
    stats_stress_test, errors_stress_test = extract_features_and_errors(classifier, stress_test_x, stress_test_y)
    auc_stress, stress_scores, stress_err = evaluate_auc(head_v2, stats_stress_test, errors_stress_test)

    print(f"  [RESULT] Clean Test AUC    : {auc_clean:.4f} (v1 was 0.5879)")
    print(f"  [RESULT] Stressed Test AUC : {auc_stress:.4f}")

    # Check Acceptance
    clean_target_met = auc_clean > 0.65
    stress_target_met = auc_stress > 0.80
    print(f"  Clean AUC target (>0.65)   : {'PASS' if clean_target_met else 'MISS'}")
    print(f"  Stress AUC target (>0.80)  : {'PASS' if stress_target_met else 'MISS'}")

    # Plot ROC
    roc_path = str(root / "figures" / "v2_introspective_roc.png")
    plot_roc_curves(clean_err, clean_scores, stress_err, stress_scores, roc_path)

    # Save checkpoint
    torch.save(head_v2.state_dict(), str(root / "data" / "intro_head_v2.pt"))

    # Results
    metrics = {
        "phase": "v2-2",
        "clean_test_auc": auc_clean,
        "stress_test_auc": auc_stress,
        "v1_clean_auc": 0.5879,
        "clean_target_met": clean_target_met,
        "stress_target_met": stress_target_met,
        "stress_calibration_error_rate": stress_err_rate,
        "figures": ["figures/v2_introspective_roc.png"],
    }
    save_results(str(root / "results" / "v2_phase2.json"), metrics, seed=0)

    print("\n" + "=" * 60)
    print("V2 Phase 2 COMPLETE")
    print(f"  Clean AUC  : {auc_clean:.4f}")
    print(f"  Stress AUC : {auc_stress:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
