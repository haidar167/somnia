"""Unit tests for Exchange v3.2 Positive Control."""

import pytest
import numpy as np
import torch
from somnia.models import ClassifierMLP, ConditionalVAE, IntrospectiveHeadV2
from somnia.sleep_v2 import SleepConsolidationV2, SoftStabilityFilter, DreamerV2
from somnia.data import make_dataloaders
from somnia.stress import apply_rotation, apply_gaussian_noise
from specimen.organism import SpecimenOrganism


def test_positive_control_learnability(tmp_path):
    """Positive control test: sleep consolidation with strong dose must improve taught sample accuracy > 5pp."""
    db_path = str(tmp_path / "test_v32.db")
    org = SpecimenOrganism(db_path=db_path)

    _, _, _, val_x_np, val_y_np = make_dataloaders()
    rot_x = apply_rotation(val_x_np[:200], max_angle=30.0, seed=42)
    stressed_x = apply_gaussian_noise(rot_x, std=0.25, seed=42)
    t_x = torch.from_numpy(stressed_x)
    t_y = torch.from_numpy(val_y_np[:200])

    with torch.no_grad():
        preds_pre = org.classifier(t_x)[0].argmax(1)
    err_mask = (preds_pre != t_y)
    err_x = t_x[err_mask]
    err_y = t_y[err_mask]

    assert len(err_y) > 10, "Need at least 10 errors to test positive control"

    # Positive control: consolidate directly on errors with LR 5e-4
    consolidator = SleepConsolidationV2(org.classifier, dream_mix=0.05, lr=5e-4, epochs=2)
    real_x = torch.randn(20, 784).clamp(0, 1)
    real_y = torch.randint(0, 10, (20,))
    dream_x = torch.randn(20, 784).clamp(0, 1)
    dream_y = torch.randint(0, 10, (20,))
    dream_w = torch.ones(20)
    taught_w = torch.full((len(err_y),), 2.0)

    # 3 cycles
    for _ in range(3):
        consolidator.sleep_cycle(
            real_x, real_y, dream_x, dream_y, dream_w,
            taught_x=err_x, taught_y=err_y, taught_weights=taught_w
        )

    with torch.no_grad():
        preds_post = org.classifier(err_x)[0].argmax(1)
    acc_post = (preds_post == err_y).float().mean().item()

    print(f"[TEST POSITIVE CONTROL] Taught accuracy post-sleep: {acc_post*100:.2f}% (was 0.00%)")
    assert acc_post >= 0.05, f"Expected post-sleep taught accuracy >= 5.0%, got {acc_post*100:.2f}%"
