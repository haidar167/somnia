"""Unit tests for Phase 1: Second Opinion cVAE Generative Vouching."""

import pytest
import numpy as np
import torch
from somnia.data import make_dataloaders
from somnia.stress import apply_rotation, apply_gaussian_noise
from somnia.second_opinion import SecondOpinionVoucher
from specimen.organism import SpecimenOrganism


class TestSecondOpinionVoucher:
    def test_corrections_plausibility_rate(self, tmp_path):
        """Corrections where classifier is wrong must land in plausible bucket >= 50% of the time."""
        db_path = str(tmp_path / "test_voucher.db")
        org = SpecimenOrganism(db_path=db_path)
        voucher = SecondOpinionVoucher(org.cvae, tolerance_ratio=1.10)

        _, _, _, val_x_np, val_y_np = make_dataloaders()
        rot_x = apply_rotation(val_x_np[:500], max_angle=30.0, seed=42)
        t_x = torch.from_numpy(rot_x)
        t_y = torch.from_numpy(val_y_np[:500])

        with torch.no_grad():
            preds = org.classifier(t_x)[0].argmax(1)
        err_mask = (preds != t_y)
        err_x = t_x[err_mask]
        err_true_y = t_y[err_mask]

        eval_res = voucher.evaluate_taught_batch(err_x, err_true_y, org.classifier)
        is_plausible = eval_res["is_plausible"]

        plausible_rate = float(is_plausible.float().mean().item())
        print(f"[TEST SECOND OPINION] True corrections plausible rate: {plausible_rate*100:.1f}% ({is_plausible.sum()}/{len(err_x)})")
        assert plausible_rate >= 0.50, f"Expected >= 50% plausible corrections, got {plausible_rate*100:.1f}%"

    def test_liars_more_implausible_than_true_corrections(self, tmp_path):
        """Synthetic liar labels (random wrong classes) must be IMPLAUSIBLE more often than real-label corrections."""
        db_path = str(tmp_path / "test_voucher_liars.db")
        org = SpecimenOrganism(db_path=db_path)
        voucher = SecondOpinionVoucher(org.cvae, tolerance_ratio=1.10)

        _, _, _, val_x_np, val_y_np = make_dataloaders()
        rot_x = apply_rotation(val_x_np[:300], max_angle=30.0, seed=42)
        t_x = torch.from_numpy(rot_x)
        t_true_y = torch.from_numpy(val_y_np[:300])

        rng = np.random.RandomState(42)
        liar_y = (t_true_y + torch.tensor([rng.randint(1, 10) for _ in range(len(t_true_y))])) % 10

        res_true = voucher.evaluate_taught_batch(t_x, t_true_y, org.classifier)
        res_liar = voucher.evaluate_taught_batch(t_x, liar_y, org.classifier)

        implausible_true = float((~res_true["is_plausible"]).float().mean().item())
        implausible_liar = float((~res_liar["is_plausible"]).float().mean().item())

        print(f"[TEST LIAR FILTER] True labels implausible: {implausible_true*100:.1f}%, Liar labels implausible: {implausible_liar*100:.1f}%")
        assert implausible_liar > implausible_true, (
            f"Expected liars ({implausible_liar*100:.1f}%) to be more implausible than true ({implausible_true*100:.1f}%)"
        )
