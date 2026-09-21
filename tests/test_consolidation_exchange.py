"""Unit tests for Phase 2: Consolidation with taught memories replay and adversarial safety."""

import pytest
import numpy as np
import torch
from pathlib import Path
from specimen.organism import SpecimenOrganism
from somnia.data import make_dataloaders


class TestConsolidationExchange:
    def test_sleep_with_taught_memories_increments_gen(self, tmp_path):
        db_path = str(tmp_path / "test_gen.db")
        org = SpecimenOrganism(db_path=db_path)
        initial_gen = org.generation
        assert initial_gen == 1

        # Feed and teach 3 samples
        for i in range(3):
            img = np.random.rand(28, 28).astype(np.float32)
            org.feed(img, visitor_id=f"teacher_{i}")
            org.reveal(true_label=i, visitor_id=f"teacher_{i}")

        assert len(org.taught_buffer) == 3

        # Run sleep cycle
        res = org.sleep_cycle()
        assert res["status"] == "success"
        assert org.generation == 2
        assert res["generation"] == 2
        assert res["taught_consolidated"] == 3
        assert "conscience_acc_before" in res
        assert "conscience_acc_after" in res

        # Verify state
        state = org.get_state()
        assert state["generation"] == 2
        assert "GEN 2" in state["specimen_id"]

    def test_adversarial_safety_defense(self, tmp_path):
        """Adversarial Safety Test: 100 taught samples with 50% corrupted (liar) labels.
        Must not drop private conscience accuracy by more than 0.5 percentage points.
        """
        db_path = str(tmp_path / "test_adversarial.db")
        org = SpecimenOrganism(db_path=db_path)

        acc_clean = org.evaluate_conscience()

        # Load real test digits for simulation
        _, _, _, val_x_np, val_y_np = make_dataloaders()
        rng = np.random.RandomState(42)

        # 100 samples
        sample_indices = rng.choice(len(val_x_np), size=100, replace=False)
        sim_x = val_x_np[sample_indices]
        sim_y = val_y_np[sample_indices].copy()

        # Corrupt 50% of them (liar labels)
        for i in range(50):
            sim_y[i] = (sim_y[i] + rng.randint(1, 10)) % 10

        # Feed and teach into organism
        for i in range(100):
            org.feed(sim_x[i].reshape(28, 28), visitor_id=f"adversary_{i}")
            org.reveal(true_label=int(sim_y[i]), visitor_id=f"adversary_{i}")

        assert len(org.taught_buffer) == 100

        # Run consolidation
        res = org.sleep_cycle()
        assert res["status"] == "success"

        acc_after = org.evaluate_conscience()
        drop = (acc_clean - acc_after) * 100.0

        print(f"[TEST ADVERSARIAL] Clean Conscience Acc: {acc_clean*100:.2f}%, After 50% Liar Sleep: {acc_after*100:.2f}%, Drop: {drop:+.2f}pp")
        assert drop <= 0.50, f"Conscience accuracy dropped by {drop:.2f}pp (exceeds 0.5pp limit)"
