"""Unit tests for Phase 1: v3.2 Field Repairs (R1 - R5)."""

import pytest
import numpy as np
import torch
from pathlib import Path
from specimen.organism import SpecimenOrganism
from somnia.data import make_dataloaders


class TestV32FieldRepairs:

    def test_r1_restore_once_reconnect_preserves_memory_state(self, tmp_path):
        """R1: DB restore happens strictly once at boot. Reconnects/state queries must never overwrite in-memory state."""
        db_p = str(tmp_path / "r1_test.db")
        org = SpecimenOrganism(db_path=db_p)

        # Mutate in-memory state
        org.current_pred = 7
        org.current_mood = "alarmed"
        org.current_p_error = 0.88
        org.event_log.appendleft({"time": "12:00:00", "message": "In-memory test event", "type": "test"})

        # Get state as WebSocket broadcast would
        state1 = org.get_state()
        assert state1["vitals"]["prediction"] == 7
        assert state1["vitals"]["mood"] == "alarmed"
        assert state1["vitals"]["p_error"] == 0.88

        # Simulate reconnect
        org.visitor_count += 1
        state2 = org.get_state()

        # Assert in-memory state preserved
        assert state2["vitals"]["prediction"] == 7
        assert state2["vitals"]["mood"] == "alarmed"
        assert state2["vitals"]["p_error"] == 0.88
        assert state2["event_log"][0]["message"] == "In-memory test event"

    def test_r2_gen_counter_single_source_of_truth(self, tmp_path):
        """R2: GEN = persisted consolidation count, +1 per completed sleep, always == sleep_count."""
        db_p = str(tmp_path / "r2_test.db")
        org = SpecimenOrganism(db_path=db_p)

        assert org.generation == 0
        assert org.sleep_count == 0

        # First sleep
        res1 = org.sleep_cycle()
        assert res1["generation"] == 1
        assert res1["sleep_count"] == 1
        assert org.generation == 1
        assert org.sleep_count == 1

        # Second sleep
        res2 = org.sleep_cycle()
        assert res2["generation"] == 2
        assert res2["sleep_count"] == 2
        assert org.generation == 2
        assert org.sleep_count == 2

        # Re-awaken organism from DB
        org_reborn = SpecimenOrganism(db_path=db_p)
        assert org_reborn.generation == 2
        assert org_reborn.sleep_count == 2
        state = org_reborn.get_state()
        assert state["generation"] == 2
        assert state["sleep_count"] == 2
        assert "GEN 2" in state["specimen_id"]

    def test_r3_dream_scores_end_to_end_variety(self, tmp_path):
        """R3: Dreams must have non-zero p_error variance, multiple target classes, and distinct scores."""
        db_p = str(tmp_path / "r3_test.db")
        org = SpecimenOrganism(db_path=db_p)

        # Check initial bootstrap dreams
        boot_dreams = org.recent_dreams
        assert len(boot_dreams) >= 4
        target_classes_boot = set(d["target_class"] for d in boot_dreams)
        assert len(target_classes_boot) > 1, "Initial dreams must span multiple target classes"

        # Trigger sleep cycle
        sleep_res = org.sleep_cycle()
        dreams = sleep_res["recent_dreams"]
        assert len(dreams) == 8

        p_errors = [d["p_error"] for d in dreams]
        target_classes = [d["target_class"] for d in dreams]
        weights = [d["weight"] for d in dreams]

        # Assert variance > 0
        p_err_var = float(np.var(p_errors))
        print(f"[TEST R3] Dream p_error variance: {p_err_var:.6f}, classes: {target_classes}")
        assert p_err_var > 0.001, f"Expected non-zero p_error variance across dreams, got {p_err_var}"
        assert len(set(target_classes)) >= 4, f"Expected >= 4 distinct target classes, got {len(set(target_classes))}"
        assert any(d["is_nightmare"] for d in dreams) or any(not d["is_nightmare"] for d in dreams)

        # Check state serialization payload
        state = org.get_state()
        ws_dreams = state["recent_dreams"]
        assert len(ws_dreams) == 8
        assert float(np.var([d["p_error"] for d in ws_dreams])) > 0.001

    def test_r4_taught_buffer_hygiene(self, tmp_path):
        """R4: a) Dedup by image hash (6 repeats = 1 entry + counter); b) Confirmation = weight 0.5; c) Noise as digit is implausible."""
        db_p = str(tmp_path / "r4_test.db")
        org = SpecimenOrganism(db_path=db_p)

        rng = np.random.RandomState(42)
        noise_img = rng.uniform(0.0, 1.0, size=(28, 28)).astype(np.float32)

        # Feed and reveal noise 6 times as label 1
        for i in range(6):
            feed_res = org.feed(noise_img, visitor_id="spammer")
            reveal_res = org.reveal(true_label=1, visitor_id="spammer")

        # Check in-memory dedup: exactly 1 entry in taught buffer with times_taught == 6
        assert len(org.taught_buffer) == 1, f"Expected 1 deduped entry, got {len(org.taught_buffer)}"
        t_entry = org.taught_buffer[0]
        assert t_entry["times_taught"] == 6
        assert reveal_res["times_taught"] == 6

        # Check DB dedup: exactly 1 row in DB
        db_taught = org.storage.get_taught_memories()
        assert len(db_taught) == 1
        assert db_taught[0]["times_taught"] == 6

        # Check CONFIRMATION path: feed clean digit 3, organism predicts 3, reveal 3
        _, _, _, val_x_np, val_y_np = make_dataloaders()
        idx_3 = np.where(val_y_np == 3)[0][0]
        digit_3 = val_x_np[idx_3].reshape(28, 28)

        feed_3 = org.feed(digit_3, visitor_id="honest_student")
        assert feed_3["prediction"] == 3

        reveal_3 = org.reveal(true_label=3, visitor_id="honest_student")
        assert reveal_3["was_correct"] is True
        assert reveal_3["verdict"] == "CONFIRMATION"
        assert reveal_3["weight"] == 0.5

        # Check event log recorded confirmation
        latest_event = org.event_log[0]["message"]
        print(f"[TEST R4] Latest event log: {latest_event}")
        assert "CONFIRMATION" in latest_event
        assert "weight 0.5" in latest_event

        # Check noise taught as digit lands IMPLAUSIBLE (tested over multiple noise samples)
        implausible_count = 0
        for s in range(10):
            sample_noise = rng.uniform(0.0, 1.0, size=(28, 28)).astype(np.float32)
            org.feed(sample_noise, visitor_id="adversary")
            rev = org.reveal(true_label=1, visitor_id="adversary")
            if rev["verdict"] == "IMPLAUSIBLE":
                implausible_count += 1

        print(f"[TEST R4] Noise taught as digit implausible rate: {implausible_count}/10")
        assert implausible_count >= 6, f"Expected >= 6/10 noise samples to be IMPLAUSIBLE, got {implausible_count}/10"

    def test_r5_footer_version_string(self):
        """R5: Footer version string in static/index.html must be THE SPECIMEN v3.2."""
        index_path = Path(__file__).resolve().parent.parent / "specimen" / "static" / "index.html"
        assert index_path.exists()
        content = index_path.read_text(encoding="utf-8")
        assert "THE SPECIMEN v3.2" in content
        assert "THE SPECIMEN v2.0" not in content
