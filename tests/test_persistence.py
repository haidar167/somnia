"""Unit test verifying that THE SPECIMEN survives restart via SQLite persistence."""

import tempfile
from pathlib import Path
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from specimen.organism import SpecimenOrganism


def test_organism_persistence_across_restarts():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_specimen.db"

        # 1. First life: instantiate organism
        org_1 = SpecimenOrganism(db_path=db_path)
        assert org_1.total_feeds == 0
        assert org_1.sleep_count == 0

        # Feed 3 samples
        rng = np.random.RandomState(42)
        for _ in range(3):
            sample = rng.rand(28, 28).astype(np.float32)
            org_1.feed(sample, visitor_id="test_visitor_001")

        # Reveal 1 label
        org_1.reveal(true_label=7, visitor_id="test_visitor_001")

        # Trigger 1 sleep cycle
        org_1.sleep_cycle()

        state_before = org_1.get_state()
        feeds_before = state_before["total_feeds"]
        sleeps_before = state_before["sleep_count"]
        dreams_before = len(state_before["recent_dreams"])
        events_before_count = len(state_before["event_log"])

        assert feeds_before == 3
        assert sleeps_before == 1
        assert dreams_before == 8

        # 2. Destroy first organism instance (simulating server restart / process death)
        del org_1

        # 3. Second life: reload organism from the same SQLite DB
        org_2 = SpecimenOrganism(db_path=db_path)
        state_after = org_2.get_state()

        # 4. Assert all counters, dreams, and event logs survived!
        assert state_after["total_feeds"] == feeds_before
        assert state_after["sleep_count"] == sleeps_before
        assert len(state_after["recent_dreams"]) == dreams_before
        assert len(state_after["event_log"]) > 0

        print("\n[PERSISTENCE SURVIVAL PROOF]")
        print(f"  Before Restart: Feeds={feeds_before}, Sleeps={sleeps_before}, Dreams={dreams_before}")
        print(f"  After Restart : Feeds={state_after['total_feeds']}, Sleeps={state_after['sleep_count']}, Dreams={len(state_after['recent_dreams'])}")
        print("  -> Organism successfully survived process restart!")
