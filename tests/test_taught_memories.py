"""Unit tests for Phase 1: Taught memories & visitor teacher tracking."""

import pytest
import numpy as np
from pathlib import Path
from specimen.organism import SpecimenOrganism
from specimen.storage import SpecimenStorage


class TestTaughtMemories:
    def test_reveal_grows_taught_buffer(self, tmp_path):
        db_path = str(tmp_path / "test_taught.db")
        org = SpecimenOrganism(db_path=db_path)
        assert len(org.taught_buffer) == 0

        # Feed image
        img = np.random.rand(28, 28).astype(np.float32)
        feed_res = org.feed(img, visitor_id="teacher_alpha")
        pred = feed_res["prediction"]

        # Reveal label
        rev_res = org.reveal(true_label=7, visitor_id="teacher_alpha")
        assert len(org.taught_buffer) == 1
        assert org.taught_buffer[-1]["true_label"] == 7
        assert org.taught_buffer[-1]["what_the_organism_said"] == pred
        assert org.taught_buffer[-1]["visitor_id"] == "teacher_alpha"
        assert rev_res["taught_count"] == 1

        # Check get_state()
        state = org.get_state()
        assert state["taught_count"] == 1

        # Check persistence across fresh instance
        org2 = SpecimenOrganism(db_path=db_path)
        assert len(org2.taught_buffer) == 1
        assert org2.taught_buffer[-1]["true_label"] == 7

    def test_independent_visitor_tracking(self, tmp_path):
        db_path = str(tmp_path / "test_multiteacher.db")
        org = SpecimenOrganism(db_path=db_path)
        storage = SpecimenStorage(db_path=Path(db_path))

        visitor_1 = "mind_uuid_111"
        visitor_2 = "mind_uuid_222"

        # Visitor 1 feeds 2 images, reveals 1
        img1 = np.zeros((28, 28), dtype=np.float32)
        org.feed(img1, visitor_id=visitor_1)
        org.reveal(true_label=0, visitor_id=visitor_1)

        img2 = np.ones((28, 28), dtype=np.float32)
        org.feed(img2, visitor_id=visitor_1)

        # Visitor 2 feeds 1 image, reveals 1
        img3 = np.random.rand(28, 28).astype(np.float32)
        org.feed(img3, visitor_id=visitor_2)
        org.reveal(true_label=4, visitor_id=visitor_2)

        p1 = storage.get_visitor_profile(visitor_1)
        p2 = storage.get_visitor_profile(visitor_2)

        assert p1 is not None
        assert p2 is not None
        assert p1["total_feeds"] == 2
        assert p1["total_reveals"] == 1
        assert p2["total_feeds"] == 1
        assert p2["total_reveals"] == 1
        assert storage.get_visitor_count() == 2
        assert len(org.taught_buffer) == 2
