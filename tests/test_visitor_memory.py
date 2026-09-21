"""Unit tests for Anonymous Visitor Memory tracking and recognition."""

import tempfile
from pathlib import Path
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from specimen.organism import SpecimenOrganism
from specimen.storage import SpecimenStorage


def test_anonymous_visitor_memory():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_visitors.db"
        storage = SpecimenStorage(db_path=db_path)
        organism = SpecimenOrganism(db_path=db_path)

        visitor_A = "mind_7f3a01"
        visitor_B = "mind_8e2c99"

        # 1. Visitor A feeds for the first time
        sample = np.zeros((28, 28), dtype=np.float32)
        res_a1 = organism.feed(sample, visitor_id=visitor_A)
        assert res_a1["visitor_info"]["is_returning"] is False
        assert res_a1["visitor_info"]["total_feeds"] == 1

        # 2. Visitor B feeds
        res_b1 = organism.feed(sample, visitor_id=visitor_B)
        assert res_b1["visitor_info"]["is_returning"] is False
        assert res_b1["visitor_info"]["total_feeds"] == 1

        # Assert 2 distinct visitors recognized
        assert storage.get_visitor_count() == 2

        # 3. Visitor A returns for a second feed
        res_a2 = organism.feed(sample, visitor_id=visitor_A)
        assert res_a2["visitor_info"]["is_returning"] is True
        assert res_a2["visitor_info"]["total_feeds"] == 2

        # Assert visitor count is still 2
        assert storage.get_visitor_count() == 2

        # 4. Visitor A reveals label
        organism.reveal(true_label=0, visitor_id=visitor_A)

        # 5. Privacy check: verify storage schema contains only anonymous visitor_id
        with storage._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("PRAGMA table_info(visitors)")
            columns = [col["name"] for col in cur.fetchall()]
            assert "visitor_id" in columns
            assert "ip_address" not in columns
            assert "email" not in columns
            assert "user_agent" not in columns
