"""Tests for Phase 0 polish: debounce, dream score variance, and memory diversity guard."""

import pytest
import numpy as np
import torch
import time
from specimen.organism import SpecimenOrganism


class TestPhase0Polish:
    def test_feed_debounce_rapid_identical(self, tmp_path):
        db_path = str(tmp_path / "test_debounce.db")
        org = SpecimenOrganism(db_path=db_path)
        img = np.random.rand(28, 28).astype(np.float32)

        results = []
        for _ in range(10):
            res = org.feed(img)
            results.append(res)

        accepted = [r for r in results if r.get("status") != "rejected"]
        rejected = [r for r in results if r.get("status") == "rejected"]

        assert len(accepted) == 1, f"Expected exactly 1 accepted feed, got {len(accepted)}"
        assert len(rejected) == 9, f"Expected 9 rejected duplicate feeds, got {len(rejected)}"
        assert rejected[0]["reason"] == "duplicate_within_300ms"
        assert rejected[0]["message"] == "FEED rejected: duplicate within 300ms."

    def test_dream_score_variance_and_badges(self, tmp_path):
        db_path = str(tmp_path / "test_dreams.db")
        org = SpecimenOrganism(db_path=db_path)

        # Generate dreams
        dreams, labels = org.dreamer.generate_random_dreams(n=16, seed=42)
        with torch.no_grad():
            logits, h = org.classifier(dreams)
            from somnia.models import extract_internal_stats_v2
            stats = extract_internal_stats_v2(h, logits)
            p_errs = org.intro_head(stats).squeeze()

        p_errs_np = p_errs.numpy()
        var = float(np.var(p_errs_np))

        # Assert nonzero variance
        assert var > 0.01, f"Dream confusion scores have near-zero variance: {var}"

        # Assert recent_dreams contains diverse scores
        dream_scores = [d["p_error"] for d in org.recent_dreams]
        assert len(set(dream_scores)) > 1, "Recent dreams all have identical scores"

    def test_memory_diversity_guard_spam(self, tmp_path):
        db_path = str(tmp_path / "test_diversity.db")
        org = SpecimenOrganism(db_path=db_path)

        # Generate 50 identical noise feeds with small sleep to bypass 300ms debounce
        noise = np.random.rand(28, 28).astype(np.float32)
        
        # We simulate 50 feeds of same noise with debounce reset
        for i in range(50):
            org.last_feed_time = 0.0  # Reset debounce timer to test buffer guard
            org.feed(noise)

        # Diversity guard should cap identical prediction + p_error within 1s to at most 5
        assert len(org.memory_buffer) <= 5, f"Expected <= 5 retained memories, got {len(org.memory_buffer)}"
        guarded = org.get_diversity_guarded_memory()
        assert len(guarded) <= 5, f"Expected <= 5 guarded memories, got {len(guarded)}"
