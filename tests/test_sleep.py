"""Unit tests for somnia.sleep -- stability filter and sleep consolidation."""

import torch
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from somnia.models import ClassifierMLP, VAE, IntrospectiveHead
from somnia.sleep import StabilityFilter, SleepConsolidation, identify_hard_subset
import numpy as np


@pytest.fixture
def classifier():
    return ClassifierMLP()


@pytest.fixture
def vae():
    return VAE()


class TestStabilityFilter:
    def test_filter_returns_correct_types(self, classifier):
        sf = StabilityFilter(classifier, n_perturbations=5, noise_std=0.05)
        dreams = torch.rand(20, 784)
        filtered, labels, mask, agreement = sf.filter(dreams, seed=0)
        assert isinstance(filtered, torch.Tensor)
        assert isinstance(labels, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
        assert isinstance(agreement, torch.Tensor)

    def test_filter_shapes(self, classifier):
        sf = StabilityFilter(classifier, n_perturbations=5)
        dreams = torch.rand(30, 784)
        filtered, labels, mask, agreement = sf.filter(dreams, seed=0)
        n_kept = mask.sum().item()
        assert filtered.shape == (n_kept, 784)
        assert labels.shape == (n_kept,)
        assert mask.shape == (30,)
        assert agreement.shape == (30,)

    def test_agreement_bounds(self, classifier):
        sf = StabilityFilter(classifier, n_perturbations=5)
        dreams = torch.rand(20, 784)
        _, _, _, agreement = sf.filter(dreams, seed=0)
        assert agreement.min() >= 0.0
        assert agreement.max() <= 1.0

    def test_high_noise_filters_more(self, classifier):
        """Higher noise should filter out more dreams (lower agreement)."""
        dreams = torch.rand(50, 784)
        sf_low = StabilityFilter(classifier, noise_std=0.01)
        sf_high = StabilityFilter(classifier, noise_std=0.5)
        _, _, mask_low, _ = sf_low.filter(dreams, seed=0)
        _, _, mask_high, _ = sf_high.filter(dreams, seed=0)
        # High noise should keep fewer or equal
        assert mask_high.sum() <= mask_low.sum() + 10  # allow some variance

    def test_deterministic(self, classifier):
        sf = StabilityFilter(classifier)
        dreams = torch.rand(10, 784)
        f1, l1, m1, a1 = sf.filter(dreams, seed=42)
        f2, l2, m2, a2 = sf.filter(dreams, seed=42)
        assert torch.equal(f1, f2)
        assert torch.equal(l1, l2)


class TestSleepConsolidation:
    def test_sleep_cycle_runs(self, classifier):
        consolidator = SleepConsolidation(classifier, dream_weight=0.25)
        real_x = torch.rand(32, 784)
        real_y = torch.randint(0, 10, (32,))
        dream_x = torch.rand(8, 784)
        dream_y = torch.randint(0, 10, (8,))
        real_loss, dream_loss = consolidator.sleep_cycle(real_x, real_y, dream_x, dream_y)
        assert real_loss > 0
        assert dream_loss > 0

    def test_empty_dreams(self, classifier):
        consolidator = SleepConsolidation(classifier)
        real_x = torch.rand(16, 784)
        real_y = torch.randint(0, 10, (16,))
        dream_x = torch.zeros(0, 784)
        dream_y = torch.zeros(0, dtype=torch.long)
        real_loss, dream_loss = consolidator.sleep_cycle(real_x, real_y, dream_x, dream_y)
        assert real_loss > 0
        assert dream_loss == 0.0


class TestHardSubset:
    def test_hard_subset_size(self, classifier):
        test_x = np.random.randn(100, 784).astype(np.float32)
        test_y = np.random.randint(0, 10, 100).astype(np.int64)
        indices = identify_hard_subset(classifier, test_x, test_y, n_hard=20)
        assert len(indices) == 20

    def test_hard_subset_valid_indices(self, classifier):
        test_x = np.random.randn(100, 784).astype(np.float32)
        test_y = np.random.randint(0, 10, 100).astype(np.int64)
        indices = identify_hard_subset(classifier, test_x, test_y, n_hard=30)
        assert all(0 <= i < 100 for i in indices)
