"""Unit tests for somnia.dreamer -- dream generation and confusion mapping."""

import torch
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from somnia.models import ClassifierMLP, VAE, IntrospectiveHead
from somnia.dreamer import Dreamer


@pytest.fixture
def dreamer():
    classifier = ClassifierMLP()
    vae = VAE()
    intro_head = IntrospectiveHead()
    return Dreamer(classifier, vae, intro_head)


class TestDreamer:
    def test_dream_shapes(self, dreamer):
        dreams, z, confusion, labels, stats = dreamer.dream(n=16, seed=42)
        assert dreams.shape == (16, 784)
        assert z.shape == (16, 16)
        assert confusion.shape == (16,)
        assert labels.shape == (16,)
        assert stats.shape == (16, 4)

    def test_dreams_in_01(self, dreamer):
        dreams, _, _, _, _ = dreamer.dream(n=8, seed=0)
        assert dreams.min() >= 0.0 and dreams.max() <= 1.0

    def test_confusion_in_01(self, dreamer):
        _, _, confusion, _, _ = dreamer.dream(n=8, seed=0)
        assert confusion.min() >= 0.0 and confusion.max() <= 1.0

    def test_labels_valid(self, dreamer):
        _, _, _, labels, _ = dreamer.dream(n=32, seed=0)
        assert labels.min() >= 0 and labels.max() <= 9

    def test_dream_from_z(self, dreamer):
        z = torch.randn(5, 16)
        dreams, confusion, labels = dreamer.dream_from_z(z)
        assert dreams.shape == (5, 784)
        assert confusion.shape == (5,)
        assert labels.shape == (5,)

    def test_deterministic(self, dreamer):
        d1, _, c1, _, _ = dreamer.dream(n=10, seed=42)
        d2, _, c2, _, _ = dreamer.dream(n=10, seed=42)
        assert torch.allclose(d1, d2)
        assert torch.allclose(c1, c2)


class TestDreamerPCA:
    def test_pca_shape(self, dreamer):
        _, z, _, _, _ = dreamer.dream(n=50, seed=0)
        z_2d, pca = Dreamer.fit_pca(z)
        assert z_2d.shape == (50, 2)

    def test_pca_explained_variance(self, dreamer):
        _, z, _, _, _ = dreamer.dream(n=50, seed=0)
        _, pca = Dreamer.fit_pca(z)
        assert sum(pca.explained_variance_ratio_) <= 1.0


class TestDreamerSelection:
    def test_select_top_k_nightmares(self, dreamer):
        dreams, _, confusion, _, _ = dreamer.dream(n=50, seed=0)
        top, scores, idx = Dreamer.select_top_k(dreams, confusion, k=5, highest=True)
        assert top.shape[0] == 5
        assert scores.shape[0] == 5
        # Verify sorted descending
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]

    def test_select_top_k_lucid(self, dreamer):
        dreams, _, confusion, _, _ = dreamer.dream(n=50, seed=0)
        top, scores, idx = Dreamer.select_top_k(dreams, confusion, k=5, highest=False)
        assert top.shape[0] == 5
        # Verify sorted ascending
        for i in range(len(scores) - 1):
            assert scores[i] <= scores[i + 1]
