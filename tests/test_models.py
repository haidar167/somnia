"""Unit tests for somnia.models — ClassifierMLP, VAE, IntrospectiveHead."""

import torch
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from somnia.models import (
    ClassifierMLP, VAE, extract_internal_stats, IntrospectiveHead,
    ConditionalVAE, extract_internal_stats_v2, IntrospectiveHeadV2,
)


class TestClassifierMLP:
    def test_output_shapes(self):
        model = ClassifierMLP(784, 256, 10)
        x = torch.randn(4, 784)
        logits, h = model(x)
        assert logits.shape == (4, 10), f"Expected (4,10), got {logits.shape}"
        assert h.shape == (4, 256), f"Expected (4,256), got {h.shape}"

    def test_predict_proba_sums_to_one(self):
        model = ClassifierMLP()
        x = torch.randn(8, 784)
        probs = model.predict_proba(x)
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones(8), atol=1e-5)

    def test_gradients_flow(self):
        model = ClassifierMLP()
        x = torch.randn(2, 784)
        logits, _ = model(x)
        loss = logits.sum()
        loss.backward()
        assert model.fc1.weight.grad is not None


class TestVAE:
    def test_output_shapes(self):
        vae = VAE(784, 256, 16)
        x = torch.randn(4, 784).clamp(0, 1)
        recon, mu, logvar = vae(x)
        assert recon.shape == (4, 784)
        assert mu.shape == (4, 16)
        assert logvar.shape == (4, 16)

    def test_recon_in_01(self):
        vae = VAE()
        x = torch.rand(4, 784)
        recon, _, _ = vae(x)
        assert recon.min() >= 0.0 and recon.max() <= 1.0

    def test_loss_positive(self):
        vae = VAE()
        x = torch.rand(4, 784)
        recon, mu, logvar = vae(x)
        loss = VAE.loss_function(recon, x, mu, logvar)
        assert loss.item() > 0

    def test_sample_shapes(self):
        vae = VAE()
        dreams = vae.sample(10, seed=42)
        assert dreams.shape == (10, 784)
        assert dreams.min() >= 0.0 and dreams.max() <= 1.0

    def test_encode_decode_roundtrip(self):
        vae = VAE()
        x = torch.rand(2, 784)
        mu, logvar = vae.encode(x)
        z = vae.reparameterize(mu, logvar)
        recon = vae.decode(z)
        assert recon.shape == x.shape


class TestInternalStats:
    def test_shape(self):
        h = torch.randn(8, 256)
        stats = extract_internal_stats(h)
        assert stats.shape == (8, 4)

    def test_alive_fraction_bounds(self):
        h = torch.randn(100, 256)
        stats = extract_internal_stats(h)
        alive = stats[:, 2]
        assert alive.min() >= 0.0 and alive.max() <= 1.0

    def test_all_positive_input(self):
        h = torch.abs(torch.randn(4, 256)) + 0.1
        stats = extract_internal_stats(h)
        assert torch.all(stats[:, 2] == 1.0)  # all neurons alive


class TestIntrospectiveHead:
    def test_output_shape(self):
        head = IntrospectiveHead(4)
        stats = torch.randn(8, 4)
        out = head(stats)
        assert out.shape == (8, 1)

    def test_output_in_01(self):
        head = IntrospectiveHead(4)
        stats = torch.randn(16, 4)
        out = head(stats)
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_gradient_flow(self):
        head = IntrospectiveHead(4)
        stats = torch.randn(4, 4)
        out = head(stats)
        out.sum().backward()
        assert head.linear.weight.grad is not None


class TestConditionalVAE:
    def test_output_shapes(self):
        cvae = ConditionalVAE(784, 256, 32, 10)
        x = torch.randn(4, 784).clamp(0, 1)
        y = torch.tensor([0, 3, 7, 9])
        recon, mu, logvar = cvae(x, y)
        assert recon.shape == (4, 784)
        assert mu.shape == (4, 32)
        assert logvar.shape == (4, 32)

    def test_recon_in_01(self):
        cvae = ConditionalVAE()
        x = torch.rand(4, 784)
        y = torch.tensor([1, 2, 4, 8])
        recon, _, _ = cvae(x, y)
        assert recon.min() >= 0.0 and recon.max() <= 1.0

    def test_sample_shapes(self):
        cvae = ConditionalVAE()
        dreams, labels = cvae.sample(n_per_class=5, seed=42)
        assert dreams.shape == (50, 784)
        assert labels.shape == (50,)
        assert dreams.min() >= 0.0 and dreams.max() <= 1.0

    def test_sample_class(self):
        cvae = ConditionalVAE()
        dreams, labels = cvae.sample_class(n=7, target_class=3, seed=42)
        assert dreams.shape == (7, 784)
        assert (labels == 3).all()


class TestInternalStatsV2:
    def test_shape(self):
        h = torch.randn(8, 256)
        logits = torch.randn(8, 10)
        from somnia.models import extract_internal_stats_v2
        stats = extract_internal_stats_v2(h, logits)
        assert stats.shape == (8, 10)


class TestIntrospectiveHeadV2:
    def test_output_shape(self):
        from somnia.models import IntrospectiveHeadV2
        head = IntrospectiveHeadV2(10, 32)
        stats = torch.randn(8, 10)
        out = head(stats)
        assert out.shape == (8, 1)
        assert out.min() >= 0.0 and out.max() <= 1.0

