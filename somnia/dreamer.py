"""Dreamer module: dream generation, confusion scoring, and latent mapping."""

import torch
import numpy as np
from sklearn.decomposition import PCA

from somnia.models import ClassifierMLP, VAE, extract_internal_stats, IntrospectiveHead


class Dreamer:
    """Generates dreams from VAE latent space and scores them via introspection.

    The dreamer samples latent vectors z ~ N(0, I), decodes them into
    synthetic images x_dream = D(z), then uses the classifier's internal
    activation statistics to predict P(error | x_dream).

    High-confusion dreams are nightmares -- exactly the training samples
    the network needs most.
    """

    def __init__(self, classifier: ClassifierMLP, vae: VAE,
                 intro_head: IntrospectiveHead):
        self.classifier = classifier
        self.vae = vae
        self.intro_head = intro_head
        self.classifier.eval()
        self.vae.eval()
        self.intro_head.eval()

    @torch.no_grad()
    def dream(self, n: int = 2000, seed: int = 42):
        """Generate n dreams and score their confusion.

        Returns:
            dreams: (n, 784) decoded images in [0,1]
            z_latent: (n, latent_dim) latent vectors
            confusion_scores: (n,) P(error) for each dream
            pseudo_labels: (n,) argmax classifier predictions
            stats: (n, 4) internal activation statistics
        """
        gen = torch.Generator().manual_seed(seed)
        z = torch.randn(n, self.vae.latent_dim, generator=gen)

        # Decode dreams
        dreams = self.vae.decode(z)

        # Get classifier response
        logits, h = self.classifier(dreams)
        probs = torch.softmax(logits, dim=1)
        pseudo_labels = probs.argmax(dim=1)

        # Extract internal stats and score confusion
        stats = extract_internal_stats(h)
        confusion = self.intro_head(stats).squeeze()  # (n,)

        return dreams, z, confusion, pseudo_labels, stats

    @torch.no_grad()
    def dream_from_z(self, z: torch.Tensor):
        """Decode specific latent vectors and score them.

        Returns:
            dreams: decoded images
            confusion_scores: P(error) for each
            pseudo_labels: argmax predictions
        """
        dreams = self.vae.decode(z)
        logits, h = self.classifier(dreams)
        pseudo_labels = torch.softmax(logits, dim=1).argmax(dim=1)
        stats = extract_internal_stats(h)
        confusion = self.intro_head(stats).squeeze()
        return dreams, confusion, pseudo_labels

    @staticmethod
    def fit_pca(z_latent: torch.Tensor, n_components: int = 2):
        """Fit PCA on latent vectors for visualization.

        Returns:
            z_2d: (n, 2) PCA-projected coordinates
            pca: fitted PCA object
        """
        pca = PCA(n_components=n_components, random_state=42)
        z_np = z_latent.numpy() if isinstance(z_latent, torch.Tensor) else z_latent
        z_2d = pca.fit_transform(z_np)
        return z_2d, pca

    @staticmethod
    def select_top_k(dreams: torch.Tensor, scores: torch.Tensor,
                     k: int = 25, highest: bool = True):
        """Select top-k dreams by confusion score.

        Args:
            highest: If True, select most confusing (nightmares).
                     If False, select least confusing (lucid dreams).
        """
        if highest:
            _, indices = scores.topk(k)
        else:
            _, indices = scores.topk(k, largest=False)
        return dreams[indices], scores[indices], indices
