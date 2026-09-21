"""SOMNIA v2 Sleep consolidation with Soft Stability Weighting and Conditional Dreamer."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from somnia.models import (
    ClassifierMLP, ConditionalVAE, IntrospectiveHeadV2, extract_internal_stats_v2
)


class SoftStabilityFilter:
    """Soft stability weighting for dream regularization.

    Instead of discarding unstable dreams (which loses diversity),
    every dream is assigned a continuous confidence weight w_i = agreement / M in [0.2, 1.0].
    Stable dreams speak loudly (w=1.0); ambiguous dreams whisper (w=0.2 - 0.6).
    """

    def __init__(self, classifier: ClassifierMLP, n_perturbations: int = 5,
                 noise_std: float = 0.05):
        self.classifier = classifier
        self.M = n_perturbations
        self.noise_std = noise_std

    @torch.no_grad()
    def compute_soft_weights(self, dreams: torch.Tensor, seed: int = 42):
        """Compute soft weights and majority pseudo-labels for dreams.

        Returns:
            pseudo_labels: (n,) majority predicted class
            weights: (n,) continuous weight in [0, 1] (agreement / M)
        """
        self.classifier.eval()
        gen = torch.Generator().manual_seed(seed)
        n = dreams.shape[0]

        all_votes = []
        for m in range(self.M):
            noise = torch.randn_like(dreams, generator=gen) * self.noise_std
            perturbed = (dreams + noise).clamp(0, 1)
            logits, _ = self.classifier(perturbed)
            preds = logits.argmax(dim=1)
            all_votes.append(preds)

        votes = torch.stack(all_votes, dim=1)  # (n, M)

        majority_labels = torch.zeros(n, dtype=torch.long)
        weights = torch.zeros(n)
        for i in range(n):
            counts = torch.bincount(votes[i], minlength=10)
            majority_labels[i] = counts.argmax()
            weights[i] = counts.max().float() / self.M

        return majority_labels, weights


class DreamerV2:
    """Class-aware and introspection-guided dream generator for SOMNIA v2."""

    def __init__(self, classifier: ClassifierMLP, cvae: ConditionalVAE,
                 intro_head: IntrospectiveHeadV2):
        self.classifier = classifier
        self.cvae = cvae
        self.intro_head = intro_head
        self.classifier.eval()
        self.cvae.eval()
        self.intro_head.eval()

    @torch.no_grad()
    def get_class_confusion_profile(self, x_samples: torch.Tensor):
        """Compute average predicted error probability per class."""
        logits, h = self.classifier(x_samples)
        preds = logits.argmax(1)
        stats = extract_internal_stats_v2(h, logits)
        p_err = self.intro_head(stats).squeeze()

        class_confusion = torch.zeros(10)
        for c in range(10):
            mask = (preds == c)
            if mask.sum() > 0:
                class_confusion[c] = p_err[mask].mean()
            else:
                class_confusion[c] = 0.5
        return class_confusion

    @torch.no_grad()
    def generate_random_dreams(self, n: int = 500, seed: int = 42):
        """Sample random dreams uniformly across all 10 classes."""
        n_per_class = max(1, n // 10)
        dreams, labels = self.cvae.sample(n_per_class=n_per_class, seed=seed)
        return dreams[:n], labels[:n]

    @torch.no_grad()
    def generate_targeted_dreams(self, class_confusion: torch.Tensor, n: int = 500, seed: int = 42):
        """Sample dreams preferentially from classes with highest introspective confusion."""
        # Convert class confusion to sampling probabilities via temperature softmax
        probs = torch.softmax(class_confusion * 5.0, dim=0).numpy()

        rng = np.random.RandomState(seed)
        sampled_classes = rng.choice(10, size=n, p=probs)

        # Generate dreams conditioned on the chosen classes
        gen = torch.Generator().manual_seed(seed)
        all_dreams = []
        all_labels = []

        for c in range(10):
            count = int((sampled_classes == c).sum())
            if count > 0:
                z = torch.randn(count, self.cvae.latent_dim, generator=gen)
                labels = torch.full((count,), c, dtype=torch.long)
                decoded = self.cvae.decode(z, labels)
                all_dreams.append(decoded)
                all_labels.append(labels)

        if len(all_dreams) == 0:
            return self.generate_random_dreams(n, seed)

        return torch.cat(all_dreams, dim=0), torch.cat(all_labels, dim=0)


class SleepConsolidationV2:
    """Gentle sleep consolidation with soft weighted dream loss.

    L_sleep = L_CE(f(x_real), y_real) + lambda_mix * sum_i (w_i * L_CE(f(x_dream_i), y_hat_i)) / sum(w_i)
    """

    def __init__(self, classifier: ClassifierMLP, dream_mix: float = 0.10,
                 lr: float = 1e-4, epochs: int = 2):
        self.classifier = classifier
        self.dream_mix = dream_mix
        self.lr = lr
        self.epochs = epochs

    def sleep_cycle(self, real_x: torch.Tensor, real_y: torch.Tensor,
                    dream_x: torch.Tensor, dream_y: torch.Tensor,
                    dream_weights: torch.Tensor,
                    taught_x: torch.Tensor = None,
                    taught_y: torch.Tensor = None,
                    taught_weights: torch.Tensor = None):
        """Run gentle sleep consolidation with soft weighted dream loss and taught memories replay."""
        self.classifier.train()
        optimizer = optim.Adam(self.classifier.parameters(), lr=self.lr)
        criterion_unreduced = nn.CrossEntropyLoss(reduction="none")
        criterion_real = nn.CrossEntropyLoss()

        real_loss_val = 0.0
        dream_loss_val = 0.0
        taught_loss_val = 0.0

        for _ in range(self.epochs):
            optimizer.zero_grad()

            logits_real, _ = self.classifier(real_x)
            loss_real = criterion_real(logits_real, real_y)

            if dream_x.shape[0] > 0 and dream_weights.sum() > 0:
                logits_dream, _ = self.classifier(dream_x)
                losses_dream = criterion_unreduced(logits_dream, dream_y)
                loss_dream = (losses_dream * dream_weights).sum() / (dream_weights.sum() + 1e-8)
            else:
                loss_dream = torch.tensor(0.0)

            loss_taught = torch.tensor(0.0)
            if taught_x is not None and taught_x.shape[0] > 0 and taught_weights is not None and taught_weights.sum() > 0:
                logits_taught, _ = self.classifier(taught_x)
                losses_taught = criterion_unreduced(logits_taught, taught_y)
                loss_taught = (losses_taught * taught_weights).sum() / (taught_weights.sum() + 1e-8)

            total_loss = loss_real + self.dream_mix * loss_dream + loss_taught
            total_loss.backward()
            optimizer.step()

            real_loss_val = loss_real.item()
            dream_loss_val = loss_dream.item()
            taught_loss_val = loss_taught.item()

        return real_loss_val, dream_loss_val
