"""Sleep module: stability filter, dream consolidation, and fine-tuning."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from somnia.models import ClassifierMLP, VAE, extract_internal_stats, IntrospectiveHead
from somnia.dreamer import Dreamer


class StabilityFilter:
    """Hallucination filter: retain only dreams with stable pseudo-labels.

    For each dream x_d, apply M stochastic perturbations:
        x_tilde_m = clip(x_d + epsilon_m, 0, 1),  epsilon ~ N(0, sigma^2 I)
    Compute classifier prediction for each perturbation.
    Retain x_d with pseudo-label = majority vote if agreement >= threshold.
    """

    def __init__(self, classifier: ClassifierMLP, n_perturbations: int = 5,
                 noise_std: float = 0.05, agreement_threshold: float = 0.80):
        self.classifier = classifier
        self.M = n_perturbations
        self.noise_std = noise_std
        self.threshold = agreement_threshold

    @torch.no_grad()
    def filter(self, dreams: torch.Tensor, seed: int = 42):
        """Apply stability filter to dreams.

        Returns:
            filtered_dreams: Dreams that pass the filter
            pseudo_labels: Majority-vote labels for filtered dreams
            keep_mask: Boolean mask of which dreams were kept
            agreement_scores: Agreement fraction for all dreams
        """
        self.classifier.eval()
        gen = torch.Generator().manual_seed(seed)
        n = dreams.shape[0]

        # Collect votes from M perturbations
        all_votes = []
        for m in range(self.M):
            noise = torch.randn_like(dreams, generator=gen) * self.noise_std
            perturbed = (dreams + noise).clamp(0, 1)
            logits, _ = self.classifier(perturbed)
            preds = logits.argmax(dim=1)  # (n,)
            all_votes.append(preds)

        votes = torch.stack(all_votes, dim=1)  # (n, M)

        # Majority vote
        majority_labels = torch.zeros(n, dtype=torch.long)
        agreement_scores = torch.zeros(n)
        for i in range(n):
            counts = torch.bincount(votes[i], minlength=10)
            majority_labels[i] = counts.argmax()
            agreement_scores[i] = counts.max().float() / self.M

        # Filter
        keep_mask = agreement_scores >= self.threshold
        filtered_dreams = dreams[keep_mask]
        filtered_labels = majority_labels[keep_mask]

        return filtered_dreams, filtered_labels, keep_mask, agreement_scores


class SleepConsolidation:
    """Sleep cycle: fine-tune classifier on real data + dream mix.

    L_sleep = L_CE(f(x_real), y_real) + lambda_dream * L_CE(f(x_dream), y_pseudo)
    """

    def __init__(self, classifier: ClassifierMLP, dream_weight: float = 0.25,
                 lr: float = 1e-3, epochs: int = 1):
        self.classifier = classifier
        self.dream_weight = dream_weight
        self.lr = lr
        self.epochs = epochs

    def sleep_cycle(self, real_x: torch.Tensor, real_y: torch.Tensor,
                    dream_x: torch.Tensor, dream_y: torch.Tensor):
        """Run one sleep consolidation cycle.

        Returns:
            real_loss: Final loss on real data
            dream_loss: Final loss on dream data
        """
        self.classifier.train()
        optimizer = optim.Adam(self.classifier.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        real_loss_val = 0.0
        dream_loss_val = 0.0

        for _ in range(self.epochs):
            optimizer.zero_grad()

            # Real data loss
            logits_real, _ = self.classifier(real_x)
            loss_real = criterion(logits_real, real_y)

            # Dream data loss
            if dream_x.shape[0] > 0:
                logits_dream, _ = self.classifier(dream_x)
                loss_dream = criterion(logits_dream, dream_y)
            else:
                loss_dream = torch.tensor(0.0)

            # Combined
            total_loss = loss_real + self.dream_weight * loss_dream
            total_loss.backward()
            optimizer.step()

            real_loss_val = loss_real.item()
            dream_loss_val = loss_dream.item()

        return real_loss_val, dream_loss_val


def identify_hard_subset(classifier: ClassifierMLP, test_x: np.ndarray,
                         test_y: np.ndarray, n_hard: int = 500):
    """Identify the hardest test samples by classification margin.

    Margin = p_(1) - p_(2) (difference between top-2 class probabilities).
    Low margin = high ambiguity = hard sample.
    """
    classifier.eval()
    with torch.no_grad():
        x_t = torch.from_numpy(test_x)
        logits, _ = classifier(x_t)
        probs = torch.softmax(logits, dim=1)
        top2 = probs.topk(2, dim=1).values
        margins = top2[:, 0] - top2[:, 1]

    _, hard_indices = margins.topk(n_hard, largest=False)
    return hard_indices.numpy()
