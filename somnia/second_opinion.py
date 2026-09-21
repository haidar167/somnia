"""SOMNIA v3.1 Second Opinion: Independent Generative Vouching via cVAE.

Provides an independent plausibility check for taught sample corrections (x, y_taught)
that does NOT rely on the classifier's own weights.

Verdict logic:
- PLAUSIBLE if BCE(x | y_taught) <= min_{c in 0..9} BCE(x | c) * 1.10
- IMPLAUSIBLE otherwise

Weight policy:
- stable (agreement >= 4/5) AND agrees with classifier: weight 2.0 (safe reinforcement)
- unstable (or disagreeing) AND PLAUSIBLE: weight 1.0 (plausible correction rescued by second opinion)
- unstable (or disagreeing) AND IMPLAUSIBLE: weight 0.1 (likely liar / noisy corruption - whisper)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any, List

from somnia.models import ConditionalVAE, ClassifierMLP


class SecondOpinionVoucher:
    """Evaluates taught samples using cVAE reconstruction likelihood as an independent second opinion."""

    def __init__(self, cvae: ConditionalVAE, tolerance_ratio: float = 1.10):
        self.cvae = cvae
        self.cvae.eval()
        self.tolerance_ratio = tolerance_ratio

    @torch.no_grad()
    def compute_all_class_bce(self, x: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction BCE for each sample under all 10 candidate class conditionings.

        Args:
            x: shape (N, 784), values in [0, 1]

        Returns:
            all_bce: shape (N, 10)
        """
        n = x.shape[0]
        all_bce = []
        for c in range(10):
            c_tensor = torch.full((n,), c, dtype=torch.long)
            recon, _, _ = self.cvae(x, c_tensor)
            # Sum BCE across pixels for each image
            bce_per_sample = F.binary_cross_entropy(recon, x, reduction="none").sum(dim=-1)  # (N,)
            all_bce.append(bce_per_sample)
        return torch.stack(all_bce, dim=1)  # (N, 10)

    @torch.no_grad()
    def evaluate_taught_batch(
        self,
        x: torch.Tensor,
        y_taught: torch.Tensor,
        classifier: ClassifierMLP,
        n_perturbations: int = 5,
        noise_std: float = 0.05,
        seed: int = 42
    ) -> Dict[str, Any]:
        """Compute second opinion verdicts and composite consolidation weights.

        Args:
            x: shape (N, 784)
            y_taught: shape (N,)
            classifier: ClassifierMLP to check nominal predictions & perturbation votes

        Returns: dict with:
            weights: (N,) tensor in {2.0, 1.0, 0.1}
            verdicts: list of str ('stable_reinforcement', 'plausible_correction', 'implausible_whisper')
            is_plausible: (N,) boolean tensor
            nominal_preds: (N,) tensor
            bce_taught: (N,) tensor
            bce_min: (N,) tensor
        """
        classifier.eval()
        n = x.shape[0]
        if n == 0:
            return {
                "weights": torch.empty((0,)),
                "verdicts": [],
                "is_plausible": torch.empty((0,), dtype=torch.bool),
                "nominal_preds": torch.empty((0,), dtype=torch.long),
            }

        # 1. Nominal classification and perturbation votes
        logits_nom, _ = classifier(x)
        nominal_preds = logits_nom.argmax(dim=1)

        gen = torch.Generator().manual_seed(seed)
        all_votes = []
        for _ in range(n_perturbations):
            noise = torch.randn_like(x, generator=gen) * noise_std
            pert = (x + noise).clamp(0, 1)
            all_votes.append(classifier(pert)[0].argmax(dim=1))
        votes_stack = torch.stack(all_votes, dim=1)  # (N, M)

        # 2. Generative cVAE Second Opinion
        all_bce = self.compute_all_class_bce(x)  # (N, 10)
        idx_range = torch.arange(n)
        bce_taught = all_bce[idx_range, y_taught]
        bce_min = all_bce.min(dim=1).values
        bce_threshold = bce_min * self.tolerance_ratio
        is_plausible = (bce_taught <= bce_threshold)

        # 3. Composite Weight Policy
        weights = torch.zeros(n, dtype=torch.float32)
        verdicts = []

        for i in range(n):
            agreement = (votes_stack[i] == y_taught[i]).float().mean().item()
            agrees_with_classifier = (nominal_preds[i].item() == y_taught[i].item())
            is_stable = (agreement >= 0.80)  # 4/5 or 5/5

            if is_stable and agrees_with_classifier:
                weights[i] = 2.0
                verdicts.append("stable_reinforcement")
            elif is_plausible[i].item():
                weights[i] = 1.0
                verdicts.append("plausible_correction")
            else:
                weights[i] = 0.1
                verdicts.append("implausible_whisper")

        return {
            "weights": weights,
            "verdicts": verdicts,
            "is_plausible": is_plausible,
            "nominal_preds": nominal_preds,
            "bce_taught": bce_taught,
            "bce_min": bce_min,
            "bce_all": all_bce,
        }
