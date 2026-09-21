"""Neural network models: ClassifierMLP, VAE, and IntrospectiveHead."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassifierMLP(nn.Module):
    """Simple 2-layer MLP classifier: 784 → 256 (ReLU) → 10.

    Exposes penultimate activations h ∈ ℝ^256 for introspective analysis.
    """

    def __init__(self, input_dim: int = 784, hidden_dim: int = 256, num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor):
        """Return (logits, penultimate_h)."""
        h = F.relu(self.fc1(x))
        logits = self.fc2(h)
        return logits, h

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities."""
        logits, _ = self.forward(x)
        return F.softmax(logits, dim=-1)


class VAE(nn.Module):
    """Variational Autoencoder: 784 → 256 → (μ, logvar) ∈ ℝ^16 → 256 → 784.

    Standard ELBO training with BCE reconstruction + KL divergence.
    """

    def __init__(self, input_dim: int = 784, hidden_dim: int = 256, latent_dim: int = 16):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.enc_fc1 = nn.Linear(input_dim, hidden_dim)
        self.enc_mu = nn.Linear(hidden_dim, latent_dim)
        self.enc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.dec_fc1 = nn.Linear(latent_dim, hidden_dim)
        self.dec_out = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: torch.Tensor):
        """Return (mu, logvar)."""
        h = F.relu(self.enc_fc1(x))
        return self.enc_mu(h), self.enc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample z via reparameterization trick: z = μ + σ·ε."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent z to reconstructed x in [0, 1]."""
        h = F.relu(self.dec_fc1(z))
        return torch.sigmoid(self.dec_out(h))

    def forward(self, x: torch.Tensor):
        """Return (x_recon, mu, logvar)."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    @staticmethod
    def loss_function(x_recon: torch.Tensor, x: torch.Tensor,
                      mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """ELBO loss = BCE reconstruction + KL divergence.

        L_VAE = E_q[-log p(x|z)] + D_KL(q(z|x) || p(z))
        """
        bce = F.binary_cross_entropy(x_recon, x, reduction="sum")
        # KL(N(μ, σ²) || N(0, I)) = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return bce + kl

    def sample(self, n: int, seed: int = 42) -> torch.Tensor:
        """Sample n dreams from the prior N(0, I)."""
        gen = torch.Generator().manual_seed(seed)
        z = torch.randn(n, self.latent_dim, generator=gen)
        with torch.no_grad():
            return self.decode(z)


def extract_internal_stats(h: torch.Tensor) -> torch.Tensor:
    """Extract the 4-dimensional somatosensory feature vector from activations h.

    z_stats = [μ(h), σ(h), ρ_{>0}(h), μ_top10%(h)]^T ∈ ℝ^4

    Args:
        h: Penultimate activations, shape (batch, hidden_dim).

    Returns:
        stats: Shape (batch, 4).
    """
    mu = h.mean(dim=1, keepdim=True)                     # mean activation
    sigma = h.std(dim=1, keepdim=True) + 1e-8             # activation spread
    alive = (h > 0).float().mean(dim=1, keepdim=True)     # fraction of alive neurons
    # Top 10% mean
    k = max(1, h.shape[1] // 10)
    topk_vals, _ = h.topk(k, dim=1)
    top_mean = topk_vals.mean(dim=1, keepdim=True)
    return torch.cat([mu, sigma, alive, top_mean], dim=1)


class IntrospectiveHead(nn.Module):
    """Logistic regression on 4 internal stats -> P(error).

    s_intro = sigma(w^T z_stats + b)
    """

    def __init__(self, input_dim: int = 4):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, stats: torch.Tensor) -> torch.Tensor:
        """Return P(error) in [0, 1], shape (batch, 1)."""
        return torch.sigmoid(self.linear(stats))


# ============================================================
# SOMNIA v2 Models
# ============================================================


class ConditionalVAE(nn.Module):
    """Conditional VAE: class-aware encoder and decoder for sharper dreams.

    Encoder: [image(784) + one-hot class(10)] -> 256 -> (mu, logvar) in R^32
    Decoder: [z(32) + one-hot class(10)] -> 256 -> 784 (sigmoid)

    Conditioning on class labels allows the decoder to produce sharper,
    class-specific reconstructions, fixing the v1 blurry-dream bottleneck.
    """

    def __init__(self, input_dim: int = 784, hidden_dim: int = 256,
                 latent_dim: int = 32, num_classes: int = 10):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # Encoder: input + one-hot class
        self.enc_fc1 = nn.Linear(input_dim + num_classes, hidden_dim)
        self.enc_mu = nn.Linear(hidden_dim, latent_dim)
        self.enc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder: z + one-hot class
        self.dec_fc1 = nn.Linear(latent_dim + num_classes, hidden_dim)
        self.dec_out = nn.Linear(hidden_dim, input_dim)

    def _one_hot(self, labels: torch.Tensor) -> torch.Tensor:
        """Convert integer labels to one-hot vectors."""
        return F.one_hot(labels, self.num_classes).float()

    def encode(self, x: torch.Tensor, labels: torch.Tensor):
        """Return (mu, logvar) conditioned on class label."""
        c = self._one_hot(labels)
        h = F.relu(self.enc_fc1(torch.cat([x, c], dim=1)))
        return self.enc_mu(h), self.enc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample z via reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Decode latent z conditioned on class label to x in [0, 1]."""
        c = self._one_hot(labels)
        h = F.relu(self.dec_fc1(torch.cat([z, c], dim=1)))
        return torch.sigmoid(self.dec_out(h))

    def forward(self, x: torch.Tensor, labels: torch.Tensor):
        """Return (x_recon, mu, logvar)."""
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, labels)
        return x_recon, mu, logvar

    @staticmethod
    def loss_function(x_recon: torch.Tensor, x: torch.Tensor,
                      mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """ELBO loss = BCE reconstruction + KL divergence."""
        bce = F.binary_cross_entropy(x_recon, x, reduction="sum")
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return bce + kl

    def sample(self, n_per_class: int, seed: int = 42) -> torch.Tensor:
        """Sample n_per_class dreams for each of the 10 classes.

        Returns: (n_per_class * 10, 784) decoded images, and labels.
        """
        gen = torch.Generator().manual_seed(seed)
        all_dreams = []
        all_labels = []
        for c in range(self.num_classes):
            z = torch.randn(n_per_class, self.latent_dim, generator=gen)
            labels = torch.full((n_per_class,), c, dtype=torch.long)
            with torch.no_grad():
                dreams = self.decode(z, labels)
            all_dreams.append(dreams)
            all_labels.append(labels)
        return torch.cat(all_dreams, dim=0), torch.cat(all_labels, dim=0)

    def sample_class(self, n: int, target_class: int, seed: int = 42) -> torch.Tensor:
        """Sample n dreams conditioned on a specific class."""
        gen = torch.Generator().manual_seed(seed)
        z = torch.randn(n, self.latent_dim, generator=gen)
        labels = torch.full((n,), target_class, dtype=torch.long)
        with torch.no_grad():
            return self.decode(z, labels), labels


def extract_internal_stats_v2(h: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """Extract 10-dimensional feature vector for v2 introspective head.

    Features:
        0. mu(h)           - mean activation
        1. sigma(h)        - activation spread
        2. rho_{>0}(h)     - fraction alive neurons
        3. mu_top10%(h)    - top 10% mean activation
        4. ||h||_2         - hidden L2 norm
        5. H(h)            - activation entropy (over softmax of h)
        6. margin           - logit margin (top1 - top2 prob) -- strongest signal
        7. max(h)          - maximum activation
        8. min(h)          - minimum activation
        9. ||h||_2 / dim   - relative norm (normalized by hidden dim)

    Args:
        h: Penultimate activations, shape (batch, hidden_dim).
        logits: Raw classifier logits, shape (batch, num_classes).

    Returns:
        stats: Shape (batch, 10).
    """
    # v1 features (0-3)
    mu = h.mean(dim=1, keepdim=True)
    sigma = h.std(dim=1, keepdim=True) + 1e-8
    alive = (h > 0).float().mean(dim=1, keepdim=True)
    k = max(1, h.shape[1] // 10)
    topk_vals, _ = h.topk(k, dim=1)
    top_mean = topk_vals.mean(dim=1, keepdim=True)

    # v2 new features (4-9)
    l2_norm = h.norm(dim=1, keepdim=True)

    # Activation entropy: softmax over hidden dims, then -sum(p log p)
    h_probs = F.softmax(h, dim=1)
    entropy = -(h_probs * (h_probs + 1e-10).log()).sum(dim=1, keepdim=True)

    # Logit margin: top1 - top2 probability (strongest signal from v1 analysis)
    probs = F.softmax(logits, dim=1)
    top2_probs = probs.topk(2, dim=1).values
    margin = (top2_probs[:, 0] - top2_probs[:, 1]).unsqueeze(1)

    max_act = h.max(dim=1, keepdim=True).values
    min_act = h.min(dim=1, keepdim=True).values
    rel_norm = l2_norm / h.shape[1]

    return torch.cat([mu, sigma, alive, top_mean,
                      l2_norm, entropy, margin,
                      max_act, min_act, rel_norm], dim=1)


class IntrospectiveHeadV2(nn.Module):
    """MLP introspective head: 10 -> 32 -> ReLU -> 1.

    Replaces v1's logistic regression with a small network trained
    on stress data to push AUC above 0.80.
    """

    def __init__(self, input_dim: int = 10, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, stats: torch.Tensor) -> torch.Tensor:
        """Return P(error) in [0, 1], shape (batch, 1)."""
        return torch.sigmoid(self.net(stats))

