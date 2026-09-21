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
    """Logistic regression on 4 internal stats → P(error).

    s_intro = σ(w^T z_stats + b)
    """

    def __init__(self, input_dim: int = 4):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, stats: torch.Tensor) -> torch.Tensor:
        """Return P(error) ∈ [0, 1], shape (batch, 1)."""
        return torch.sigmoid(self.linear(stats))
