"""GAN (Goodfellow et al., 2014) para generar escenarios de rendimientos.

Arquitectura de perceptrón multicapa y pérdida no saturante, siguiendo el curso de
Google (developers.google.com/machine-learning/gan). Con `window > 1` cada muestra es
una ventana de días consecutivos (idea de Takahashi et al., 2019 / TimeGAN) y el
escenario a 1 día es el último día de la ventana generada.
"""
import copy

import numpy as np
import torch
from torch import nn


def _mlp(n_in: int, n_out: int, hidden: int, dropout: float = 0.0) -> nn.Sequential:
    layers = [nn.Linear(n_in, hidden), nn.LeakyReLU(0.2)]
    if dropout:
        layers.append(nn.Dropout(dropout))
    layers += [nn.Linear(hidden, hidden), nn.LeakyReLU(0.2)]
    if dropout:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden, n_out))
    return nn.Sequential(*layers)


class ReturnGAN:
    """Entrena y muestrea una GAN sobre rendimientos log estandarizados."""

    def __init__(self, n_assets: int, window=1, latent_dim=8, hidden=64, epochs=400,
                 batch_size=64, lr_g=2e-4, lr_d=2e-4, beta1=0.5, ema=0.0, seed=0):
        self.n_assets, self.window = n_assets, window
        self.latent_dim, self.epochs, self.batch_size = latent_dim, epochs, batch_size
        self.seed, self.ema_decay = seed, ema
        dim = n_assets * window
        torch.manual_seed(seed)
        self.G = _mlp(latent_dim, dim, hidden)
        self.D = _mlp(dim, 1, hidden, dropout=0.2)
        self.opt_g = torch.optim.Adam(self.G.parameters(), lr=lr_g, betas=(beta1, 0.999))
        self.opt_d = torch.optim.Adam(self.D.parameters(), lr=lr_d, betas=(beta1, 0.999))
        self.loss = nn.BCEWithLogitsLoss()
        # Copia del generador con media móvil exponencial de pesos (estabiliza el muestreo)
        self.G_ema = copy.deepcopy(self.G)
        self.mu = self.sigma = None
        self.history = []

    def _windows(self, z: np.ndarray) -> np.ndarray:
        w = self.window
        return np.stack([z[i : i + w].reshape(-1) for i in range(len(z) - w + 1)])

    def fit(self, train_returns: np.ndarray, verbose: bool = False):
        self.mu, self.sigma = train_returns.mean(0), train_returns.std(0, ddof=1)
        z = (train_returns - self.mu) / self.sigma
        x = torch.tensor(self._windows(z), dtype=torch.float32)
        n = len(x)
        gen = torch.Generator().manual_seed(self.seed)
        for epoch in range(self.epochs):
            perm = torch.randperm(n, generator=gen)
            d_losses, g_losses = [], []
            for i in range(0, n, self.batch_size):
                real = x[perm[i : i + self.batch_size]]
                b = len(real)
                noise = torch.randn(b, self.latent_dim, generator=gen)
                fake = self.G(noise)
                # --- Discriminador: reales -> 1 (suavizado 0.9), falsas -> 0
                d_loss = (self.loss(self.D(real), torch.full((b, 1), 0.9))
                          + self.loss(self.D(fake.detach()), torch.zeros(b, 1)))
                self.opt_d.zero_grad(); d_loss.backward(); self.opt_d.step()
                # --- Generador (no saturante): que D clasifique las falsas como 1
                g_loss = self.loss(self.D(fake), torch.ones(b, 1))
                self.opt_g.zero_grad(); g_loss.backward(); self.opt_g.step()
                self._update_ema()
                d_losses.append(d_loss.item()); g_losses.append(g_loss.item())
            self.history.append((epoch + 1, np.mean(d_losses), np.mean(g_losses)))
            if verbose and (epoch + 1) % 100 == 0:
                print(f"  época {epoch + 1:4d}  D={self.history[-1][1]:.3f}  G={self.history[-1][2]:.3f}")
        return self

    @torch.no_grad()
    def _update_ema(self):
        d = self.ema_decay
        for p_ema, p in zip(self.G_ema.parameters(), self.G.parameters()):
            p_ema.mul_(d).add_(p, alpha=1 - d) if d > 0 else p_ema.copy_(p)

    @torch.no_grad()
    def sample(self, n_scenarios: int, seed: int = 0) -> np.ndarray:
        """`n_scenarios` vectores de rendimientos log a 1 día (n, n_assets)."""
        self.G_ema.eval()
        gen = torch.Generator().manual_seed(seed)
        noise = torch.randn(n_scenarios, self.latent_dim, generator=gen)
        out = self.G_ema(noise).numpy().reshape(n_scenarios, self.window, self.n_assets)
        return out[:, -1, :] * self.sigma + self.mu
