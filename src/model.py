"""
model.py
--------
LSTM Autoencoder for unsupervised anomaly detection on
financial time-series windows.

Architecture
~~~~~~~~~~~~
Encoder : stacked LSTM  →  last hidden state  →  linear bottleneck
Decoder : repeat latent →  stacked LSTM  →  linear projection to input dim
Loss    : MSE reconstruction error per window
Anomaly : windows whose error > µ_train + σ * THRESH_SIGMA are flagged
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int,
                 latent_dim: int, n_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):                          # x: (B, W, F)
        _, (h_n, _) = self.lstm(x)                # h_n: (layers, B, H)
        h_last = h_n[-1]                          # (B, H)
        z      = self.fc(h_last)                  # (B, latent)
        return z


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int,
                 output_dim: int, window_size: int,
                 n_layers: int, dropout: float):
        super().__init__()
        self.window_size = window_size
        self.fc   = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):                          # z: (B, latent)
        h0   = self.fc(z).unsqueeze(0)            # (1, B, H) – seed hidden state
        inp  = h0.squeeze(0).unsqueeze(1).repeat(1, self.window_size, 1)  # (B, W, H)
        out, _ = self.lstm(inp)                   # (B, W, H)
        return self.proj(out)                     # (B, W, F)


class LSTMAutoencoder(nn.Module):
    """
    Full autoencoder.  Call .encode(x) to get latent vectors
    for SHAP attribution.
    """
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int,
                 window_size: int, n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, n_layers, dropout)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim,
                               window_size, n_layers, dropout)

    def forward(self, x):
        z    = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def encode(self, x):
        return self.encoder(x)

    @staticmethod
    def reconstruction_error(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """Mean squared error per window, averaged across time and features."""
        return ((x - x_hat) ** 2).mean(dim=(1, 2))   # (B,)


def build_model(n_features: int, config) -> LSTMAutoencoder:
    model = LSTMAutoencoder(
        input_dim   = n_features,
        hidden_dim  = config.HIDDEN_DIM,
        latent_dim  = config.LATENT_DIM,
        window_size = config.WINDOW_SIZE,
        n_layers    = config.N_LAYERS,
        dropout     = config.DROPOUT,
    )
    return model
