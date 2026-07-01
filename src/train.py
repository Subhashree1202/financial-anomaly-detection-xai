"""
train.py
--------
Training loop, model checkpointing, and anomaly-threshold calibration.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config


def train(model: nn.Module,
          loader_train: DataLoader,
          loader_full: DataLoader,
          device: torch.device,
          save_path: str = "results/model.pt") -> dict:
    """
    Train the LSTM autoencoder, returning a results dict with:
      - train_losses  : loss per epoch
      - full_errors   : reconstruction error for every window (full series)
      - threshold     : calibrated anomaly threshold (train-set µ + σ*THRESH_SIGMA)
    """
    model  = model.to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=config.LR)
    sched  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    loss_fn = nn.MSELoss()

    train_losses = []
    best_loss    = float("inf")

    print(f"Training on {device}  |  {config.EPOCHS} epochs")
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for batch in loader_train:
            batch = batch.to(device)
            opt.zero_grad()
            x_hat = model(batch)
            loss  = loss_fn(x_hat, batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            epoch_loss += loss.item()

        epoch_loss /= len(loader_train)
        sched.step(epoch_loss)
        train_losses.append(epoch_loss)

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:>3}/{config.EPOCHS}  loss={epoch_loss:.6f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)

    # ── Calibrate threshold on train windows ─────────────────────────────────
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()

    train_errors = _compute_errors(model, loader_train, device)
    threshold    = train_errors.mean() + config.THRESH_SIGMA * train_errors.std()
    print(f"\nAnomaly threshold (µ + {config.THRESH_SIGMA}σ) = {threshold:.6f}")

    # ── Score the full series ─────────────────────────────────────────────────
    full_errors = _compute_errors(model, loader_full, device)

    return {
        "train_losses" : train_losses,
        "full_errors"  : full_errors,
        "threshold"    : float(threshold),
    }


def _compute_errors(model: nn.Module,
                    loader: DataLoader,
                    device: torch.device) -> np.ndarray:
    """Return reconstruction errors as a 1-D numpy array."""
    errors = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            x_hat = model(batch)
            err   = model.reconstruction_error(batch, x_hat)
            errors.append(err.cpu().numpy())
    return np.concatenate(errors)
