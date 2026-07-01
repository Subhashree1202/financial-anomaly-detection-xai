"""
explain.py
----------
SHAP-based explainability for the LSTM autoencoder.

We explain the *encoder* — mapping a (W, F) window to a latent
vector — because that is where the model "decides" whether a
pattern is normal or anomalous.

Strategy
~~~~~~~~
We use shap.GradientExplainer (gradient × input), which works
natively with PyTorch and handles the sequential input without
needing to flatten it.  Attribution shape is (W, F): for each
window we get one SHAP value per (time-step, feature) pair.

For reporting, we aggregate over time steps to get per-feature
importance, allowing direct comparison with tabular SHAP plots
familiar to finance / risk audiences.
"""

import numpy as np
import torch
import shap

from src.data import FEATURE_COLS


def _encoder_wrapper(model):
    """
    Return a thin callable that maps (B, W, F) → scalar anomaly proxy
    so GradientExplainer has a single output to attribute.

    We use the L2 norm of the latent vector as the proxy: a larger
    norm signals a more unusual encoding.
    """
    class EncoderProxy(torch.nn.Module):
        def forward(self, x):
            z = model.encode(x)            # (B, latent)
            return z.norm(dim=-1, keepdim=True)   # (B, 1)
    return EncoderProxy()


def compute_shap_values(model: torch.nn.Module,
                        anomaly_windows: torch.Tensor,
                        background_windows: torch.Tensor,
                        device: torch.device,
                        n_background: int = 100) -> np.ndarray:
    """
    Compute SHAP values for a set of detected anomaly windows.

    Parameters
    ----------
    model             : trained LSTMAutoencoder
    anomaly_windows   : (N_anom, W, F) tensor of anomalous windows
    background_windows: (N_bg,   W, F) tensor sampled from normal windows
    device            : torch device
    n_background      : how many background samples to use (speed / accuracy trade-off)

    Returns
    -------
    shap_vals : np.ndarray of shape (N_anom, W, F)
    """
    proxy = _encoder_wrapper(model).to(device)
    proxy.eval()
    model.eval()

    # Sub-sample background for speed
    idx  = torch.randperm(len(background_windows))[:n_background]
    bg   = background_windows[idx].to(device)

    explainer  = shap.GradientExplainer(proxy, bg)
    shap_vals  = explainer.shap_values(anomaly_windows.to(device))
    # shap_vals shape: (N_anom, W, F)
    return np.array(shap_vals)


def aggregate_feature_importance(shap_vals: np.ndarray) -> dict:
    """
    Aggregate (N, W, F) SHAP values to per-feature mean |SHAP|,
    averaged across all anomaly windows and time steps.

    Returns a dict {feature_name: mean_abs_shap} sorted descending.
    """
    mean_abs = np.abs(shap_vals).mean(axis=(0, 1))   # (F,)
    importance = {f: float(v) for f, v in zip(FEATURE_COLS, mean_abs)}
    return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))


def top_driver_per_anomaly(shap_vals: np.ndarray) -> list[str]:
    """
    For each anomaly window return the feature with the highest
    mean |SHAP| across time steps.  Useful for per-event narrative.
    """
    mean_abs = np.abs(shap_vals).mean(axis=1)         # (N, F)
    top_idx  = mean_abs.argmax(axis=1)                # (N,)
    return [FEATURE_COLS[i] for i in top_idx]
