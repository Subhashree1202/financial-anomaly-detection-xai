"""
visualize.py
------------
Publication-quality plots for the anomaly detection pipeline.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from src.data import FEATURE_COLS

sns.set_theme(style="whitegrid", palette="muted")
FIGSIZE_WIDE = (14, 5)
FIGSIZE_SQ   = (8, 6)


def plot_price_with_anomalies(prices: pd.Series,
                               window_dates: pd.DatetimeIndex,
                               errors: np.ndarray,
                               threshold: float,
                               known_events: dict,
                               save_path: str = None):
    """
    Main result chart: S&P 500 price with detected anomaly windows
    shaded red and known events annotated.
    """
    anomaly_dates = window_dates[errors > threshold]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                    gridspec_kw={"height_ratios": [3, 1]},
                                    sharex=True)
    # ── Price ──
    ax1.plot(prices.index, prices.values, color="#1f77b4", linewidth=0.9, label="S&P 500")
    for d in anomaly_dates:
        ax1.axvspan(d - pd.Timedelta(days=15), d + pd.Timedelta(days=15),
                    alpha=0.15, color="crimson", linewidth=0)
    for date_str, label in known_events.items():
        dt = pd.Timestamp(date_str)
        if prices.index[0] <= dt <= prices.index[-1]:
            ax1.axvline(dt, color="darkred", linestyle="--", linewidth=0.8, alpha=0.7)
            ax1.text(dt, prices.max() * 0.98, label, fontsize=7,
                     rotation=90, va="top", color="darkred")
    ax1.set_ylabel("Price (USD)", fontsize=11)
    ax1.set_title("S&P 500 with LSTM Autoencoder Anomaly Detection", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=9)

    # ── Reconstruction error ──
    ax2.plot(window_dates, errors, color="steelblue", linewidth=0.8, label="Reconstruction error")
    ax2.axhline(threshold, color="crimson", linestyle="--", linewidth=1.2, label=f"Threshold (µ+3σ)")
    ax2.fill_between(window_dates, errors, threshold,
                     where=errors > threshold, color="crimson", alpha=0.3)
    ax2.set_ylabel("MSE", fontsize=11)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()


def plot_shap_feature_importance(importance: dict,
                                  save_path: str = None):
    """
    Horizontal bar chart of mean |SHAP| per feature across all anomalies.
    Mirrors the standard SHAP bar plot but is fully reproducible
    without the shap.plots API.
    """
    features = list(importance.keys())
    values   = list(importance.values())

    fig, ax = plt.subplots(figsize=FIGSIZE_SQ)
    bars = ax.barh(features[::-1], values[::-1], color="steelblue", edgecolor="white")
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title("Feature Importance Across Detected Anomalies\n(SHAP — encoder attribution)", fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()


def plot_shap_heatmap(shap_vals: np.ndarray,
                       anomaly_labels: list,
                       save_path: str = None):
    """
    Heatmap: anomaly events × features, coloured by mean |SHAP|.
    Useful for comparing which features drove each specific event.
    """
    mean_abs = np.abs(shap_vals).mean(axis=1)     # (N, F)
    df = pd.DataFrame(mean_abs, index=anomaly_labels, columns=FEATURE_COLS)

    fig, ax = plt.subplots(figsize=(12, max(4, len(anomaly_labels) * 0.6)))
    sns.heatmap(df, annot=True, fmt=".3f", cmap="YlOrRd",
                linewidths=0.4, ax=ax, cbar_kws={"label": "Mean |SHAP|"})
    ax.set_title("SHAP Attribution per Anomaly Event × Feature", fontsize=12)
    ax.set_xlabel("Feature", fontsize=11)
    ax.set_ylabel("Anomaly Event", fontsize=11)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()


def plot_training_loss(losses: list, save_path: str = None):
    """Simple training curve."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses, color="steelblue", linewidth=1.5)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("MSE Loss", fontsize=11)
    ax.set_title("Training Loss — LSTM Autoencoder", fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()
