"""Validation plot helpers.

Ports ``generate_val_plots`` and ``avg_flux_confidence`` from upstream
``neugk/plot_utils.py`` so the JAX evaluators produce the same wandb
figures as the torch pipeline.
"""

from __future__ import annotations

import io
from typing import Optional, Sequence

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from neugk_jax.utils import recombine_zf as _recombine_zf


def _plot_nd_local(x: np.ndarray, y: np.ndarray, *, cmap: str = "RdBu_r"):
    """N-D cross-section grid plot — self-contained port of upstream ``plot_nd``.

    Both inputs are ``(C, *spatial)``. We average over the non-displayed
    spatial axes and lay out a grid of pairwise (pred, gt) heatmaps so the
    spatial coverage stays comparable across runs.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim < 3 or y.ndim < 3:
        # fallback: 1D line plot
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(np.asarray(x).ravel(), label="pred")
        ax.plot(np.asarray(y).ravel(), label="gt", linestyle="--")
        ax.legend(); ax.set_title("recon vs gt")
        return fig
    # collapse to (C, H, W) by averaging the leading non-display axes
    spatial = x.shape[1:]
    if len(spatial) > 2:
        agg_axes = tuple(range(1, len(spatial) - 1))
        x = np.mean(x, axis=tuple(a + 1 for a in agg_axes)) if False else np.mean(
            x, axis=tuple(1 + a for a in range(len(spatial) - 2))
        )
        y = np.mean(y, axis=tuple(1 + a for a in range(len(spatial) - 2)))
    C = x.shape[0]
    fig, axes = plt.subplots(C, 3, figsize=(11, 3 * C), constrained_layout=True)
    if C == 1:
        axes = axes[None, :]
    for c in range(C):
        vmax = float(max(np.abs(x[c]).max(), np.abs(y[c]).max(), 1e-30))
        vmin = -vmax if cmap == "RdBu_r" else 0.0
        axes[c, 0].imshow(y[c], cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        axes[c, 0].set_title(f"channel {c} — gt")
        axes[c, 1].imshow(x[c], cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        axes[c, 1].set_title(f"channel {c} — pred")
        diff = x[c] - y[c]
        d = float(np.abs(diff).max() + 1e-30)
        axes[c, 2].imshow(diff, cmap="RdBu_r", vmin=-d, vmax=d, aspect="auto")
        axes[c, 2].set_title(f"channel {c} — pred − gt")
    return fig


def _plt_to_wandb_image(fig):
    """Try to return a wandb.Image; fall back to the bare figure if wandb is missing."""
    try:
        import wandb
        from PIL import Image as PILImage
        buf = io.BytesIO()
        fig.savefig(buf, bbox_inches="tight", format="png", dpi=120, pad_inches=0.01)
        buf.seek(0)
        img = PILImage.open(buf)
        plt.close(fig)
        return wandb.Image(img)
    except Exception:
        return fig


def generate_val_plots(
    rollout: dict[str, np.ndarray],
    gt: dict[str, np.ndarray],
    phase: str,
    *,
    ts: Optional[np.ndarray] = None,
    to_wandb: bool = True,
) -> dict[str, object]:
    """Cross-section grid plots — port of ``neugk.plot_utils.generate_val_plots``.

    Operates on numpy arrays. ``df`` is recombined from separate-zf back
    to 2-channel before plotting; ``phi`` is passed through as-is.
    """
    plots: dict[str, object] = {}
    time_str = f"T={float(ts[0]):.2f}, " if ts is not None and ts.size > 0 else ""
    field_configs = {
        "df": {"name": f"df ({time_str}{phase})", "recombine": True, "cmap": "RdBu_r"},
        "phi": {"name": f"phi ({time_str}{phase})", "recombine": False, "cmap": "plasma"},
    }
    for key, cfg in field_configs.items():
        if key not in rollout or key not in gt:
            continue
        x = np.asarray(rollout[key])
        y = np.asarray(gt[key])
        if cfg["recombine"]:
            # df may have a 4-channel separated layout — collapse back to 2-channel
            if y.shape[0] != 2:
                y = _recombine_zf(y, axis=0)
            if x.shape[1 if x.ndim == 7 else 0] != 2:
                # rollout may have a leading time axis (T, C, ...); pick channel axis accordingly
                axis = 1 if x.ndim == 7 else 0
                x = _recombine_zf(x, axis=axis)
        if x.ndim == 7:
            x = x[0]
        x = np.squeeze(x)
        y = np.squeeze(y)
        fig = _plot_nd_local(x, y, cmap=cfg["cmap"])
        plots[cfg["name"]] = _plt_to_wandb_image(fig) if to_wandb else fig
    return plots


def avg_flux_confidence(
    pred_means: np.ndarray,
    pred_stds: np.ndarray,
    tgt_vals: np.ndarray,
    traj_ids: Sequence[str],
    *,
    to_wandb: bool = True,
) -> object:
    """Per-trajectory flux UQ scatter plot — port of upstream's
    ``avg_flux_confidence``."""
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    x_pos = np.arange(len(traj_ids))
    ax.errorbar(
        x_pos, pred_means, yerr=pred_stds,
        fmt="o", capsize=6, label="Predicted (Mean ± Std)",
        color="#1f77b4", mfc="white", mew=2, alpha=0.8,
    )
    ax.scatter(
        x_pos, tgt_vals,
        marker="x", s=80, color="#d62728",
        label="Ground Truth", zorder=3,
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(traj_ids, rotation=45, ha="right")
    ax.set_xlabel("Trajectory ID", fontsize=12)
    ax.set_ylabel("Average Flux", fontsize=12)
    ax.set_title("Flux Prediction Accuracy across Trajectories", fontsize=14)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=True, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3, ls="--")
    return _plt_to_wandb_image(fig) if to_wandb else fig
