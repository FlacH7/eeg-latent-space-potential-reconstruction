"""
CD-HSA Specific Modes — Stage 2 Plots
=======================================

Diagnostic and illustrative plots for the ``cdhsa_specific_modes``
Stage-2 dynamics strategy.  All plot functions follow the same
convention as the existing ``src.plotters`` module:

* ``stage2_meta : dict`` — the metadata dict returned by
  ``CDHSASpecificModesDynamics.fit_transform()``.
* ``out_dir : str | Path`` — directory where figures are saved.
* Failures are **non-fatal** (wrapped in try/except by the caller).
* Each function saves a PNG and returns the path.

Place this file at ``src/plotters/cdhsa_plots.py`` and add to
``src/plotters/__init__.py``::

    from src.plotters.cdhsa_plots import (
        plot_cdhsa_eigenvalue_spectrum,
        plot_cdhsa_mode_structure,
        plot_cdhsa_projection_power,
    )
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    pass  # stage2_meta is a plain dict


# ---------------------------------------------------------------------------
# Plot 1: Eigenvalue spectrum of specific modes
# ---------------------------------------------------------------------------

def plot_cdhsa_eigenvalue_spectrum(
    stage2_meta: dict,
    out_dir: str | Path,
) -> Path | None:
    """
    Bar chart of the specific-mode eigenvalues (lambda_specific) for
    the target condition.  The selected top-N modes are highlighted.

    This is analogous to ``plot_dmd_eigenvalue_unit_circle`` /
    ``plot_diffusion_eigenvalue_spectrum`` but for CD-HSA specific-mode
    eigenvalues.
    """
    all_eigs = stage2_meta.get("all_condition_eigenvalues")
    selected_eigs = stage2_meta.get("eigenvalues")
    selected_indices = stage2_meta.get("mode_indices", [])
    condition = stage2_meta.get("condition", "unknown")
    top_n = stage2_meta.get("top_n", len(selected_indices))

    if all_eigs is None or len(all_eigs) == 0:
        return None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    r = len(all_eigs)
    x = np.arange(r)
    colors = [
        "#e74c3c" if i in selected_indices else "#bdc3c7"
        for i in range(r)
    ]

    bars = ax.bar(x, all_eigs, color=colors, edgecolor="black", linewidth=0.5)

    # Annotate selected modes
    for idx in selected_indices:
        ax.annotate(
            f"{all_eigs[idx]:.4f}",
            xy=(idx, all_eigs[idx]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center", fontsize=9, fontweight="bold",
            color="#c0392b",
        )

    ax.set_xlabel("Mode index (sorted by eigenvalue)", fontsize=11)
    ax.set_ylabel("Eigenvalue (\u03bb)", fontsize=11)
    ax.set_title(
        f"CD-HSA Specific Modes — {condition}\n"
        f"Total modes: {r}, Selected: {top_n}",
        fontsize=12, fontweight="bold",
    )
    ax.set_xticks(x)
    ax.axhline(0, color="gray", linewidth=0.5)

    # Legend proxy
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#e74c3c", edgecolor="black", label="Selected (top-N)"),
        Patch(facecolor="#bdc3c7", edgecolor="black", label="Not selected"),
    ]
    ax.legend(handles=legend_elements, loc="best", fontsize=9)

    path = out_dir / "stage2_cdhsa_eigenvalue_spectrum.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Plot 2: Mode structure — spatial distribution of selected modes
# ---------------------------------------------------------------------------

def plot_cdhsa_mode_structure(
    stage2_meta: dict,
    out_dir: str | Path,
) -> Path | None:
    """
    Heatmap of the selected mode vectors W_sel reshaped as
    ``(L, p, top_n)``.  Each panel shows one selected mode with the
    block-Hankel lag on the y-axis and the Hankel feature dimension
    on the x-axis.

    This reveals the temporal structure captured by each specific mode:
    which lag blocks and which channel-delay combinations contribute most.
    """
    W_sel = stage2_meta.get("W_sel")
    if W_sel is None:
        return None

    L = stage2_meta.get("L")
    p_hankel = stage2_meta.get("p_hankel")
    if L is None or p_hankel is None:
        return None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    top_n = W_sel.shape[1]
    condition = stage2_meta.get("condition", "unknown")

    # Reshape: (p*L, top_n) -> (L, p, top_n)
    W_3d = W_sel.reshape(L, p_hankel, top_n)

    # Aggregate over p (mean absolute value per lag block per mode)
    lag_profile = np.mean(np.abs(W_3d), axis=1)  # (L, top_n)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    # Panel 1: Lag activation profile (which lags contribute most)
    ax1 = axes[0]
    lags = np.arange(L)
    for m in range(top_n):
        ax1.bar(
            lags + m * 0.8 / top_n,
            lag_profile[:, m],
            width=0.7 / top_n,
            label=f"Mode {m} (\u03bb={stage2_meta.get('eigenvalues', [None])[m]:.4f})"
            if stage2_meta.get("eigenvalues")
            else f"Mode {m}",
            alpha=0.85,
        )
    ax1.set_xlabel("Block-Hankel lag", fontsize=11)
    ax1.set_ylabel("Mean |W| per lag block", fontsize=11)
    ax1.set_title("Lag Activation Profile", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=8)

    # Panel 2: Full mode heatmap (L x p_hankel, first mode)
    # Use a downsampled version if p_hankel is too large
    ax2 = axes[1]
    mode_to_show = 0
    W_mode = W_3d[:, :, mode_to_show]  # (L, p_hankel)

    # Downsample p axis for readability if needed
    max_cols = 100
    if p_hankel > max_cols:
        step = max(1, p_hankel // max_cols)
        W_show = W_mode[:, ::step]
    else:
        W_show = W_mode

    vmax = np.percentile(np.abs(W_show), 98)
    im = ax2.imshow(
        W_show, aspect="auto", cmap="RdBu_r",
        vmin=-vmax, vmax=vmax, interpolation="nearest",
    )
    ax2.set_xlabel("Hankel feature dim (downsampled)" if p_hankel > max_cols
                    else "Hankel feature dim", fontsize=10)
    ax2.set_ylabel("Block-Hankel lag", fontsize=10)
    ax2.set_title(f"Mode {mode_to_show} spatial structure", fontsize=12,
                   fontweight="bold")
    fig.colorbar(im, ax=ax2, shrink=0.8, label="W weight")

    fig.suptitle(
        f"CD-HSA Mode Structure — {condition} (L={L}, p={p_hankel})",
        fontsize=13, fontweight="bold", y=1.02,
    )

    path = out_dir / "stage2_cdhsa_mode_structure.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Plot 3: Projection power — variance explained by each selected mode
# ---------------------------------------------------------------------------

def plot_cdhsa_projection_power(
    stage2_meta: dict,
    Y2: np.ndarray | None = None,
    out_dir: str | Path | None = None,
) -> Path | None:
    """
    Diagnostic showing the "projection power" of each selected mode:
    - Bar chart of mode norms (how much each mode vector weighs)
    - If Y2 (the Stage-2 output) is provided, also shows the variance
      explained by each mode's time course.

    Analogous to ``plot_pca_variance_explained`` but for CD-HSA modes.
    """
    mode_norms = stage2_meta.get("mode_norms")
    selected_eigs = stage2_meta.get("eigenvalues")
    condition = stage2_meta.get("condition", "unknown")
    output_shape = stage2_meta.get("output_shape")
    top_n = stage2_meta.get("top_n", 2)

    if mode_norms is None:
        return None

    out_dir = Path(out_dir) if out_dir else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    # Panel 1: Mode norms
    ax1 = axes[0]
    modes = np.arange(top_n)
    bars = ax1.bar(modes, mode_norms, color="#3498db", edgecolor="black",
                   linewidth=0.5)
    for bar, norm in zip(bars, mode_norms):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{norm:.1f}", ha="center", va="bottom", fontsize=9)
    ax1.set_xlabel("Mode", fontsize=11)
    ax1.set_ylabel("||W_mode|| (Frobenius norm)", fontsize=11)
    ax1.set_title("Mode Vector Norms", fontsize=12, fontweight="bold")
    ax1.set_xticks(modes)
    ax1.set_xticklabels([f"Mode {i}" for i in modes])

    # Panel 2: Eigenvalues of selected modes
    ax2 = axes[1]
    if selected_eigs and len(selected_eigs) == top_n:
        bars2 = ax2.bar(modes, selected_eigs, color="#e74c3c",
                        edgecolor="black", linewidth=0.5)
        for bar, ev in zip(bars2, selected_eigs):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{ev:.4f}", ha="center", va="bottom", fontsize=9)
        ax2.set_ylabel("Eigenvalue (\u03bb)", fontsize=11)
        ax2.set_title("Specific Mode Eigenvalues", fontsize=12,
                       fontweight="bold")
    else:
        # Fallback: variance of Y2 rows if available
        if Y2 is not None and Y2.shape[0] >= top_n:
            variances = np.var(Y2[:top_n, :], axis=1)
            ax2.bar(modes, variances, color="#2ecc71", edgecolor="black",
                    linewidth=0.5)
            ax2.set_ylabel("Variance of mode time course", fontsize=11)
            ax2.set_title("Mode Time-Course Variance", fontsize=12,
                           fontweight="bold")
        else:
            ax2.text(0.5, 0.5, "No eigenvalue data available",
                     transform=ax2.transAxes, ha="center", va="center",
                     fontsize=11, color="gray")

    ax2.set_xlabel("Mode", fontsize=11)
    ax2.set_xticks(modes)
    ax2.set_xticklabels([f"Mode {i}" for i in modes])

    fig.suptitle(
        f"CD-HSA Projection Power — {condition}",
        fontsize=13, fontweight="bold", y=1.02,
    )

    path = out_dir / "stage2_cdhsa_projection_power.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
