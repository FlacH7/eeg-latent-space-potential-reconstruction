"""
Latent Subspace Extraction — End-to-End Orchestrator
=====================================================

Single-call function that runs the complete pipeline:
    raw EEG → filtering → ICA → ICLabel artifact rejection
    → subspace selection (Markov / Conservative / Weighted / Sequential / Pareto)
    → latent space as np.ndarray (n_samples, n_dim)

Typical usage::

    from extract_latent_subspace import extract_latent_space

    # From file
    latent, meta = extract_latent_space(
        "/path/to/recording_raw.fif",
        n_dim=2,
        scoring_method="markov",
        n_bins=5,
        n_workers=4,
    )
    # latent.shape == (n_samples, 2)

    # From existing mne.Raw object
    latent, meta = extract_latent_space(
        raw,
        n_dim=3,
        scoring_method="weighted",
        alpha=0.5,
        fc_metric="total_variance",
        n_bins=5,
    )

Return value
------------
latent : np.ndarray, shape (n_samples, n_dim)
    The extracted latent subspace, one column per latent dimension.
    Time runs along the rows (same axis as the original time series).

meta : dict
    Metadata including the selected component indices, scores, timing,
    preprocessing info, etc.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Literal

import mne
import numpy as np

#Preprocessing
from eeg_preprocessing import (
    load_raw_eeg,
    load_sample_mne_data,
    run_full_preprocessing,
)

# Conservative fraction (Stage II)
from conservative_fraction import (
    find_best_subspace_fc,
    greedy_forward_selection_fc,
    score_many_combinations_fc,
)

# Markov time (Stage III)
from markov_subspace import (
    find_best_subspace_markov,
    greedy_forward_selection_markov,
    score_many_combinations_markov,
)

# Unified scoring strategies
from scoring import (
    independent_selection,
    pareto_frontier_selection,
    sequential_filtering_selection,
    weighted_score_selection,
)


# ---------------------------------------------------------------------------
# Default constants
# ---------------------------------------------------------------------------

DEFAULT_L_FREQ: float = 1.0
DEFAULT_H_FREQ: float = 40.0
DEFAULT_ICA_METHOD: str = "picard"
DEFAULT_ICA_RANDOM_STATE: int = 42
DEFAULT_N_BINS: int = 5
DEFAULT_FC_METRIC: str = "variance_sum"


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def extract_latent_space(
    raw_input: mne.io.Raw | str | Path,
    *,
    # ---- Subspace dimension ----
    n_dim: int = 2,
    # ---- Scoring method ----
    scoring_method: Literal[
        "markov", "conservative", "weighted", "sequential", "pareto", "independent"
    ] = "markov",
    # ---- Conservative-fraction params ----
    fc_metric: str = DEFAULT_FC_METRIC,
    # ---- Markov-time params ----
    n_bins: int = DEFAULT_N_BINS,
    # ---- Weighted strategy params ----
    alpha: float | None = None,
    # ---- Sequential strategy params ----
    primary_criterion: Literal["fc", "markov"] = "markov",
    sequential_K: int | None = None,
    # ---- Preprocessing params ----
    l_freq: float = DEFAULT_L_FREQ,
    h_freq: float = DEFAULT_H_FREQ,
    n_components: int | float | None = None,
    ica_method: str = DEFAULT_ICA_METHOD,
    ica_random_state: int | None = DEFAULT_ICA_RANDOM_STATE,
    retained_labels: list[str] | None = None,
    # ---- Search / compute params ----
    search_strategy: Literal["exhaustive", "greedy"] = "exhaustive",
    n_workers: int | None = None,
    verbose: bool | str | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Extract a low-dimensional latent subspace from an EEG recording.

    Parameters
    ----------
    raw_input : mne.io.Raw | str | Path
        Either an MNE Raw object already loaded in memory, or a path to a
        raw data file (``.fif``, ``.edf``, ``.bdf``, ``.set``, etc.).
    n_dim : int, default 2
        Target dimensionality of the latent subspace (``N`` in the paper).
    scoring_method : str, default ``"markov"``
        Subspace selection strategy. One of:

        * ``"markov"``       — minimise Markov relaxation time (fastest
          dynamics). Recommended based on empirical results.
        * ``"conservative"`` — maximise conservative fraction (variance
          retention). Falls back to greedy when ``n_dim >= 4``.
        * ``"weighted"``     — convex combination of normalised fc and
          1/tau. Requires *alpha*.
        * ``"sequential"``   — filter top-K with one criterion, re-rank
          with the other.
        * ``"pareto"``       — return the full Pareto frontier (useful
          for downstream multi-objective analysis).
        * ``"independent"``  — run both criteria independently and return
          the Markov winner (discards the FC winner).

    fc_metric : str, default ``"variance_sum"``
        Variance measure for the conservative fraction. One of:
        ``"variance_sum"``, ``"first_pc_var"``, ``"total_variance"``.
    n_bins : int, default 5
        Number of quantile bins per component for Markov discretisation.
    alpha : float or None, optional
        Trade-off parameter for the weighted strategy in ``[0, 1]``.
        ``0`` = pure Markov, ``1`` = pure conservative fraction.
        Required when *scoring_method* is ``"weighted"``.
    primary_criterion : ``"fc"`` | ``"markov"``, default ``"markov"``
        Which criterion leads the sequential filtering strategy.
    sequential_K : int or None, optional
        Number of candidates to retain in the first filtering stage of
        sequential strategy. ``None`` auto-computes ``min(20, 5% of combos)``.
    l_freq : float, default 1.0
        High-pass filter cutoff (Hz).
    h_freq : float, default 40.0
        Low-pass filter cutoff (Hz).
    n_components : int | float | None, optional
        Number of ICA components. ``None`` = ``n_channels - 1``.
    ica_method : str, default ``"picard"``
        ICA algorithm (``"picard"``, ``"fastica"``, ``"infomax"``).
    ica_random_state : int | None, default 42
        Random seed for ICA reproducibility.
    retained_labels : list of str, optional
        ICLabel classes to keep. Defaults to ``["brain", "other"]``.
    search_strategy : ``"exhaustive"`` | ``"greedy"``, default ``"exhaustive"``
        Combinatorial search strategy. Use ``"greedy"`` when
        ``n_dim >= 4`` or for faster approximate results.
    n_workers : int or None, optional
        Number of parallel processes. ``None`` uses all CPU cores.
    verbose : bool | str | None, optional
        MNE verbosity level.

    Returns
    -------
    latent : np.ndarray, shape (n_samples, n_dim)
        The extracted latent subspace. Each column is one latent
        dimension (component time series). Ready for downstream analyses.
    meta : dict
        Dictionary with keys:

        * ``"selected_indices"``    — component indices in the clean Y
          matrix that form the latent subspace.
        * ``"scoring_method"``      — which strategy was used.
        * ``"latent_scores"``       — criterion values (fc, tau, etc.).
        * ``"preprocessing"``       — Stage-I info (D, T, sfreq,
          excluded indices, label counts).
        * ``"elapsed_time"``        — total wall-clock time in seconds.
        * ``"Y_shape"``             — shape of the clean component
          matrix before subspace selection.
    """
    t0 = time.time()

    # =====================================================================
    # Stage I — Preprocessing
    # =====================================================================
    if isinstance(raw_input, (str, Path)):
        raw = load_raw_eeg(str(raw_input), verbose=verbose)
    else:
        raw = raw_input

    if retained_labels is None:
        retained_labels = ["brain", "other"]

    prep = run_full_preprocessing(
        raw,
        l_freq=l_freq,
        h_freq=h_freq,
        n_components=n_components,
        ica_method=ica_method,
        ica_random_state=ica_random_state,
        retained_labels=retained_labels,
        verbose=verbose,
    )

    Y = prep["Y"]  # shape (D, T), zero-mean rows
    D, T = Y.shape

    # =====================================================================
    # Stage II — Subspace Selection
    # =====================================================================
    use_greedy = search_strategy == "greedy" or n_dim >= 4

    if scoring_method == "markov":
        selected_idx, meta_scores = _select_markov(
            Y, n_dim, n_bins, use_greedy, n_workers
        )

    elif scoring_method == "conservative":
        selected_idx, meta_scores = _select_conservative(
            Y, n_dim, fc_metric, use_greedy, n_workers
        )

    elif scoring_method == "weighted":
        if alpha is None:
            raise ValueError(
                "alpha must be provided when scoring_method='weighted'"
            )
        selected_idx, meta_scores = _select_weighted(
            Y, n_dim, alpha, fc_metric, n_bins, use_greedy, n_workers
        )

    elif scoring_method == "sequential":
        selected_idx, meta_scores = _select_sequential(
            Y, n_dim, primary_criterion, sequential_K,
            fc_metric, n_bins, n_workers,
        )

    elif scoring_method == "pareto":
        selected_idx, meta_scores = _select_pareto(
            Y, n_dim, fc_metric, n_bins, n_workers
        )

    elif scoring_method == "independent":
        selected_idx, meta_scores = _select_independent(
            Y, n_dim, fc_metric, n_bins, use_greedy, n_workers
        )

    else:
        raise ValueError(f"Unknown scoring_method: {scoring_method!r}")

    # =====================================================================
    # Build latent space: transpose to (n_samples, n_dim)
    # =====================================================================
    latent = Y[list(selected_idx), :].T  # (T, N) → (n_samples, n_dim)

    elapsed = time.time() - t0

    meta = {
        "selected_indices": list(selected_idx),
        "scoring_method": scoring_method,
        "latent_scores": meta_scores,
        "preprocessing": {
            "D": D,
            "T": T,
            "sfreq": prep["sfreq"],
            "n_channels": prep["n_channels"],
            "excluded_indices": prep["excluded_indices"],
            "kept_indices": prep["kept_indices"],
            "Y": Y,               # clean component matrix for downstream reuse
        },
        "Y": Y,                   # also at top level for convenience
        "Y_shape": (D, T),
        "elapsed_time": elapsed,
    }

    return latent, meta


# ---------------------------------------------------------------------------
# Quick sampled conservative-fraction diagnostic
# ---------------------------------------------------------------------------

def sample_conservative_fraction(
    Y: np.ndarray,
    n_dim: int,
    *,
    n_samples: int = 500,
    metric: str = "variance_sum",
    n_workers: int | None = None,
) -> dict:
    """
    Evaluate the conservative fraction on the first *n_samples* combinations
    of *n_dim* components.

    This is a lightweight diagnostic to check whether the conservative
    fraction discriminates on a given dataset without running the full
    exhaustive search.

    Parameters
    ----------
    Y : np.ndarray, shape (D, T)
        Clean component matrix (output of Stage I).
    n_dim : int
        Subspace dimensionality.
    n_samples : int, default 500
        Number of combinations to evaluate.
    metric : str, default ``"variance_sum"``
        Conservative-fraction metric to use.
    n_workers : int or None, optional
        Parallel processes.

    Returns
    -------
    report : dict
        Keys:

        * ``"fc_values"``     — array of all computed fc values.
        * ``"combinations"``  — list of evaluated combinations.
        * ``"min"`` / ``"max"`` / ``"range"`` / ``"mean"`` / ``"std"`` / ``"cv"``
          — descriptive statistics.
        * ``"constant"``      — ``True`` if all values are identical
          (within machine precision).
        * ``"degenerate"``    — ``True`` if CV < 1e-6.
    """
    from itertools import combinations
    import math

    D, T = Y.shape
    all_combs = list(combinations(range(D), n_dim))
    total = len(all_combs)

    if n_samples >= total:
        sampled = all_combs
        n_samples = total
    else:
        sampled = all_combs[:n_samples]

    print(f"\n[SampleFC] Sampling {n_samples}/{total} combinations (N={n_dim}, D={D})")

    scored = score_many_combinations_fc(
        Y, sampled, metric=metric, n_workers=n_workers, show_progress=False
    )
    fc_values = np.array([fc for _, fc in scored])

    vmin, vmax = float(fc_values.min()), float(fc_values.max())
    vmean = float(fc_values.mean())
    vstd = float(fc_values.std())
    cv = vstd / vmean if vmean > 0 else 0.0

    report = {
        "fc_values": fc_values,
        "combinations": sampled,
        "n_evaluated": n_samples,
        "n_total": total,
        "metric": metric,
        "min": vmin,
        "max": vmax,
        "range": vmax - vmin,
        "mean": vmean,
        "std": vstd,
        "cv": cv,
        "constant": np.allclose(fc_values, fc_values[0]),
        "degenerate": cv < 1e-6,
    }

    print(f"  min={vmin:.6f}  max={vmax:.6f}  range={report['range']:.6f}")
    print(f"  mean={vmean:.6f}  std={vstd:.6f}  CV={cv:.6f}")
    print(f"  constant={report['constant']}  degenerate={report['degenerate']}")

    return report


# ---------------------------------------------------------------------------
# Internal selection helpers
# ---------------------------------------------------------------------------

def _select_markov(
    Y: np.ndarray,
    n_dim: int,
    n_bins: int,
    use_greedy: bool,
    n_workers: int | None,
) -> tuple[tuple[int, ...], dict]:
    """Select subspace by minimising Markov relaxation time."""
    if use_greedy:
        comb, tau = greedy_forward_selection_markov(Y, n_dim, n_bins=n_bins)
        comb = tuple(comb)
    else:
        comb, tau = find_best_subspace_markov(
            Y, n_dim, n_bins=n_bins, n_workers=n_workers
        )
    return comb, {"tau": float(tau), "search": "greedy" if use_greedy else "exhaustive"}


def _select_conservative(
    Y: np.ndarray,
    n_dim: int,
    fc_metric: str,
    use_greedy: bool,
    n_workers: int | None,
) -> tuple[tuple[int, ...], dict]:
    """Select subspace by maximising conservative fraction."""
    if use_greedy:
        comb, fc = greedy_forward_selection_fc(Y, n_dim, metric=fc_metric)
        comb = tuple(comb)
    else:
        comb, fc = find_best_subspace_fc(
            Y, n_dim, metric=fc_metric, n_workers=n_workers
        )
    return comb, {"fc": float(fc), "metric": fc_metric,
                  "search": "greedy" if use_greedy else "exhaustive"}


def _select_weighted(
    Y: np.ndarray,
    n_dim: int,
    alpha: float,
    fc_metric: str,
    n_bins: int,
    use_greedy: bool,
    n_workers: int | None,
) -> tuple[tuple[int, ...], dict]:
    """Select subspace via weighted scalarisation."""
    result = weighted_score_selection(
        Y, n_dim,
        alphas=[alpha],
        metric=fc_metric,
        n_bins=n_bins,
        n_workers=n_workers,
        search_strategy="greedy" if use_greedy else "exhaustive",
    )
    comb = result["best_combinations"][0]
    return comb, {
        "alpha": float(alpha),
        "score": float(result["best_scores"][0]),
        "fc": float(result["fc_values"][0]),
        "tau": float(result["tau_values"][0]),
    }


def _select_sequential(
    Y: np.ndarray,
    n_dim: int,
    primary: str,
    K: int | None,
    fc_metric: str,
    n_bins: int,
    n_workers: int | None,
) -> tuple[tuple[int, ...], dict]:
    """Select subspace via sequential filtering."""
    D = Y.shape[0]
    if K is None:
        import math
        K = min(20, max(5, int(np.prod([D - i for i in range(n_dim)]) / math.factorial(n_dim) * 0.05)))
        K = max(K, 5)

    result = sequential_filtering_selection(
        Y, n_dim, K=K, primary=primary,
        metric=fc_metric, n_bins=n_bins, n_workers=n_workers,
    )
    comb = result["best_combination"]
    return comb, {
        "primary": primary,
        "K": K,
        "fc": result["best_fc"],
        "tau": result["best_tau"],
    }


def _select_pareto(
    Y: np.ndarray,
    n_dim: int,
    fc_metric: str,
    n_bins: int,
    n_workers: int | None,
) -> tuple[tuple[int, ...], dict]:
    """Select the first point on the Pareto frontier (best trade-off)."""
    result = pareto_frontier_selection(
        Y, n_dim, metric=fc_metric, n_bins=n_bins, n_workers=n_workers,
    )
    comb = result["frontier_combinations"][0]
    return comb, {
        "frontier_size": int(result["frontier_size"]),
        "fc": float(result["frontier_fc"][0]),
        "tau": float(result["frontier_tau"][0]),
    }


def _select_independent(
    Y: np.ndarray,
    n_dim: int,
    fc_metric: str,
    n_bins: int,
    use_greedy: bool,
    n_workers: int | None,
) -> tuple[tuple[int, ...], dict]:
    """Run both criteria, return the Markov winner (empirically better)."""
    result = independent_selection(
        Y, n_dim,
        metric=fc_metric,
        n_bins=n_bins,
        n_workers=n_workers,
        fc_search="greedy" if use_greedy else "exhaustive",
        markov_search="greedy" if use_greedy else "exhaustive",
    )
    # Return Markov winner — it discriminates better than FC
    comb = result["markov_combination"]
    return comb, {
        "fc_combination": result["fc_combination"],
        "fc_value": result["fc_value"],
        "markov_combination": result["markov_combination"],
        "markov_tau": result["markov_tau"],
    }


# ---------------------------------------------------------------------------
# Diagnostic: does the dataset support discriminative subspace selection?
# ---------------------------------------------------------------------------

def diagnose_subspace_discrimination(
    Y: np.ndarray,
    n_dim: int = 2,
    *,
    fc_metrics: list[str] | None = None,
    n_bins: int = 5,
    n_workers: int | None = None,
    cv_threshold: float = 0.05,
) -> dict:
    """
    Exhaustively evaluate *all* combinations and report whether each
    criterion discriminates on this dataset.

    For each metric the function reports:
        * min / max / range / std / CV
        * a discrimination verdict (``True`` if CV > *cv_threshold*)

    It ends with a recommended scoring method based on the evidence.

    Parameters
    ----------
    Y : np.ndarray, shape (D, T)
        Clean component matrix (output of Stage I).
    n_dim : int, default 2
        Subspace dimensionality to evaluate.
    fc_metrics : list of str, optional
        Conservative-fraction metrics to test. Defaults to all three.
    n_bins : int, default 5
        Quantile bins for Markov discretisation.
    n_workers : int or None, optional
        Parallel processes.
    cv_threshold : float, default 0.05
        A criterion is considered "discriminative" if its coefficient of
        variation (std / mean) exceeds this value.

    Returns
    -------
    report : dict
        Structured results ready for inspection or logging. Key fields:

        * ``"conservative_fraction"`` — dict per metric with
          ``discriminates`` (bool), ``min``, ``max``, ``range``, ``std``,
          ``cv``, ``best_comb``, ``worst_comb``.
        * ``"markov_time"`` — same structure for tau.
        * ``"recommendation"`` — suggested scoring method and rationale.
        * ``"n_combinations_evaluated"`` — total combos tested.
    """
    from itertools import combinations
    import math

    if fc_metrics is None:
        fc_metrics = ["variance_sum", "first_pc_var", "total_variance"]

    D, T = Y.shape
    all_combs = list(combinations(range(D), n_dim))
    n_combs = len(all_combs)

    print(f"\n[Diagnostic] Evaluating discriminative power on {n_combs:,} combinations")
    print(f"[Diagnostic] D={D}, T={T}, N={n_dim}, n_bins={n_bins}\n")

    report: dict = {
        "n_combinations_evaluated": n_combs,
        "n_dim": n_dim,
        "D": D,
        "conservative_fraction": {},
        "markov_time": {},
        "recommendation": {},
    }

    # ===================================================================
    # Conservative fraction — all metrics
    # ===================================================================
    print("=" * 60)
    print("  CONSERVATIVE FRACTION ANALYSIS")
    print("=" * 60)

    for metric in fc_metrics:
        scored = score_many_combinations_fc(
            Y, all_combs, metric=metric, n_workers=n_workers, show_progress=False
        )
        fc_values = np.array([fc for _, fc in sorted(scored, key=lambda x: x[0])])
        # Sort by combination to align with markov later

        vmin, vmax = float(fc_values.min()), float(fc_values.max())
        vrange = vmax - vmin
        vmean = float(fc_values.mean())
        vstd = float(fc_values.std())
        cv = vstd / vmean if vmean > 0 else 0.0
        discriminates = cv > cv_threshold

        best_idx = int(np.argmax(fc_values))
        worst_idx = int(np.argmin(fc_values))

        report["conservative_fraction"][metric] = {
            "discriminates": discriminates,
            "min": vmin,
            "max": vmax,
            "range": vrange,
            "mean": vmean,
            "std": vstd,
            "cv": cv,
            "best_comb": all_combs[best_idx],
            "best_value": vmax,
            "worst_comb": all_combs[worst_idx],
            "worst_value": vmin,
        }

        verdict = "YES" if discriminates else "NO"
        print(f"\n  Metric: {metric}")
        print(f"    Range  : {vmin:.6f} → {vmax:.6f} (span = {vrange:.6f})")
        print(f"    Mean   : {vmean:.6f}")
        print(f"    Std    : {vstd:.6f}")
        print(f"    CV     : {cv:.4f}")
        print(f"    Best   : {all_combs[best_idx]} (fc={vmax:.6f})")
        print(f"    Worst  : {all_combs[worst_idx]} (fc={vmin:.6f})")
        print(f"    Discriminates? {verdict} (CV threshold = {cv_threshold})")

    # ===================================================================
    # Markov time
    # ===================================================================
    print("\n" + "=" * 60)
    print("  MARKOV TIME ANALYSIS")
    print("=" * 60)

    scored_tau = score_many_combinations_markov(
        Y, all_combs, n_bins=n_bins, n_workers=n_workers, show_progress=False
    )
    tau_values = np.array([tau for _, tau in sorted(scored_tau, key=lambda x: x[0])])

    # Filter out inf values for statistics
    tau_finite = tau_values[np.isfinite(tau_values)]
    n_inf = int(np.sum(~np.isfinite(tau_values)))

    if len(tau_finite) > 0:
        vmin_tau, vmax_tau = float(tau_finite.min()), float(tau_finite.max())
        vrange_tau = vmax_tau - vmin_tau
        vmean_tau = float(tau_finite.mean())
        vstd_tau = float(tau_finite.std())
        cv_tau = vstd_tau / vmean_tau if vmean_tau > 0 else 0.0
    else:
        vmin_tau = vmax_tau = vrange_tau = vmean_tau = vstd_tau = cv_tau = float("nan")

    discriminates_tau = cv_tau > cv_threshold
    best_idx_tau = int(np.argmin(tau_values))  # min tau = best
    worst_idx_tau = int(np.argmax(np.where(np.isfinite(tau_values), tau_values, -np.inf)))

    report["markov_time"] = {
        "discriminates": discriminates_tau,
        "min": vmin_tau,
        "max": vmax_tau,
        "range": vrange_tau,
        "mean": vmean_tau,
        "std": vstd_tau,
        "cv": cv_tau,
        "n_inf": n_inf,
        "best_comb": all_combs[best_idx_tau],
        "best_value": float(tau_values[best_idx_tau]),
        "worst_comb": all_combs[worst_idx_tau],
        "worst_value": float(tau_values[worst_idx_tau]),
    }

    verdict_tau = "YES" if discriminates_tau else "NO"
    print(f"\n  Metric: Markov relaxation time (n_bins={n_bins})")
    print(f"    Range    : {vmin_tau:.6f} → {vmax_tau:.6f} (span = {vrange_tau:.6f})")
    print(f"    Mean     : {vmean_tau:.6f}")
    print(f"    Std      : {vstd_tau:.6f}")
    print(f"    CV       : {cv_tau:.4f}")
    print(f"    Inf count: {n_inf}/{n_combs} non-ergodic")
    print(f"    Best     : {all_combs[best_idx_tau]} (tau={tau_values[best_idx_tau]:.6f})")
    print(f"    Worst    : {all_combs[worst_idx_tau]} (tau={tau_values[worst_idx_tau]:.6f})")
    print(f"    Discriminates? {verdict_tau} (CV threshold = {cv_threshold})")

    # ===================================================================
    # Recommendation
    # ===================================================================
    print("\n" + "=" * 60)
    print("  RECOMMENDATION")
    print("=" * 60)

    fc_discriminates_any = any(
        r["discriminates"] for r in report["conservative_fraction"].values()
    )

    if discriminates_tau and not fc_discriminates_any:
        method = "markov"
        rationale = (
            "Markov time discriminates strongly (CV={cv:.3f}) while "
            "conservative fraction does not across all metrics. "
            "Use 'markov' as the primary scoring method."
        ).format(cv=cv_tau)
    elif fc_discriminates_any and not discriminates_tau:
        method = "conservative"
        best_fc_metric = max(
            report["conservative_fraction"].items(),
            key=lambda kv: kv[1]["cv"],
        )[0]
        rationale = (
            "Conservative fraction discriminates (CV above threshold) while "
            "Markov time does not. Use 'conservative' with "
            f"fc_metric='{best_fc_metric}'."
        )
    elif discriminates_tau and fc_discriminates_any:
        method = "weighted"
        rationale = (
            "Both criteria discriminate. Use 'weighted' with alpha≈0.5 "
            "to balance variance retention and dynamical speed, "
            "or 'pareto' to explore the trade-off frontier."
        )
    else:
        method = "markov"
        rationale = (
            "Neither criterion discriminates strongly on this dataset. "
            "Defaulting to 'markov' as it tends to be more robust "
            "when the signal has rich temporal structure."
        )

    report["recommendation"] = {
        "method": method,
        "rationale": rationale,
        "fc_discriminates_any": fc_discriminates_any,
        "markov_discriminates": discriminates_tau,
    }

    print(f"\n  Proposed scoring method : {method}")
    print(f"  Rationale               : {rationale}\n")

    return report


# ---------------------------------------------------------------------------
# CLI entry point (optional, for quick testing)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract latent subspace from EEG"
    )
    parser.add_argument("--file", type=str, default=None,
                        help="Path to raw EEG file (uses MNE sample data if omitted)")
    parser.add_argument("--n-dim", type=int, default=2,
                        help="Target dimensionality (default: 2)")
    parser.add_argument("--method", type=str, default="markov",
                        choices=["markov", "conservative", "weighted",
                                 "sequential", "pareto", "independent"],
                        help="Scoring method (default: markov)")
    parser.add_argument("--fc-metric", type=str, default="variance_sum",
                        choices=["variance_sum", "first_pc_var", "total_variance"])
    parser.add_argument("--n-bins", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--greedy", action="store_true")

    args = parser.parse_args()

    if args.file:
        raw_input = args.file
    else:
        raw_input = load_sample_mne_data()

    search = "greedy" if args.greedy else "exhaustive"

    print(f"Extracting {args.n_dim}-D latent space using '{args.method}'...")
    latent, meta = extract_latent_space(
        raw_input,
        n_dim=args.n_dim,
        scoring_method=args.method,
        fc_metric=args.fc_metric,
        n_bins=args.n_bins,
        alpha=args.alpha,
        search_strategy=search,
        n_workers=args.workers,
    )

    print(f"\nLatent space shape: {latent.shape}")
    print(f"Selected components: {meta['selected_indices']}")
    print(f"Scores: {meta['latent_scores']}")
    print(f"Total time: {meta['elapsed_time']:.1f} s")
