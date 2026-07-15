"""
Chapman-Kolmogorov Test for Markovianity in Latent Space
==========================================================

Tests whether a low-dimensional trajectory satisfies the Chapman-Kolmogorov
equation, a necessary condition for the process to be Markovian.

The CK equation states that for a Markov process the transition probability
can be composed:

    T(tau) = T(tau/2) @ T(tau/2)

If this does not hold, the process has memory beyond the current state and
the Kramers-Moyal expansion may not be valid.

Typical usage from the IGA pipeline::

    from ck_test import chapman_kolmogorov_test

    ck_result = chapman_kolmogorov_test(
        data,               # latent trajectory, shape (n_samples, n_dim)
        dt=dt,              # sampling interval [s]
        tau_candidates=None,  # auto-generated lags
        n_bins=20,          # discretisation bins per dimension
        threshold=0.15,     # acceptance threshold for CK error
        plot=True,
        out_dir=out_dir,
    )

    if ck_result["is_markovian"]:
        print(f"  Markovian at tau* = {ck_result['tau_star']:.4f} s")
    else:
        print(f"  NOT Markovian — min error: {ck_result['min_error']:.4f}")
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence, Literal

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_N_BINS: int = 20
DEFAULT_THRESHOLD: float = 0.15
DEFAULT_TAU_MULTIPLIERS: Sequence[int] = (1, 2, 3, 4, 5, 6, 8, 10, 12, 16,
                                             20, 25, 32, 40, 50, 64, 80, 100,
                                             128, 160, 200, 256)


# ---------------------------------------------------------------------------
# Discretisation
# ---------------------------------------------------------------------------

def _discretise_uniform(
    data: np.ndarray,
    n_bins: int,
    pad: float = 0.05,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Discretise each dimension of *data* into *n_bins* uniform-width bins.

    Parameters
    ----------
    data : ndarray, shape (n_samples, D)
        Continuous trajectory.
    n_bins : int
        Number of bins per dimension.
    pad : float
        Fractional padding beyond [min, max] to ensure all points fit.

    Returns
    -------
    states : ndarray, shape (n_samples,)
        Flattened state index for each sample (mixed-radix enumeration).
    edges : list of ndarrays
        Bin edges for each dimension.
    """
    D = data.shape[1]
    edges = []
    digitised = np.empty((data.shape[0], D), dtype=int)

    for d in range(D):
        dmin, dmax = data[:, d].min(), data[:, d].max()
        margin = pad * (dmax - dmin) if dmax > dmin else 1.0
        e = np.linspace(dmin - margin, dmax + margin, n_bins + 1)
        edges.append(e)
        digitised[:, d] = np.digitize(data[:, d], e) - 1
        digitised[:, d] = np.clip(digitised[:, d], 0, n_bins - 1)

    # Mixed-radix encoding
    multipliers = np.cumprod([1] + [n_bins] * (D - 1))
    states = np.sum(digitised * multipliers, axis=1)

    return states, edges


def _discretise_by_quantiles(
    data: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Discretise each dimension of *data* into *n_bins* quantile-based bins.
    Guarantees balanced occupancy.
    """
    D = data.shape[1]
    edges = []
    digitised = np.empty((data.shape[0], D), dtype=int)

    for d in range(D):
        e = np.quantile(data[:, d], np.linspace(0, 1, n_bins + 1))
        e[-1] += 1e-12
        edges.append(e)
        digitised[:, d] = np.digitize(data[:, d], e) - 1
        digitised[:, d] = np.clip(digitised[:, d], 0, n_bins - 1)

    multipliers = np.cumprod([1] + [n_bins] * (D - 1))
    states = np.sum(digitised * multipliers, axis=1)

    return states, edges


# ---------------------------------------------------------------------------
# Transition matrix estimation
# ---------------------------------------------------------------------------

def _estimate_transition_matrix(
    states: np.ndarray,
    lag: int,
    n_states: int,
) -> np.ndarray:
    """
    Estimate the row-normalised transition matrix T^{(lag)} by counting.
    """
    n_samples = len(states)
    counts = np.zeros((n_states, n_states), dtype=float)

    src = states[:n_samples - lag]
    dst = states[lag:]

    flat_idx = src.astype(int) * n_states + dst.astype(int)
    np.add.at(counts.ravel(), flat_idx, 1.0)

    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    T = counts / row_sums

    # Unvisited states -> uniform (prevents NaN in composition)
    unvisited = (counts.sum(axis=1) == 0)
    T[unvisited, :] = 1.0 / n_states

    return T


# ---------------------------------------------------------------------------
# CK error: population-weighted Frobenius norm
# ---------------------------------------------------------------------------

def _ck_error_weighted(
    T_tau: np.ndarray,
    T_half: np.ndarray,
    pi: np.ndarray | None = None,
) -> float:
    """
    Population-weighted CK error.

    Measures the discrepancy between the directly estimated transition
    matrix T(tau) and the composed matrix T(tau/2)^2.  Only the *K*
    most populated states contribute, weighted by their stationary
    probability, which makes the metric robust against poorly sampled
    tail states.

    Parameters
    ----------
    T_tau : ndarray, shape (n_states, n_states)
        Transition matrix at lag tau.
    T_half : ndarray
        Transition matrix at lag tau/2.
    pi : ndarray, optional
        Stationary distribution (eigenvector of T_tau^T with eigenvalue 1).
        If None, estimated from row sums of the count matrix (approximation).

    Returns
    -------
    error : float in [0, 1]
        Weighted Frobenius-norm CK error normalised by ||T_tau||_F.
    """
    T_composed = T_half @ T_half
    diff = T_tau - T_composed

    # Use stationary distribution as weights (focus on well-sampled states)
    if pi is None:
        # Approximate pi from row-normalised counts (symmetrise)
        eigvals, eigvecs = np.linalg.eig(T_tau.T)
        idx = np.argmax(np.real(eigvals))
        pi = np.real(eigvecs[:, idx])
        pi = np.abs(pi)
        pi = pi / pi.sum()

    # Weight the error by stationary probability of each initial state
    weighted_diff = diff * np.sqrt(pi[:, None])
    numerator = np.linalg.norm(weighted_diff, ord="fro")
    denominator = np.linalg.norm(T_tau * np.sqrt(pi[:, None]), ord="fro")

    if denominator == 0:
        return 1.0
    return numerator / denominator


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def chapman_kolmogorov_test(
    data: np.ndarray,
    *,
    dt: float,
    tau_candidates: Sequence[float] | np.ndarray | None = None,
    n_bins: int = DEFAULT_N_BINS,
    discretisation: Literal["uniform", "quantile"] = "uniform",
    threshold: float = DEFAULT_THRESHOLD,
    top_k_states: int | None = None,
    min_occupancy: float = 0.3,
    plot: bool = True,
    out_dir: str | Path | None = None,
    verbose: bool = True,
) -> dict:
    """
    Test whether a low-dimensional trajectory is Markovian via the
    Chapman-Kolmogorov equation.

    For each candidate lag *tau*, the function:
    1. Discretises the state space into bins (uniform or quantile).
    2. Estimates T(tau) and T(tau/2).
    3. Compares T(tau) with T(tau/2) @ T(tau/2).
    4. Reports the population-weighted Frobenius-norm error.

    The process is declared Markovian if the CK error falls below
    *threshold* for some lag.  The smallest such lag is tau*.

    Parameters
    ----------
    data : ndarray, shape (n_samples, D)
        Latent trajectory (continuous).  Typically D=2 or 3.
    dt : float
        Sampling interval in seconds.
    tau_candidates : sequence or ndarray, optional
        Candidate lags *in seconds*.  If None, a geometric progression
        of multipliers is applied to *dt* (see DEFAULT_TAU_MULTIPLIERS).
    n_bins : int, default 20
        Number of bins per dimension.
    discretisation : ``"uniform"`` | ``"quantile"``, default ``"uniform"``
        Binning strategy.  ``"uniform"`` uses equal-width bins (better
        for Gaussian data); ``"quantile"`` uses equal-count bins.
    threshold : float, default 0.15
        CK error threshold for declaring Markovianity.
        Values in the 0.10--0.20 range are typical for continuous
        stochastic processes; 0.15 is a pragmatic default.
    top_k_states : int or None, optional
        Number of most-populated states to use in the CK error.
        ``None`` uses all states.
    min_occupancy : float, default 0.3
        Minimum fraction of bins that must be occupied.
    plot : bool, default True
        Whether to generate the CK error vs tau plot.
    out_dir : str or Path, optional
        Directory to save the plot.  If None, plot is returned but not saved.
    verbose : bool, default True
        Print progress and results.

    Returns
    -------
    result : dict
        Dictionary with keys:

        * ``is_markovian``     — bool, True if CK error < threshold for some tau.
        * ``tau_star``         — float [s], smallest tau with CK error < threshold.
          ``np.inf`` if no such tau exists.
        * ``tau_star_idx``     — int, index in ``tau_candidates`` of tau*.
        * ``min_error``        — float, minimum CK error across all taus.
        * ``min_error_tau``    — float [s], tau at which min_error occurs.
        * ``tau_values``       — ndarray [s], all candidate taus tested.
        * ``ck_errors``        — ndarray, CK error for each tau.
        * ``n_bins``           — int, bins per dimension.
        * ``n_occupied_bins``  — int, number of occupied bins.
        * ``occupancy``        — float, fraction of bins occupied.
        * ``threshold``        — float, threshold used.
        * ``fig``              — matplotlib Figure (if plot=True) or None.
        * ``transition_matrices`` — dict mapping tau (s) to T matrix.

    Interpretation guidelines
    -------------------------
    * ``is_markovian == True`` and ``tau_star`` small (few dt):
      The latent space is well-approximated as Markovian.  Use ``tau_star``
      as the *lag* for Kramers-Moyal estimation.
    * ``is_markovian == True`` but ``tau_star`` large (> 50 dt):
      The process is Markovian but only at coarse time resolution.  The
      Kramers-Moyal estimate may lose fine temporal structure.
    * ``is_markovian == False``:
      The latent space retains memory.  Consider:
      - Increasing the latent dimension.
      - Increasing the embedding depth (for Hankel+DMD).
      - Using the lag at minimum error as a pragmatic compromise.
    """
    t0 = time.time()

    if data.ndim == 1:
        data = data.reshape(-1, 1)
    n_samples, D = data.shape

    if verbose:
        print("\n" + "=" * 70)
        print("  CHAPMAN-KOLMOGOROV MARKOVIANITY TEST")
        print("=" * 70)
        print(f"  Data shape      : {data.shape}")
        print(f"  dt              : {dt:.6f} s  ({1/dt:.1f} Hz)")
        print(f"  Duration        : {n_samples * dt:.1f} s")
        print(f"  Bins per dim    : {n_bins}")
        print(f"  Total states    : {n_bins ** D}")
        print(f"  Discretisation  : {discretisation}")
        print(f"  CK threshold    : {threshold}")

    # ------------------------------------------------------------------
    # 1. Discretise
    # ------------------------------------------------------------------
    if discretisation == "uniform":
        states, edges = _discretise_uniform(data, n_bins)
    else:
        states, edges = _discretise_by_quantiles(data, n_bins)

    n_states = n_bins ** D
    occupied = len(np.unique(states))
    occupancy = occupied / n_states

    if verbose:
        print(f"\n  Occupied bins   : {occupied}/{n_states} ({occupancy:.1%})")

    if occupancy < min_occupancy:
        print(f"  [WARN] Occupancy {occupancy:.1%} < {min_occupancy} — "
              f"consider reducing n_bins (current: {n_bins})")

    # ------------------------------------------------------------------
    # 2. Candidate lags
    # ------------------------------------------------------------------
    if tau_candidates is None:
        tau_steps = np.array(DEFAULT_TAU_MULTIPLIERS, dtype=int)
        tau_candidates = tau_steps * dt
    else:
        tau_candidates = np.asarray(tau_candidates, dtype=float)

    max_tau_samples = n_samples // 3
    max_tau = max_tau_samples * dt
    valid_mask = tau_candidates <= max_tau
    tau_values = tau_candidates[valid_mask]

    if len(tau_values) == 0:
        raise ValueError(
            f"No tau candidates fit in data. "
            f"Max feasible tau: {max_tau:.4f} s "
            f"({max_tau_samples} samples)."
        )

    tau_steps = (tau_values / dt).astype(int)
    tau_steps = np.maximum(tau_steps, 1)

    if verbose:
        print(f"\n  Testing {len(tau_values)} tau values:")
        print(f"    min = {tau_values[0]:.4f} s ({tau_steps[0]} steps)")
        print(f"    max = {tau_values[-1]:.4f} s ({tau_steps[-1]} steps)")

    # ------------------------------------------------------------------
    # 3. Compute CK error for each tau
    # ------------------------------------------------------------------
    ck_errors = np.full(len(tau_values), np.nan)
    T_matrices = {}

    if verbose:
        print(f"\n  {'Tau (s)':>10}  {'Steps':>7}  {'CK error':>12}  {'Status'}")
        print(f"  {'-'*10}  {'-'*7}  {'-'*12}  {'-'*8}")

    for i, (tau_s, lag) in enumerate(zip(tau_values, tau_steps)):
        T_tau = _estimate_transition_matrix(states, lag, n_states)

        half_lag = max(lag // 2, 1)
        T_half = _estimate_transition_matrix(states, half_lag, n_states)

        err = _ck_error_weighted(T_tau, T_half)
        ck_errors[i] = err
        T_matrices[float(tau_s)] = T_tau

        status = "OK" if err < threshold else "---"
        if verbose:
            print(f"  {tau_s:10.4f}  {lag:7d}  {err:12.6f}  {status}")

    # ------------------------------------------------------------------
    # 4. Determine tau*
    # ------------------------------------------------------------------
    min_err_idx = int(np.nanargmin(ck_errors))
    min_error = float(ck_errors[min_err_idx])
    min_error_tau = float(tau_values[min_err_idx])

    below_threshold = ck_errors < threshold
    if np.any(below_threshold):
        tau_star_idx = int(np.argmax(below_threshold))
        tau_star = float(tau_values[tau_star_idx])
        is_markovian = True
    else:
        tau_star_idx = -1
        tau_star = float("inf")
        is_markovian = False

    # ------------------------------------------------------------------
    # 5. Veredicto
    # ------------------------------------------------------------------
    elapsed = time.time() - t0

    if verbose:
        print(f"\n  {'='*50}")
        print(f"  RESULTS")
        print(f"  {'='*50}")
        print(f"  Minimum CK error : {min_error:.6f}  (at tau = {min_error_tau:.4f} s)")
        if is_markovian:
            print(f"  tau*             : {tau_star:.4f} s  ({int(tau_star/dt)} steps)")
            print(f"  CK error at tau* : {ck_errors[tau_star_idx]:.6f}  <  {threshold}")
            print(f"  VERDICT          : MARKOVIAN  (use tau* for KM)")
        else:
            print(f"  tau*             : INF  (no tau satisfies threshold)")
            print(f"  Best CK error    : {min_error:.6f}  >=  {threshold}")
            print(f"  VERDICT          : NOT MARKOVIAN")
            print(f"  Recommendation   : Increase latent dimension or embedding depth")
        print(f"  Elapsed time     : {elapsed:.1f} s")
        print(f"  {'='*50}")

    # ------------------------------------------------------------------
    # 6. Plot
    # ------------------------------------------------------------------
    fig = None
    if plot:
        fig = _plot_ck_results(
            tau_values=tau_values,
            ck_errors=ck_errors,
            threshold=threshold,
            tau_star=tau_star if is_markovian else None,
            min_error_tau=min_error_tau,
            min_error=min_error,
            is_markovian=is_markovian,
            dt=dt,
            D=D,
            n_bins=n_bins,
            occupancy=occupancy,
            discretisation=discretisation,
        )
        if out_dir is not None:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_dir / "ck_test.png", dpi=150)
            if verbose:
                print(f"\n  [PLOT] Saved CK test plot to: {out_dir / 'ck_test.png'}")

    return {
        "is_markovian": is_markovian,
        "tau_star": tau_star,
        "tau_star_idx": tau_star_idx,
        "min_error": min_error,
        "min_error_tau": min_error_tau,
        "tau_values": tau_values,
        "ck_errors": ck_errors,
        "n_bins": n_bins,
        "n_occupied_bins": occupied,
        "occupancy": occupancy,
        "threshold": threshold,
        "discretisation": discretisation,
        "fig": fig,
        "transition_matrices": T_matrices,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_ck_results(
    tau_values: np.ndarray,
    ck_errors: np.ndarray,
    threshold: float,
    tau_star: float | None,
    min_error_tau: float,
    min_error: float,
    is_markovian: bool,
    dt: float,
    D: int,
    n_bins: int,
    occupancy: float,
    discretisation: str,
) -> plt.Figure:
    """Generate the CK error vs tau plot."""
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot CK error curve
    ax.semilogy(tau_values / dt, ck_errors, "ko-", markersize=8,
                markerfacecolor="steelblue", linewidth=1.5, label="CK error",
                zorder=3)

    # Threshold line
    ax.axhline(threshold, color="crimson", linestyle="--", linewidth=2,
               label=f"threshold = {threshold}", zorder=2)

    # tau* marker
    if is_markovian and tau_star is not None:
        tau_star_idx = int(np.argmin(np.abs(tau_values - tau_star)))
        ax.plot(tau_star / dt, ck_errors[tau_star_idx], "*", color="green",
                markersize=20, markeredgecolor="black", markeredgewidth=1.5,
                label=f"$\\tau^*$ = {tau_star:.2f} s", zorder=4)

    # Min error marker
    min_idx = int(np.nanargmin(ck_errors))
    ax.plot(tau_values[min_idx] / dt, min_error, "s", color="orange",
            markersize=10, markeredgecolor="black", markeredgewidth=1,
            label=f"min error = {min_error:.4f}", zorder=4)

    ax.set_xlabel(r"Lag $\tau$  [$\Delta t$ steps]", fontsize=12)
    ax.set_ylabel(
        r"CK error  $\|\sqrt{\pi}(T(\tau) - T(\tau/2)^2)\|_F \,/\, "
        r"\|\sqrt{\pi}\,T(\tau)\|_F$",
        fontsize=11,
    )

    verdict_str = "MARKOVIAN" if is_markovian else "NOT MARKOVIAN"
    verdict_color = "green" if is_markovian else "crimson"
    ax.set_title(
        f"Chapman-Kolmogorov Test  —  {D}D  —  "
        f"bins={n_bins}$^{D}$, {discretisation}, occ={occupancy:.0%}  —  "
        f"$\\mathbf{{{verdict_str}}}$",
        fontsize=12, color=verdict_color, fontweight="bold",
    )

    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="best", fontsize=9)
    ax.set_ylim(bottom=max(ck_errors.min() * 0.3, 1e-4))

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

def _demo():
    """Run a quick demo with synthetic data."""
    np.random.seed(42)
    n = 150000
    dt = 1.0 / 250.0

    # 2D Ornstein-Uhlenbeck (known to be Markovian)
    tau = 0.05
    sigma = 0.5
    x = np.zeros(n)
    y = np.zeros(n)
    for t in range(1, n):
        x[t] = x[t-1] - x[t-1] * dt / tau + sigma * np.sqrt(dt) * np.random.randn()
        y[t] = y[t-1] - y[t-1] * dt / tau + sigma * np.sqrt(dt) * np.random.randn()
    data = np.column_stack([x, y])

    print("Demo: 2D Ornstein-Uhlenbeck (should be Markovian)\n")
    result = chapman_kolmogorov_test(data, dt=dt, n_bins=15,
                                     threshold=0.15, plot=False, verbose=True)
    print(f"\n  is_markovian : {result['is_markovian']}")
    print(f"  tau*         : {result['tau_star']:.4f} s")
    print(f"  min_error    : {result['min_error']:.4f}")


if __name__ == "__main__":
    _demo()
