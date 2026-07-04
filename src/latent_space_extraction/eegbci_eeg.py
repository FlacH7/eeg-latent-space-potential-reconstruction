"""
eegbci_eeg.py
=============
Loader for the **EEG Motor Movement/Imagery Dataset** (EEGBCI / eegmmidb)
from PhysioNet.  The dataset contains EEG recordings from 109 healthy
subjects performing motor movement/imagery tasks and baseline recordings.

The runs most useful for resting-state / interictal comparison are:
    * Run 1 – baseline, eyes open
    * Run 2 – baseline, eyes closed

Other runs (3-14) contain motor execution / imagery tasks.

Data are fetched automatically via ``mne.datasets.eegbci`` on first use.

Typical workflow::

    # Load baseline eyes-closed for subject 1
    raw = load_eegbci(subject=1, runs=[2], t_minutes=(0, 10))

    # Or use the convenience wrapper with metadata
    raw, info = load_eegbci_with_info(subject=1, runs=[1, 2], t_minutes=(0, 5))
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import mne
from mne.datasets import eegbci


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Run descriptions (from PhysioNet eegmmidb)
RUN_DESCRIPTIONS: dict[int, str] = {
    1:  "baseline eyes open",
    2:  "baseline eyes closed",
    3:  "motor execution: left vs right hand",
    4:  "motor imagery: left vs right hand",
    5:  "motor execution: hands vs feet",
    6:  "motor imagery: hands vs feet",
    7:  "motor execution: left vs right hand (rep 2)",
    8:  "motor imagery: left vs right hand (rep 2)",
    9:  "motor execution: hands vs feet (rep 2)",
    10: "motor imagery: hands vs feet (rep 2)",
    11: "motor execution: left vs right hand (rep 3)",
    12: "motor imagery: left vs right hand (rep 3)",
    13: "motor execution: hands vs feet (rep 3)",
    14: "motor imagery: hands vs feet (rep 3)",
}

# Runs recommended for resting-state analysis
BASELINE_RUNS: list[int] = [1, 2]

# Number of subjects
_N_SUBJECTS: int = 109


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def list_eegbci_runs() -> dict[int, str]:
    """Return a mapping of run number to description."""
    return RUN_DESCRIPTIONS.copy()


def list_eegbci_subjects() -> list[int]:
    """Return the list of available subject IDs (1-109)."""
    return list(range(1, _N_SUBJECTS + 1))


# ---------------------------------------------------------------------------
# Low-level loader
# ---------------------------------------------------------------------------

def load_eegbci(
    subject: int = 1,
    runs: Sequence[int] | None = None,
    *,
    t_start: float | None = None,
    t_stop: float | None = None,
    t_minutes: tuple[float, float] | None = None,
    first_n_minutes: float | None = None,
    concatenate: bool = True,
    preload: bool = True,
    verbose: bool | str | None = None,
) -> mne.io.Raw | list[mne.io.Raw]:
    """
    Load EEGBCI ``.edf`` file(s) for a given subject and run(s).

    Channels are automatically renamed to the 10-20 standard and the
    ``standard_1005`` montage is set via :func:`mne.datasets.eegbci.standardize`.

    Time selection (mutually exclusive, checked in order):

    * ``t_minutes``       – ``(start, stop)`` in minutes.
    * ``first_n_minutes`` – keep only the first *N* minutes.
    * ``t_start`` / ``t_stop`` – absolute seconds.

    Parameters
    ----------
    subject : int, default 1
        Subject ID between 1 and 109.
    runs : Sequence[int] | None, default (1, 2)
        List of run numbers to load.  Defaults to baseline runs 1 and 2
        (eyes open + eyes closed).
    t_start : float | None
        Start time in seconds (lowest priority).
    t_stop : float | None
        Stop time in seconds.
    t_minutes : tuple[float, float] | None
        ``(start_min, stop_min)`` convenience shortcut.
    first_n_minutes : float | None
        Keep only the first *N* minutes.
    concatenate : bool, default True
        If True and multiple runs are requested, concatenate them into
        a single ``Raw`` object.
    preload : bool, default True
        Passed to MNE.
    verbose : bool | str | None
        MNE verbosity level.

    Returns
    -------
    raw : mne.io.Raw or list[mne.io.Raw]
        The loaded (and optionally concatenated) raw object(s).

    Examples
    --------
    >>> # Baseline eyes-closed, subject 1
    >>> raw = load_eegbci(subject=1, runs=[2], t_minutes=(0, 10))

    >>> # Both baseline runs concatenated
    >>> raw = load_eegbci(subject=5, runs=[1, 2], t_minutes=(0, 5))

    >>> # Motor imagery, first 3 minutes only
    >>> raw = load_eegbci(subject=10, runs=[4, 8, 12], first_n_minutes=3)
    """
    if runs is None:
        runs = BASELINE_RUNS

    runs = list(runs)
    if not runs:
        raise ValueError("At least one run must be specified.")

    if not 1 <= subject <= _N_SUBJECTS:
        raise ValueError(
            f"Subject ID must be between 1 and {_N_SUBJECTS}, got {subject}"
        )

    for r in runs:
        if r not in RUN_DESCRIPTIONS:
            raise ValueError(
                f"Invalid run: {r}. Available: {sorted(RUN_DESCRIPTIONS.keys())}"
            )

    # ------------------------------------------------------------------
    # Resolve time window (minutes take precedence)
    # ------------------------------------------------------------------
    _t_start: float | None = t_start
    _t_stop: float | None = t_stop

    if t_minutes is not None:
        if len(t_minutes) != 2:
            raise ValueError("t_minutes must be a tuple of (start, stop)")
        _t_start, _t_stop = t_minutes[0] * 60.0, t_minutes[1] * 60.0
    elif first_n_minutes is not None:
        _t_start, _t_stop = 0.0, first_n_minutes * 60.0

    # ------------------------------------------------------------------
    # Download / locate and load
    # ------------------------------------------------------------------
    try:
        fnames = eegbci.load_data(subject, runs, verbose=verbose)
    except Exception as exc:
        raise FileNotFoundError(
            "Could not download/load EEGBCI data. "
            "Ensure you have an internet connection for the first download."
        ) from exc

    raw_list: list[mne.io.Raw] = []
    for fname in fnames:
        raw = mne.io.read_raw_edf(fname, preload=preload, verbose=verbose)

        # ------------------------------------------------------------------
        # Rename channels to standard 10-20 names and set montage
        # ------------------------------------------------------------------
        # EEGBCI channels come with trailing dots, e.g. "Fc5.", "Cz."
        rename_map = {
            ch: ch.rstrip(".").upper() for ch in raw.ch_names
        }
        raw.rename_channels(rename_map)

        # Try multiple montages until one works
        montage_set = False
        for montage_name in ("standard_1005", "standard_1020"):
            try:
                raw.set_montage(
                    montage_name,
                    match_case=False,
                    on_missing="warn",
                    verbose=verbose,
                )
                if raw.get_montage() is not None:
                    montage_set = True
                    if verbose:
                        print(f"  [EEGBCI] Montage '{montage_name}' set.")
                    break
            except Exception:
                continue

        if not montage_set:
            raise RuntimeError(
                "Could not set electrode montage for EEGBCI data. "
                "This is required for ICLabel component classification."
            )

        raw_list.append(raw)

        if verbose:
            print(f"  [EEGBCI] Loaded {fname.name}: "
                  f"{len(raw.ch_names)} ch, {raw.times[-1] / 60:.1f} min")

    # ------------------------------------------------------------------
    # Crop to requested time window
    # ------------------------------------------------------------------
    if _t_start is not None or _t_stop is not None:
        for i, raw in enumerate(raw_list):
            total_dur = raw.times[-1]
            start = _t_start if _t_start is not None else 0.0
            stop = _t_stop if _t_stop is not None else total_dur

            start = max(0.0, start)
            stop = min(stop, total_dur)

            if start >= stop:
                raise ValueError(
                    f"Invalid crop window: t_start={start:.1f}s >= "
                    f"t_stop={stop:.1f}s (duration: {total_dur:.1f}s)"
                )

            raw.crop(tmin=start, tmax=stop)

            if verbose:
                print(f"  [EEGBCI] Cropped run to {start:.1f}s-{stop:.1f}s "
                      f"({(stop - start) / 60:.1f} min)")

    # ------------------------------------------------------------------
    # Concatenate or return list
    # ------------------------------------------------------------------
    if concatenate and len(raw_list) > 1:
        raw_out = mne.concatenate_raws(raw_list, verbose=verbose)
        if verbose:
            print(f"  [EEGBCI] Concatenated {len(raw_list)} runs: "
                  f"{len(raw_out.ch_names)} ch, {raw_out.times[-1] / 60:.1f} min total")
        return raw_out

    return raw_list[0] if len(raw_list) == 1 else raw_list


# ---------------------------------------------------------------------------
# High-level convenience loader
# ---------------------------------------------------------------------------

def load_eegbci_with_info(
    subject: int = 1,
    runs: Sequence[int] | None = None,
    *,
    t_minutes: tuple[float, float] | None = None,
    first_n_minutes: float | None = None,
    t_start: float | None = None,
    t_stop: float | None = None,
    concatenate: bool = True,
    preload: bool = True,
    verbose: bool | str | None = None,
) -> tuple[mne.io.Raw, dict]:
    """
    Load EEGBCI data and return the raw object together with metadata.

    Parameters
    ----------
    subject : int, default 1
        Subject ID (1-109).
    runs : Sequence[int] | None
        Run numbers.  Defaults to baseline runs [1, 2].
    t_minutes, first_n_minutes, t_start, t_stop
        Passed through to :func:`load_eegbci`.
    concatenate : bool, default True
        Concatenate multiple runs.
    preload : bool, default True
        Passed to MNE.
    verbose : bool | str | None
        MNE verbosity level.

    Returns
    -------
    raw : mne.io.Raw
        The loaded (and optionally concatenated) raw object.
    info : dict
        Metadata dictionary with keys ``'subject'``, ``'runs'``,
        ``'run_descriptions'``, ``'sfreq'``, ``'n_channels'``,
        ``'ch_names'``, ``'duration_sec'``, ``'concatenated'``.

    Examples
    --------
    >>> raw, info = load_eegbci_with_info(subject=1, runs=[2], t_minutes=(0, 10))
    >>> print(info)
    {'subject': 1, 'runs': [2], 'run_descriptions': ['baseline eyes closed'], ...}
    """
    if runs is None:
        runs = BASELINE_RUNS

    raw = load_eegbci(
        subject=subject,
        runs=runs,
        t_minutes=t_minutes,
        first_n_minutes=first_n_minutes,
        t_start=t_start,
        t_stop=t_stop,
        concatenate=concatenate,
        preload=preload,
        verbose=verbose,
    )

    # If load_eegbci returned a list (concatenate=False), wrap it
    if isinstance(raw, list):
        raw = mne.concatenate_raws(raw, verbose=verbose)

    info = {
        "subject": subject,
        "runs": list(runs),
        "run_descriptions": [RUN_DESCRIPTIONS.get(r, f"run-{r}") for r in runs],
        "sfreq": raw.info["sfreq"],
        "n_channels": len(raw.ch_names),
        "ch_names": raw.ch_names.copy(),
        "duration_sec": raw.times[-1],
        "concatenated": concatenate and len(runs) > 1,
    }

    return raw, info
