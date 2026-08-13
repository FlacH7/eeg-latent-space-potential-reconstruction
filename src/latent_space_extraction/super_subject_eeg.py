"""
super_subject_eeg.py
====================
Loader for **super-subject** EEG recordings from the test-retest Gedai
dataset (EEGLAB ``.set``/``.fdt``).

A *super-subject* is the time-axis concatenation of multiple individual
subjects' EEG recordings sharing the same ``session`` and ``task``.  The
concatenated ``mne.io.Raw`` is then processed by the IgA pipeline **as if
it were a single subject's recording from the start** -- exactly as the
user requested: *"primero unir los eeg y despues hacer todo el
procesamiento como si ese eeg fuera de un unico sujeto desde el inicio"*.

Two helpers are provided:

* :func:`resolve_super_subject_subject_ids` -- maps a super-subject ID
  (1, 2, 3, ...) to the list of individual subject indices it contains,
  either via an explicit ``groups`` mapping or via an auto-generated
  contiguous partition of the subject pool.

* :func:`load_super_subject_eeg` -- loads every individual raw (cropped
  to ``[t_start, t_stop]``), harmonises their channel sets to the
  intersection, and concatenates them along the time axis via
  ``mne.concatenate_raws``.

Typical workflow (per-subject crop window of 300 s)::

    raw = load_super_subject_eeg(
        super_subject_id=1,
        session="session1",
        task="eyesclosed",
        subjects_per_super_subject=20,
        subject_start_offset=1,
        t_start=0.0,
        t_stop=300.0,
        verbose=True,
    )
    # raw is now 20 * 300 s = 6000 s long, ready for the IgA pipeline.

Explicit ``groups`` override (useful for non-contiguous partitions)::

    raw = load_super_subject_eeg(
        super_subject_id=2,
        session="session1",
        task="eyesclosed",
        groups={
            1: [1, 5, 9, 13, ...],
            2: [2, 6, 10, 14, ...],
            3: [3, 7, 11, 15, ...],
        },
        t_start=0.0,
        t_stop=300.0,
    )
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import mne


# ----- REAL-TIME LOG FLUSH (Fix 1) -----
# Ensure print() output appears immediately even when stdout is redirected
# (nohup, pipe, subprocess).  Without this, logs accumulate in Python's
# internal buffer and all flush at once, making the pipeline appear frozen.
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(line_buffering=True)
logger = logging.getLogger("super_subject_eeg")


# ---------------------------------------------------------------------------
# Super-subject -> subject-ids resolution
# ---------------------------------------------------------------------------

def resolve_super_subject_subject_ids(
    super_subject_id: int,
    *,
    subjects_per_super_subject: int = 20,
    subject_start_offset: int = 1,
    groups: dict[int, list[int]] | None = None,
) -> list[int]:
    """
    Resolve the list of individual subject indices that compose a
    super-subject.

    Parameters
    ----------
    super_subject_id : int
        1-indexed super-subject identifier (1, 2, 3, ...).
    subjects_per_super_subject : int, default 20
        Number of individual subjects per super-subject (used when
        ``groups`` is not provided).
    subject_start_offset : int, default 1
        Index of the first subject in the dataset (typically 1 for
        ``sub-01``).
    groups : dict[int, list[int]] | None
        Optional explicit mapping ``{super_subject_id: [subject_indices]}``.
        When provided, takes precedence over the auto-generated
        contiguous partition.

    Returns
    -------
    subject_ids : list[int]
        Ordered list of individual subject indices belonging to the
        requested super-subject.

    Raises
    ------
    KeyError
        If ``groups`` is provided and ``super_subject_id`` is not a key.
    ValueError
        If ``subjects_per_super_subject`` is non-positive.
    """
    if groups is not None:
        if super_subject_id not in groups:
            raise KeyError(
                f"super_subject_id={super_subject_id} not found in groups "
                f"(available: {sorted(groups.keys())})"
            )
        return list(groups[super_subject_id])

    if subjects_per_super_subject <= 0:
        raise ValueError("subjects_per_super_subject must be > 0")

    start_idx = subject_start_offset + (super_subject_id - 1) * subjects_per_super_subject
    end_idx = start_idx + subjects_per_super_subject - 1
    return list(range(start_idx, end_idx + 1))


# ---------------------------------------------------------------------------
# Channel harmonisation across raws
# ---------------------------------------------------------------------------

def _harmonise_channels(raws: list[mne.io.Raw]) -> list[mne.io.Raw]:
    """Restrict every raw to the **intersection** of their channel sets.

    All raws are picked in-place to the same ordered channel list so that
    :func:`mne.concatenate_raws` can stack them along the time axis.

    Notes
    -----
    * Channel order is taken from the first raw.
    * Sampling frequency is verified to be the same across raws.
    * The intersection (rather than the union) is used because
      ``mne.concatenate_raws`` requires identical channel sets; missing
      channels in any single subject would otherwise break concatenation.
    """
    if not raws:
        raise ValueError("Cannot harmonise channels of an empty list.")

    # Verify sfreq consistency
    sfreqs = {float(r.info["sfreq"]) for r in raws}
    if len(sfreqs) != 1:
        raise ValueError(
            f"Inconsistent sampling frequencies across raws: {sfreqs}. "
            f"All subjects in a super-subject must share the same sfreq."
        )

    # Intersection of channel names (preserving first raw's order)
    common = list(raws[0].ch_names)
    for r in raws[1:]:
        common = [ch for ch in common if ch in r.ch_names]

    if not common:
        raise ValueError(
            "No common channels across subjects in super-subject. "
            "Channel sets are completely disjoint."
        )

    n_dropped_per_raw: list[int] = []
    harmonised: list[mne.io.Raw] = []
    for r in raws:
        n_before = len(r.ch_names)
        r.pick(common)
        n_dropped_per_raw.append(n_before - len(r.ch_names))
        harmonised.append(r)

    n_total = len(raws)
    n_dropped_total = sum(n_dropped_per_raw)
    if n_dropped_total > 0:
        logger.info(
            "  [SuperSubject] Harmonised channels: dropped %d channel-instances "
            "across %d raws (final common set: %d channels).",
            n_dropped_total, n_total, len(common),
        )

    return harmonised


# ---------------------------------------------------------------------------
# High-level loader
# ---------------------------------------------------------------------------

def load_super_subject_eeg(
    super_subject_id: int,
    session: str,
    task: str,
    *,
    subject_ids: list[int] | None = None,
    subjects_per_super_subject: int = 20,
    subject_start_offset: int = 1,
    groups: dict[int, list[int]] | None = None,
    db_path: str | Path | None = None,
    t_start: float | None = None,
    t_stop: float | None = None,
    preload: bool = False,
    verbose: bool | str | None = None,
) -> mne.io.Raw:
    """
    Load and concatenate the EEG recordings of all individual subjects
    composing a super-subject.

    The result is a single :class:`mne.io.Raw` whose duration equals the
    sum of the individual (cropped) durations, ready to be processed by
    the IgA pipeline as if it were a single subject.

    Parameters
    ----------
    super_subject_id : int
        1-indexed super-subject identifier.
    session, task : str
        BIDS-style session and task labels shared by every subject in
        the super-subject.
    subject_ids : list[int] | None
        Explicit list of subject indices.  When provided, takes
        precedence over the auto-resolution (``subjects_per_super_subject``
        + ``subject_start_offset`` + ``groups``).
    subjects_per_super_subject : int, default 20
        Used by the auto-resolution if ``subject_ids`` is None and
        ``groups`` is None.
    subject_start_offset : int, default 1
        Index of the first subject in the dataset (used by the
        auto-resolution).
    groups : dict[int, list[int]] | None
        Optional explicit ``{super_subject_id: [subject_indices]}``
        mapping used by the auto-resolution.
    db_path : str | Path | None
        Root of the Gedai-preprocessed dataset.  If ``None``, the
        loader falls back to ``DB_TEST_RETEST_GEDAI_PATH``.
    t_start, t_stop : float | None
        Per-subject crop window (seconds).  Each subject's recording is
        cropped to ``[t_start, t_stop]`` **before** concatenation, so the
        final concatenated raw has length
        ``N_subjects * (t_stop - t_start)`` seconds.  When ``None``, the
        full recording of each subject is used.
    preload : bool, default False
        Forwarded to :func:`load_test_retest_gedai_eeg_from_ids`.
    verbose : bool | str | None
        Verbosity level.

    Returns
    -------
    raw_concat : mne.io.Raw
        Concatenated raw object.

    Raises
    ------
    FileNotFoundError
        If none of the individual ``.set`` files could be loaded.
    ValueError
        If the resolved subject list is empty or the channel sets are
        incompatible.
    """
    # Local import to avoid circular dependencies
    from src.latent_space_extraction.test_retest_gedai_eeg import (
        load_test_retest_gedai_eeg_from_ids,
    )

    # Resolve subject indices
    if subject_ids is None:
        subject_ids = resolve_super_subject_subject_ids(
            super_subject_id,
            subjects_per_super_subject=subjects_per_super_subject,
            subject_start_offset=subject_start_offset,
            groups=groups,
        )

    if not subject_ids:
        raise ValueError(
            f"super_subject_id={super_subject_id} resolved to an empty "
            f"subject list. Check subjects_per_super_subject / groups."
        )

    if verbose:
        print(f"  [SuperSubject] super_subject_id={super_subject_id}")
        print(f"  [SuperSubject] session={session}, task={task}")
        print(f"  [SuperSubject] subjects ({len(subject_ids)}): "
              f"{subject_ids[0]}..{subject_ids[-1]}")

    # Load each individual raw
    raws: list[mne.io.Raw] = []
    _load_t0 = time.time()
    _n_total = len(subject_ids)
    for _load_i, subj_idx in enumerate(subject_ids, 1):
        subject = f"sub-{subj_idx:02d}"
        if verbose:
            _elapsed = time.time() - _load_t0
            _eta = (_elapsed / _load_i) * (_n_total - _load_i) if _load_i > 0 else 0
            print(f"  [SuperSubject]   Loading {_load_i}/{_n_total} {subject}/{session}/{task}..."
                  f"  (elapsed: {_elapsed:.1f}s, ETA: {_eta:.0f}s)")
        try:
            raw_i = load_test_retest_gedai_eeg_from_ids(
                subject=subject,
                session=session,
                task=task,
                db_path=db_path,
                t_start=t_start,
                t_stop=t_stop,
                preload=preload,
                verbose=verbose,
            )
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("  [SuperSubject] Skipping %s: %s", subject, exc)
            continue
        raws.append(raw_i)
    if verbose:
        print(f"  [SuperSubject] Loaded {len(raws)}/{_n_total} subjects in "
              f"{time.time() - _load_t0:.1f}s")

    if not raws:
        raise FileNotFoundError(
            f"Could not load any raw EEG for super_subject_id={super_subject_id} "
            f"(session={session}, task={task})."
        )

    if len(raws) < len(subject_ids):
        logger.warning(
            "  [SuperSubject] Loaded %d / %d subjects (some missing).",
            len(raws), len(subject_ids),
        )

    # Harmonise channels across all raws (intersection)
    raws = _harmonise_channels(raws)

    # Concatenate along time axis
    if verbose:
        print(f"  [SuperSubject] Concatenating {len(raws)} raws along time axis...")
    _concat_t0 = time.time()
    raw_concat = mne.concatenate_raws(raws)
    if verbose:
        print(f"  [SuperSubject] Concatenation done in {time.time() - _concat_t0:.1f}s")

    if verbose:
        total_dur = raw_concat.times[-1]
        print(f"  [SuperSubject] Concatenated raw: "
              f"{len(raw_concat.ch_names)} channels, "
              f"{total_dur:.1f} s ({total_dur / 60:.1f} min)")

    return raw_concat
