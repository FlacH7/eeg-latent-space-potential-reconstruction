"""
src.pipelines.run_cdhsa - Orchestrator for the full CD-HSA pipeline
===================================================================

Provides a high-level interface that runs all steps (A through D)
with sensible defaults, plus a dataclass for configuration.

Usage (programmatic, original API preserved)::

    from src.pipelines.run_cdhsa import CDHSAConfig, run_cdhsa, CDHSAResult
    cfg = CDHSAConfig(fixed_rank=10, a6_n_null=100, bc_n_perm=5000)
    result = run_cdhsa(X, L, cfg)
    print(result.summary())

Usage (CLI - builds Hankel matrices internally from EEG)::

    python -m src.pipelines.run_cdhsa \\
        --session session1 \\
        --tasks eyesclosed eyesopen \\
        --subjects 1 2 3 4 5 \\
        --t-start 0 --t-end 300 \\
        --L 10 \\
        --l-freq 1.0 --h-freq 40.0 \\
        --hankel-depth 250 \\
        --fixed-rank 10 --a6-n-null 100 --bc-n-perm 5000

Pipeline interno hasta las matrices de Hankel
-----------------------------------------------
Para cada par (sujeto, condicion)::

    1. Cargar EEG bruto         -> load_test_retest_gedai_eeg_from_ids()
    2. Filtro pasa-banda        -> extract_filtered_data_matrix()
       (X_filtered: n_channels x n_times, centrada a media cero)
    3. Matriz de Hankel         -> _build_multivariate_hankel()
       (H: (n_channels * depth) x (n_times - depth + 1))
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


# =====================================================================
# CDHSAConfig - sin cambios
# =====================================================================

@dataclass
class CDHSAConfig:
    """Configuration for the full CD-HSA pipeline."""
    fixed_rank: int = 10
    rank_method: str = "fixed"
    rmax: int = 20
    n_blocks: int = 4
    repro_threshold: float = 0.80
    repro_strategy: str = "consecutive"
    max_common: int = 30
    prevalence_quantile: float = 0.10
    a6_max_common: int = 0
    a6_n_folds: int = 5
    a6_n_null: int = 100
    a6_alpha: float = 0.05
    a6_seed: int = 1234
    bc_blocks: list | None = None
    bc_energy_metric: str = "log_absolute"
    bc_geometry_metric: str = "adjusted"
    bc_n_perm: int = 5000
    bc_seed: int = 20260812
    bc_alpha: float = 0.05
    bc_condition_names: list[str] = field(default_factory=list)
    tangent_blocks: list | None = None
    tangent_n_perm: int = 5000
    tangent_seed: int = 9999
    tangent_alpha: float = 0.05
    d_max_specific: int = 10
    d_residual_rank_method: str = "local_gap"
    d_residual_rank_threshold: float = 0.1
    d_fixed_residual_rank: int = 5
    skip_bc: bool = False
    skip_tangent: bool = False
    skip_d: bool = False
    seed: int = 42


# =====================================================================
# CDHSAResult - sin cambios
# =====================================================================

class CDHSAResult:
    """Container for all CD-HSA results."""

    def __init__(self, config, R, A6=None, BC=None, G=None, D=None):
        self.config = config
        self.R = R
        self.A6 = A6
        self.BC = BC
        self.G = G
        self.D = D

    def summary(self) -> str:
        lines = []
        lines.append("CD-HSA Results Summary")
        lines.append("=" * 60)
        R = self.R
        lines.append(f"  Subjects: {R['S']}, Conditions: {R['C']}, "
                     f"Channels: {R['p']}, d = {R['d']}")
        lines.append(f"  Common directions estimated: {len(R['lambda_'])}")
        if self.A6 is not None:
            lines.append(f"\n  A6 Common rank r0 = {self.A6['r0']}")
            if self.A6['r0'] > 0:
                for j in range(self.A6['r0']):
                    lines.append(
                        f"    j={j+1}: lambda={self.A6['lambda0'][j]:.4f}"
                    )
        if self.BC is not None:
            lines.append(f"\n  B/C Condition tests (alpha={self.config.bc_alpha}):")
            for k, name in enumerate(self.BC['block_names']):
                sig_e = "*" if self.BC['sig_energy_maxF'][k] else ""
                sig_g = "*" if self.BC['sig_geometry_maxF'][k] else ""
                lines.append(
                    f"    {name}: energy_F={self.BC['summary']['energy_F'][k]:.2f}"
                    f"{sig_e}  geom_F={self.BC['summary']['geometry_F'][k]:.2f}"
                    f"{sig_g}"
                )
        if self.G is not None:
            lines.append(f"\n  Tangent geometry test:")
            for k in range(len(self.G['T_obs'])):
                sig = "*" if self.G['sig_maxT'][k] else ""
                lines.append(
                    f"    block {k+1}: T={self.G['T_obs'][k]:.4f}"
                    f"  p_maxT={self.G['p_maxT'][k]:.4f}{sig}"
                )
        if self.D is not None:
            lines.append(f"\n  D Condition-specific modes:")
            for c in range(self.D['C']):
                rc = self.D['r_specific'][c]
                pc = self.D['prevalence_contrast'][c]
                lines.append(
                    f"    Condition {c+1}: {rc} modes,"
                    f"  prevalence contrast = {pc:.4f}"
                )
        return "\n".join(lines)


# =====================================================================
# run_cdhsa - sin cambios
# =====================================================================

def run_cdhsa(
    X: list[list[NDArray[np.floating]]],
    L: int,
    config: CDHSAConfig | None = None,
) -> CDHSAResult:
    """Run the full CD-HSA pipeline.

    Parameters
    ----------
    X : list of list of arrays, shape (p, T)
        X[s][c] is the data matrix (e.g. Hankel) for subject s,
        condition c, with shape (p, T).
    L : int
        Subspace dimension to analyse.
    config : CDHSAConfig, optional

    Returns
    -------
    result : CDHSAResult
    """
    from src.cdhsa.a_common_subspace import cdhsa_A1_A5
    from src.cdhsa.a6_common_rank import cdhsa_A6_common_rank
    from src.cdhsa.b_energy import cdhsa_BC_condition_tests
    from src.cdhsa.c_geometry import cdhsa_tangent_geometry_test
    from src.cdhsa.d_condition_specific import cdhsa_D_condition_specific_modes

    if config is None:
        config = CDHSAConfig()

    R = cdhsa_A1_A5(
        X, L,
        rank_method=config.rank_method,
        fixed_rank=config.fixed_rank,
        rmax=config.rmax,
        n_blocks=config.n_blocks,
        repro_threshold=config.repro_threshold,
        repro_strategy=config.repro_strategy,
        max_common=config.max_common,
        prevalence_quantile=config.prevalence_quantile,
    )

    a6_max = config.a6_max_common if config.a6_max_common > 0 else min(20, len(R['lambda_']))
    A6 = cdhsa_A6_common_rank(R, opts={
        'max_common': a6_max,
        'n_folds': config.a6_n_folds,
        'n_null': config.a6_n_null,
        'alpha': config.a6_alpha,
        'seed': config.a6_seed,
    })

    BC = None
    G = None
    D = None

    if not config.skip_bc and A6['r0'] >= 1:
        bc_opts = {
            'blocks': config.bc_blocks,
            'energy_metric': config.bc_energy_metric,
            'geometry_metric': config.bc_geometry_metric,
            'n_perm': config.bc_n_perm,
            'seed': config.bc_seed,
            'alpha': config.bc_alpha,
        }
        if config.bc_condition_names:
            bc_opts['condition_names'] = config.bc_condition_names
        BC = cdhsa_BC_condition_tests(X, L, R, A6, opts=bc_opts)

    if not config.skip_tangent and A6['r0'] >= 1:
        t_blocks = config.tangent_blocks
        if t_blocks is None:
            t_blocks = [np.arange(1, A6['r0'] + 1, dtype=int)]
        G = cdhsa_tangent_geometry_test(R, A6, blocks=t_blocks, opts={
            'n_perm': config.tangent_n_perm,
            'seed': config.tangent_seed,
            'alpha': config.tangent_alpha,
        })

    if not config.skip_d and A6['r0'] >= 1:
        D = cdhsa_D_condition_specific_modes(X, L, R, A6, opts={
            'max_specific': config.d_max_specific,
            'residual_rank_method': config.d_residual_rank_method,
            'residual_rank_threshold': config.d_residual_rank_threshold,
            'fixed_residual_rank': config.d_fixed_residual_rank,
            'prevalence_quantile': config.prevalence_quantile,
        })

    return CDHSAResult(config=config, R=R, A6=A6, BC=BC, G=G, D=D)


# =====================================================================
# Construccion de matrices de Hankel desde EEG
# =====================================================================

def build_hankel_from_eeg(
    *,
    session: str,
    tasks: list[str],
    subject_ids: list[int],
    db_path: str | Path | None = None,
    t_start: float | None = None,
    t_stop: float | None = None,
    l_freq: float = 1.0,
    h_freq: float = 40.0,
    hankel_depth: int | None = None,
    verbose: bool | str | None = None,
) -> tuple[list[list[NDArray[np.floating]]], dict]:
    """Construir las matrices de Hankel para cada par (sujeto, condicion).

    Pipeline interno (reproducido del pipeline de latent-space extraction):

        1. Cargar EEG bruto (por sujeto individual)
           -> load_test_retest_gedai_eeg_from_ids()
        2. Filtro pasa-banda (l_freq, h_freq)
           -> extract_filtered_data_matrix()
           Resultado: X_filtered de forma (n_channels, n_times), centrada.
        3. Matriz de Hankel bloque-multivariada
           -> _build_multivariate_hankel(X_filtered, depth)
           Resultado: H de forma (n_channels * depth, n_times - depth + 1)

    Parameters
    ----------
    session : str
    tasks : list of str
    subject_ids : list of int
    db_path : str or Path or None
    t_start, t_stop : float or None
    l_freq, h_freq : float
    hankel_depth : int or None
        None -> auto: clip(sfreq * 0.25, 50, 200) capped at n_times // 10.
    verbose : bool or str or None

    Returns
    -------
    X : list[list[NDArray]]
        X[s][c] = Hankel matrix (p, T_samples) para sujeto s, condicion c.
    info : dict
    """
    from src.latent_space_extraction.test_retest_gedai_eeg import (
        load_test_retest_gedai_eeg_from_ids,
    )
    from src.latent_space_extraction.eeg_preprocessing import (
        extract_filtered_data_matrix,
    )
    from src.latent_space_extraction.hankel_dmd_extractor import (
        _build_multivariate_hankel,
        _auto_embedding_depth,
    )

    # Resolver db_path
    if db_path is None:
        try:
            from src.utils.config import DB_TEST_RETEST_GEDAI_PATH
            db_path = DB_TEST_RETEST_GEDAI_PATH
        except ImportError:
            raise ValueError(
                "--db-path es obligatorio si src.utils.config "
                "no define DB_TEST_RETEST_GEDAI_PATH."
            )

    S = len(subject_ids)
    C = len(tasks)

    X: list[list[NDArray[np.floating]]] = []
    shapes: list[list[tuple[int, int] | None]] = []
    sfreqs: list[float] = []
    depths_used: list[int] = []
    n_channels_list: list[int] = []
    n_times_filtered_list: list[int] = []
    skipped: list[tuple[int, int, str]] = []

    t0_global = time.time()

    for s_idx, subj_id in enumerate(subject_ids):
        subject_label = f"sub-{subj_id:02d}"
        X_s: list[NDArray[np.floating]] = []
        shapes_s: list[tuple[int, int] | None] = []

        for c_idx, task in enumerate(tasks):
            tag = f"[{s_idx + 1}/{S}] {subject_label}/{session}/{task}"
            print(f"  {tag} ...", end=" ")
            sys.stdout.flush()

            # --- Paso 1: Cargar EEG bruto del sujeto individual ---
            try:
                raw = load_test_retest_gedai_eeg_from_ids(
                    subject=subject_label,
                    session=session,
                    task=task,
                    db_path=db_path,
                    t_start=t_start,
                    t_stop=t_stop,
                    preload=True,
                    verbose=False,
                )
            except (FileNotFoundError, ValueError) as exc:
                print(f"SKIP ({exc})")
                X_s.append(np.empty((0, 0)))
                shapes_s.append(None)
                skipped.append((s_idx, c_idx, str(exc)))
                continue

            sfreq = float(raw.info["sfreq"])

            # --- Paso 2: Filtro pasa-banda ---
            X_filtered, _raw_filt, sfreq = extract_filtered_data_matrix(
                raw, l_freq=l_freq, h_freq=h_freq, verbose=False,
            )
            n_ch, n_times = X_filtered.shape

            # --- Paso 3: Construir matriz de Hankel ---
            if hankel_depth is None:
                depth = _auto_embedding_depth(sfreq, n_times)
            else:
                depth = int(hankel_depth)

            if depth >= n_times:
                print(f"SKIP (depth={depth} >= n_times={n_times})")
                X_s.append(np.empty((0, 0)))
                shapes_s.append(None)
                skipped.append((s_idx, c_idx,
                    f"depth={depth} >= n_times={n_times}"))
                continue

            H = _build_multivariate_hankel(X_filtered, depth)

            X_s.append(H)
            shapes_s.append(H.shape)
            sfreqs.append(sfreq)
            depths_used.append(depth)
            n_channels_list.append(n_ch)
            n_times_filtered_list.append(n_times)

            print(f"OK  ch={n_ch} T_raw={n_times} "
                  f"depth={depth} -> H={H.shape}")
            sys.stdout.flush()

        X.append(X_s)
        shapes.append(shapes_s)

    elapsed = time.time() - t0_global

    info = {
        "session": session,
        "tasks": tasks,
        "subject_ids": subject_ids,
        "S": S,
        "C": C,
        "l_freq": l_freq,
        "h_freq": h_freq,
        "hankel_depth_requested": hankel_depth,
        "shapes": shapes,
        "sfreqs": sfreqs,
        "depths_used": depths_used,
        "n_channels": n_channels_list,
        "n_times_filtered": n_times_filtered_list,
        "skipped": skipped,
        "elapsed_build": elapsed,
    }
    if sfreqs:
        info["sfreq_common"] = sfreqs[0] if len(set(sfreqs)) == 1 else None
        info["depth_common"] = (
            depths_used[0] if len(set(depths_used)) == 1 else None
        )
        info["n_channels_common"] = (
            n_channels_list[0] if len(set(n_channels_list)) == 1 else None
        )

    return X, info


def characterize_hankel_matrices(
    X: list[list[NDArray[np.floating]]],
    info: dict,
) -> str:
    """Resumen legible de las matrices de Hankel construidas.

    Para cada par (sujeto, condicion) valido calcula:
    - Forma (p, T_samples)
    - Rango numerico (SVD parcial, k=min(p,T,50))
    - Ratio de compresion p/rank
    - Top-5 valores singulares
    """
    lines = []
    lines.append("")
    lines.append("=" * 70)
    lines.append("  CARACTERIZACION DE MATRICES DE HANKEL")
    lines.append("=" * 70)

    S = info["S"]
    C = info["C"]
    tasks = info["tasks"]
    subject_ids = info["subject_ids"]

    lines.append(f"  Sujetos           : {S}")
    lines.append(f"  Condiciones       : {C}  ({', '.join(tasks)})")
    lines.append(f"  Filtro            : {info['l_freq']}-{info['h_freq']} Hz")
    lines.append(f"  Depth pedido      : {info['hankel_depth_requested']}")
    if info.get("sfreq_common") is not None:
        lines.append(f"  sfreq             : {info['sfreq_common']:.2f} Hz")
    if info.get("depth_common") is not None:
        lines.append(f"  Depth real        : {info['depth_common']}")
    if info.get("n_channels_common") is not None:
        lines.append(f"  Canales           : {info['n_channels_common']}")
    lines.append(f"  Saltados          : {len(info['skipped'])}")
    lines.append(f"  Tiempo construccion: {info['elapsed_build']:.1f} s")

    header = (f"  {'Sujeto':<12} {'Cond':<15} {'Forma H':<30} "
              f"{'Rank':>8} {'Compres.':>10}  {'Top-5 sing.vals'}")
    lines.append("")
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    n_valid = 0
    for s in range(S):
        for c in range(C):
            H = X[s][c]
            subj_label = f"sub-{subject_ids[s]:02d}"
            if H.size == 0:
                lines.append(
                    f"  {subj_label:<12} {tasks[c]:<15} {"(vacio)":<30}"
                )
                continue
            n_valid += 1
            p, T_samples = H.shape

            k_svd = min(p, T_samples, 50)
            try:
                from scipy.sparse.linalg import svds
                svals = svds(H, k=k_svd, return_singular_vectors=False)
                svals = np.sort(svals)[::-1]
                rank_est = int(np.sum(svals > svals[0] * 1e-6))
            except Exception:
                rank_est = -1
                svals = np.array([])

            comp = p / rank_est if rank_est > 0 else float("inf")
            top5 = ", ".join(f"{v:.1f}" for v in svals[:5])
            lines.append(
                f"  {subj_label:<12} {tasks[c]:<15} {str(H.shape):<30} "
                f"{rank_est:>8} {comp:>9.2f}x  {top5}"
            )

    lines.append("")
    lines.append(f"  Total matrices validas: {n_valid} / {S * C}")
    if info.get("sfreq_common") is not None:
        lines.append(f"  sfreq consistente   : SI")
    else:
        lines.append("  sfreq consistente   : NO")
    if info.get("depth_common") is not None:
        lines.append(f"  Depth consistente   : SI")
    else:
        lines.append(f"  Depth consistente   : NO")

    return "\n".join(lines)


# =====================================================================
# CLI
# =====================================================================

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "CD-HSA: construye matrices de Hankel desde EEG del "
            "dataset Gedai y ejecuta el analisis CD-HSA."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Fuente de datos ---
    parser.add_argument("--session", type=str, required=True,
                        help="Session ID (ej: session1)")
    parser.add_argument("--tasks", type=str, nargs="+", required=True,
                        help="Condiciones/tareas (ej: eyesclosed eyesopen)")
    parser.add_argument("--subjects", type=int, nargs="+", required=True,
                        help="Indices de sujetos (ej: 1 2 3 ... 20)")
    parser.add_argument("--db-path", type=str, default=None,
                        help="Raiz del dataset Gedai")
    parser.add_argument("--t-start", type=float, default=None,
                        help="Inicio del segmento (s)")
    parser.add_argument("--t-end", type=float, default=None,
                        help="Fin del segmento (s)")

    # --- CDHSA ---
    parser.add_argument("--L", type=int, required=True,
                        help="Dimension del subespacio para CD-HSA")

    # --- Preprocesamiento ---
    parser.add_argument("--l-freq", type=float, default=1.0,
                        help="Corte inferior del filtro (Hz). Default: 1.0")
    parser.add_argument("--h-freq", type=float, default=40.0,
                        help="Corte superior del filtro (Hz). Default: 40.0")

    # --- Hankel ---
    parser.add_argument("--hankel-depth", type=int, default=10,
                        help="Profundidad Hankel. None = auto")

    # --- Config CDHSA ---
    g_cdhsa = parser.add_argument_group("Parametros CDHSA (Steps A-D)")
    g_cdhsa.add_argument("--fixed-rank", type=int, default=20)
    g_cdhsa.add_argument("--rank-method", type=str, default="reproducibility",
                         choices=["fixed", "reproducibility"])
    g_cdhsa.add_argument("--a6-n-null", type=int, default=500)
    g_cdhsa.add_argument("--bc-n-perm", type=int, default=5000)
    g_cdhsa.add_argument("--skip-bc", action="store_true")
    g_cdhsa.add_argument("--skip-tangent", action="store_true")
    g_cdhsa.add_argument("--skip-d", action="store_true")

    parser.add_argument("--verbose", action="store_true", default=True)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    verbose = "INFO" if args.verbose else None

    # 1. Construir matrices de Hankel
    print("=" * 70)
    print("  CONSTRUYENDO MATRICES DE HANKEL DESDE EEG")
    print("=" * 70)

    X, hankel_info = build_hankel_from_eeg(
        session=args.session,
        tasks=args.tasks,
        subject_ids=args.subjects,
        db_path=args.db_path,
        t_start=args.t_start,
        t_stop=args.t_end,
        l_freq=args.l_freq,
        h_freq=args.h_freq,
        hankel_depth=args.hankel_depth,
        verbose=verbose,
    )

    # 2. Caracterizar
    print(characterize_hankel_matrices(X, hankel_info))

    n_valid = sum(
        1 for s in range(len(X)) for c in range(len(X[s]))
        if X[s][c].size > 0
    )
    if n_valid == 0:
        print("\n[ERROR] No se construyeron matrices validas.")
        return 1

    # 3. Configurar y ejecutar CD-HSA
    cfg = CDHSAConfig(
        fixed_rank=args.fixed_rank,
        rank_method=args.rank_method,
        a6_n_null=args.a6_n_null,
        bc_n_perm=args.bc_n_perm,
        bc_condition_names=list(args.tasks),
        skip_bc=args.skip_bc,
        skip_tangent=args.skip_tangent,
        skip_d=args.skip_d,
    )

    print("\n" + "=" * 70)
    print("  EJECUTANDO CD-HSA")
    print("=" * 70)
    print(f"  L (subespacio)  : {args.L}")
    print(f"  fixed_rank      : {cfg.fixed_rank}")
    print(f"  a6_n_null       : {cfg.a6_n_null}")
    print(f"  bc_n_perm       : {cfg.bc_n_perm}")
    print("")
    sys.stdout.flush()

    result = run_cdhsa(X, args.L, cfg)

    # 4. Resultados
    print("")
    print(result.summary())

    return 0


if __name__ == "__main__":
    sys.exit(main())
