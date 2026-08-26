"""
src.pipelines.run_cdhsa_v2 - Memory-Optimized CD-HSA Orchestrator
================================================================

Same pipeline as run_cdhsa.py but uses the _v2 modules that avoid
materializing the block-Hankel matrix.  The two changes are:

  1. cdhsa_A1_A5  →  cdhsa_A1_A5 from a_common_subspace_v2.py
     (LinearOperator + svds, H never materialized)
  2. cdhsa_BC_condition_tests  →  cdhsa_BC_condition_tests from b_energy_v2.py
     (energy computed block-by-block, H never materialized)
  3. cdhsa_D_condition_specific_modes → from d_condition_specific_v2.py
     (removed dead import of build_block_hankel; algorithm unchanged
      since D operates only on small U/W0 matrices)

Expected memory reduction: from ~97 GB peak to ~13-15 GB peak
(with the same parameters that produced 97 GB before).

Usage (programmatic)::

    from run_cdhsa_v2 import run_cdhsa_v2, CDHSAConfig
    cfg = CDHSAConfig(fixed_rank=10, a6_n_null=100, bc_n_perm=5000)
    result = run_cdhsa_v2(X, L, cfg)
    print(result.summary())

The API is 100% compatible with run_cdhsa().  Only the internal
implementation of steps A1-A5 and B/C changes.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


# =====================================================================
# Reuse CDHSAConfig and CDHSAResult from the original module
# =====================================================================
try:
    from src.pipelines.run_cdhsa import CDHSAConfig, CDHSAResult
except ImportError:
    from run_cdhsa import CDHSAConfig, CDHSAResult


# =====================================================================
# run_cdhsa_v2 (memory-optimized)
# =====================================================================

def run_cdhsa_v2(
    X: list[list[NDArray[np.floating]]],
    L: int,
    config: CDHSAConfig | None = None,
) -> CDHSAResult:
    """Run the full CD-HSA pipeline (memory-optimized version).

    Parameters
    ----------
    X : list of list of arrays, shape (p, T)
        X[s][c] is the data matrix (e.g. first-level Hankel) for subject s,
        condition c, with shape (p, T).
    L : int
        Block-Hankel embedding depth.
    config : CDHSAConfig, optional

    Returns
    -------
    result : CDHSAResult
        Identical structure to run_cdhsa().

    Memory optimization
    -------------------
    - Steps A1-A5: uses LinearOperator + svds instead of materializing
      the block-Hankel H of shape (p*L, K).  Peak memory per (s,c):
      O(p*K) instead of O(p*L*K).
    - Steps B/C: computes H^T W_k and ||H||_F^2 block-by-block from X
      instead of materializing H.  Peak memory per (s,c):
      O(K * dim_k) instead of O(p*L*K).
    - Step D: uses d_condition_specific_v2.py which removes the dead
      import of build_block_hankel. The algorithm is identical since D
      only operates on small U_sc and W0 matrices.
    - Steps A6, C (tangent) are unchanged (they operate on small
      matrices already).
    """
    # Import v2 modules
    try:
        from src.cdhsa.a_common_subspace_v2 import cdhsa_A1_A5 as cdhsa_A1_A5_v2
    except ImportError:
        from src.cdhsa.a_common_subspace_v2 import cdhsa_A1_A5 as cdhsa_A1_A5_v2

    try:
        from src.cdhsa.b_energy_v2 import cdhsa_BC_condition_tests as cdhsa_BC_v2
    except ImportError:
        from src.cdhsa.b_energy_v2 import cdhsa_BC_condition_tests as cdhsa_BC_v2

    # Steps A6, C are unchanged (operate on small matrices)
    try:
        from src.cdhsa.a6_common_rank import cdhsa_A6_common_rank
    except ImportError:
        from src.cdhsa.a6_common_rank import cdhsa_A6_common_rank

    try:
        from src.cdhsa.c_geometry import cdhsa_tangent_geometry_test
    except ImportError:
        from src.cdhsa.c_geometry import cdhsa_tangent_geometry_test

    # Step D: v2 (removed dead build_block_hankel import, algorithm identical)
    try:
        from src.cdhsa.d_condition_specific_v2 import cdhsa_D_condition_specific_modes as cdhsa_D_v2
    except ImportError:
        from src.cdhsa.d_condition_specific_v2 import cdhsa_D_condition_specific_modes as cdhsa_D_v2

    if config is None:
        config = CDHSAConfig()

    # ---- A1-A5 (v2: LinearOperator) ----
    print("[v2] Running A1-A5 (memory-optimized: LinearOperator + svds)...")
    R = cdhsa_A1_A5_v2(
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
    print(f"[v2] A1-A5 done. {R['S']}x{R['C']} recordings, d={R['d']}, "
          f"p={R['p']}, qmax={len(R['lambda_'])}")

    # ---- A6 (unchanged) ----
    a6_max = (config.a6_max_common if config.a6_max_common > 0
             else min(20, len(R['lambda_'])))
    A6 = cdhsa_A6_common_rank(R, opts={
        'max_common': a6_max,
        'n_folds': config.a6_n_folds,
        'n_null': config.a6_n_null,
        'alpha': config.a6_alpha,
        'seed': config.a6_seed,
    })
    print(f"[v2] A6 done. r0 = {A6['r0']}")

    # ---- B/C (v2: block-wise energy) ----
    BC = None
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
        print(f"[v2] Running B/C (memory-optimized: block-wise energy, "
              f"{config.bc_n_perm} perms)...")
        BC = cdhsa_BC_v2(X, L, R, A6, opts=bc_opts)
        n_sig_e = int(np.sum(BC['sig_energy_maxF']))
        n_sig_g = int(np.sum(BC['sig_geometry_maxF']))
        print(f"[v2] B/C done. sig_energy={n_sig_e}, sig_geom={n_sig_g}")

    # ---- C: tangent geometry (unchanged) ----
    G = None
    if not config.skip_tangent and A6['r0'] >= 1:
        t_blocks = config.tangent_blocks
        if t_blocks is None:
            t_blocks = [np.arange(1, A6['r0'] + 1, dtype=int)]
        G = cdhsa_tangent_geometry_test(R, A6, blocks=t_blocks, opts={
            'n_perm': config.tangent_n_perm,
            'seed': config.tangent_seed,
            'alpha': config.tangent_alpha,
        })

    # ---- D: condition-specific (v2) ----
    D = None
    if not config.skip_d and A6['r0'] >= 1:
        D = cdhsa_D_v2(X, L, R, A6, opts={
            'max_specific': config.d_max_specific,
            'residual_rank_method': config.d_residual_rank_method,
            'residual_rank_threshold': config.d_residual_rank_threshold,
            'fixed_residual_rank': config.d_fixed_residual_rank,
            'prevalence_quantile': config.prevalence_quantile,
        })

    return CDHSAResult(config=config, R=R, A6=A6, BC=BC, G=G, D=D)


# =====================================================================
# Standalone run (for quick testing without the full project)
# =====================================================================

if __name__ == "__main__":
    print("run_cdhsa_v2.py is a library module. Use run_cdhsa.py CLI with v2 imports,")
    print("or call run_cdhsa_v2(X, L, config) programmatically.")
