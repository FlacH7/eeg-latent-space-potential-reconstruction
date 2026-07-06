"""
km_tools_v2.py
================
Pipeline completo para estimación de coeficientes de Kramers-Moyal,
reconstrucción de potenciales via IgA (Galerkin-B-spline), y simulación
de datos estocásticos.

Versión 4.0 — Reconstrucción por Análisis Isogeométrico.

Cambios principales respecto a v3.0:
  - Reemplaza integración BFS/path por Galerkin con B-splines tensoriales.
  - Inversión robusta de D via SVD truncado adaptativo (pseudo-inversa).
  - Descomposición de Helmholtz explícita con métrica de no-equilibrio η.
  - Compatible D=1..5, bins no equidistantes, sin ground truth.
"""

import numpy as np
from kramersmoyal import km
from kramersmoyal.kernels import epanechnikov, gaussian
import matplotlib.pyplot as plt

import hashlib
import json
import os
import warnings

from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from scipy.stats import pearsonr
import scipy.ndimage as ndi

# ---------------------------------------------------------------------------
# Importar núcleo IgA
# ---------------------------------------------------------------------------
from iga_reconstructor import (
    reconstruct_potential_iga,
    compute_g_robust,
    compute_helmholtz_residual,
    BSplineTensorSpace
)

from src.utils.config import(
    BASE_CACHE_PATH
)

# =============================================================================
# SECCIÓN 0: UTILIDADES GENERALES
# =============================================================================

def _param_to_str(param):
    if np.isscalar(param):
        return f"scalar:{float(param):.12g}"
    else:
        arr = np.asarray(param)
        return f"array:{arr.shape}:{np.array2string(arr, precision=12, separator=',', suppress_small=True)}"

def _build_cache_path(config, cache_dir='sim_cache'):
    param_dict = {
        'model': config['model'],
        'D': config['D'],
        'dt': config['dt'],
        'T': config['T'],
        'burn_ratio': config.get('burn_ratio', 0.1),
        'seed': config.get('seed', None),
        'params': {k: _param_to_str(v) for k, v in config.get('params', {}).items()}
    }
    param_str = json.dumps(param_dict, sort_keys=True, ensure_ascii=True)
    param_hash = hashlib.md5(param_str.encode('utf-8')).hexdigest()
    cache_filename = f"sim_D{config['D']}_{config['model']}_{param_hash}.npz"
    return os.path.join(BASE_CACHE_PATH, cache_dir, cache_filename), param_dict

def _rebuild_theoretical(model_info, D):
    """Reconstruye el dict 'theoretical' a partir de model_info (serializable)."""
    name = model_info['name']
    if name == 'ou':
        theta_mat = model_info['theta']
        sigma_mat = model_info['sigma']
        D_diff = model_info['D_diff']
        def drift_func(x):
            x = np.asarray(x)
            if x.ndim == 1: return -theta_mat @ x
            else: return (-theta_mat @ x.T).T
        def diffusion_func(x): return D_diff
        def potential_func(x):
            x = np.asarray(x)
            D_inv = np.linalg.inv(D_diff + 1e-10 * np.eye(D))
            A = D_inv @ theta_mat
            A_sym = (A + A.T) / 2
            if x.ndim == 1: return 0.5 * x @ A_sym @ x
            else: return 0.5 * np.sum(x * (A_sym @ x.T).T, axis=1)
    elif name == 'double_well':
        a_vec = model_info['a']; b_vec = model_info['b']; c = model_info['c']
        sigma_mat = model_info['sigma']; D_diff = (sigma_mat @ sigma_mat.T) / 2.0
        def drift_func(x):
            x = np.asarray(x)
            if x.ndim == 1:
                d = 2 * a_vec * x - 4 * b_vec * x**3
                if c != 0:
                    for i in range(D): d[i] -= 2 * c * np.sum(x[i] - x)
                return d
            else:
                d = 2 * a_vec * x - 4 * b_vec * x**3
                if c != 0:
                    for i in range(D): d[:, i] -= 2 * c * np.sum(x[:, i:i+1] - x, axis=1)
                return d
        def diffusion_func(x): return D_diff
        def potential_func(x):
            x = np.asarray(x)
            if x.ndim == 1:
                U = np.sum(-a_vec * x**2 + b_vec * x**4)
                if c != 0:
                    for i in range(D):
                        for j in range(i+1, D): U += c * (x[i] - x[j])**2
                return U
            else:
                U = np.sum(-a_vec * x**2 + b_vec * x**4, axis=1)
                if c != 0:
                    for i in range(D):
                        for j in range(i+1, D): U += c * (x[:, i] - x[:, j])**2
                return U
    elif name == 'ring_attractor':
        alpha = model_info['alpha']; r0 = model_info['r0']; omega = model_info['omega']
        theta_rest_vec = model_info['theta_rest']
        sigma_mat = model_info['sigma']; D_diff = (sigma_mat @ sigma_mat.T) / 2.0
        def drift_func(x):
            x = np.asarray(x)
            if x.ndim == 1:
                d = np.zeros(D)
                if D >= 2:
                    x1, x2 = x[0], x[1]; r = np.sqrt(x1**2 + x2**2) + 1e-12
                    d[0] = -alpha * (r - r0) * (x1 / r) - omega * x2
                    d[1] = -alpha * (r - r0) * (x2 / r) + omega * x1
                for i in range(2, D): d[i] = -theta_rest_vec[i-2] * x[i]
                return d
            else:
                d = np.zeros_like(x)
                if D >= 2:
                    x1, x2 = x[:, 0], x[:, 1]; r = np.sqrt(x1**2 + x2**2) + 1e-12
                    d[:, 0] = -alpha * (r - r0) * (x1 / r) - omega * x2
                    d[:, 1] = -alpha * (r - r0) * (x2 / r) + omega * x1
                for i in range(2, D): d[:, i] = -theta_rest_vec[i-2] * x[:, i]
                return d
        def diffusion_func(x): return D_diff
        def potential_func(x):
            x = np.asarray(x)
            if x.ndim == 1:
                U = 0.0
                if D >= 2: r = np.sqrt(x[0]**2 + x[1]**2); U += 0.5 * alpha * (r - r0)**2
                for i in range(2, D): U += 0.5 * theta_rest_vec[i-2] * x[i]**2
                return U
            else:
                U = np.zeros(x.shape[0])
                if D >= 2: r = np.sqrt(x[:, 0]**2 + x[:, 1]**2); U += 0.5 * alpha * (r - r0)**2
                for i in range(2, D): U += 0.5 * theta_rest_vec[i-2] * x[:, i]**2
                return U
    elif name == 'multi_stable':
        a = model_info['a']; b = model_info['b']; c = model_info['c']
        sigma_mat = model_info['sigma']; D_diff = (sigma_mat @ sigma_mat.T) / 2.0
        def drift_func(x):
            x = np.asarray(x)
            if x.ndim == 1:
                d = 2 * a * x - 4 * b * x**3; sum_sq = np.sum(x**2)
                d -= 2 * c * x * (sum_sq - x**2); return d
            else:
                d = 2 * a * x - 4 * b * x**3; sum_sq = np.sum(x**2, axis=1, keepdims=True)
                d -= 2 * c * x * (sum_sq - x**2); return d
        def diffusion_func(x): return D_diff
        def potential_func(x):
            x = np.asarray(x)
            if x.ndim == 1:
                U = np.sum(-a * x**2 + b * x**4)
                for i in range(D):
                    for j in range(i+1, D): U += c * x[i]**2 * x[j]**2
                return U
            else:
                U = np.sum(-a * x**2 + b * x**4, axis=1)
                for i in range(D):
                    for j in range(i+1, D): U += c * x[:, i]**2 * x[:, j]**2
                return U
    elif name == 'stochastic_oscillator':
        lam = model_info['lambda']; omega = model_info['omega']
        theta_rest_vec = model_info['theta_rest']
        sigma_mat = model_info['sigma']; D_diff = (sigma_mat @ sigma_mat.T) / 2.0
        def drift_func(x):
            x = np.asarray(x)
            if x.ndim == 1:
                d = np.zeros(D)
                if D >= 2:
                    x1, x2 = x[0], x[1]; r2 = x1**2 + x2**2
                    d[0] = lam * x1 - omega * x2 - r2 * x1
                    d[1] = lam * x2 + omega * x1 - r2 * x2
                for i in range(2, D): d[i] = -theta_rest_vec[i-2] * x[i]
                return d
            else:
                d = np.zeros_like(x)
                if D >= 2:
                    x1, x2 = x[:, 0], x[:, 1]; r2 = x1**2 + x2**2
                    d[:, 0] = lam * x1 - omega * x2 - r2 * x1
                    d[:, 1] = lam * x2 + omega * x1 - r2 * x2
                for i in range(2, D): d[:, i] = -theta_rest_vec[i-2] * x[:, i]
                return d
        def diffusion_func(x): return D_diff
        def potential_func(x):
            x = np.asarray(x)
            if x.ndim == 1:
                U = 0.0
                if D >= 2: r2 = x[0]**2 + x[1]**2; U += -0.5 * lam * r2 + 0.25 * r2**2
                for i in range(2, D): U += 0.5 * theta_rest_vec[i-2] * x[i]**2
                return U
            else:
                U = np.zeros(x.shape[0])
                if D >= 2: r2 = x[:, 0]**2 + x[:, 1]**2; U += -0.5 * lam * r2 + 0.25 * r2**2
                for i in range(2, D): U += 0.5 * theta_rest_vec[i-2] * x[:, i]**2
                return U
    elif name == 'single_well':
        k = model_info['k']
        sigma_mat = model_info['sigma']; D_diff = (sigma_mat @ sigma_mat.T) / 2.0
        def drift_func(x):
            x = np.asarray(x)
            return -k * x if x.ndim == 1 else -k * x
        def diffusion_func(x): return D_diff
        def potential_func(x):
            x = np.asarray(x)
            return 0.5 * k * np.sum(x**2) if x.ndim == 1 else 0.5 * k * np.sum(x**2, axis=1)
    elif name == 'asymmetric_double_well':
        a = model_info['a']; b = model_info['b']; c = model_info['c']; d = model_info['d']
        sigma_mat = model_info['sigma']; D_diff = (sigma_mat @ sigma_mat.T) / 2.0
        def drift_func(x):
            x = np.asarray(x)
            return -(4*a*x**3 - 3*b*x**2 + 2*c*x + d) if x.ndim == 1 else -(4*a*x**3 - 3*b*x**2 + 2*c*x + d)
        def diffusion_func(x): return D_diff
        def potential_func(x):
            x = np.asarray(x)
            return a*x**4 - b*x**3 + c*x**2 + d*x if x.ndim == 1 else a*x**4 - b*x**3 + c*x**2 + d*x
    elif name == 'triple_well_3d':
        a = model_info['a']; b = model_info['b']; c = model_info['c']; k_rest = model_info['k_rest']
        sigma_mat = model_info['sigma']; D_diff = (sigma_mat @ sigma_mat.T) / 2.0
        def drift_func(x):
            x = np.asarray(x)
            if x.ndim == 1:
                d = np.zeros(D)
                for i in range(D):
                    d[i] = 2*a*x[i] - 4*b*x[i]**3
                if D >= 2:
                    for i in range(D):
                        for j in range(i+1, D):
                            if j < 3:
                                d[i] -= 2*c*x[i]*x[j]**2
                                d[j] -= 2*c*x[j]*x[i]**2
                for i in range(2, D):
                    d[i] -= 2*k_rest*x[i]
                return d
            else:
                d = np.zeros_like(x)
                for i in range(D):
                    d[:, i] = 2*a*x[:, i] - 4*b*x[:, i]**3
                if D >= 2:
                    for i in range(D):
                        for j in range(i+1, D):
                            if j < 3:
                                d[:, i] -= 2*c*x[:, i]*x[:, j]**2
                                d[:, j] -= 2*c*x[:, j]*x[:, i]**2
                for i in range(2, D):
                    d[:, i] -= 2*k_rest*x[:, i]
                return d
        def diffusion_func(x): return D_diff
        def potential_func(x):
            x = np.asarray(x)
            if x.ndim == 1:
                U = np.sum(-a*x**2 + b*x**4)
                for i in range(D):
                    for j in range(i+1, D):
                        if j < 3:
                            U += c*x[i]**2*x[j]**2
                for i in range(2, D):
                    U += k_rest*x[i]**2
                return U
            else:
                U = np.sum(-a*x**2 + b*x**4, axis=1)
                for i in range(D):
                    for j in range(i+1, D):
                        if j < 3:
                            U += c*x[:, i]**2*x[:, j]**2
                for i in range(2, D):
                    U += k_rest*x[:, i]**2
                return U
    else:
        raise ValueError(f"Modelo {name} no reconocido para reconstrucción")
    return {
        'drift_func': drift_func,
        'diffusion_func': diffusion_func,
        'potential_func': potential_func,
        'drift_grid': lambda edges: _eval_drift_on_grid(drift_func, edges),
        'diffusion_grid': lambda edges: _eval_diffusion_on_grid(diffusion_func, edges),
        'potential_grid': lambda edges: _eval_potential_on_grid(potential_func, edges),
    }

def _eval_drift_on_grid(drift_func, edges):
    grid_shape = tuple(len(e) for e in edges)
    N = np.prod(grid_shape)
    mesh = np.meshgrid(*edges, indexing='ij')
    points = np.stack([m.ravel() for m in mesh], axis=1)
    try:
        vals = drift_func(points)
    except Exception:
        vals = np.array([drift_func(p) for p in points])
    vals = np.asarray(vals)
    if vals.shape[:1] != (N,):
        vals = np.broadcast_to(vals, (N,) + vals.shape)
    return vals.reshape(grid_shape + (vals.shape[-1],)).transpose(-1, *range(len(grid_shape)))

def _eval_diffusion_on_grid(diffusion_func, edges):
    grid_shape = tuple(len(e) for e in edges)
    N = np.prod(grid_shape)
    mesh = np.meshgrid(*edges, indexing='ij')
    points = np.stack([m.ravel() for m in mesh], axis=1)
    try:
        vals = diffusion_func(points)
    except Exception:
        vals = np.array([diffusion_func(p) for p in points])
    vals = np.asarray(vals)
    if vals.shape[:1] != (N,):
        vals = np.broadcast_to(vals, (N,) + vals.shape)
    D_out = vals.shape[-2]
    return vals.reshape(grid_shape + (D_out, D_out)).transpose(-2, -1, *range(len(grid_shape)))

def _eval_potential_on_grid(potential_func, edges):
    grid_shape = tuple(len(e) for e in edges)
    N = np.prod(grid_shape)
    mesh = np.meshgrid(*edges, indexing='ij')
    points = np.stack([m.ravel() for m in mesh], axis=1)
    try:
        vals = potential_func(points)
    except Exception:
        vals = np.array([potential_func(p) for p in points])
    vals = np.asarray(vals)
    if vals.shape[:1] != (N,):
        return np.broadcast_to(vals, grid_shape)
    return vals.reshape(grid_shape)

# =============================================================================
# SECCIÓN 1: SIMULACIÓN DE DATOS (sin cambios funcionales)
# =============================================================================

def simulate_data(config, cache_dir='sim_cache', use_cache=True):
    """Orquestador de simulación (sin cambios respecto a v3.0)."""
    model = config['model']
    D = config['D']
    dt = config['dt']
    T = config['T']
    burn_ratio = config.get('burn_ratio', 0.1)
    seed = config.get('seed', None)
    params = config.get('params', {})

    if not (1 <= D <= 5):
        raise ValueError(f"D={D} no soportado. Use 1 <= D <= 5.")

    cache_path, param_dict = _build_cache_path(config, cache_dir)
    if use_cache and os.path.exists(cache_path):
        try:
            loaded = np.load(cache_path, allow_pickle=True)
            data = loaded['data']
            model_info = loaded['model_info'].item()
            theoretical = _rebuild_theoretical(model_info, D)
            print(f"[CACHE] Cargando simulación existente: {os.path.basename(cache_path)}")
            return {'data': data, 'theoretical': theoretical, 'config': config, 'model_info': model_info}
        except Exception as e:
            print(f"[CACHE] Archivo corrupto ({e}), regenerando...")

    print(f"[SIM] Generando {model} (D={D})...")
    if model == 'ou':
        data, theoretical, model_info = _simulate_ou(D, dt, T, burn_ratio, seed, params)
    elif model == 'double_well':
        data, theoretical, model_info = _simulate_double_well(D, dt, T, burn_ratio, seed, params)
    elif model == 'single_well':
        data, theoretical, model_info = _simulate_single_well(D, dt, T, burn_ratio, seed, params)
    elif model == 'asymmetric_double_well':
        data, theoretical, model_info = _simulate_asymmetric_double_well(D, dt, T, burn_ratio, seed, params)
    elif model == 'triple_well_3d':
        data, theoretical, model_info = _simulate_triple_well_3d(D, dt, T, burn_ratio, seed, params)
    elif model == 'ring_attractor':
        data, theoretical, model_info = _simulate_ring_attractor(D, dt, T, burn_ratio, seed, params)
    elif model == 'multi_stable':
        data, theoretical, model_info = _simulate_multi_stable(D, dt, T, burn_ratio, seed, params)
    elif model == 'stochastic_oscillator':
        data, theoretical, model_info = _simulate_stochastic_oscillator(D, dt, T, burn_ratio, seed, params)
    else:
        raise ValueError(f"Modelo '{model}' no reconocido.")

    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez_compressed(cache_path, data=data, model_info=model_info)
        print(f"[CACHE] Guardada en: {cache_path}")

    return {'data': data, 'theoretical': theoretical, 'config': config, 'model_info': model_info}

# --- Modelos privados (sin cambios) ---

def _simulate_ou(D, dt, T, burn_ratio, seed, params):
    theta = params.get('theta', 1.0)
    sigma = params.get('sigma', 0.5)
    if seed is not None: np.random.seed(seed)
    n_steps = int(T / dt); burn = int(burn_ratio * n_steps)
    theta_mat = theta * np.eye(D) if np.isscalar(theta) else np.asarray(theta)
    sigma_mat = sigma * np.eye(D) if np.isscalar(sigma) else np.asarray(sigma)
    X = np.zeros((n_steps, D))
    for t in range(1, n_steps):
        dW = np.sqrt(dt) * np.random.randn(D)
        X[t] = X[t-1] - theta_mat @ X[t-1] * dt + sigma_mat @ dW
    data = X[burn:]
    D_diff = (sigma_mat @ sigma_mat.T) / 2.0
    def drift_func(x):
        x = np.asarray(x)
        return -theta_mat @ x if x.ndim == 1 else (-theta_mat @ x.T).T
    def diffusion_func(x): return D_diff
    def potential_func(x):
        x = np.asarray(x)
        D_inv = np.linalg.inv(D_diff + 1e-10 * np.eye(D))
        A_sym = (D_inv @ theta_mat + theta_mat.T @ D_inv.T) / 2
        return 0.5 * x @ A_sym @ x if x.ndim == 1 else 0.5 * np.sum(x * (A_sym @ x.T).T, axis=1)
    theoretical = {
        'drift_func': drift_func, 'diffusion_func': diffusion_func, 'potential_func': potential_func,
        'drift_grid': lambda edges: _eval_drift_on_grid(drift_func, edges),
        'diffusion_grid': lambda edges: _eval_diffusion_on_grid(diffusion_func, edges),
        'potential_grid': lambda edges: _eval_potential_on_grid(potential_func, edges),
    }
    return data, theoretical, {'name': 'ou', 'theta': theta_mat, 'sigma': sigma_mat, 'D_diff': D_diff}

def _simulate_single_well(D, dt, T, burn_ratio, seed, params):
    """
    Potencial armónico simple: U(x) = 0.5 * k * x^2.
    Ideal para validación 1D básica (equilibrio detallado, J=0).
    """
    k = params.get('k', 1.0)
    sigma = params.get('sigma', 0.5)
    sigma_mat = sigma * np.eye(D) if np.isscalar(sigma) else np.asarray(sigma)
    if sigma_mat.shape != (D, D):
        raise ValueError(f"sigma debe ser escalar o matriz {D}x{D}")
    if seed is not None:
        np.random.seed(seed)
    n_steps = int(T / dt)
    burn = int(burn_ratio * n_steps)
    X = np.zeros((n_steps, D))
    for t in range(1, n_steps):
        x = X[t-1]
        drift = -k * x
        dW = np.sqrt(dt) * np.random.randn(D)
        X[t] = x + drift * dt + sigma_mat @ dW
    data = X[burn:]
    D_diff = (sigma_mat @ sigma_mat.T) / 2.0
    def drift_func(x):
        x = np.asarray(x)
        return -k * x if x.ndim == 1 else -k * x
    def diffusion_func(x):
        return D_diff
    def potential_func(x):
        x = np.asarray(x)
        return 0.5 * k * np.sum(x**2) if x.ndim == 1 else 0.5 * k * np.sum(x**2, axis=1)
    theoretical = {
        'drift_func': drift_func, 'diffusion_func': diffusion_func, 'potential_func': potential_func,
        'drift_grid': lambda edges: _eval_drift_on_grid(drift_func, edges),
        'diffusion_grid': lambda edges: _eval_diffusion_on_grid(diffusion_func, edges),
        'potential_grid': lambda edges: _eval_potential_on_grid(potential_func, edges),
    }
    return data, theoretical, {'name': 'single_well', 'k': k, 'sigma': sigma_mat}


def _simulate_asymmetric_double_well(D, dt, T, burn_ratio, seed, params):
    """
    Potencial doble pozo asimétrico: U(x) = a*x^4 - b*x^3 + c*x^2 + d*x.
    Los pozos tienen profundidades diferentes (test de sensibilidad a asimetría).
    """
    a = params.get('a', 1.0)
    b = params.get('b', 0.5)
    c = params.get('c', -1.0)
    d = params.get('d', 0.2)
    sigma = params.get('sigma', 0.5)
    sigma_mat = sigma * np.eye(D) if np.isscalar(sigma) else np.asarray(sigma)
    if sigma_mat.shape != (D, D):
        raise ValueError(f"sigma debe ser escalar o matriz {D}x{D}")
    if seed is not None:
        np.random.seed(seed)
    n_steps = int(T / dt)
    burn = int(burn_ratio * n_steps)
    X = np.zeros((n_steps, D))
    for t in range(1, n_steps):
        x = X[t-1]
        drift = -(4*a*x**3 - 3*b*x**2 + 2*c*x + d)
        dW = np.sqrt(dt) * np.random.randn(D)
        X[t] = x + drift * dt + sigma_mat @ dW
    data = X[burn:]
    D_diff = (sigma_mat @ sigma_mat.T) / 2.0
    def drift_func(x):
        x = np.asarray(x)
        return -(4*a*x**3 - 3*b*x**2 + 2*c*x + d) if x.ndim == 1 else -(4*a*x**3 - 3*b*x**2 + 2*c*x + d)
    def diffusion_func(x):
        return D_diff
    def potential_func(x):
        x = np.asarray(x)
        return a*x**4 - b*x**3 + c*x**2 + d*x if x.ndim == 1 else a*x**4 - b*x**3 + c*x**2 + d*x
    theoretical = {
        'drift_func': drift_func, 'diffusion_func': diffusion_func, 'potential_func': potential_func,
        'drift_grid': lambda edges: _eval_drift_on_grid(drift_func, edges),
        'diffusion_grid': lambda edges: _eval_diffusion_on_grid(diffusion_func, edges),
        'potential_grid': lambda edges: _eval_potential_on_grid(potential_func, edges),
    }
    return data, theoretical, {'name': 'asymmetric_double_well', 'a': a, 'b': b, 'c': c, 'd': d, 'sigma': sigma_mat}


def _simulate_triple_well_3d(D, dt, T, burn_ratio, seed, params):
    """
    Potencial triple pozo en 3D: pozos en (±1,0,0) y (0,±1,0) con acoplamiento.
    Extensible a D>=3 (últimas D-2 dimensiones son armónicas).
    """
    a = params.get('a', 1.0)
    b = params.get('b', 1.0)
    c = params.get('c', 0.3)
    k_rest = params.get('k_rest', 1.0)
    sigma = params.get('sigma', 0.4)
    sigma_mat = sigma * np.eye(D) if np.isscalar(sigma) else np.asarray(sigma)
    if sigma_mat.shape != (D, D):
        raise ValueError(f"sigma debe ser escalar o matriz {D}x{D}")
    if seed is not None:
        np.random.seed(seed)
    n_steps = int(T / dt)
    burn = int(burn_ratio * n_steps)
    X = np.zeros((n_steps, D))
    for t in range(1, n_steps):
        x = X[t-1].copy()
        drift = np.zeros(D)
        for i in range(D):
            drift[i] = 2*a*x[i] - 4*b*x[i]**3
        if D >= 2:
            for i in range(D):
                for j in range(i+1, D):
                    if j < 3:
                        drift[i] -= 2*c*x[i]*x[j]**2
                        drift[j] -= 2*c*x[j]*x[i]**2
        for i in range(2, D):
            drift[i] -= 2*k_rest*x[i]
        dW = np.sqrt(dt) * np.random.randn(D)
        X[t] = x + drift * dt + sigma_mat @ dW
    data = X[burn:]
    D_diff = (sigma_mat @ sigma_mat.T) / 2.0
    def drift_func(x):
        x = np.asarray(x)
        if x.ndim == 1:
            d = np.zeros(D)
            for i in range(D):
                d[i] = 2*a*x[i] - 4*b*x[i]**3
            if D >= 2:
                for i in range(D):
                    for j in range(i+1, D):
                        if j < 3:
                            d[i] -= 2*c*x[i]*x[j]**2
                            d[j] -= 2*c*x[j]*x[i]**2
            for i in range(2, D):
                d[i] -= 2*k_rest*x[i]
            return d
        else:
            d = np.zeros_like(x)
            for i in range(D):
                d[:, i] = 2*a*x[:, i] - 4*b*x[:, i]**3
            if D >= 2:
                for i in range(D):
                    for j in range(i+1, D):
                        if j < 3:
                            d[:, i] -= 2*c*x[:, i]*x[:, j]**2
                            d[:, j] -= 2*c*x[:, j]*x[:, i]**2
            for i in range(2, D):
                d[:, i] -= 2*k_rest*x[:, i]
            return d
    def diffusion_func(x):
        return D_diff
    def potential_func(x):
        x = np.asarray(x)
        if x.ndim == 1:
            U = np.sum(-a*x**2 + b*x**4)
            for i in range(D):
                for j in range(i+1, D):
                    if j < 3:
                        U += c*x[i]**2*x[j]**2
            for i in range(2, D):
                U += k_rest*x[i]**2
            return U
        else:
            U = np.sum(-a*x**2 + b*x**4, axis=1)
            for i in range(D):
                for j in range(i+1, D):
                    if j < 3:
                        U += c*x[:, i]**2*x[:, j]**2
            for i in range(2, D):
                U += k_rest*x[:, i]**2
            return U
    theoretical = {
        'drift_func': drift_func, 'diffusion_func': diffusion_func, 'potential_func': potential_func,
        'drift_grid': lambda edges: _eval_drift_on_grid(drift_func, edges),
        'diffusion_grid': lambda edges: _eval_diffusion_on_grid(diffusion_func, edges),
        'potential_grid': lambda edges: _eval_potential_on_grid(potential_func, edges),
    }
    return data, theoretical, {'name': 'triple_well_3d', 'a': a, 'b': b, 'c': c, 'k_rest': k_rest, 'sigma': sigma_mat}


def _simulate_double_well(D, dt, T, burn_ratio, seed, params):
    a = params.get('a', 1.0); b = params.get('b', 1.0); c = params.get('c', 0.1); sigma = params.get('sigma', 0.5)
    a_vec = np.full(D, a) if np.isscalar(a) else np.asarray(a)
    b_vec = np.full(D, b) if np.isscalar(b) else np.asarray(b)
    sigma_mat = sigma * np.eye(D) if np.isscalar(sigma) else np.asarray(sigma)
    if seed is not None: np.random.seed(seed)
    n_steps = int(T / dt); burn = int(burn_ratio * n_steps)
    X = np.zeros((n_steps, D))
    for t in range(1, n_steps):
        x = X[t-1]
        drift = 2 * a_vec * x - 4 * b_vec * x**3
        if c != 0:
            for i in range(D): drift[i] -= 2 * c * np.sum(x[i] - x)
        dW = np.sqrt(dt) * np.random.randn(D)
        X[t] = x + drift * dt + sigma_mat @ dW
    data = X[burn:]
    D_diff = (sigma_mat @ sigma_mat.T) / 2.0
    def drift_func(x):
        x = np.asarray(x)
        if x.ndim == 1:
            d = 2 * a_vec * x - 4 * b_vec * x**3
            if c != 0:
                for i in range(D): d[i] -= 2 * c * np.sum(x[i] - x)
            return d
        else:
            d = 2 * a_vec * x - 4 * b_vec * x**3
            if c != 0:
                for i in range(D): d[:, i] -= 2 * c * np.sum(x[:, i:i+1] - x, axis=1)
            return d
    def diffusion_func(x): return D_diff
    def potential_func(x):
        x = np.asarray(x)
        if x.ndim == 1:
            U = np.sum(-a_vec * x**2 + b_vec * x**4)
            if c != 0:
                for i in range(D):
                    for j in range(i+1, D): U += c * (x[i] - x[j])**2
            return U
        else:
            U = np.sum(-a_vec * x**2 + b_vec * x**4, axis=1)
            if c != 0:
                for i in range(D):
                    for j in range(i+1, D): U += c * (x[:, i] - x[:, j])**2
            return U
    theoretical = {
        'drift_func': drift_func, 'diffusion_func': diffusion_func, 'potential_func': potential_func,
        'drift_grid': lambda edges: _eval_drift_on_grid(drift_func, edges),
        'diffusion_grid': lambda edges: _eval_diffusion_on_grid(diffusion_func, edges),
        'potential_grid': lambda edges: _eval_potential_on_grid(potential_func, edges),
    }
    return data, theoretical, {'name': 'double_well', 'a': a_vec, 'b': b_vec, 'c': c, 'sigma': sigma_mat}

def _simulate_ring_attractor(D, dt, T, burn_ratio, seed, params):
    alpha = params.get('alpha', 1.0); r0 = params.get('r0', 1.0); omega = params.get('omega', 1.0)
    theta_rest = params.get('theta_rest', 1.0); sigma = params.get('sigma', 0.5)
    sigma_mat = sigma * np.eye(D) if np.isscalar(sigma) else np.asarray(sigma)
    theta_rest_vec = np.full(max(0, D-2), theta_rest) if np.isscalar(theta_rest) else np.asarray(theta_rest)
    if seed is not None: np.random.seed(seed)
    n_steps = int(T / dt); burn = int(burn_ratio * n_steps)
    X = np.zeros((n_steps, D))
    for t in range(1, n_steps):
        x = X[t-1].copy(); drift = np.zeros(D)
        if D >= 2:
            x1, x2 = x[0], x[1]; r = np.sqrt(x1**2 + x2**2) + 1e-12
            drift[0] = -alpha * (r - r0) * (x1 / r) - omega * x2
            drift[1] = -alpha * (r - r0) * (x2 / r) + omega * x1
        for i in range(2, D): drift[i] = -theta_rest_vec[i-2] * x[i]
        dW = np.sqrt(dt) * np.random.randn(D)
        X[t] = x + drift * dt + sigma_mat @ dW
    data = X[burn:]
    D_diff = (sigma_mat @ sigma_mat.T) / 2.0
    def drift_func(x):
        x = np.asarray(x)
        if x.ndim == 1:
            d = np.zeros(D)
            if D >= 2:
                x1, x2 = x[0], x[1]; r = np.sqrt(x1**2 + x2**2) + 1e-12
                d[0] = -alpha * (r - r0) * (x1 / r) - omega * x2
                d[1] = -alpha * (r - r0) * (x2 / r) + omega * x1
            for i in range(2, D): d[i] = -theta_rest_vec[i-2] * x[i]
            return d
        else:
            d = np.zeros_like(x)
            if D >= 2:
                x1, x2 = x[:, 0], x[:, 1]; r = np.sqrt(x1**2 + x2**2) + 1e-12
                d[:, 0] = -alpha * (r - r0) * (x1 / r) - omega * x2
                d[:, 1] = -alpha * (r - r0) * (x2 / r) + omega * x1
            for i in range(2, D): d[:, i] = -theta_rest_vec[i-2] * x[:, i]
            return d
    def diffusion_func(x): return D_diff
    def potential_func(x):
        x = np.asarray(x)
        if x.ndim == 1:
            U = 0.0
            if D >= 2: r = np.sqrt(x[0]**2 + x[1]**2); U += 0.5 * alpha * (r - r0)**2
            for i in range(2, D): U += 0.5 * theta_rest_vec[i-2] * x[i]**2
            return U
        else:
            U = np.zeros(x.shape[0])
            if D >= 2: r = np.sqrt(x[:, 0]**2 + x[:, 1]**2); U += 0.5 * alpha * (r - r0)**2
            for i in range(2, D): U += 0.5 * theta_rest_vec[i-2] * x[:, i]**2
            return U
    theoretical = {
        'drift_func': drift_func, 'diffusion_func': diffusion_func, 'potential_func': potential_func,
        'drift_grid': lambda edges: _eval_drift_on_grid(drift_func, edges),
        'diffusion_grid': lambda edges: _eval_diffusion_on_grid(diffusion_func, edges),
        'potential_grid': lambda edges: _eval_potential_on_grid(potential_func, edges),
    }
    return data, theoretical, {'name': 'ring_attractor', 'alpha': alpha, 'r0': r0, 'omega': omega, 'theta_rest': theta_rest_vec, 'sigma': sigma_mat}

def _simulate_multi_stable(D, dt, T, burn_ratio, seed, params):
    a = params.get('a', 1.0); b = params.get('b', 1.0); c = params.get('c', 0.5); sigma = params.get('sigma', 0.5)
    sigma_mat = sigma * np.eye(D) if np.isscalar(sigma) else np.asarray(sigma)
    if seed is not None: np.random.seed(seed)
    n_steps = int(T / dt); burn = int(burn_ratio * n_steps)
    X = np.zeros((n_steps, D))
    for t in range(1, n_steps):
        x = X[t-1]
        drift = 2 * a * x - 4 * b * x**3
        sum_sq = np.sum(x**2)
        drift -= 2 * c * x * (sum_sq - x**2)
        dW = np.sqrt(dt) * np.random.randn(D)
        X[t] = x + drift * dt + sigma_mat @ dW
    data = X[burn:]
    D_diff = (sigma_mat @ sigma_mat.T) / 2.0
    def drift_func(x):
        x = np.asarray(x)
        if x.ndim == 1:
            d = 2 * a * x - 4 * b * x**3; sum_sq = np.sum(x**2)
            d -= 2 * c * x * (sum_sq - x**2); return d
        else:
            d = 2 * a * x - 4 * b * x**3; sum_sq = np.sum(x**2, axis=1, keepdims=True)
            d -= 2 * c * x * (sum_sq - x**2); return d
    def diffusion_func(x): return D_diff
    def potential_func(x):
        x = np.asarray(x)
        if x.ndim == 1:
            U = np.sum(-a * x**2 + b * x**4)
            for i in range(D):
                for j in range(i+1, D): U += c * x[i]**2 * x[j]**2
            return U
        else:
            U = np.sum(-a * x**2 + b * x**4, axis=1)
            for i in range(D):
                for j in range(i+1, D): U += c * x[:, i]**2 * x[:, j]**2
            return U
    theoretical = {
        'drift_func': drift_func, 'diffusion_func': diffusion_func, 'potential_func': potential_func,
        'drift_grid': lambda edges: _eval_drift_on_grid(drift_func, edges),
        'diffusion_grid': lambda edges: _eval_diffusion_on_grid(diffusion_func, edges),
        'potential_grid': lambda edges: _eval_potential_on_grid(potential_func, edges),
    }
    return data, theoretical, {'name': 'multi_stable', 'a': a, 'b': b, 'c': c, 'sigma': sigma_mat}

def _simulate_stochastic_oscillator(D, dt, T, burn_ratio, seed, params):
    lam = params.get('lambda', 1.0); omega = params.get('omega', 1.0); theta_rest = params.get('theta_rest', 1.0); sigma = params.get('sigma', 0.5)
    sigma_mat = sigma * np.eye(D) if np.isscalar(sigma) else np.asarray(sigma)
    theta_rest_vec = np.full(max(0, D-2), theta_rest) if np.isscalar(theta_rest) else np.asarray(theta_rest)
    if seed is not None: np.random.seed(seed)
    n_steps = int(T / dt); burn = int(burn_ratio * n_steps)
    X = np.zeros((n_steps, D))
    for t in range(1, n_steps):
        x = X[t-1].copy(); drift = np.zeros(D)
        if D >= 2:
            x1, x2 = x[0], x[1]; r2 = x1**2 + x2**2
            drift[0] = lam * x1 - omega * x2 - r2 * x1
            drift[1] = lam * x2 + omega * x1 - r2 * x2
        for i in range(2, D): drift[i] = -theta_rest_vec[i-2] * x[i]
        dW = np.sqrt(dt) * np.random.randn(D)
        X[t] = x + drift * dt + sigma_mat @ dW
    data = X[burn:]
    D_diff = (sigma_mat @ sigma_mat.T) / 2.0
    def drift_func(x):
        x = np.asarray(x)
        if x.ndim == 1:
            d = np.zeros(D)
            if D >= 2:
                x1, x2 = x[0], x[1]; r2 = x1**2 + x2**2
                d[0] = lam * x1 - omega * x2 - r2 * x1
                d[1] = lam * x2 + omega * x1 - r2 * x2
            for i in range(2, D): d[i] = -theta_rest_vec[i-2] * x[i]
            return d
        else:
            d = np.zeros_like(x)
            if D >= 2:
                x1, x2 = x[:, 0], x[:, 1]; r2 = x1**2 + x2**2
                d[:, 0] = lam * x1 - omega * x2 - r2 * x1
                d[:, 1] = lam * x2 + omega * x1 - r2 * x2
            for i in range(2, D): d[:, i] = -theta_rest_vec[i-2] * x[:, i]
            return d
    def diffusion_func(x): return D_diff
    def potential_func(x):
        x = np.asarray(x)
        if x.ndim == 1:
            U = 0.0
            if D >= 2: r2 = x[0]**2 + x[1]**2; U += -0.5 * lam * r2 + 0.25 * r2**2
            for i in range(2, D): U += 0.5 * theta_rest_vec[i-2] * x[i]**2
            return U
        else:
            U = np.zeros(x.shape[0])
            if D >= 2: r2 = x[:, 0]**2 + x[:, 1]**2; U += -0.5 * lam * r2 + 0.25 * r2**2
            for i in range(2, D): U += 0.5 * theta_rest_vec[i-2] * x[:, i]**2
            return U
    theoretical = {
        'drift_func': drift_func, 'diffusion_func': diffusion_func, 'potential_func': potential_func,
        'drift_grid': lambda edges: _eval_drift_on_grid(drift_func, edges),
        'diffusion_grid': lambda edges: _eval_diffusion_on_grid(diffusion_func, edges),
        'potential_grid': lambda edges: _eval_potential_on_grid(potential_func, edges),
    }
    return data, theoretical, {'name': 'stochastic_oscillator', 'lambda': lam, 'omega': omega, 'theta_rest': theta_rest_vec, 'sigma': sigma_mat}

# =============================================================================
# SECCIÓN 2: ESTIMACIÓN KM (sin cambios funcionales)
# =============================================================================

def extract_km_coefficients(data, bins, p=2, bw=None, kernel='epanechnikov', dt=1.0,
                            sigma_smooth=0.0, density_threshold=0.0):
    """
    Estima coeficientes de Kramers-Moyal con post-suavizado opcional
    para eliminar artefactos de borde.
    """
    if isinstance(kernel, str):
        if kernel == 'epanechnikov':
            kernel_func = epanechnikov
        elif kernel == 'gaussian':
            kernel_func = gaussian
        else:
            raise ValueError(f"Kernel {kernel} no reconocido")
    else:
        kernel_func = kernel

    kmc, edges, bw_used, powers = km(
        data, bins=bins, powers=p, kernel=kernel_func,
        bw=bw, center_edges=True, full=True
    )
    kmc = kmc.copy()
    for idx in range(1, len(kmc)):
        kmc[idx] /= dt

    D = data.shape[1]
    sum_powers = np.sum(powers, axis=1)
    first_order_idx = np.where(sum_powers == 1)[0]
    comp_idx = [np.argmax(powers[i]) for i in first_order_idx]
    order = np.argsort(comp_idx)
    first_order_idx = first_order_idx[order]
    drift = np.array([kmc[idx] for idx in first_order_idx])

    second_order_idx = np.where(sum_powers == 2)[0]
    diffusion = np.full((D, D), None, dtype=object)
    for idx in second_order_idx:
        comb = powers[idx]
        nonzero = np.where(comb > 0)[0]
        if len(nonzero) == 1:
            i = nonzero[0]
            diffusion[i, i] = kmc[idx]
        elif len(nonzero) == 2:
            i, j = nonzero[0], nonzero[1]
            diffusion[i, j] = kmc[idx]
            diffusion[j, i] = kmc[idx]

    diff_numeric = np.full((D, D) + drift.shape[1:], np.nan, dtype=float)
    for i in range(D):
        for j in range(D):
            if diffusion[i, j] is not None:
                diff_numeric[i, j] = diffusion[i, j]
    diffusion = diff_numeric

    # ================================================================
    # POST-SUAVIZADO: elimina picos de borde por boundary bias
    # ================================================================
    if sigma_smooth > 0:
        # 'reflect' simula datos simétricos fuera del dominio, evitando
        # el sesgo de "corte" que produce los picos.
        for i in range(D):
            drift[i] = ndi.gaussian_filter(
                np.nan_to_num(drift[i], nan=0.0),
                sigma=sigma_smooth, mode='reflect'
            )
        for i in range(D):
            for j in range(D):
                if not np.all(np.isnan(diffusion[i, j])):
                    diffusion[i, j] = ndi.gaussian_filter(
                        np.nan_to_num(diffusion[i, j], nan=0.0),
                        sigma=sigma_smooth, mode='reflect'
                    )

    # ================================================================
    # MÁSCARA DE DENSIDAD: anular celdas con soporte estadístico bajo
    # ================================================================
    if density_threshold > 0:
        hist_edges = []
        for d in range(D):
            c = edges[d]
            half = (c[1] - c[0]) / 2.0 if len(c) > 1 else 0.5
            e = np.concatenate([[c[0] - half], c + half])
            hist_edges.append(e)

        density, _ = np.histogramdd(data, bins=hist_edges)
        density = density.astype(float)
        mask = density >= density_threshold * density.max()

        # Dilatamos la máscara unas pocas celdas para que el borde "sano"
        # no se contamine al difundirse hacia las celdas anuladas.
        pad = int(np.ceil(sigma_smooth)) + 1
        if D == 1:
            struct = np.ones(3)
            mask = ndi.binary_dilation(mask, structure=struct, iterations=pad)
        elif D >= 2:
            mask = ndi.binary_dilation(mask, iterations=pad)

        for i in range(D):
            drift[i] = np.where(mask, drift[i], np.nan)
        for i in range(D):
            for j in range(D):
                if not np.all(np.isnan(diffusion[i, j])):
                    diffusion[i, j] = np.where(mask, diffusion[i, j], np.nan)

    return drift, diffusion, edges

# =============================================================================
# SECCIÓN 3: RECONSTRUCCIÓN DE POTENCIAL (wrappers a IgA)
# =============================================================================

def reconstruct_potential_1D(drift, diffusion, edges, method='iga',
                             sigma_smooth=1.0, mask_density=None, density=None,
                             **iga_kwargs):
    """
    Reconstruye potencial 1D por componente (cortes del potencial D-dimensional).

    Ahora usa IgA internamente. El parámetro density es obligatorio para method='iga'.
    """
    if density is None:
        raise ValueError("reconstruct_potential_1D requiere 'density' (histograma empírico)")
    if sigma_smooth > 0 and method == 'iga':
        warnings.warn("sigma_smooth se ignora en reconstrucción IgA (el suavizado es global)", UserWarning)

    decompose_helmholtz = iga_kwargs.pop('decompose_helmholtz', False)
    result = reconstruct_potential_iga(
        drift, diffusion, edges, density,
        decompose_helmholtz=decompose_helmholtz,
        **iga_kwargs
    )
    U = result['potential']
    D = drift.shape[0]
    U_list = []
    for i in range(D):
        slice_idx = []
        for d in range(D):
            if d == i:
                slice_idx.append(slice(None))
            else:
                idx0 = np.argmin(np.abs(edges[d]))
                slice_idx.append(idx0)
        U_list.append(U[tuple(slice_idx)])
    return U_list


def reconstruct_potential(drift, diffusion, edges, sigma_smooth=1.0,
                          method='iga', mask_density=None, n_iter=5000,
                          center=True, post_smooth=0.5, density=None,
                          return_full=False, **iga_kwargs):
    """
    Reconstruye el potencial escalar U via Galerkin-B-spline (IgA).

    Parámetros legacy (sigma_smooth, post_smooth, n_iter, center) se mantienen
    para compatibilidad backward pero son ignorados por el método IgA.

    Parameters
    ----------
    drift, diffusion, edges : arrays
    density : ndarray (n1, ..., nD)
        Histograma empírico de densidad. OBLIGATORIO para method='iga'.
    return_full : bool
        Si True, retorna el dict completo del pipeline IgA (incluye eta, rank_D, etc.)
    **iga_kwargs : passed to reconstruct_potential_iga

    Returns
    -------
    U : ndarray o dict
    """
    if method != 'iga':
        warnings.warn(f"method='{method}' no soportado en v4.0. Usando 'iga'.", UserWarning)

    if density is None:
        raise ValueError("reconstruct_potential requiere 'density' (histograma empírico) para method='iga'")

    if sigma_smooth > 0:
        warnings.warn("sigma_smooth se ignora en IgA (suavizado global implícito)", UserWarning)
    if post_smooth > 0:
        warnings.warn("post_smooth se ignora en IgA (resultado ya es C^{p-1})", UserWarning)

    # Extraer decompose_helmholtz de iga_kwargs o usar default True
    decompose_helmholtz = iga_kwargs.pop('decompose_helmholtz', True)
    result = reconstruct_potential_iga(
        drift, diffusion, edges, density,
        decompose_helmholtz=decompose_helmholtz,
        **iga_kwargs
    )

    U = result['potential']
    if center:
        U_valid = U[~np.isnan(U)]
        if len(U_valid) > 0:
            U = U - np.nanmean(U)
            result['potential'] = U

    if return_full:
        return result
    return U


# =============================================================================
# SECCIÓN 4: BW ÓPTIMO (actualizado para IgA)
# =============================================================================

def _worker_bw(args):
    """Worker para optimal_bw usando IgA."""
    data, bins, bw, dt, p, kernel, sigma_smooth, density = args
    drift, diffusion, edges = extract_km_coefficients(
        data, bins=bins, p=p, bw=bw, kernel=kernel, dt=dt
    )
    result = reconstruct_potential_iga(
        drift, diffusion, edges, density=density,
        degree=3, decompose_helmholtz=False
    )
    return result['potential'], edges, bw


def optimal_bw(data, bins, dt=1.0, p=2, kernel='epanechnikov',
               sigma_smooth=1.0, n_candidates=30, n_jobs=1,
               plot=True, figsize=(16, 4.5),
               w_corr=0.35, w_stability=0.35, w_smoothness=0.30):
    """
    Busca el bw óptimo usando un criterio híbrido:
      - Correlación con -log(densidad empírica)  [35%]
      - Estabilidad del potencial ante variaciones de bw  [35%]
      - Suavidad / penalización de rangos extremos  [30%]

    El criterio no requiere ground truth y es robusto ante sobreajuste.

    Parameters
    ----------
    w_corr, w_stability, w_smoothness : float
        Pesos del score híbrido (deben sumar 1.0).
    n_candidates : int
        Número de candidatos (default 30, log-espaciados).
    """
    D = data.shape[1]
    bins_arr = np.atleast_1d(bins)

    # ── Rango de búsqueda mejorado ──────────────────────────────────
    data_range = np.ptp(data, axis=0)
    data_range[data_range == 0] = 1.0
    dx = data_range / bins_arr
    # bw mínimo: kernel debe cubrir al menos ~1.5 celdas en la dim más gruesa
    bw_min = np.max(dx) * 1.5

    std_data = np.std(data, axis=0)
    std_data = std_data[std_data > 0]
    if len(std_data) == 0:
        std_data = np.array([1.0])
    # bw máximo: usar std MÁXIMA (no mínima) para no restringir artificialmente
    bw_max = np.max(std_data) * 0.6
    if bw_max <= bw_min:
        bw_max = bw_min * 5.0

    # Candidatos log-espaciados (mejor cobertura de rangos amplios)
    bw_candidates = np.geomspace(bw_min, bw_max, n_candidates)
    print(f"[optimal_bw] Rango log-espaciado: [{bw_min:.4f}, {bw_max:.4f}] ({n_candidates} candidatos)")

    # ── Histograma base (con primer candidato) ──────────────────────
    drift0, diffusion0, edges = extract_km_coefficients(
        data, bins=bins, p=p, bw=bw_candidates[0], kernel=kernel, dt=dt
    )

    hist_edges = []
    for d in range(D):
        c = edges[d]
        if len(c) > 1:
            half = (c[1] - c[0]) / 2.0
        else:
            half = 0.5
        e = np.concatenate([[c[0] - half], c + half])
        hist_edges.append(e)

    rho, _ = np.histogramdd(data, bins=hist_edges)
    rho = rho.astype(float)
    rho_floor = np.percentile(rho[rho > 0], 1) * 0.1
    U_density = -np.log(rho + rho_floor)
    mask_density = rho > (0.05 * rho.max())
    range_logrho = np.ptp(U_density[mask_density]) if np.any(mask_density) else 1.0

    args_list = [(data, bins, bw, dt, p, kernel, sigma_smooth, rho)
                 for bw in bw_candidates]

    potentials = []
    edges_list = []
    bw_out = []

    if n_jobs != 1 and len(bw_candidates) > 1:
        max_workers = os.cpu_count() if n_jobs == -1 else n_jobs
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_worker_bw, arg): i
                       for i, arg in enumerate(args_list)}
            with tqdm(total=len(bw_candidates), desc="Scanning bw", unit="bw") as pbar:
                for future in as_completed(futures):
                    U, edges_i, bw_i = future.result()
                    potentials.append(U)
                    edges_list.append(edges_i)
                    bw_out.append(bw_i)
                    pbar.update(1)
    else:
        with tqdm(total=len(bw_candidates), desc="Scanning bw", unit="bw") as pbar:
            for arg in args_list:
                U, edges_i, bw_i = _worker_bw(arg)
                potentials.append(U)
                edges_list.append(edges_i)
                bw_out.append(bw_i)
                pbar.update(1)

    results = sorted(zip(bw_out, potentials, edges_list), key=lambda x: x[0])
    bw_candidates = np.array([r[0] for r in results])
    potentials = [r[1] for r in results]
    edges = results[0][2]

    # ── Métricas por candidato ──────────────────────────────────────
    n = len(bw_candidates)
    correlations = np.full(n, np.nan)
    stability = np.full(n, np.nan)
    smoothness = np.full(n, np.nan)
    score = np.full(n, np.nan)

    for i, U in enumerate(potentials):
        if U.shape != U_density.shape:
            continue

        valid = mask_density & ~np.isnan(U)
        n_valid = np.sum(valid)
        if n_valid < 10:
            continue

        # 1. Correlación con -log ρ
        if np.std(U[valid]) > 1e-12 and np.std(U_density[valid]) > 1e-12:
            corr, _ = pearsonr(U[valid], U_density[valid])
            correlations[i] = corr

        # 2. Estabilidad: variación respecto a vecinos
        if 0 < i < n - 1:
            # Comparar con vecino anterior y posterior (ambos deben existir)
            U_prev = potentials[i - 1]
            U_next = potentials[i + 1]
            if (U_prev.shape == U.shape and U_next.shape == U.shape and
                not np.all(np.isnan(U_prev)) and not np.all(np.isnan(U_next))):
                # Diferencias en región válida común
                valid3 = valid & ~np.isnan(U_prev) & ~np.isnan(U_next)
                if np.sum(valid3) > 10:
                    diff_prev = np.mean((U[valid3] - U_prev[valid3])**2)
                    diff_next = np.mean((U[valid3] - U_next[valid3])**2)
                    mean_diff = 0.5 * (diff_prev + diff_next)
                    # Normalizar por varianza de U para hacer adimensional
                    var_U = np.var(U[valid3])
                    if var_U > 1e-12:
                        stability[i] = np.exp(-mean_diff / var_U)
                    else:
                        stability[i] = 1.0
        elif n == 1:
            stability[i] = 1.0

        # 3. Suavidad: penalizar rangos extremos
        range_U = np.ptp(U[valid])
        if range_logrho > 1e-12 and range_U > 1e-12:
            ratio = range_U / range_logrho
            # Ideal: ratio ≈ 1 (misma escala que -log ρ)
            smoothness[i] = 1.0 / (1.0 + np.abs(np.log(ratio + 1e-12)))
        else:
            smoothness[i] = 1.0

    # ─- Score híbrido ──────────────────────────────────────────────
    # Normalizar cada métrica a [0, 1] sobre candidatos válidos
    def _normalize(arr):
        valid = ~np.isnan(arr)
        if not np.any(valid):
            return np.zeros_like(arr)
        a_min, a_max = np.nanmin(arr), np.nanmax(arr)
        if a_max - a_min < 1e-12:
            return np.where(valid, 1.0, np.nan)
        return np.where(valid, (arr - a_min) / (a_max - a_min), np.nan)

    corr_norm = _normalize(correlations)
    stab_norm = _normalize(stability)
    smooth_norm = _normalize(smoothness)

    # Combinar
    for i in range(n):
        if not (np.isnan(corr_norm[i]) or np.isnan(stab_norm[i]) or np.isnan(smooth_norm[i])):
            score[i] = (w_corr * corr_norm[i] +
                        w_stability * stab_norm[i] +
                        w_smoothness * smooth_norm[i])

    if np.any(~np.isnan(score)):
        optimal_idx = int(np.nanargmax(score))
    else:
        optimal_idx = n // 2
        warnings.warn("Ningún candidato produjo score válido. Usando candidato central.")

    # ── Plotting ────────────────────────────────────────────────────
    fig = None
    if plot:
        fig, axes = plt.subplots(1, 4, figsize=figsize)

        ax = axes[0]
        valid = ~np.isnan(correlations)
        ax.plot(bw_candidates[valid], correlations[valid], 'bo-', markersize=5, zorder=3)
        ax.axvline(bw_candidates[optimal_idx], color='r', linestyle='--', linewidth=2,
                   label=f'Óptimo = {bw_candidates[optimal_idx]:.4f}', zorder=4)
        ax.set_xlabel('bw')
        ax.set_ylabel(r'$\rho(U, -\log \rho_{\rm emp})$')
        ax.set_title('Correlación con densidad')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-1.05, 1.05])

        ax = axes[1]
        valid = ~np.isnan(stability)
        ax.plot(bw_candidates[valid], stability[valid], 'gs-', markersize=5, zorder=3)
        ax.axvline(bw_candidates[optimal_idx], color='r', linestyle='--', linewidth=2, zorder=4)
        ax.set_xlabel('bw')
        ax.set_ylabel('Estabilidad')
        ax.set_title('Estabilidad del potencial')
        ax.grid(True, alpha=0.3)

        ax = axes[2]
        valid = ~np.isnan(score)
        ax.plot(bw_candidates[valid], score[valid], 'm^-', markersize=6, zorder=3, linewidth=2)
        ax.axvline(bw_candidates[optimal_idx], color='r', linestyle='--', linewidth=2,
                   label=f'Score máx = {score[optimal_idx]:.3f}', zorder=4)
        ax.set_xlabel('bw')
        ax.set_ylabel('Score híbrido')
        ax.set_title(f'Score = {w_corr:.0%}corr + {w_stability:.0%}stab + {w_smoothness:.0%}smooth')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        ax = axes[3]
        U_opt = potentials[optimal_idx]
        if D == 2:
            X, Y = np.meshgrid(edges[0], edges[1], indexing='ij')
            U_plot = np.nan_to_num(U_opt, nan=np.nanmedian(U_opt))
            levels = np.linspace(U_plot.min(), U_plot.max(), 20)
            cnt = ax.contourf(X, Y, U_plot, levels=levels, cmap='viridis')
            plt.colorbar(cnt, ax=ax, label='U')
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
        elif D == 1:
            ax.plot(edges[0], U_opt, 'b-', label='Potencial óptimo')
            ax.set_xlabel('x')
            ax.set_ylabel('U')
        ax.set_title(f'Potencial óptimo (bw={bw_candidates[optimal_idx]:.4f})')

        fig.tight_layout()
        plt.show()

    return {
        'bw_candidates': bw_candidates,
        'potentials': potentials,
        'edges': edges,
        'correlations': correlations,
        'stability': stability,
        'smoothness': smoothness,
        'score': score,
        'optimal_bw': float(bw_candidates[optimal_idx]),
        'optimal_idx': int(optimal_idx),
        'fig': fig
    }


# =============================================================================
# SECCIÓN 5: PLOTEO (sin cambios funcionales)
# =============================================================================

def plot_km_components(drift, diffusion, edges,
                       drift_components=None,
                       diff_components=None,
                       fixed_coords=None,
                       theoretical=None,
                       figsize=None):
    if fixed_coords is None:
        fixed_coords = {}
    D = drift.shape[0]
    centers = edges
    n_plots = 0
    if drift_components:
        n_plots += len(drift_components)
    if diff_components:
        n_plots += len(diff_components)
    if n_plots == 0:
        raise ValueError("Debe especificarse al menos drift_components o diff_components")
    max_cols = 3
    ncols = min(max_cols, n_plots)
    nrows = (n_plots + ncols - 1) // ncols
    if figsize is None:
        figsize = (ncols * 5, nrows * 4)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    plot_idx = 0

    if drift_components:
        for i in drift_components:
            ax = axes.flat[plot_idx]
            varying_dim = i
            x_vals = centers[varying_dim]
            slice_idx = []
            for d in range(D):
                if d == varying_dim:
                    slice_idx.append(slice(None))
                else:
                    fixed_val = fixed_coords.get(d, 0.0)
                    idx_fixed = np.argmin(np.abs(centers[d] - fixed_val))
                    slice_idx.append(idx_fixed)
            drift_cut = drift[i][tuple(slice_idx)]
            ax.plot(x_vals, drift_cut, 'b-', label='Estimated', linewidth=2)
            if theoretical is not None:
                x_eval = np.zeros((len(x_vals), D))
                x_eval[:, varying_dim] = x_vals
                for d in range(D):
                    if d != varying_dim:
                        x_eval[:, d] = fixed_coords.get(d, 0.0)
                drift_theo = theoretical['drift_func'](x_eval)[:, i]
                ax.plot(x_vals, drift_theo, 'r--', label='Theoretical', linewidth=2)
            ax.set_xlabel(f'$x_{varying_dim+1}$')
            ax.set_ylabel(f'Drift $D^{{(e_{i+1})}}$')
            ax.legend()
            ax.grid(True, alpha=0.3)
            fixed_str = ', '.join([f"$x_{d+1}={fixed_coords.get(d,0):.2f}$"
                                   for d in range(D) if d != varying_dim])
            ax.set_title(f'Drift comp. {i+1}\n({fixed_str})')
            plot_idx += 1

    if diff_components:
        for (i, j) in diff_components:
            ax = axes.flat[plot_idx]
            varying_dim = i
            x_vals = centers[varying_dim]
            slice_idx = []
            for d in range(D):
                if d == varying_dim:
                    slice_idx.append(slice(None))
                else:
                    fixed_val = fixed_coords.get(d, 0.0)
                    idx_fixed = np.argmin(np.abs(centers[d] - fixed_val))
                    slice_idx.append(idx_fixed)
            diff_cut = diffusion[i, j][tuple(slice_idx)]
            ax.plot(x_vals, diff_cut, 'g-', label='Estimated', linewidth=2)
            if theoretical is not None:
                x_eval = np.zeros((len(x_vals), D))
                x_eval[:, varying_dim] = x_vals
                for d in range(D):
                    if d != varying_dim:
                        x_eval[:, d] = fixed_coords.get(d, 0.0)
                diff_theo = theoretical['diffusion_func'](x_eval)
                if diff_theo.ndim == 2:
                    diff_theo_val = diff_theo[i, j]
                else:
                    diff_theo_val = diff_theo[:, i, j]
                ax.axhline(y=diff_theo_val, color='r', linestyle='--',
                           linewidth=2, label=f'Theoretical = {np.mean(diff_theo_val):.3f}')
            ax.set_xlabel(f'$x_{varying_dim+1}$')
            ax.set_ylabel(f'Diffusion $D^{{({i+1},{j+1})}}$')
            ax.legend()
            ax.grid(True, alpha=0.3)
            fixed_str = ', '.join([f"$x_{d+1}={fixed_coords.get(d,0):.2f}$"
                                   for d in range(D) if d != varying_dim])
            ax.set_title(f'Diffusion ({i+1},{j+1})\n({fixed_str})')
            plot_idx += 1

    for idx in range(plot_idx, len(axes.flat)):
        axes.flat[idx].axis('off')
    fig.tight_layout()
    return fig


def plot_potential(U_reconstructed, edges, theoretical=None, component_labels=None,
                   figsize=(10, 6), fixed_coords=None):
    if fixed_coords is None:
        fixed_coords = {}
    D = len(U_reconstructed)
    if component_labels is None:
        component_labels = [f'x{i+1}' for i in range(D)]
    fig, axes = plt.subplots(1, D, figsize=figsize, squeeze=False)
    axes = axes[0]
    for i in range(D):
        ax = axes[i]
        x = edges[i]
        U_i = U_reconstructed[i]
        ax.plot(x, U_i, 'b-', label='Reconstructed', linewidth=2)
        if theoretical is not None:
            x_eval = np.zeros((len(x), D))
            x_eval[:, i] = x
            for d in range(D):
                if d != i:
                    x_eval[:, d] = fixed_coords.get(d, 0.0)
            U_theo = theoretical['potential_func'](x_eval)
            ax.plot(x, U_theo, 'r--', label='Teórico', linewidth=2)
        ax.set_xlabel(component_labels[i])
        ax.set_ylabel('Potential U')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Potential at {component_labels[i]}')
    fig.tight_layout()
    return fig


def _crop_to_valid(U_arr, edge_list):
    """Recorta un grid 2D y sus edges a la bounding box de celdas no-NaN."""
    valid = ~np.isnan(U_arr)
    if not np.any(valid):
        return U_arr, edge_list
    x_mask = np.any(valid, axis=1)
    y_mask = np.any(valid, axis=0)
    x_start = np.argmax(x_mask)
    x_end   = len(x_mask) - np.argmax(x_mask[::-1])
    y_start = np.argmax(y_mask)
    y_end   = len(y_mask) - np.argmax(y_mask[::-1])
    U_crop = U_arr[x_start:x_end, y_start:y_end]
    edges_crop = [edge_list[0][x_start:x_end], edge_list[1][y_start:y_end]]
    return U_crop, edges_crop


# def plot_potential_2d(U, edges, U_theoretical=None, title_est="Reconstructed Potential",
#                       title_theo="Theoretical Potential", levels=50, cmap='viridis',
#                       figsize=(12, 5), unify_colorbar=False, align_minima=False,
#                       align_to_zero=False, crop_to_valid=False):
#     """
#     Plot 2D potential. If crop_to_valid=True, automatically zooms to the
#     region where U is not NaN (useful when density_threshold masks large areas).
#     """
#     if U.ndim != 2:
#         raise ValueError("This function is only for 2D potentials.")

#     U_plot = U.copy()
#     edges_plot = [e.copy() for e in edges]

#     if crop_to_valid:
#         U_plot, edges_plot = _crop_to_valid(U_plot, edges_plot)

#     x_centers, y_centers = edges_plot
#     X, Y = np.meshgrid(x_centers, y_centers, indexing='ij')

#     if U_theoretical is not None:
#         # Resolvemos la función teórica si viene como dict
#         if isinstance(U_theoretical, dict) and 'potential_grid' in U_theoretical:
#             U_theo = U_theoretical['potential_grid'](edges).copy()
#         else:
#             U_theo = U_theoretical.copy()

#         if crop_to_valid:
#             valid = ~np.isnan(U)
#             x_mask = np.any(valid, axis=1)
#             y_mask = np.any(valid, axis=0)
#             x_start = np.argmax(x_mask); x_end = len(x_mask) - np.argmax(x_mask[::-1])
#             y_start = np.argmax(y_mask); y_end = len(y_mask) - np.argmax(y_mask[::-1])
#             U_theo = U_theo[x_start:x_end, y_start:y_end]

#         # --- Alineación de mínimos hacia Z=0 ---
#         if align_to_zero:
#             align_minima = False
#             if np.any(~np.isnan(U_plot)):
#                 U_plot = U_plot - np.nanmin(U_plot)
#             if np.any(~np.isnan(U_theo)):
#                 U_theo = U_theo - np.nanmin(U_theo)

#         # --- Alineación de mínimos (desplazamiento en Z) ---
#         if align_minima:
#             if np.any(~np.isnan(U_plot)) and np.any(~np.isnan(U_theo)):
#                 min_rec  = np.nanmin(U_plot)
#                 min_theo = np.nanmin(U_theo)
#                 offset   = min_theo - min_rec
#                 U_plot   = U_plot + offset

#         # --- Unificación de colorbar ---
#         if unify_colorbar:
#             vmin = min(np.nanmin(U_plot), np.nanmin(U_theo))
#             vmax = max(np.nanmax(U_plot), np.nanmax(U_theo))
#             contour_levels = np.linspace(vmin, vmax, levels)
#         else:
#             vmin = None
#             vmax = None
#             contour_levels = levels

#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

#         contour1 = ax1.contourf(X, Y, U_plot, levels=contour_levels, cmap=cmap, vmin=vmin, vmax=vmax)
#         plt.colorbar(contour1, ax=ax1, label='Potential U')
#         ax1.set_xlabel('x')
#         ax1.set_ylabel('y')
#         ax1.set_title(title_est)
#         ax1.axis('equal')

#         contour2 = ax2.contourf(X, Y, U_theo, levels=contour_levels, cmap=cmap, vmin=vmin, vmax=vmax)
#         plt.colorbar(contour2, ax=ax2, label='Potential U')
#         ax2.set_xlabel('x')
#         ax2.set_ylabel('y')
#         ax2.set_title(title_theo)
#         ax2.axis('equal')

#     else:
#         fig, ax = plt.subplots(1, 1, figsize=figsize)
#         contour = ax.contourf(X, Y, U_plot, levels=levels, cmap=cmap)
#         plt.colorbar(contour, ax=ax, label='Potential U')
#         ax.set_xlabel('x')
#         ax.set_ylabel('y')
#         ax.set_title(title_est)
#         ax.axis('equal')

#     fig.tight_layout()
#     return fig

def plot_potential_2d(U, edges, U_theoretical=None, title_est="Reconstructed Potential",
                      title_theo="Theoretical Potential", levels=50, cmap='viridis',
                      figsize=(12, 5), unify_colorbar=False, align_minima=False,
                      align_to_zero=False, crop_to_valid=False, clip_percentile=None):
    """
    Plot 2D potential. If crop_to_valid=True, automatically zooms to the
    region where U is not NaN (useful when density_threshold masks large areas).
    """
    if U.ndim != 2:
        raise ValueError("This function is only for 2D potentials.")

    U_plot = U.copy()
    edges_plot = [e.copy() for e in edges]

    if crop_to_valid:
        U_plot, edges_plot = _crop_to_valid(U_plot, edges_plot)

    x_centers, y_centers = edges_plot
    X, Y = np.meshgrid(x_centers, y_centers, indexing='ij')

    if U_theoretical is not None:
        # Resolvemos la función teórica si viene como dict
        if isinstance(U_theoretical, dict) and 'potential_grid' in U_theoretical:
            U_theo = U_theoretical['potential_grid'](edges).copy()
        else:
            U_theo = U_theoretical.copy()

        if crop_to_valid:
            valid = ~np.isnan(U)
            x_mask = np.any(valid, axis=1)
            y_mask = np.any(valid, axis=0)
            x_start = np.argmax(x_mask); x_end = len(x_mask) - np.argmax(x_mask[::-1])
            y_start = np.argmax(y_mask); y_end = len(y_mask) - np.argmax(y_mask[::-1])
            U_theo = U_theo[x_start:x_end, y_start:y_end]

        # --- Alineación de mínimos hacia Z=0 ---
        if align_to_zero:
            align_minima = False
            if np.any(~np.isnan(U_plot)):
                U_plot = U_plot - np.nanmin(U_plot)
            if np.any(~np.isnan(U_theo)):
                U_theo = U_theo - np.nanmin(U_theo)

        # --- Alineación de mínimos (desplazamiento en Z) ---
        if align_minima:
            if np.any(~np.isnan(U_plot)) and np.any(~np.isnan(U_theo)):
                min_rec  = np.nanmin(U_plot)
                min_theo = np.nanmin(U_theo)
                offset   = min_theo - min_rec
                U_plot   = U_plot + offset

        # --- Determinación de vmin/vmax para cada plot ---
        if clip_percentile is not None:
            if unify_colorbar:
                # Unificar rango recortado sobre ambos conjuntos
                valid_rec  = U_plot[~np.isnan(U_plot)]
                valid_theo = U_theo[~np.isnan(U_theo)]
                all_valid  = np.concatenate([valid_rec, valid_theo])
                if len(all_valid) > 0:
                    vmin = np.percentile(all_valid, 100 - clip_percentile)
                    vmax = np.percentile(all_valid, clip_percentile)
                else:
                    vmin = None; vmax = None
                vmin1 = vmin2 = vmin
                vmax1 = vmax2 = vmax
            else:
                # Cada plot con su propio recorte independiente
                valid_rec = U_plot[~np.isnan(U_plot)]
                if len(valid_rec) > 0:
                    vmin1 = np.percentile(valid_rec, 100 - clip_percentile)
                    vmax1 = np.percentile(valid_rec, clip_percentile)
                else:
                    vmin1 = None; vmax1 = None

                valid_theo = U_theo[~np.isnan(U_theo)]
                if len(valid_theo) > 0:
                    vmin2 = np.percentile(valid_theo, 100 - clip_percentile)
                    vmax2 = np.percentile(valid_theo, clip_percentile)
                else:
                    vmin2 = None; vmax2 = None
        elif unify_colorbar:
            vmin = min(np.nanmin(U_plot), np.nanmin(U_theo))
            vmax = max(np.nanmax(U_plot), np.nanmax(U_theo))
            vmin1 = vmin2 = vmin
            vmax1 = vmax2 = vmax
        else:
            vmin1 = vmax1 = None
            vmin2 = vmax2 = None

        # --- Construcción de levels ---
        if vmin1 is not None and vmax1 is not None:
            levels1 = np.linspace(vmin1, vmax1, levels)
        else:
            levels1 = levels
        if vmin2 is not None and vmax2 is not None:
            levels2 = np.linspace(vmin2, vmax2, levels)
        else:
            levels2 = levels

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        contour1 = ax1.contourf(X, Y, U_plot, levels=levels1, cmap=cmap, vmin=vmin1, vmax=vmax1)
        plt.colorbar(contour1, ax=ax1, label='Potential U')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title(title_est)
        ax1.axis('equal')

        contour2 = ax2.contourf(X, Y, U_theo, levels=levels2, cmap=cmap, vmin=vmin2, vmax=vmax2)
        plt.colorbar(contour2, ax=ax2, label='Potential U')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_title(title_theo)
        ax2.axis('equal')

    else:
        # --- Caso sin teórico: clipping independiente ---
        if clip_percentile is not None:
            valid_data = U_plot[~np.isnan(U_plot)]
            if len(valid_data) > 0:
                vmin = np.percentile(valid_data, 100 - clip_percentile)
                vmax = np.percentile(valid_data, clip_percentile)
                contour_levels = np.linspace(vmin, vmax, levels)
            else:
                vmin = None; vmax = None; contour_levels = levels
        else:
            vmin = None; vmax = None; contour_levels = levels

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        contour = ax.contourf(X, Y, U_plot, levels=contour_levels, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(contour, ax=ax, label='Potential U')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(title_est)
        ax.axis('equal')

    fig.tight_layout()
    return fig


def plot_potential_slice(U, edges, dims=(0, 1), fixed_coords=None,
                         U_theoretical=None, title="Potencial (corte 2D)",
                         levels=50, cmap='viridis', figsize=(10, 4.5),
                         unify_colorbar=False, align_minima=False,
                         align_to_zero=False, crop_to_valid=False):
    """
    Visualiza un corte 2D de un potencial D-dimensional.
    Si crop_to_valid=True, recorta automáticamente a la región con datos válidos.
    """
    D = len(edges)
    i, j = dims
    if i == j or i >= D or j >= D:
        raise ValueError(f"dims={dims} inválido para D={D}")

    if fixed_coords is None:
        fixed_coords = {}

    # --- 1. Extraer corte 2D del potencial reconstruido ---
    slice_idx = []
    for d in range(D):
        if d == i or d == j:
            slice_idx.append(slice(None))
        else:
            fixed_val = fixed_coords.get(d, 0.0)
            idx = np.argmin(np.abs(edges[d] - fixed_val))
            slice_idx.append(idx)

    U_slice = U[tuple(slice_idx)]

    # --- 2. Recortar a región válida (opcional) ---
    x = edges[i].copy()
    y = edges[j].copy()
    if crop_to_valid:
        U_slice, x, y = _crop_to_valid(U_slice, [x, y])
    X, Y = np.meshgrid(x, y, indexing='ij')

    # --- 3. Extraer corte 2D del teórico (si existe) ---
    U_theo_slice = None
    if U_theoretical is not None:
        if isinstance(U_theoretical, dict) and 'potential_grid' in U_theoretical:
            U_theo_full = U_theoretical['potential_grid'](edges)
            U_theo_slice = U_theo_full[tuple(slice_idx)]
        elif isinstance(U_theoretical, np.ndarray):
            U_theo_slice = U_theoretical[tuple(slice_idx)]
        elif callable(U_theoretical):
            U_theo_full = U_theoretical(edges)
            if isinstance(U_theo_full, np.ndarray):
                U_theo_slice = U_theo_full[tuple(slice_idx)]
        else:
            warnings.warn("U_theoretical no es ndarray, dict ni callable. Ignorando.", UserWarning)

        if crop_to_valid and U_theo_slice is not None:
            valid = ~np.isnan(U_slice)
            x_mask = np.any(valid, axis=1)
            y_mask = np.any(valid, axis=0)
            x_start = np.argmax(x_mask); x_end = len(x_mask) - np.argmax(x_mask[::-1])
            y_start = np.argmax(y_mask); y_end = len(y_mask) - np.argmax(y_mask[::-1])
            U_theo_slice = U_theo_slice[x_start:x_end, y_start:y_end]

    # --- 4. Alineación de potenciales ---
    U_plot = U_slice.copy()
    if U_theo_slice is not None:
        U_theo_plot = U_theo_slice.copy()

    if align_to_zero:
        align_minima = False
        if np.any(~np.isnan(U_plot)):
            U_plot = U_plot - np.nanmin(U_plot)
        if U_theo_slice is not None and np.any(~np.isnan(U_theo_plot)):
            U_theo_plot = U_theo_plot - np.nanmin(U_theo_plot)

    if align_minima and U_theo_slice is not None:
        if np.any(~np.isnan(U_plot)) and np.any(~np.isnan(U_theo_plot)):
            min_rec = np.nanmin(U_plot)
            min_theo = np.nanmin(U_theo_plot)
            offset = min_theo - min_rec
            U_plot = U_plot + offset

    # --- 5. Unificación de colorbar ---
    if unify_colorbar and U_theo_slice is not None:
        vmin = min(np.nanmin(U_plot), np.nanmin(U_theo_plot))
        vmax = max(np.nanmax(U_plot), np.nanmax(U_theo_plot))
        contour_levels = np.linspace(vmin, vmax, levels)
    else:
        vmin = None
        vmax = None
        contour_levels = levels

    # --- 6. Plotting ---
    if U_theo_slice is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        cnt1 = ax1.contourf(X, Y, U_plot, levels=contour_levels, cmap=cmap,
                            vmin=vmin, vmax=vmax)
        plt.colorbar(cnt1, ax=ax1, label='U')
        ax1.set_aspect('equal', adjustable='box')
        ax1.set_xlabel(f'x{i+1}')
        ax1.set_ylabel(f'x{j+1}')
        ax1.set_title(title)

        cnt2 = ax2.contourf(X, Y, U_theo_plot, levels=contour_levels, cmap=cmap,
                            vmin=vmin, vmax=vmax)
        plt.colorbar(cnt2, ax=ax2, label='U')
        ax2.set_aspect('equal', adjustable='box')
        ax2.set_xlabel(f'x{i+1}')
        ax2.set_ylabel(f'x{j+1}')
        ax2.set_title("Theoretical")

    else:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        cnt = ax.contourf(X, Y, U_plot, levels=contour_levels, cmap=cmap,
                          vmin=vmin, vmax=vmax)
        plt.colorbar(cnt, ax=ax, label='U')
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel(f'x{i+1}')
        ax.set_ylabel(f'x{j+1}')
        ax.set_title(title)

    fig.tight_layout()
    return fig

# def plot_potential_slice(U, edges, dims=(0, 1), fixed_coords=None,
#                          U_theoretical=None, title="Potencial (corte 2D)",
#                          levels=50, cmap='viridis', figsize=(10, 4.5)):
#     """
#     Visualiza un corte 2D de un potencial D-dimensional fijando las otras
#     dimensiones en valores constantes (default: 0 o coordenadas especificadas).

#     Parameters
#     ----------
#     U : ndarray, shape (n1, ..., nD)
#         Potencial D-dimensional.
#     edges : list of D arrays
#         Centros de bins por dimensión.
#     dims : tuple (i, j)
#         Índices de las dos dimensiones a plotear.
#     fixed_coords : dict, optional
#         {dim_index: valor} para dimensiones fijas. Si None, usa 0.
#     U_theoretical : ndarray o callable, optional
#         Potencial teórico del mismo shape, o callable(edges) -> grid.
#     """
#     D = len(edges)
#     i, j = dims
#     if i == j or i >= D or j >= D:
#         raise ValueError(f"dims={dims} inválido para D={D}")

#     if fixed_coords is None:
#         fixed_coords = {}

#     # Construir índices de slice para extraer el corte 2D
#     slice_idx = []
#     for d in range(D):
#         if d == i:
#             slice_idx.append(slice(None))
#         elif d == j:
#             slice_idx.append(slice(None))
#         else:
#             fixed_val = fixed_coords.get(d, 0.0)
#             idx = np.argmin(np.abs(edges[d] - fixed_val))
#             slice_idx.append(idx)

#     U_slice = U[tuple(slice_idx)]

#     x = edges[i]
#     y = edges[j]
#     X, Y = np.meshgrid(x, y, indexing='ij')

#     if U_theoretical is not None:
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
#         for ax, U_plot, ttl in [(ax1, U_slice, title),
#                                  (ax2, U_theoretical[tuple(slice_idx)] if isinstance(U_theoretical, np.ndarray) else U_theoretical, "Teórico")]:
#             if isinstance(U_plot, np.ndarray):
#                 #U_plot = np.nan_to_num(U_plot, nan=np.nanmedian(U_plot))
#                 cnt = ax.contourf(X, Y, U_plot, levels=levels, cmap=cmap)
#                 plt.colorbar(cnt, ax=ax, label='U')
#             ax.set_aspect('equal', adjustable='box')
#             ax.set_xlabel(f'x{i+1}')
#             ax.set_ylabel(f'x{j+1}')
#             ax.set_title(ttl)
#     else:
#         fig, ax = plt.subplots(1, 1, figsize=figsize)
#         #U_plot = np.nan_to_num(U_slice, nan=np.nanmedian(U_slice))
#         cnt = ax.contourf(X, Y, U_slice, levels=levels, cmap=cmap)
#         plt.colorbar(cnt, ax=ax, label='U')
#         ax.set_aspect('equal', adjustable='box')
#         ax.set_xlabel(f'x{i+1}')
#         ax.set_ylabel(f'x{j+1}')
#         ax.set_title(title)

#     fig.tight_layout()
#     plt.show()
#     return fig
