"""
extract_mode_indices.py
======================
Lee los archivos de salida de CD-HSA (hankel_info.json, config.json,
cdhsa_arrays.npz) y genera un JSON que mapea cada par
(super-sujeto, condicion) a los indices de los modos especificos
mas relevantes de esa condicion.

Uso::

    python extract_mode_indices.py --results-dir results/cdhsa/session1/nSS5_.../.../eyesclosed_music
    python extract_mode_indices.py --results-dir <PATH> --top-n 2

El JSON de salida (mode_map.json) contiene, para cada condicion,
los indices de los modos especificos ordenados por eigenvalor,
y para cada (super-sujeto, condicion) la clave de la Hankel
asociada y la formula de proyeccion.

Contexto matematico
--------------------
Step D de CD-HSA calcula, para cada condicion c:

  1. Residuos: R(s,c) = (I - U0 U0^T) H(s,c)  para cada SS s
  2. Covarianza residual pool: R_bar(c) = (1/S) sum_s R(s,c) R(s,c)^T
  3. Descomposicion: R_bar(c) = Phi(c) Lambda(c) Phi(c)^T
  4. Los modos especificos son las columnas de Phi(c) en R^p.

Los modos son COMUNES a todos los super-sujetos dentro de una
condicion (se calculan haciendo pooling sobre S super-sujetos).

Para obtener las series temporales de dimension r_c de un par (s,c)::

    alpha = Phi(c)^T @ H(s,c)   ->  shape (r_c, T)

Las series se ordenan por eigenvalor (Lambda(c)) de mayor a menor.
Tomar los top-n da las n series que mas capturan la dinamica
especifica de la condicion c.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _find_d_keys(npz_data: np.NpzFile) -> dict[str, list[tuple[str, tuple]]]:
    """Clasificar todas las keys D__ del NPZ por su estructura.

    Devuelve un dict como::
        {
            'scalar_keys': [('D__C', ()), ...],
            'array_keys': [('D__r_specific', (5,)), ...],
            'nested_dicts': {
                'D__Phi_specific': ['0', '1', ...],  # sub-keys
                'D__Lambda_specific': ['0', '1', ...],
            }
        }
    """
    all_keys = [k for k in npz_data.files if k.startswith('D__')]

    scalar_keys = []
    array_keys = []
    nested_prefixes: dict[str, list[str]] = {}

    for k in all_keys:
        parts = k.split('__')
        # D__C, D__r_specific -> len(parts) == 2
        # D__Phi_specific__0 -> len(parts) == 3
        # D__Phi_specific__0__extra -> len(parts) == 4

        if len(parts) == 2:
            # Top-level: D__key
            shape = npz_data[k].shape
            if shape == () or (len(shape) == 1 and shape[0] == 1):
                scalar_keys.append((k, shape))
            else:
                array_keys.append((k, shape))
        elif len(parts) >= 3:
            # Nested: D__prefix__subkey
            prefix = '__'.join(parts[:2])  # e.g. 'D__Phi_specific'
            subkey = '__'.join(parts[2:])   # e.g. '0' or '0__extra'
            if prefix not in nested_prefixes:
                nested_prefixes[prefix] = []
            nested_prefixes[prefix].append(subkey)

    return {
        'scalar_keys': scalar_keys,
        'array_keys': array_keys,
        'nested_dicts': nested_prefixes,
    }


def _identify_mode_arrays(
    d_info: dict,
    n_conditions: int,
) -> dict[int, str]:
    """Identificar cual key anidada contiene los modos Phi por condicion.

    Busca keys anidadas donde los sub-keys sean strings numericos 0..C-1
    y cada array tenga shape (p, r_c) con p grande y r_c variable.

    Returns dict {condition_idx: npz_key}  o dict vacio si no se encuentran.
    """
    nested = d_info['nested_dicts']
    candidates = []

    for prefix, subkeys in nested.items():
        # Verificar que los sub-keys son numericos consecutivos
        try:
            indices = sorted(int(sk) for sk in subkeys)
        except ValueError:
            continue

        if indices != list(range(len(indices))):
            continue

        if len(indices) != n_conditions:
            continue

        # Verificar shapes: (p, r_c) con p constante y r_c variable
        # (o al menos que p sea grande > 10)
        shapes_ok = True
        p_vals = []
        for idx in indices:
            key = f'{prefix}__{idx}'
            # La shape se verificara al cargar, por ahora aceptamos
            p_vals.append(idx)

        candidates.append(prefix)

    return candidates


def _identify_eigenvalue_arrays(
    d_info: dict,
    n_conditions: int,
) -> list[str]:
    """Identificar keys anidadas que contienen eigenvalores Lambda por condicion.

    Similar a _identify_mode_arrays pero buscando shapes (r_c,)  (1D).
    """
    nested = d_info['nested_dicts']
    candidates = []

    for prefix, subkeys in nested.items():
        try:
            indices = sorted(int(sk) for sk in subkeys)
        except ValueError:
            continue

        if indices != list(range(len(indices))):
            continue
        if len(indices) != n_conditions:
            continue

        candidates.append(prefix)

    return candidates


def _json_safe(obj):
    """Convertir tipos numpy a tipos nativos de Python para JSON."""
    if isinstance(obj, (np.integer, )):
        return int(obj)
    if isinstance(obj, (np.floating, )):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def build_mode_map(
    results_dir: str | Path,
    top_n: int = 2,
) -> dict:
    """Construir el modo de mapeo de los resultados de CD-HSA.

    Parameters
    ----------
    results_dir : path
        Directorio que contiene hankel_info.json, config.json,
        hankel_matrices.npz y cdhsa_arrays.npz.
    top_n : int
        Cuantos modos especificos conservar por condicion (ordenados
        por eigenvalor descendente).

    Returns
    -------
    mode_map : dict
        Estructura JSON completa (ver docstring del modulo).
    """
    results_dir = Path(results_dir)

    # ------------------------------------------------------------------
    # 1. Cargar metadata
    # ------------------------------------------------------------------
    with open(results_dir / 'hankel_info.json') as f:
        hankel_info = json.load(f)

    with open(results_dir / 'config.json') as f:
        config = json.load(f)

    tasks = hankel_info.get('tasks', [])
    S = hankel_info.get('n_super_subjects',
                        hankel_info.get('S', len(tasks) and 1))
    C = len(tasks)
    p = hankel_info.get('n_channels_common',
                        hankel_info.get('n_global_channels', None))
    depth = hankel_info.get('depth_common',
                            hankel_info.get('hankel_depth_requested', None))
    if p is not None and depth is not None:
        p_hankel = p * depth
    else:
        p_hankel = None

    # ------------------------------------------------------------------
    # 2. Cargar cdhsa_arrays.npz
    # ------------------------------------------------------------------
    npz_path = results_dir / 'cdhsa_arrays.npz'
    npz_data = np.load(npz_path, allow_pickle=True)

    # ------------------------------------------------------------------
    # 3. Extraer informacion de Step D
    # ------------------------------------------------------------------
    d_info = _find_d_keys(npz_data)

    # Numero de condiciones desde el NPZ (D__C)
    n_conditions_npz = None
    for k, _ in d_info['scalar_keys']:
        if k == 'D__C':
            n_conditions_npz = int(npz_data[k].flat[0])
            break
    if n_conditions_npz is None:
        for k, shape in d_info['array_keys']:
            if 'r_specific' in k:
                n_conditions_npz = len(npz_data[k])
                break

    if n_conditions_npz is None:
        raise ValueError(
            'No se pudo determinar el numero de condiciones desde '
            'cdhsa_arrays.npz. Verifique que Step D se ejecuto correctamente.'
        )

    assert n_conditions_npz == C, (
        f'Inconsistencia: hankel_info dice C={C} pero '
        f'cdhsa_arrays.npz dice C={n_conditions_npz}'
    )

    # r_specific: cuantos modos se encontraron por condicion
    r_specific = None
    for k, shape in d_info['array_keys']:
        if 'r_specific' in k:
            r_specific = npz_data[k]
            break

    # prevalence_contrast
    prev_contrast = None
    for k, shape in d_info['array_keys']:
        if 'prevalence_contrast' in k:
            prev_contrast = npz_data[k]
            break

    # ------------------------------------------------------------------
    # 4. Identificar arrays de modos (Phi) y eigenvalores (Lambda)
    # ------------------------------------------------------------------
    mode_candidates = _identify_mode_arrays(d_info, C)
    eigen_candidates = _identify_eigenvalue_arrays(d_info, C)

    # Separar: los que tienen shape 2D (p, r_c) son Phi,
    # los que tienen shape 1D (r_c,) son Lambda
    phi_key = None
    lambda_key = None

    for prefix in mode_candidates:
        sample_key = f'{prefix}__0'
        shape = npz_data[sample_key].shape
        if len(shape) == 2 and shape[0] > shape[1]:
            phi_key = prefix
            break
        elif len(shape) == 1:
            if lambda_key is None:
                lambda_key = prefix

    for prefix in eigen_candidates:
        if prefix == phi_key:
            continue
        sample_key = f'{prefix}__0'
        shape = npz_data[sample_key].shape
        if len(shape) == 1:
            lambda_key = prefix
            break

    if phi_key is None:
        # Fallback: buscar cualquier key anidada con sub-keys numericos
        # y shape 2D
        for prefix, subkeys in d_info['nested_dicts'].items():
            try:
                indices = [int(sk) for sk in subkeys]
            except ValueError:
                continue
            sample_key = f'{prefix}__{indices[0]}'
            shape = npz_data[sample_key].shape
            if len(shape) == 2:
                phi_key = prefix
                break

    if phi_key is None:
        raise ValueError(
            'No se encontraron arrays de modos especificos (Phi) en '
            'cdhsa_arrays.npz. Posibles causas: Step D no se ejecuto, '
            'o la estructura de keys no coincide con lo esperado.\n'
            f'Keys D__ encontradas: {[k for k in npz_data.files if k.startswith("D__")]}'
        )

    # ------------------------------------------------------------------
    # 5. Para cada condicion: obtener modos, eigenvalores, y ordenar
    # ------------------------------------------------------------------
    conditions_data = {}

    for c_idx in range(C):
        task_name = tasks[c_idx]

        # Cargar Phi(c) y Lambda(c) si existen
        phi_npz_key = f'{phi_key}__{c_idx}'
        Phi_c = npz_data[phi_npz_key]
        r_c = Phi_c.shape[1] if len(Phi_c.shape) == 2 else Phi_c.shape[0]

        # Eigenvalores
        eigenvals = None
        if lambda_key is not None:
            lam_npz_key = f'{lambda_key}__{c_idx}'
            if lam_npz_key in npz_data:
                eigenvals = npz_data[lam_npz_key]

        # Si hay eigenvalores, ordenar por ellos (descendente)
        if eigenvals is not None and len(eigenvals) == r_c:
            order = np.argsort(eigenvals)[::-1]  # descendente
        else:
            # Sin eigenvalores, asumir que ya vienen ordenados
            order = np.arange(r_c)

        # Top-N modos
        n_actual = min(top_n, r_c)
        top_indices = [int(order[i]) for i in range(n_actual)]

        # Construir info
        cond_info = {
            'task': task_name,
            'condition_index': c_idx,
            'total_specific_modes': int(r_c),
            'top_n_requested': top_n,
            'top_n_actual': n_actual,
            'mode_indices_in_Phi': top_indices,
            'mode_npz_keys': [f'{phi_key}__{c_idx}'],
            'Phi_shape': list(Phi_c.shape),
            'eigenvalues_npz_key': (f'{lambda_key}__{c_idx}'
                                    if lambda_key is not None else None),
            'eigenvalues': _json_safe(eigenvals) if eigenvals is not None else None,
            'prevalence_contrast': (float(prev_contrast[c_idx])
                                    if prev_contrast is not None else None),
            'r_specific_from_pipeline': (int(r_specific[c_idx])
                                          if r_specific is not None else None),
        }

        # Para cada modo en top_indices, guardar la info
        cond_info['modes'] = []
        for rank_pos, mode_idx in enumerate(top_indices):
            mode_info = {
                'rank_position': rank_pos + 1,  # 1 = mas importante
                'column_index_in_Phi': mode_idx,
                'eigenvalue': (float(eigenvals[mode_idx])
                               if eigenvals is not None else None),
            }
            cond_info['modes'].append(mode_info)

        conditions_data[task_name] = cond_info

    # ------------------------------------------------------------------
    # 6. Para cada (super-sujeto, condicion): mapeo a Hankel y proyeccion
    # ------------------------------------------------------------------
    per_pair = []

    for s_idx in range(S):
        ss_label = f'SS{s_idx + 1}'
        for c_idx in range(C):
            task_name = tasks[c_idx]
            hankel_key = f'H_ss{s_idx + 1}_c{c_idx + 1}'

            cond = conditions_data[task_name]
            mode_indices = cond['mode_indices_in_Phi']
            phi_key_for_cond = cond['mode_npz_keys'][0]

            entry = {
                'super_subject': s_idx + 1,
                'super_subject_label': ss_label,
                'task': task_name,
                'task_index': c_idx,
                'hankel_npz_key': hankel_key,
                'hankel_npz_file': 'hankel_matrices.npz',
                'specific_modes_npz_key': phi_key_for_cond,
                'specific_modes_npz_file': 'cdhsa_arrays.npz',
                'columns_to_extract': mode_indices,
                'n_output_dimensions': cond['top_n_actual'],
                'projection_formula': (
                    f'alpha = {phi_key_for_cond}[:, {mode_indices}].T @ {hankel_key}\n'
                    f'  -> alpha shape: ({cond["top_n_actual"]}, T)'
                ),
            }
            per_pair.append(entry)

    # ------------------------------------------------------------------
    # 7. Info del espacio comun (A6)
    # ------------------------------------------------------------------
    common_info = {}
    a6_keys = [k for k in npz_data.files if k.startswith('A6__')]
    for k in a6_keys:
        short_name = k.replace('A6__', '')
        arr = npz_data[k]
        common_info[short_name] = _json_safe(arr)

    # ------------------------------------------------------------------
    # 8. Ensamblar JSON final
    # ------------------------------------------------------------------
    mode_map = {
        'metadata': {
            'source_dir': str(results_dir),
            'n_super_subjects': S,
            'n_conditions': C,
            'tasks': tasks,
            'p_hankel': p_hankel,
            'n_channels': p,
            'hankel_depth': depth,
            'top_n_requested': top_n,
            'pipeline_config': {
                'fixed_rank': config.get('fixed_rank'),
                'L': config.get('L'),
                'd_max_specific': config.get('d_max_specific', 10),
                'a6_n_null': config.get('a6_n_null'),
                'bc_n_perm': config.get('bc_n_perm'),
            },
        },
        'common_subspace': common_info,
        'conditions': conditions_data,
        'per_super_subject_task': per_pair,
        'npz_structure': {
            'd_keys_found': [k for k in npz_data.files if k.startswith('D__')],
            'd_scalar_keys': [k for k, _ in d_info['scalar_keys']],
            'd_array_keys': [(k, list(sh)) for k, sh in d_info['array_keys']],
            'd_nested_prefixes': d_info['nested_dicts'],
            'phi_key_identified': phi_key,
            'lambda_key_identified': lambda_key,
        },
        'usage_instructions': {
            'description': (
                'Para obtener las series temporales de dimension top_n '
                'del par (super_subject, task), proyectar la Hankel '
                'sobre los modos especificos indicados en columns_to_extract.'
            ),
            'python_example': (
                'import numpy as np\n'
                'hankel = np.load("hankel_matrices.npz")\n'
                'cdhsa = np.load("cdhsa_arrays.npz")\n'
                'H = hankel["H_ss1_c1"]  # (p, T)\n'
                'Phi = cdhsa["D__Phi_specific__0"]  # (p, r_c)\n'
                'idx = [0, 1]  # columns_to_extract\n'
                'alpha = Phi[:, idx].T @ H  # (2, T)\n'
                '# alpha[0] = serie temporal del modo mas importante\n'
                '# alpha[1] = serie temporal del segundo modo\n'
            ),
        },
    }

    return mode_map


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            'Extraer indices de modos especificos de CD-HSA y generar '
            'mode_map.json'
        ),
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        required=True,
        help='Directorio con los archivos de salida de CD-HSA',
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=2,
        help=(
            'Cuantos modos especificos conservar por condicion '
            '(default: 2)'
        ),
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help=(
            'Ruta del JSON de salida. Default: mode_map.json '
            'en el mismo results-dir'
        ),
    )

    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        print(f'[ERROR] Directorio no encontrado: {results_dir}',
              file=sys.stderr)
        return 1

    # Verificar archivos necesarios
    required = ['hankel_info.json', 'config.json', 'cdhsa_arrays.npz']
    for fname in required:
        if not (results_dir / fname).exists():
            print(f'[ERROR] Archivo requerido no encontrado: {fname}',
                  file=sys.stderr)
            return 1

    print(f'Leyendo resultados de: {results_dir}')
    print(f'Top-N modos por condicion: {args.top_n}')
    print()

    mode_map = build_mode_map(results_dir, top_n=args.top_n)

    # Guardar
    out_path = Path(args.output) if args.output else results_dir / 'mode_map.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(mode_map, f, indent=2, ensure_ascii=False, default=str)

    print(f'[OK] mode_map.json guardado en: {out_path}')
    print()

    # Resumen
    meta = mode_map['metadata']
    print(f'  Super-sujetos: {meta["n_super_subjects"]}')
    print(f'  Condiciones:   {meta["n_conditions"]} ({meta["tasks"]})')
    print(f'  p (Hankel):    {meta["p_hankel"]}')
    print()

    print('  Modos especificos por condicion:')
    for task_name, cond in mode_map['conditions'].items():
        r_c = cond['total_specific_modes']
        n_actual = cond['top_n_actual']
        indices = cond['mode_indices_in_Phi']
        pc = cond['prevalence_contrast']
        print(f'    {task_name}: {r_c} modos encontrados, '
              f'top-{n_actual} = columnas {indices}, '
              f'prev_contrast = {pc}')
        for m in cond['modes']:
            ev = m['eigenvalue']
            ev_str = f'ev={ev:.6f}' if ev is not None else 'ev=N/A'
            print(f'      rank {m["rank_position"]}: columna {m["column_index_in_Phi"]} ({ev_str})')

    print()
    print('  Mapeo (super-sujeto, task) -> Hankel key:')
    for entry in mode_map['per_super_subject_task']:
        print(f'    SS{entry["super_subject"]}/{entry["task"]}: '
              f'{entry["hankel_npz_key"]} -> cols {entry["columns_to_extract"]}')

    # Mostrar estructura NPZ detectada
    npz_struct = mode_map['npz_structure']
    print()
    print('  Estructura NPZ detectada:')
    print(f'    Phi (modos) key:  {npz_struct["phi_key_identified"]}')
    print(f'    Lambda (eigenvals) key: {npz_struct["lambda_key_identified"]}')
    print(f'    D__ keys totales: {len(npz_struct["d_keys_found"])}')
    if npz_struct['d_nested_prefixes']:
        print('    Prefijos anidados:')
        for prefix, subkeys in npz_struct['d_nested_prefixes'].items():
            sample = f'{prefix}__{subkeys[0]}'
            shape = 'N/A'
            print(f'      {prefix}: {len(subkeys)} sub-keys '
                  f'(ej: {sample})')

    return 0


if __name__ == '__main__':
    sys.exit(main())
