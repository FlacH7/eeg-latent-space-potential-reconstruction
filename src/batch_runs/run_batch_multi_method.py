#!/usr/bin/env python3
"""
run_batch_multi_method.py
===========================
Ejecutor de batch multi-método para el pipeline IgA sobre el dataset
test-retest preprocesado con **Gedai** (formato EEGLAB .set/.fdt).

A diferencia de ``run_batch_iga_test_retest_gedai.py``, este script:

1. **Itera sobre múltiples scoring-methods** en cada corrida (por defecto
   markov, markov_inverted, hankel_dmd, diffusion_maps), generando un
   resultado independiente por método.

2. **Permite parametrizar** el rango de sujetos, las sesiones a procesar,
   las tareas y los métodos, haciéndolo escalable.

3. **Post-procesamiento automático**: al finalizar todas las corridas,
   genera gráficos compuestos de 2 filas × N columnas por cada
   (sujeto, sesión, tarea), donde:
   - Cada **columna** corresponde a un método.
   - La **fila 1** muestra el potencial 2D (``potential_2d.png``).
   - La **fila 2** muestra la fuerza no-conservativa / rotacional
     (``potential_2d_nonconservative_force.png``).

Configuración
--------------
Todo se controla mediante variables de entorno (o los defaults del script):

+------------------------------------------+---------------------------+--------------------------------------------------+
| Variable de entorno                      | Default                   | Descripción                                      |
+==========================================+===========================+==================================================+
| ``BATCH_MM_PARAMS_FILE``                 | (ninguno)                 | Archivo .txt/.csv de parámetros. Si se da, se    |
|                                          |                           | usa en lugar de la generación automática.        |
+------------------------------------------+---------------------------+--------------------------------------------------+
| ``BATCH_MM_SUBJ_START``                  | ``1``                     | Índice inicial de sujeto (sub-01, sub-02, …).    |
+------------------------------------------+---------------------------+--------------------------------------------------+
| ``BATCH_MM_SUBJ_END``                    | ``5``                     | Índice final de sujeto (inclusive).              |
+------------------------------------------+---------------------------+--------------------------------------------------+
| ``BATCH_MM_SESSIONS``                    | ``["session1"]``          | Lista JSON de sesiones a procesar.               |
+------------------------------------------+---------------------------+--------------------------------------------------+
| ``BATCH_MM_TASKS``                       | ``["eyesclosed", ...]``   | Lista JSON de tareas. Si vacío, usa las 5 por    |
|                                          |                           | defecto.                                         |
+------------------------------------------+---------------------------+--------------------------------------------------+
| ``BATCH_MM_METHODS``                     | ``["markov", ...]``       | Lista JSON de scoring-methods.                   |
+------------------------------------------+---------------------------+--------------------------------------------------+
| ``BATCH_MM_T_START``                     | ``0.0``                   | Inicio de la ventana temporal (s).               |
+------------------------------------------+---------------------------+--------------------------------------------------+
| ``BATCH_MM_T_END``                       | ``300.0``                 | Fin de la ventana temporal (s).                  |
+------------------------------------------+---------------------------+--------------------------------------------------+
| ``BATCH_MM_DB_PATH``                     | ``DB_TEST_RETEST_GEDAI_PATH`` | Ruta a la base de datos Gedai.                |
+------------------------------------------+---------------------------+--------------------------------------------------+
| ``BATCH_MM_OUTPUT_DIR``                  | ``BASE_RESULTS_PATH``     | Directorio base de resultados.                   |
+------------------------------------------+---------------------------+--------------------------------------------------+
| ``BATCH_MM_CACHE_DIR``                   | ``BASE_CACHE_PATH``       | Directorio base de caché.                        |
+------------------------------------------+---------------------------+--------------------------------------------------+
| ``BATCH_MM_LATENT_DIM``                  | ``2``                     | Dimensión del espacio latente.                   |
+------------------------------------------+---------------------------+--------------------------------------------------+
| ``BATCH_MM_ICA_METHOD``                  | ``picard``                | Método ICA.                                      |
+------------------------------------------+---------------------------+--------------------------------------------------+
| ``BATCH_MM_L_FREQ`` / ``BATCH_MM_H_FREQ``| ``1.0`` / ``40.0``       | Frecuencias de filtro.                          |
+------------------------------------------+---------------------------+--------------------------------------------------+
| ``BATCH_MM_MAX_WORKERS``                 | ``1``                     | Workers paralelos (1 = secuencial).              |
+------------------------------------------+---------------------------+--------------------------------------------------+
| ``BATCH_MM_DELAY``                       | ``2.0``                   | Pausa entre ejecuciones (solo secuencial).       |
+------------------------------------------+---------------------------+--------------------------------------------------+
| ``BATCH_MM_IGNORE_CACHE``                | ``false``                 | Forzar recálculo del espacio latente.            |
+------------------------------------------+---------------------------+--------------------------------------------------+
| ``BATCH_MM_RUN_POSTPROCESS``             | ``true``                  | Ejecutar post-proc de gráficos compuestos.       |
+------------------------------------------+---------------------------+--------------------------------------------------+
| ``BATCH_MM_LOG_LEVEL``                   | ``INFO``                  | Nivel de logging.                                |
+------------------------------------------+---------------------------+--------------------------------------------------+

Uso
---
Desde la raíz del proyecto::

    # Usando defaults (sub-01..sub-05, session1, 4 métodos, 5 tareas → 100 runs)
    python -m src.batch_runs.run_batch_multi_method

    # Personalizando vía entorno
    export BATCH_MM_SUBJ_END=10
    export BATCH_MM_METHODS='["markov", "hankel_dmd"]'
    python -m src.batch_runs.run_batch_multi_method

    # Usando archivo de parámetros externo (comportamiento clásico)
    export BATCH_MM_PARAMS_FILE=/ruta/a/mis_params.txt
    python -m src.batch_runs.run_batch_multi_method

Autor: Generado para el pipeline IgA multi-método.
"""

from __future__ import annotations

import ast
import csv
import json
import logging
import os
import subprocess
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import numpy as np

# ---------------------------------------------------------------------------
# Asegurar importabilidad del paquete
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent  # batch_runs/ -> src/ -> raíz

for _p in (_PROJECT_ROOT, _PROJECT_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ---------------------------------------------------------------------------
# Imports del proyecto
# ---------------------------------------------------------------------------
try:
    from src.utils.config import (
        BASE_CACHE_PATH,
        BASE_RESULTS_PATH,
        DB_TEST_RETEST_GEDAI_PATH,
        BASE_PARAMS_FILE,
        LOGGING_BASE_PATH,
        LOGGING_LEVEL,
    )
except ImportError as _exc:
    print(
        f"[ERROR] No se pudieron importar los módulos del proyecto. "
        f"Asegúrate de ejecutar este script desde la raíz del repositorio "
        f"o de que 'src' esté en PYTHONPATH.\n{_exc}",
        file=sys.stderr,
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuración de logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=LOGGING_LEVEL.upper(),
    format="[%(asctime)s] %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("batch_multi_method")

logging_path = (
    Path(LOGGING_BASE_PATH + "/batch_multi_method") if LOGGING_BASE_PATH else None
)
if not logging_path:
    logger.warning("No se definió LOGGING_BASE_PATH; logs no se guardarán en archivo.")
else:
    os.makedirs(logging_path, exist_ok=True)

# ===========================================================================
# DEFAULTS ESCALABLES
# ===========================================================================

DEFAULT_TASKS = ["eyesclosed", "eyesopen", "mathematic", "memory", "music"]
DEFAULT_METHODS = ["markov", "markov_inverted", "hankel_dmd", "diffusion_maps"]
DEFAULT_SESSIONS = ["session1"]

# ===========================================================================
# HELPERS de lectura de entorno
# ===========================================================================


def _env(var: str, default: str | None = None) -> str | None:
    return os.environ.get(var, default)


def _env_bool(var: str, default: bool = False) -> bool:
    val = os.environ.get(var, "").strip().lower()
    return val in ("1", "true", "yes", "on") if val else default


def _env_float(var: str, default: float) -> float:
    try:
        return float(os.environ[var])
    except (KeyError, ValueError):
        return default


def _env_int(var: str, default: int) -> int:
    try:
        return int(os.environ[var])
    except (KeyError, ValueError):
        return default


def _env_json_list(var: str, default: list) -> list:
    """Lee una variable de entorno como lista JSON. Si falla, devuelve default."""
    val = os.environ.get(var, "").strip()
    if not val:
        return default
    try:
        parsed = json.loads(val)
        if isinstance(parsed, list):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass
    return default


# ===========================================================================
# CONFIGURACIÓN GLOBAL
# ===========================================================================

# --- Rango de sujetos (parametrizable) ---
SUBJ_START: int = _env_int("BATCH_MM_SUBJ_START", 1)
SUBJ_END: int = _env_int("BATCH_MM_SUBJ_END", 5)

# --- Sesiones, tareas y métodos (parametrizable vía JSON en env) ---
SESSIONS: list[str] = _env_json_list("BATCH_MM_SESSIONS", DEFAULT_SESSIONS)
TASKS: list[str] = _env_json_list("BATCH_MM_TASKS", DEFAULT_TASKS)
METHODS: list[str] = _env_json_list("BATCH_MM_METHODS", DEFAULT_METHODS)

# --- Ventana temporal ---
T_START: float = _env_float("BATCH_MM_T_START", 0.0)
T_END: float = _env_float("BATCH_MM_T_END", 300.0)

# --- Parámetros del pipeline ---
LATENT_DIM: int = _env_int("BATCH_MM_LATENT_DIM", 2)
ICA_METHOD: str = _env("BATCH_MM_ICA_METHOD", "picard")
L_FREQ: float = _env_float("BATCH_MM_L_FREQ", 1.0)
H_FREQ: float = _env_float("BATCH_MM_H_FREQ", 40.0)

# --- Rutas ---
PARAMS_FILE: Path | None = None
if _env("BATCH_MM_PARAMS_FILE"):
    PARAMS_FILE = Path(_env("BATCH_MM_PARAMS_FILE"))
DB_PATH: Path = Path(_env("BATCH_MM_DB_PATH", DB_TEST_RETEST_GEDAI_PATH))
OUTPUT_DIR: Path = Path(_env("BATCH_MM_OUTPUT_DIR", BASE_RESULTS_PATH))
CACHE_DIR: Path = Path(_env("BATCH_MM_CACHE_DIR", BASE_CACHE_PATH))

# --- Ejecución ---
DELAY: float = _env_float("BATCH_MM_DELAY", 2.0)
IGNORE_CACHE: bool = _env_bool("BATCH_MM_IGNORE_CACHE", False)
RUN_POSTPROCESS: bool = _env_bool("BATCH_MM_RUN_POSTPROCESS", True)
MAX_WORKERS: int = _env_int("BATCH_MM_MAX_WORKERS", 1)

# --- Logging ---
if os.environ.get("BATCH_MM_LOG_LEVEL", LOGGING_LEVEL).upper() == "DEBUG":
    logger.setLevel(logging.DEBUG)

# --- Pipeline module ---
PIPELINE_MODULE: str = "src.pipelines.test_iga_from_eeg_latent_test_retest_gedai"
_PIPELINE_OUT_SUBPATH = "test_retest_gedai"

# --- Archivos de imagen para el post-procesamiento ---
POTENTIAL_IMG = "potential_2d.png"
NONCONS_IMG = "potential_2d_nonconservative_force.png"
COMPOSITE_FILENAME = "methods_comparison_potential_rotational.png"

# --- Checkpoint ---
CHECKPOINT_FILE: Path = CACHE_DIR / f"batch_checkpoint_multi_method.json"


# ===========================================================================
# GENERACIÓN DE JOBS
# ===========================================================================


def _generate_jobs_from_params() -> list[dict]:
    """Genera la lista de jobs a partir de los parámetros escalables.

    Cada job es un dict con:
        subject, session, task, method, t_start, t_end
    """
    jobs: list[dict] = []
    for subj_idx in range(SUBJ_START, SUBJ_END + 1):
        subject = f"sub-{subj_idx:02d}"
        for session in SESSIONS:
            for task in TASKS:
                for method in METHODS:
                    jobs.append({
                        "subject": subject,
                        "session": session,
                        "task": task,
                        "method": method,
                        "t_start": str(T_START),
                        "t_end": str(T_END),
                    })
    return jobs


def _generate_jobs_from_file(filepath: Path) -> list[dict]:
    """Lee jobs desde un archivo de parámetros y los combina con cada método.

    El archivo tiene el formato original:
        ['sub-01', 'session1_eyesclosed', '0.00', '300.00', 'eyesclosed']
    """
    rows: list[list[str]] = []
    ext = filepath.suffix.lower()
    if ext == ".csv":
        with open(filepath, "r", encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            next(reader, None)  # header
            for row in reader:
                if len(row) >= 5:
                    rows.append(row[:5])
    else:
        with open(filepath, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    row = ast.literal_eval(line)
                    if isinstance(row, list) and len(row) >= 5:
                        rows.append(row[:5])
                except (SyntaxError, ValueError):
                    continue

    jobs: list[dict] = []
    for row in rows:
        subject, session_task, t_start, t_end, label = row[:5]
        if "_" in session_task:
            session, task = session_task.rsplit("_", 1)
        else:
            session, task = session_task, label
        for method in METHODS:
            jobs.append({
                "subject": subject,
                "session": session,
                "task": task,
                "method": method,
                "t_start": t_start,
                "t_end": t_end,
            })
    return jobs


# ===========================================================================
# BATCH RUNNER
# ===========================================================================


class MultiMethodBatchRunner:
    """Orquesta la ejecución multi-método del pipeline IgA."""

    CSV_FIELDS = [
        "timestamp", "subject", "session", "task", "method",
        "t_start", "t_end", "success", "returncode", "elapsed_s", "command",
    ]

    def __init__(self) -> None:
        # --- Validaciones ---
        if PARAMS_FILE is not None and not PARAMS_FILE.exists():
            logger.error("Archivo de parámetros no encontrado: %s", PARAMS_FILE)
            sys.exit(1)
        if not _PROJECT_ROOT.exists():
            logger.error("Raíz del proyecto no encontrada: %s", _PROJECT_ROOT)
            sys.exit(1)
        if not DB_PATH.exists():
            logger.warning("Ruta de test-retest Gedai no encontrada: %s", DB_PATH)

        self.checkpoint: set[str] = self._load_checkpoint()

        ts = datetime.now().strftime("Y%m%d_%H%M%S")
        self.log_file = logging_path / f"batch_log_multi_method_{ts}.csv"
        self._init_csv_log()

        # --- Generar jobs ---
        if PARAMS_FILE is not None:
            logger.info("Generando jobs desde archivo: %s", PARAMS_FILE)
            self.all_jobs = _generate_jobs_from_file(PARAMS_FILE)
        else:
            logger.info("Generando jobs desde parámetros escalables")
            self.all_jobs = _generate_jobs_from_params()

        self._print_banner()

    # ------------------------------------------------------------------
    # Banner
    # ------------------------------------------------------------------

    def _print_banner(self) -> None:
        n_subjects = len({j["subject"] for j in self.all_jobs})
        n_sessions = len({j["session"] for j in self.all_jobs})
        n_tasks = len({j["task"] for j in self.all_jobs})
        n_methods = len({j["method"] for j in self.all_jobs})

        logger.info("=" * 70)
        logger.info("  BATCH MULTI-METHOD -- Test-Retest Gedai (EEGLAB .set/.fdt)")
        logger.info("=" * 70)
        logger.info("  Pipeline    : %s", PIPELINE_MODULE)
        logger.info("  DB path     : %s", DB_PATH)
        logger.info("  Output dir  : %s", OUTPUT_DIR)
        logger.info("  Cache dir   : %s", CACHE_DIR)
        logger.info("  Checkpoint  : %s", CHECKPOINT_FILE)
        logger.info("  ---")
        logger.info("  Subjects    : %d (sub-%02d .. sub-%02d)",
                    n_subjects, SUBJ_START, SUBJ_END)
        logger.info("  Sessions    : %s", SESSIONS)
        logger.info("  Tasks       : %s", TASKS)
        logger.info("  Methods     : %s", METHODS)
        logger.info("  Time window : %.1f s -> %.1f s", T_START, T_END)
        logger.info("  ---")
        logger.info("  Total jobs  : %d (%d subj × %d sess × %d task × %d meth)",
                    len(self.all_jobs), n_subjects, n_sessions, n_tasks, n_methods)
        logger.info("  Delay       : %.1f s", DELAY)
        logger.info("  Ignore cache: %s", IGNORE_CACHE)
        logger.info("  Max workers : %d (%s)",
                    MAX_WORKERS, "paralelo" if MAX_WORKERS > 1 else "secuencial")
        logger.info("  Post-process: %s", RUN_POSTPROCESS)
        logger.info("=" * 70)

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def _load_checkpoint(self) -> set[str]:
        if CHECKPOINT_FILE.exists():
            try:
                with open(CHECKPOINT_FILE, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                ck = set(data.get("completed", []))
                logger.info("Checkpoint cargado: %d jobs previos completados", len(ck))
                return ck
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Checkpoint corrupto, empezando de cero: %s", exc)
        return set()

    def _save_checkpoint(self) -> None:
        try:
            CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(CHECKPOINT_FILE, "w", encoding="utf-8") as fh:
                json.dump({"completed": sorted(self.checkpoint)}, fh, indent=2)
        except OSError as exc:
            logger.warning("No se pudo guardar checkpoint: %s", exc)

    @staticmethod
    def _checkpoint_key(subject: str, session: str, task: str,
                        method: str, t_start: str, t_end: str) -> str:
        return f"{subject}|{session}|{task}|{method}|{t_start}|{t_end}"

    # ------------------------------------------------------------------
    # Logs CSV
    # ------------------------------------------------------------------

    def _init_csv_log(self) -> None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.log_file, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=self.CSV_FIELDS)
                writer.writeheader()
        except OSError as exc:
            logger.warning("No se pudo inicializar log CSV: %s", exc)

    def _write_csv_log(self, job: dict, success: bool,
                       returncode: int, elapsed: float, cmd: list[str]) -> None:
        try:
            with open(self.log_file, "a", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=self.CSV_FIELDS)
                writer.writerow({
                    "timestamp": datetime.now().isoformat(),
                    "subject": job["subject"],
                    "session": job["session"],
                    "task": job["task"],
                    "method": job["method"],
                    "t_start": job["t_start"],
                    "t_end": job["t_end"],
                    "success": success,
                    "returncode": returncode,
                    "elapsed_s": round(elapsed, 2),
                    "command": " ".join(cmd),
                })
        except OSError as exc:
            logger.warning("Fallo al escribir log CSV: %s", exc)

    # ------------------------------------------------------------------
    # Filtro de jobs ya completados
    # ------------------------------------------------------------------

    def _filter_todo(self, jobs: list[dict]) -> list[dict]:
        todo: list[dict] = []
        for job in jobs:
            key = self._checkpoint_key(
                job["subject"], job["session"], job["task"],
                job["method"], job["t_start"], job["t_end"],
            )

            if key in self.checkpoint:
                logger.debug(
                    "SKIP (checkpoint): %s/%s/%s [%s]",
                    job["subject"], job["session"], job["task"], job["method"],
                )
                continue

            # Verificar si el directorio de salida ya tiene resultados
            out_sub = self._get_output_dir(job)
            if out_sub.exists() and any(out_sub.glob("*.png")):
                logger.debug(
                    "SKIP (output exists): %s/%s/%s [%s]",
                    job["subject"], job["session"], job["task"], job["method"],
                )
                self.checkpoint.add(key)
                continue

            todo.append(job)

        if skipped := len(jobs) - len(todo):
            logger.info("Jobs ya completados (skip): %d / %d", skipped, len(jobs))
        return todo

    # ------------------------------------------------------------------
    # Ruta de salida para un job
    # ------------------------------------------------------------------

    @staticmethod
    def _get_output_dir(job: dict) -> Path:
        """Devuelve el directorio de salida donde el pipeline guarda sus resultados.

        Mismo patrón que ``test_iga_from_eeg_latent_test_retest_gedai.py``:
        ``test_retest_gedai/{subject}/{session}/{latent_dim}_latent_dim_{method}/from{t_start}s_to_{t_end}s_{task}``
        """
        return (
            OUTPUT_DIR
            / _PIPELINE_OUT_SUBPATH
            / job["subject"]
            / job["session"]
            / f"{LATENT_DIM}_latent_dim_{job["method"]}"
            / f"from{job["t_start"]}s_to_{job["t_end"]}s_{job["task"]}"
        )

    # ------------------------------------------------------------------
    # Construcción del comando
    # ------------------------------------------------------------------

    def _build_command(self, job: dict) -> list[str]:
        cmd = [
            sys.executable,
            "-m", PIPELINE_MODULE,
            "--subject", job["subject"],
            "--session", job["session"],
            "--task", job["task"],
            "--t-start", job["t_start"],
            "--t-end", job["t_end"],
            "--latent-dim", str(LATENT_DIM),
            "--scoring-method", job["method"],
            "--l-freq", str(L_FREQ),
            "--h-freq", str(H_FREQ),
            "--ica-method", ICA_METHOD,
        ]
        if IGNORE_CACHE:
            cmd.append("--ignore-cache")
        return cmd

    # ------------------------------------------------------------------
    # Ejecución de un job individual
    # ------------------------------------------------------------------

    def _run_single_job(self, job: dict) -> tuple[str, bool]:
        key = self._checkpoint_key(
            job["subject"], job["session"], job["task"],
            job["method"], job["t_start"], job["t_end"],
        )

        try:
            cmd = self._build_command(job)
        except FileNotFoundError as exc:
            logger.error(
                "No se encontró .set para %s/%s/%s [%s]: %s",
                job["subject"], job["session"], job["task"], job["method"], exc,
            )
            return key, False

        logger.info(
            "RUN | %s/%s/%s | method=%s | [%s-%s] s",
            job["subject"], job["session"], job["task"],
            job["method"], job["t_start"], job["t_end"],
        )
        logger.debug("CMD: %s", " ".join(cmd))

        t0 = time.time()
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=_PROJECT_ROOT,
            )

            # Stream stdout en tiempo real
            for line in proc.stdout:
                logger.info("  [PIPE] %s", line.rstrip())

            proc.wait()
            elapsed = time.time() - t0
            success = proc.returncode == 0

            self._write_csv_log(job, success, proc.returncode, elapsed, cmd)

            if success:
                logger.info(
                    "OK | %s/%s/%s [%s] (%.1f s)",
                    job["subject"], job["session"], job["task"],
                    job["method"], elapsed,
                )
            else:
                logger.error(
                    "ERROR | %s/%s/%s [%s] -- código %d",
                    job["subject"], job["session"], job["task"],
                    job["method"], proc.returncode,
                )

            return key, success

        except Exception as exc:
            elapsed = time.time() - t0
            logger.error(
                "EXCEPTION | %s/%s/%s [%s]: %s",
                job["subject"], job["session"], job["task"],
                job["method"], exc,
            )
            self._write_csv_log(job, False, -1, elapsed, cmd)
            return key, False

    # ------------------------------------------------------------------
    # Post-procesamiento: gráficos compuestos 2×N
    # ------------------------------------------------------------------

    def _run_postprocessing(self) -> None:
        """Genera gráficos compuestos por (sujeto, sesión, tarea).

        Layout: 2 filas × N columnas (N = número de métodos)
          - Fila 0: potencial 2D (potential_2d.png)
          - Fila 1: fuerza no-conservativa / rotacional (potential_2d_nonconservative_force.png)
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image

        logger.info("")
        logger.info("=" * 70)
        logger.info("  INICIANDO POST-PROCESAMIENTO: GRÁFICOS COMPUESTOS")
        logger.info("=" * 70)

        # Agrupar jobs por (subject, session, task)
        groups: dict[tuple[str, str, str], list[dict]] = {}
        for job in self.all_jobs:
            key = (job["subject"], job["session"], job["task"])
            groups.setdefault(key, []).append(job)

        total_groups = len(groups)
        generated = 0
        missing = 0
        partial = 0

        for group_idx, ((subject, session, task), jobs) in enumerate(groups.items(), 1):
            logger.info(
                "[%d/%d] Procesando %s/%s/%s ...",
                group_idx, total_groups, subject, session, task,
            )

            n_methods = len(jobs)
            fig, axes = plt.subplots(
                2, n_methods,
                figsize=(5 * n_methods, 8),
                constrained_layout=True,
            )

            # Si solo hay 1 método, asegurarse de que axes sea 2D
            if n_methods == 1:
                axes = axes[:, np.newaxis] if hasattr(axes, 'shape') and axes.ndim == 1 else axes.reshape(2, 1)

            row_labels = ["Potential (U)", "Non-conservative force (v)"]
            found_any = False
            n_found = 0

            for col_idx, job in enumerate(jobs):
                method = job["method"]
                out_dir = self._get_output_dir(job)

                # --- Fila 0: Potencial 2D ---
                pot_path = out_dir / POTENTIAL_IMG
                if pot_path.exists():
                    try:
                        img = Image.open(pot_path)
                        axes[0, col_idx].imshow(img)
                        axes[0, col_idx].set_title(f"{method}", fontsize=11, fontweight="bold")
                        axes[0, col_idx].axis("off")
                        n_found += 1
                        found_any = True
                    except Exception as exc:
                        logger.warning("  No se pudo cargar %s: %s", pot_path, exc)
                        axes[0, col_idx].text(
                            0.5, 0.5, f"ERROR\n{pot_path.name}",
                            ha="center", va="center", fontsize=9, color="red",
                            transform=axes[0, col_idx].transAxes,
                        )
                        axes[0, col_idx].set_title(f"{method}", fontsize=11)
                        axes[0, col_idx].axis("off")
                else:
                    logger.warning("  Falta: %s", pot_path)
                    axes[0, col_idx].text(
                        0.5, 0.5, f"MISSING\n{pot_path.name}",
                        ha="center", va="center", fontsize=9, color="gray",
                        transform=axes[0, col_idx].transAxes,
                    )
                    axes[0, col_idx].set_title(f"{method}", fontsize=11)
                    axes[0, col_idx].axis("off")

                # --- Fila 1: Fuerza no-conservativa ---
                ncf_path = out_dir / NONCONS_IMG
                if ncf_path.exists():
                    try:
                        img = Image.open(ncf_path)
                        axes[1, col_idx].imshow(img)
                        axes[1, col_idx].axis("off")
                        found_any = True
                    except Exception as exc:
                        logger.warning("  No se pudo cargar %s: %s", ncf_path, exc)
                        axes[1, col_idx].text(
                            0.5, 0.5, f"ERROR\n{ncf_path.name}",
                            ha="center", va="center", fontsize=9, color="red",
                            transform=axes[1, col_idx].transAxes,
                        )
                        axes[1, col_idx].axis("off")
                else:
                    logger.warning("  Falta: %s", ncf_path)
                    axes[1, col_idx].text(
                        0.5, 0.5, f"MISSING\n{ncf_path.name}",
                        ha="center", va="center", fontsize=9, color="gray",
                        transform=axes[1, col_idx].transAxes,
                    )
                    axes[1, col_idx].axis("off")

            # --- Etiquetas de fila ---
            for row_idx, label in enumerate(row_labels):
                axes[row_idx, 0].set_ylabel(label, fontsize=12, fontweight="bold",
                                             rotation=90, labelpad=15)

            # --- Título general ---
            fig.suptitle(
                f"{subject} | {session} | {task}  --  Methods comparison",
                fontsize=14, fontweight="bold", y=1.01,
            )

            # --- Guardar ---
            # Se guarda al nivel de task: en el directorio del primer método
            # encontrado, o en un directorio dedicado si no hay ninguno
            save_dir = self._get_output_dir(jobs[0]).parent
            save_path = save_dir / COMPOSITE_FILENAME
            save_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                fig.savefig(str(save_path), dpi=150, bbox_inches="tight",
                            facecolor="white", edgecolor="none")
                plt.close(fig)
                logger.info("  Guardado: %s", save_path)
                generated += 1
            except Exception as exc:
                logger.error("  Error guardando %s: %s", save_path, exc)
                plt.close(fig)

            # Conteo parcial
            if n_found == 0:
                missing += 1
            elif n_found < 2 * n_methods:
                partial += 1

        # --- Resumen ---
        logger.info("")
        logger.info("=" * 70)
        logger.info("  POST-PROCESAMIENTO COMPLETADO")
        logger.info("=" * 70)
        logger.info("  Grupos procesados  : %d", total_groups)
        logger.info("  Completos (todas img): %d", generated - partial - (generated - generated + missing - missing + partial))
        logger.info("  Parciales (algunas img): %d", partial)
        logger.info("  Sin imágenes        : %d", missing)
        logger.info("  Archivos generados  : %d", generated)

    # ------------------------------------------------------------------
    # Orquestación principal
    # ------------------------------------------------------------------

    def run(self) -> int:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        if not self.all_jobs:
            logger.error("No se generaron jobs. Revisa la configuración.")
            return 1

        todo = self._filter_todo(self.all_jobs)
        total = len(todo)

        if total == 0:
            logger.info("Todos los jobs ya están completados.")
            if RUN_POSTPROCESS:
                self._run_postprocessing()
            return 0

        logger.info("Total a ejecutar: %d / %d", total, len(self.all_jobs))

        completed = 0
        failed = 0

        if MAX_WORKERS > 1:
            completed, failed = self._run_parallel(todo, total)
        else:
            completed, failed = self._run_sequential(todo, total)

        self._save_checkpoint()

        logger.info("=" * 70)
        logger.info(
            "BATCH COMPLETADO -- OK: %d | Fallos: %d | Total: %d",
            completed, failed, total,
        )
        logger.info("Log CSV: %s", self.log_file)

        if RUN_POSTPROCESS:
            self._run_postprocessing()

        return 0 if failed == 0 else 1

    def _run_sequential(self, todo: list[dict], total: int) -> tuple[int, int]:
        completed = 0
        failed = 0

        for idx, job in enumerate(todo, start=1):
            logger.info("")
            logger.info("-" * 70)
            logger.info("Progreso: %d / %d", idx, total)
            logger.info("-" * 70)

            key, success = self._run_single_job(job)
            if success:
                self.checkpoint.add(key)
                completed += 1
            else:
                failed += 1

            self._save_checkpoint()

            if idx < total and DELAY > 0:
                logger.debug("Pausa %.1f s...", DELAY)
                time.sleep(DELAY)

        return completed, failed

    def _run_parallel(self, todo: list[dict], total: int) -> tuple[int, int]:
        completed = 0
        failed = 0

        logger.info("Modo PARALELO con %d workers", MAX_WORKERS)

        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_job = {
                executor.submit(self._run_single_job, job): job for job in todo
            }

            for future in as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    key, success = future.result()
                    if success:
                        self.checkpoint.add(key)
                        completed += 1
                    else:
                        failed += 1
                except Exception as exc:
                    logger.error(
                        "FUTURE EXCEPTION | %s/%s/%s [%s]: %s",
                        job["subject"], job["session"], job["task"],
                        job["method"], exc,
                    )
                    failed += 1

                self._save_checkpoint()
                logger.info(
                    "Progreso: %d / %d completados", completed + failed, total
                )

        return completed, failed


# ===========================================================================
# ENTRY POINT
# ===========================================================================


def main() -> int:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        runner = MultiMethodBatchRunner()
        return runner.run()


if __name__ == "__main__":
    sys.exit(main())
