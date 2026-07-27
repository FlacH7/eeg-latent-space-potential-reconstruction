#!/usr/bin/env python3
"""
run_batch_multi_method.py
===========================
Ejecutor de batch multi-metodo para el pipeline IgA sobre el dataset
test-retest preprocesado con **Gedai** (formato EEGLAB .set/.fdt).

A diferencia de ``run_batch_iga_test_retest_gedai.py``, este script:

1. **Itera sobre multiples scoring-methods** en cada corrida (por defecto
   markov, markov_inverted, hankel_dmd, diffusion_maps), generando un
   resultado independiente por metodo.

2. **Permite parametrizar** el rango de sujetos, las sesiones a procesar,
   las tareas y los metodos, haciendolo escalable.

3. **Post-procesamiento automatico**: al finalizar todas las corridas,
   genera graficos compuestos de 2 filas x N columnas por cada
   (sujeto, sesion, tarea), donde:
   - Cada **columna** corresponde a un metodo.
   - La **fila 1** muestra el potencial 2D (``potential_2d.png``).
   - La **fila 2** muestra la fuerza no-conservativa / rotacional
     (``potential_2d_nonconservative_force.png``).

   Los graficos se nombran dinamicamente incluyendo sujeto, sesion y tarea:
   ``methods_comparison_{subject}_{session}_{task}.png``

Configuracion
--------------
Todo se controla mediante variables de entorno (o los defaults del script):

BATCH_MM_PARAMS_FILE     : Archivo .txt/.csv de parametros externo.
BATCH_MM_SUBJ_START      : Indice inicial de sujeto (default: 1)
BATCH_MM_SUBJ_END        : Indice final de sujeto inclusive (default: 5)
BATCH_MM_SESSIONS        : Lista JSON de sesiones (default: ["session1"])
BATCH_MM_TASKS           : Lista JSON de tareas (default: las 5)
BATCH_MM_METHODS         : Lista JSON de metodos (default: los 4)
BATCH_MM_T_START/T_END   : Ventana temporal (default: 0.0 / 300.0)
BATCH_MM_DB_PATH         : Ruta a la base de datos Gedai
BATCH_MM_OUTPUT_DIR      : Directorio base de resultados
BATCH_MM_CACHE_DIR       : Directorio base de cache
BATCH_MM_LATENT_DIM      : Dimension del espacio latente (default: 2)
BATCH_MM_ICA_METHOD      : Metodo ICA (default: picard)
BATCH_MM_L_FREQ/H_FREQ   : Frecuencias de filtro (default: 1.0 / 40.0)
BATCH_MM_MAX_WORKERS     : Workers paralelos (default: 1, secuencial)
BATCH_MM_DELAY           : Pausa entre ejecuciones (default: 2.0 s)
BATCH_MM_IGNORE_CACHE    : Forzar recalculo (default: false)
BATCH_MM_RUN_POSTPROCESS : Ejecutar post-proc (default: true)
BATCH_MM_LOG_LEVEL       : Nivel de logging (default: INFO)

Uso
---
Desde la raiz del proyecto::

    # Usando defaults (sub-01..sub-05, session1, 4 metodos, 5 tareas -> 100 runs)
    python -m src.batch_runs.run_batch_multi_method

    # Personalizando via entorno
    export BATCH_MM_SUBJ_END=10
    export BATCH_MM_METHODS='["markov", "hankel_dmd"]'
    python -m src.batch_runs.run_batch_multi_method

    # Usando archivo de parametros externo (comportamiento clasico)
    export BATCH_MM_PARAMS_FILE=/ruta/a/mis_params.txt
    python -m src.batch_runs.run_batch_multi_method
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

# ---------------------------------------------------------------------------
# Asegurar importabilidad del paquete
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent  # batch_runs/ -> src/ -> raiz

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
        f"[ERROR] No se pudieron importar los modulos del proyecto. "
        f"Asegurate de ejecutar este script desde la raiz del repositorio "
        f"o de que 'src' este en PYTHONPATH.\n{_exc}",
        file=sys.stderr,
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuracion de logging
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
    logger.warning("No se definio LOGGING_BASE_PATH; logs no se guardaran en archivo.")
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
# CONFIGURACION GLOBAL
# ===========================================================================

# --- Rango de sujetos (parametrizable) ---
SUBJ_START: int = _env_int("BATCH_MM_SUBJ_START", 1)
SUBJ_END: int = _env_int("BATCH_MM_SUBJ_END", 5)

# --- Sesiones, tareas y metodos (parametrizable via JSON en env) ---
SESSIONS: list[str] = _env_json_list("BATCH_MM_SESSIONS", DEFAULT_SESSIONS)
TASKS: list[str] = _env_json_list("BATCH_MM_TASKS", DEFAULT_TASKS)
METHODS: list[str] = _env_json_list("BATCH_MM_METHODS", DEFAULT_METHODS)

# --- Ventana temporal ---
T_START: float = _env_float("BATCH_MM_T_START", 0.0)
T_END: float = _env_float("BATCH_MM_T_END", 300.0)

# --- Parametros del pipeline ---
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

# --- Ejecucion ---
DELAY: float = _env_float("BATCH_MM_DELAY", 2.0)
IGNORE_CACHE: bool = _env_bool("BATCH_MM_IGNORE_CACHE", True)
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
COMPOSITE_PREFIX = "methods_comparison"

# --- Checkpoint ---
CHECKPOINT_FILE: Path = CACHE_DIR / "batch_checkpoint_multi_method.json"


# ===========================================================================
# GENERACION DE JOBS
# ===========================================================================


def _generate_jobs_from_params() -> list[dict]:
    """Genera la lista de jobs a partir de los parametros escalables.

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
    """Lee jobs desde un archivo de parametros y los combina con cada metodo.

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
    """Orquesta la ejecucion multi-metodo del pipeline IgA."""

    CSV_FIELDS = [
        "timestamp", "subject", "session", "task", "method",
        "t_start", "t_end", "success", "returncode", "elapsed_s", "command",
    ]

    def __init__(self) -> None:
        # --- Validaciones ---
        if PARAMS_FILE is not None and not PARAMS_FILE.exists():
            logger.error("Archivo de parametros no encontrado: %s", PARAMS_FILE)
            sys.exit(1)
        if not _PROJECT_ROOT.exists():
            logger.error("Raiz del proyecto no encontrada: %s", _PROJECT_ROOT)
            sys.exit(1)
        if not DB_PATH.exists():
            logger.warning("Ruta de test-retest Gedai no encontrada: %s", DB_PATH)

        self.checkpoint: set[str] = self._load_checkpoint()

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = logging_path / f"batch_log_multi_method_{ts}.csv"
        self._init_csv_log()

        # --- Generar jobs ---
        if PARAMS_FILE is not None:
            logger.info("Generando jobs desde archivo: %s", PARAMS_FILE)
            self.all_jobs = _generate_jobs_from_file(PARAMS_FILE)
        else:
            logger.info("Generando jobs desde parametros escalables")
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
        logger.info("  Total jobs  : %d (%d subj x %d sess x %d task x %d meth)",
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

        Mismo patron que ``test_iga_from_eeg_latent_test_retest_gedai.py``:
        ``test_retest_gedai/{subject}/{session}/{latent_dim}_latent_dim_{method}/from{t_start}s_to_{t_end}s_{task}``
        """
        method = job["method"]
        return (
            OUTPUT_DIR
            / _PIPELINE_OUT_SUBPATH
            / job["subject"]
            / job["session"]
            / f"{LATENT_DIM}_latent_dim_{method}"
            / f"from{job['t_start']}s_to_{job['t_end']}s_{job['task']}"
        )

    # ------------------------------------------------------------------
    # Construccion del comando
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
    # Ejecucion de un job individual
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
                "No se encontro .set para %s/%s/%s [%s]: %s",
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
                    "ERROR | %s/%s/%s [%s] -- codigo %d",
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
    # Post-procesamiento: graficos compuestos 2xN
    # ------------------------------------------------------------------

    def _run_postprocessing(self) -> None:
        """Genera graficos compuestos por (sujeto, sesion, tarea).

        Layout: 2 filas x N columnas (N = numero de metodos)
          - Fila 0: potencial 2D (potential_2d.png)
          - Fila 1: fuerza no-conservativa / rotacional (potential_2d_nonconservative_force.png)

        El nombre del archivo incluye sujeto, sesion y tarea para unicidad:
          methods_comparison_{subject}_{session}_{task}.png
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image

        logger.info("")
        logger.info("=" * 70)
        logger.info("  INICIANDO POST-PROCESAMIENTO: GRAFICOS COMPUESTOS")
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

            # Si solo hay 1 metodo, asegurarse de que axes sea 2D
            if n_methods == 1:
                axes = axes.reshape(2, 1)

            row_labels = ["Potential (U)", "Non-conservative force (v)"]
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
                        n_found += 1
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

            # --- Titulo general ---
            fig.suptitle(
                f"{subject} | {session} | {task}  --  Methods comparison",
                fontsize=14, fontweight="bold", y=1.01,
            )

            # --- Guardar con nombre dinamico ---
            save_dir = self._get_output_dir(jobs[0]).parent
            composite_name = f"{COMPOSITE_PREFIX}_{subject}_{session}_{task}.png"
            save_path = save_dir / composite_name
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
            expected_images = 2 * n_methods
            if n_found == 0:
                missing += 1
            elif n_found < expected_images:
                partial += 1

        # --- Resumen ---
        complete = generated - partial - missing
        logger.info("")
        logger.info("=" * 70)
        logger.info("  POST-PROCESAMIENTO COMPLETADO")
        logger.info("=" * 70)
        logger.info("  Grupos procesados   : %d", total_groups)
        logger.info("  Completos           : %d", complete)
        logger.info("  Parciales           : %d", partial)
        logger.info("  Sin imagenes        : %d", missing)
        logger.info("  Archivos generados  : %d", generated)

    # ------------------------------------------------------------------
    # Orquestacion principal
    # ------------------------------------------------------------------

    def run(self) -> int:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        if not self.all_jobs:
            logger.error("No se generaron jobs. Revisa la configuracion.")
            return 1

        todo = self._filter_todo(self.all_jobs)
        total = len(todo)

        if total == 0:
            logger.info("Todos los jobs ya estan completados.")
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
