#!/usr/bin/env python3
"""
verify_corruption.py
--------------------
Verifica si los archivos GEDAI .set de Subject 41, Session 3 (memory, music)
están corruptos (header MATLAB válido de 128 bytes + resto todo ceros).
Si están corruptos, verifica que los originales BrainVision sean legibles.

Uso:
    python verify_corruption.py
"""

import os
import struct
from pathlib import Path
import sys

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.utils.config import DB_TEST_RETEST_GEDAI_PATH, DB_TEST_RETEST_PATH

# ---------------------------------------------------------------------------
# Configuración de rutas
# ---------------------------------------------------------------------------
GEDAI_ROOT = Path(DB_TEST_RETEST_GEDAI_PATH) / "sub-41" / "ses-session3" / "eeg"
ORIG_ROOT  = Path(DB_TEST_RETEST_PATH)      / "sub-41" / "ses-session3" / "eeg"

TASKS = ["memory", "music"]

# Nombres de los archivos GEDAI
GEDAI_FILES = {
    task: f"sub-41_ses-session3_task-{task}_eeg_01-40_Gedai.set"
    for task in TASKS
}

# Nombres de los archivos originales BrainVision (header)
ORIG_FILES = {
    task: f"sub-41_ses-session3_task-{task}_eeg.vhdr"
    for task in TASKS
}

# ---------------------------------------------------------------------------
# Funciones de verificación binaria para .set (MATLAB v5)
# ---------------------------------------------------------------------------
def is_valid_matlab_header(data: bytes) -> bool:
    """
    Un archivo .mat v5 debe tener:
      - Bytes 0-3:  'MATLAB 5.0 MAT-file' (texto legible en los primeros 116 bytes)
      - Bytes 124-125: versión (0x0100 para v5)
      - Bytes 126-127: endian indicator ('IM' o 'MI')
    """
    if len(data) < 128:
        return False

    # Text header (primeros 116 bytes) debe contener "MATLAB"
    text_header = data[:116]
    if b"MATLAB" not in text_header:
        return False

    # Versión en bytes 124-125 (little-endian uint16 == 0x0100)
    version = struct.unpack("<H", data[124:126])[0]
    if version != 0x0100:
        return False

    # Endian indicator en bytes 126-127
    endian = data[126:128]
    if endian not in (b"IM", b"MI"):
        return False

    return True


def check_set_corruption(filepath: Path) -> dict:
    """
    Inspección binaria de un .set:
      1. ¿Existe?
      2. ¿Tiene header MATLAB válido de 128 bytes?
      3. ¿El resto del archivo está todo en cero?
      4. ¿Cuántas variables reporta MNE/EEGLAB (si se puede leer)?
    """
    result = {
        "exists": False,
        "size_bytes": 0,
        "valid_matlab_header": False,
        "nonzero_after_header": None,
        "all_zeros_after_header": False,
        "mne_loadable": False,
        "mne_error": None,
    }

    if not filepath.exists():
        return result

    result["exists"] = True
    result["size_bytes"] = filepath.stat().st_size

    with open(filepath, "rb") as f:
        data = f.read()

    if len(data) < 128:
        result["valid_matlab_header"] = False
        return result

    result["valid_matlab_header"] = is_valid_matlab_header(data)

    payload = data[128:]
    if payload:
        nonzero_count = sum(1 for b in payload if b != 0)
        result["nonzero_after_header"] = nonzero_count
        result["all_zeros_after_header"] = (nonzero_count == 0)
    else:
        result["nonzero_after_header"] = 0
        result["all_zeros_after_header"] = True

    # Intento de carga con MNE (read_raw_eeglab o read_epochs_eeglab)
    try:
        import mne
        # Los .set de GEDAI suelen ser Raw; si falla, probamos Epochs
        try:
            raw = mne.io.read_raw_eeglab(str(filepath), preload=False, verbose="ERROR")
            result["mne_loadable"] = True
        except Exception:
            epochs = mne.read_epochs_eeglab(str(filepath), verbose="ERROR")
            result["mne_loadable"] = True
    except Exception as e:
        result["mne_loadable"] = False
        result["mne_error"] = str(e)

    return result


# ---------------------------------------------------------------------------
# Funciones de verificación para originales BrainVision
# ---------------------------------------------------------------------------
def check_original_integrity(vhdr_path: Path) -> dict:
    """
    Intenta cargar el archivo BrainVision con MNE y reporta diagnóstico.
    """
    result = {
        "exists": False,
        "mne_loadable": False,
        "mne_error": None,
        "n_channels": None,
        "sfreq": None,
        "n_times": None,
        "duration_sec": None,
    }

    if not vhdr_path.exists():
        return result

    result["exists"] = True

    # Verificar que los archivos auxiliares (.eeg, .vmrk) existan
    eeg_path = vhdr_path.with_suffix(".eeg")
    vmrk_path = vhdr_path.with_suffix(".vmrk")
    missing_aux = []
    if not eeg_path.exists():
        missing_aux.append(str(eeg_path.name))
    if not vmrk_path.exists():
        missing_aux.append(str(vmrk_path.name))

    if missing_aux:
        result["mne_error"] = f"Faltan archivos auxiliares: {', '.join(missing_aux)}"
        return result

    try:
        import mne
        raw = mne.io.read_raw_brainvision(str(vhdr_path), preload=False, verbose="ERROR")
        result["mne_loadable"] = True
        result["n_channels"] = raw.info["nchan"]
        result["sfreq"] = raw.info["sfreq"]
        result["n_times"] = raw.n_times
        result["duration_sec"] = raw.n_times / raw.info["sfreq"]
    except Exception as e:
        result["mne_loadable"] = False
        result["mne_error"] = str(e)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("DIAGNÓSTICO DE CORRUPCIÓN – Subject 41, Session 3")
    print("=" * 70)

    # --- 1. Verificar archivos GEDAI (.set) ---
    print("\n[1] VERIFICACIÓN DE ARCHIVOS GEDAI (.set)\n")
    gedai_corrupt = {}

    for task in TASKS:
        fpath = GEDAI_ROOT / GEDAI_FILES[task]
        print(f"  Tarea: {task.upper()}")
        print(f"  Ruta : {fpath}")

        diag = check_set_corruption(fpath)

        if not diag["exists"]:
            print("  → Estado: NO ENCONTRADO\n")
            gedai_corrupt[task] = None
            continue

        print(f"  → Tamaño archivo     : {diag['size_bytes']:,} bytes")
        print(f"  → Header MATLAB válido: {'SÍ' if diag['valid_matlab_header'] else 'NO'}")
        print(f"  → Bytes != 0 tras header: {diag['nonzero_after_header']}")
        print(f"  → Todo ceros tras header: {'SÍ' if diag['all_zeros_after_header'] else 'NO'}")
        print(f"  → Cargable con MNE    : {'SÍ' if diag['mne_loadable'] else 'NO'}")

        if diag["mne_error"]:
            print(f"  → Error MNE           : {diag['mne_error'][:120]}")

        # Criterio de corrupción según el auditor:
        #   header MATLAB OK  AND  todo ceros después del header
        is_corrupt = diag["valid_matlab_header"] and diag["all_zeros_after_header"]
        gedai_corrupt[task] = is_corrupt
        print(f"  → CORRUPTO (criterio) : {'SÍ' if is_corrupt else 'NO'}\n")

    # --- 2. Verificar archivos originales BrainVision ---
    print("[2] VERIFICACIÓN DE ARCHIVOS ORIGINALES (BrainVision)\n")

    for task in TASKS:
        vhdr = ORIG_ROOT / ORIG_FILES[task]
        print(f"  Tarea: {task.upper()}")
        print(f"  Ruta : {vhdr}")

        diag = check_original_integrity(vhdr)

        if not diag["exists"]:
            print("  → Estado: NO ENCONTRADO\n")
            continue

        if diag["mne_error"] and not diag["mne_loadable"]:
            print(f"  → Cargable con MNE: NO")
            print(f"  → Error MNE       : {diag['mne_error'][:120]}\n")
            continue

        print(f"  → Cargable con MNE: SÍ")
        print(f"  → Canales         : {diag['n_channels']}")
        print(f"  → Frec. muestreo  : {diag['sfreq']} Hz")
        print(f"  → Muestras        : {diag['n_times']}")
        print(f"  → Duración        : {diag['duration_sec']:.2f} s")
        print(f"  → Estado          : {'CORRUPTO' if not diag['mne_loadable'] else 'OK'}\n")

    # --- 3. Resumen final ---
    print("=" * 70)
    print("RESUMEN")
    print("=" * 70)

    any_gedai_corrupt = any(v for v in gedai_corrupt.values() if v is True)

    if any_gedai_corrupt:
        print("\n  Los siguientes archivos GEDAI están corruptos:")
        for task, corrupt in gedai_corrupt.items():
            if corrupt:
                print(f"    • {GEDAI_FILES[task]}")
        print("\n  Verificando si los originales están sanos...")

        originals_ok = True
        for task in TASKS:
            if gedai_corrupt.get(task):
                vhdr = ORIG_ROOT / ORIG_FILES[task]
                diag = check_original_integrity(vhdr)
                if diag["mne_loadable"]:
                    print(f"    • Original {task}: OK (cargable, {diag['n_channels']} ch, {diag['duration_sec']:.1f} s)")
                else:
                    print(f"    • Original {task}: FALLA – {diag.get('mne_error', 'desconocido')}")
                    originals_ok = False

        if originals_ok:
            print("\n  >>> CONCLUSIÓN: Los .set de GEDAI están corruptos, pero los")
            print("      archivos originales BrainVision están sanos. Puedes regenerar")
            print("      los GEDAI desde los originales o solicitar al estudiante que")
            print("      reenvíe los .set/.fdt correctos.")
        else:
            print("\n  >>> CONCLUSIÓN: Tanto los GEDAI como los originales tienen problemas.")
            print("      Revisa el dataset fuente antes de continuar.")
    else:
        print("\n  >>> CONCLUSIÓN: No se detectó corrupción en los archivos GEDAI revisados.")
        print("      (o no cumplen el patrón header-válido + payload-todo-ceros).")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()