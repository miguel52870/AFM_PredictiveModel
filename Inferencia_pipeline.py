"""
inferencia_pipeline.py — Orquestador de inferencia de los 3 modelos predictivos
Tesis: Evolución de los Dominios Ferroeléctricos con Deep Learning

Ejecuta secuencialmente:
  1. inferencia_regresion_C2   → genera C2 predicho + diffs encadenados
  2. inferencia_regresion_C3   → usa diffs de C2, genera C3 predicho
  3. inferencia_segmentacion   → usa diffs de C2, genera máscaras predichas

El orden es importante:
  - C2 genera los diffs encadenados Y el C2_prep predicho
  - C3 usa el C2_prep predicho de C2 como input (frames sin GT)
  - Seg usa el C2_prep predicho de C2 y el C3_prep predicho de C3 (frames sin GT)
Para frames sin GT los 3 canales de entrada son completamente predichos.

Uso:
  python inferencia_pipeline.py

Modos disponibles (INFERENCE_MODE):
  'independiente'  → inputs reales en cada frame (solo frames con GT)
  'autoregresiva'  → C predicho anterior como input (solo frames con GT)
  'ambos'          → ambos modos con comparación (solo frames con GT)

Frames sin ground truth (PREDICT_TO > total de archivos):
  Se procesan automáticamente en modo encadenado después de los frames con GT.
  C2 genera los diffs reales encadenados para que C3 y seg los usen.
"""

import sys
import time
import importlib
import importlib.util
from pathlib import Path

# =================================================================
# CONFIGURACIÓN GLOBAL
# (editar aquí — se propaga a los 3 scripts de inferencia)
# =================================================================

BASE_DIR = Path(r'C:\Users\migue\Desktop\modelo_predictivo\scripts')

# ── Recorte ──────────────────────────────────────────────────────
CROP_MODE   = 'cuadrado'   # 'cuadrado' | 'rectangular'
CROP_SIZE   = 80
CROP_WIDTH  = 80           # solo si CROP_MODE = 'rectangular' (divisible por 32)
CROP_HEIGHT = 64           # solo si CROP_MODE = 'rectangular' (divisible por 32)

# ── Frames a predecir ────────────────────────────────────────────
PREDICT_FROM   = 36
PREDICT_TO     = 45

# ── Modo de inferencia ───────────────────────────────────────────
INFERENCE_MODE = 'ambos'

# ── Threshold de binarización (segmentación) ─────────────────────
THRESHOLD    = 0.5

# ── Modelos a correr ─────────────────────────────────────────────
CORRER_C2  = True
CORRER_C3  = True
CORRER_SEG = True

# =================================================================
# RUTAS DE SCRIPTS (no modificar)
# =================================================================

SCRIPT_C2  = BASE_DIR / 'inferencia_regresion_C2.py'
SCRIPT_C3  = BASE_DIR / 'inferencia_regresion_C3.py'
SCRIPT_SEG = BASE_DIR / 'inferencia_segmentacion.py'

# =================================================================
# UTILIDADES
# =================================================================

def fmt_tiempo(segundos):
    h = int(segundos // 3600)
    m = int((segundos % 3600) // 60)
    s = int(segundos % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"

def cargar_modulo(nombre, script_path):
    spec = importlib.util.spec_from_file_location(nombre, str(script_path))
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def inyectar_config(mod):
    mod.PREDICT_FROM   = PREDICT_FROM
    mod.PREDICT_TO     = PREDICT_TO
    mod.INFERENCE_MODE = INFERENCE_MODE
    mod.THRESHOLD      = THRESHOLD
    mod.CROP_MODE      = CROP_MODE
    mod.CROP_SIZE      = CROP_SIZE
    mod.CROP_WIDTH     = CROP_WIDTH
    mod.CROP_HEIGHT    = CROP_HEIGHT
    if CROP_MODE == 'cuadrado':
        mod.CROP_W = CROP_SIZE
        mod.CROP_H = CROP_SIZE
    else:
        mod.CROP_W = CROP_WIDTH
        mod.CROP_H = CROP_HEIGHT

def correr_inferencia(nombre, script_path):
    print(f"\n{'='*65}")
    print(f"  INFERENCIA: {nombre}")
    print(f"{'='*65}")
    if not script_path.exists():
        print(f"  ERROR: script no encontrado en {script_path}")
        return False
    try:
        mod = cargar_modulo(nombre, script_path)
        inyectar_config(mod)
        t0 = time.time()
        mod.main()
        elapsed = time.time() - t0
        print(f"\n  [{nombre}] Completado en {fmt_tiempo(elapsed)}")
        return True
    except Exception as e:
        import traceback
        print(f"\n  [{nombre}] ERROR: {e}")
        traceback.print_exc()
        return False

# =================================================================
# MAIN
# =================================================================

def main():
    print("=" * 65)
    print("  PIPELINE DE INFERENCIA")
    print("=" * 65)
    crop_label = (f"{CROP_SIZE}px" if CROP_MODE == 'cuadrado'
                  else f"{CROP_WIDTH}x{CROP_HEIGHT}px")
    print(f"\nConfiguración global:")
    print(f"  Frames a predecir : {PREDICT_FROM}–{PREDICT_TO}")
    print(f"  Modo inferencia   : {INFERENCE_MODE}")
    print(f"  Recorte           : {CROP_MODE} ({crop_label})")
    print(f"  Threshold         : {THRESHOLD}")
    print()

    if not CORRER_C2 and (CORRER_C3 or CORRER_SEG):
        print("AVISO: CORRER_C2=False — C3 y segmentación no tendrán")
        print("  diffs encadenados NI C2_prep predicho para frames sin GT.")
        print("  Usarán el último C2 real disponible como alternativa.")
        print()
    if not CORRER_C3 and CORRER_SEG:
        print("AVISO: CORRER_C3=False — segmentación no tendrá")
        print("  C3_prep predicho para frames sin GT.")
        print("  Usará el último C3 real disponible como alternativa.")
        print()

    pipeline = [
        ('Regresion_C2',  SCRIPT_C2,  CORRER_C2),
        ('Regresion_C3',  SCRIPT_C3,  CORRER_C3),
        ('Segmentacion',  SCRIPT_SEG, CORRER_SEG),
    ]
    resumen  = []
    t_total  = time.time()

    for nombre, script_path, activado in pipeline:
        if not activado:
            print(f"  [{nombre}] Omitido (desactivado)")
            resumen.append((nombre, 'omitido'))
            continue
        ok = correr_inferencia(nombre, script_path)
        resumen.append((nombre, 'ok' if ok else 'ERROR'))

    elapsed_total = time.time() - t_total
    print(f"\n{'='*65}")
    print(f"  RESUMEN FINAL  ({fmt_tiempo(elapsed_total)} total)")
    print(f"{'='*65}")
    for nombre, estado in resumen:
        icono = '✓' if estado == 'ok' else ('—' if estado == 'omitido' else '✗')
        print(f"  [{icono}] {nombre:<20} : {estado}")

    print(f"\nCarpetas de resultados:")
    carpetas = [
        BASE_DIR / 'resultados' / 'modelo_regresion_C2' / 'predicciones',
        BASE_DIR / 'resultados' / 'modelo_regresion_c3' / 'predicciones',
        BASE_DIR / 'resultados' / 'modelo_segmentacion' / 'predicciones',
    ]
    for c in carpetas:
        existe = '✓' if c.exists() else '✗'
        print(f"  [{existe}] {c}")

if __name__ == '__main__':
    main()