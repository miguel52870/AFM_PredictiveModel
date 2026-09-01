"""
inferencia_pipeline.py — Orquestador de inferencia de los 3 modelos predictivos
Tesis: Evolución de los Dominios Ferroeléctricos con Deep Learning

Ejecuta secuencialmente:
  1. inferencia_regresion_C2   → genera C2 predicho + diffs encadenados
  2. inferencia_regresion_C3   → usa diffs de C2, genera C3 predicho
  3. inferencia_segmentacion   → usa diffs de C2, genera máscaras predichas

El orden es OBLIGATORIO, no solo recomendado:
  - C2 genera los diffs encadenados, el C2 autoregresivo y el C2_prep predicho
  - C3 usa el C2 autoregresivo y los diffs de C2 como input
  - Seg usa el C2 de C2 y el C3 de C3 como input

Esto aplica tanto al modo autoregresivo (frames con GT) como al modo
sin GT. En ambos, a partir del paso 1 los 3 canales de entrada son
completamente predichos: ninguno es real.

Si se desactiva un modelo, los siguientes NO caen a datos reales —
leen lo que haya quedado en disco de una corrida previa. Ver avisos.

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

# ── Diff encadenado (solo aplica a inferencia_regresion_C2) ──────
# Renormaliza el diff encadenado a [0,1] para igualar la escala de los
# diffs reales. True = consistente con el entrenamiento.
NORMALIZAR_DIFF_ENCADENADO = True
# Guarda el diff del modo autoregresivo — C3 y Seg lo necesitan.
GUARDAR_DIFF_AUTO          = True

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
    # Solo inferencia_regresion_C2 define estos flags
    if hasattr(mod, 'NORMALIZAR_DIFF_ENCADENADO'):
        mod.NORMALIZAR_DIFF_ENCADENADO = NORMALIZAR_DIFF_ENCADENADO
    if hasattr(mod, 'GUARDAR_DIFF_AUTO'):
        mod.GUARDAR_DIFF_AUTO = GUARDAR_DIFF_AUTO

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
        print("!" * 65)
        print("AVISO: CORRER_C2=False")
        print("  C3 y segmentacion NO caeran a datos reales: van a leer los")
        print("  archivos que ya esten en disco de una corrida anterior de C2")
        print("    resultados/modelo_regresion_C2/predicciones/npy/pred/")
        print("    resultados/modelo_regresion_C2/predicciones/npy/pred_c2/")
        print("    resultados/modelo_regresion_C2/predicciones/npy/diff_pred/")
        print("  Si PREDICT_FROM/PREDICT_TO o el checkpoint cambiaron desde")
        print("  entonces, se mezclaran predicciones de dos configuraciones")
        print("  distintas SIN aviso. Los nombres de archivo son iguales.")
        print("  Correr C2 tambien, o borrar esas tres carpetas antes.")
        print("!" * 65)
        print()
    if not CORRER_C3 and CORRER_SEG:
        print("!" * 65)
        print("AVISO: CORRER_C3=False")
        print("  Segmentacion leera el C3 predicho que ya este en disco:")
        print("    resultados/modelo_regresion_c3/predicciones/npy/pred/")
        print("  Mismo riesgo: archivos viejos con el mismo nombre pasan")
        print("  desapercibidos. Correr C3 tambien, o borrar esa carpeta.")
        print("!" * 65)
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
    root = BASE_DIR.parent   # BASE_DIR apunta a scripts/, resultados/ esta un nivel arriba
    carpetas = [
        root / 'resultados' / 'modelo_regresion_C2' / 'predicciones',
        root / 'resultados' / 'modelo_regresion_c3' / 'predicciones',
        root / 'resultados' / 'modelo_segmentacion' / 'predicciones',
    ]
    for c in carpetas:
        existe = '✓' if c.exists() else '✗'
        print(f"  [{existe}] {c}")

if __name__ == '__main__':
    main()