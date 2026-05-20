"""
entrenar_pipeline.py — Orquestador de entrenamiento de los 3 modelos predictivos
Tesis: Evolución de los Dominios Ferroeléctricos con Deep Learning

Entrena secuencialmente:
  1. Modelo_Segmentacion  → predice C3_mask[N+1]   (loss: BCE + Dice)
  2. Modelo_Regresion_C2  → predice C2_prep[N+1]   (loss: MSE + SSIM)
  3. Modelo_Regresion_C3  → predice C3_prep[N+1]   (loss: MSE + SSIM)

Los 3 modelos comparten la misma arquitectura (U-Net + EfficientNet-B0)
y el mismo conjunto de inputs: C2_prep[N], C2_diff[N], C3_prep[N].

Uso:
  python entrenar_pipeline.py

Comportamiento de skip:
  Si ya existe best_model.pth para un modelo, se omite su entrenamiento.
  Para forzar el reentrenamiento de un modelo específico, elimina su checkpoint.
"""

import sys
import time
import importlib
from pathlib import Path

# =================================================================
# CONFIGURACIÓN GLOBAL
# (se aplica a los 3 modelos — editar aquí, no en los scripts individuales)
# =================================================================

BASE_DIR = Path(r'C:\Users\migue\Desktop\modelo_predictivo')

# ── Recorte ──────────────────────────────────────────────────────
CROP_MODE   = 'cuadrado'   # 'cuadrado' | 'rectangular'
CROP_SIZE   = 80           # usado si CROP_MODE = 'cuadrado'
CROP_WIDTH  = 80           # usado si CROP_MODE = 'rectangular' (divisible por 32)
CROP_HEIGHT = 64           # usado si CROP_MODE = 'rectangular' (divisible por 32)

# ── Entrenamiento ─────────────────────────────────────────────────
EPOCHS       = 100
BATCH_SIZE   = 4
LR           = 1e-4
WEIGHT_DECAY = 1e-4

# ── Dataset ───────────────────────────────────────────────────────
VAL_FRAMES   = 6     # últimos N pares usados para validación (split cronológico)

# ── Reproducibilidad ──────────────────────────────────────────────
SEED         = 42

# ── Threshold de binarización (segmentación) ─────────────────────
THRESHOLD    = 0.5

# ── Modelos a entrenar ────────────────────────────────────────────
# True = entrenar si no tiene checkpoint | False = omitir siempre
ENTRENAR_SEG = True
ENTRENAR_C2  = True
ENTRENAR_C3  = True

# ── Forzar reentrenamiento aunque ya exista checkpoint ───────────
# True = reentrenar aunque exista best_model.pth
FORZAR_SEG   = False
FORZAR_C2    = False
FORZAR_C3    = False

# =================================================================
# RUTAS DE CHECKPOINTS (no modificar)
# =================================================================

CKPT_SEG = BASE_DIR / 'resultados' / 'modelo_segmentacion' / 'checkpoints' / 'best_model.pth'
CKPT_C2  = BASE_DIR / 'resultados' / 'modelo_regresion_C2' / 'checkpoints' / 'best_model.pth'
CKPT_C3  = BASE_DIR / 'resultados' / 'modelo_regresion_c3' / 'checkpoints' / 'best_model.pth'

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

def inyectar_config(modulo):
    """Sobreescribe las variables de configuración del módulo con los valores globales."""
    modulo.CROP_MODE   = CROP_MODE
    modulo.CROP_SIZE   = CROP_SIZE
    modulo.CROP_WIDTH  = CROP_WIDTH
    modulo.CROP_HEIGHT = CROP_HEIGHT
    modulo.EPOCHS      = EPOCHS
    modulo.BATCH_SIZE  = BATCH_SIZE
    modulo.LR          = LR
    modulo.WEIGHT_DECAY= WEIGHT_DECAY
    modulo.VAL_FRAMES  = VAL_FRAMES
    modulo.SEED        = SEED
    modulo.THRESHOLD   = THRESHOLD

    # Recalcular dimensiones derivadas
    if CROP_MODE == 'cuadrado':
        modulo.CROP_W = CROP_SIZE
        modulo.CROP_H = CROP_SIZE
    else:
        modulo.CROP_W = CROP_WIDTH
        modulo.CROP_H = CROP_HEIGHT

def entrenar_modelo(nombre, script_path, ckpt_path, forzar):
    """
    Importa el script de entrenamiento, inyecta la configuración global
    y llama a main(). Retorna True si se entrenó, False si se omitió.
    """
    if not forzar and ckpt_path.exists():
        print(f"  [{nombre}] Checkpoint encontrado — omitiendo")
        print(f"    {ckpt_path}")
        return False

    if forzar and ckpt_path.exists():
        print(f"  [{nombre}] Forzando reentrenamiento (FORZAR=True)")

    print(f"\n{'='*65}")
    print(f"  ENTRENANDO: {nombre}")
    print(f"{'='*65}")

    # Agregar el directorio del script al path si es necesario
    script_dir = str(script_path.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    # Importar o recargar el módulo
    spec = importlib.util.spec_from_file_location(nombre, str(script_path))
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Inyectar configuración global
    inyectar_config(mod)

    # Ejecutar entrenamiento
    t0 = time.time()
    mod.main()
    elapsed = time.time() - t0

    print(f"\n  [{nombre}] Completado en {fmt_tiempo(elapsed)}")
    return True

# =================================================================
# MAIN
# =================================================================

def main():
    print("=" * 65)
    print("  PIPELINE DE ENTRENAMIENTO")
    print("=" * 65)
    print(f"\nConfiguración global:")
    crop_label = (f"{CROP_SIZE}px" if CROP_MODE == 'cuadrado'
                  else f"{CROP_WIDTH}x{CROP_HEIGHT}px")
    print(f"  Recorte      : {CROP_MODE} ({crop_label})")
    print(f"  Epochs       : {EPOCHS}")
    print(f"  Batch size   : {BATCH_SIZE}")
    print(f"  LR           : {LR}")
    print(f"  Weight decay : {WEIGHT_DECAY}")
    print(f"  Val frames   : {VAL_FRAMES}")
    print(f"  Seed         : {SEED}")
    print()

    # Configurar rutas de scripts
    scripts = [
        ('Segmentacion', BASE_DIR / 'Modelo_Segmentacion.py', CKPT_SEG, ENTRENAR_SEG, FORZAR_SEG),
        ('Regresion_C2', BASE_DIR / 'Modelo_Regresion_C2.py', CKPT_C2,  ENTRENAR_C2,  FORZAR_C2),
        ('Regresion_C3', BASE_DIR / 'Modelo_Regresion_C3.py', CKPT_C3,  ENTRENAR_C3,  FORZAR_C3),
    ]

    resumen = []
    t_total = time.time()

    for nombre, script_path, ckpt_path, activado, forzar in scripts:
        if not activado:
            print(f"\n  [{nombre}] Desactivado (ENTRENAR_{nombre.split('_')[-1]}=False)")
            resumen.append((nombre, 'omitido (desactivado)'))
            continue

        if not script_path.exists():
            print(f"\n  [{nombre}] ERROR: script no encontrado en {script_path}")
            resumen.append((nombre, 'ERROR: script no encontrado'))
            continue

        try:
            entrenado = entrenar_modelo(nombre, script_path, ckpt_path, forzar)
            estado = 'entrenado' if entrenado else 'omitido (checkpoint existente)'
            resumen.append((nombre, estado))
        except Exception as e:
            print(f"\n  [{nombre}] ERROR durante entrenamiento: {e}")
            resumen.append((nombre, f'ERROR: {e}'))

    # Resumen final
    elapsed_total = time.time() - t_total
    print(f"\n{'='*65}")
    print(f"  RESUMEN FINAL  ({fmt_tiempo(elapsed_total)} total)")
    print(f"{'='*65}")
    for nombre, estado in resumen:
        print(f"  {nombre:<20} : {estado}")

    # Checkpoints existentes
    print(f"\nCheckpoints:")
    for nombre, script, ckpt, _, _ in scripts:
        existe = '✓' if ckpt.exists() else '✗'
        print(f"  [{existe}] {nombre:<20} : {ckpt}")

if __name__ == '__main__':
    main()