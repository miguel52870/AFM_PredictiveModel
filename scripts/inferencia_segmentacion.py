"""
inferencia_segmentacion.py — Inferencia del modelo de segmentación de dominios
Tesis: Evolución de los Dominios Ferroeléctricos con Deep Learning

Modos disponibles (INFERENCE_MODE):
  'independiente'  → cada predicción usa datos reales del frame N como input
  'autoregresiva'  → usa la máscara predicha del frame anterior como input
  'ambos'          → corre ambos modos y genera comparación lado a lado

Paneles por frame (3 filas x 4 columnas):
  Fila 0 — Inputs:      C2 prep | C2 diff | C3 prep (fase continua)
  Fila 1 — Resultados:  Binaria predicha | Mascara Otsu real |
                        C3_prep real | Probabilidad (salida continua)
  Fila 2 — Errores:     |Binaria - Mascara| (error duro) |
                        |Probabilidad - Mascara| (error pesado por confianza)

Genera en resultados/modelo_segmentacion/predicciones/:
  - PNG por frame con 8 paneles
  - PNG resumen: grilla predicción vs. real
  - PNG comparacion_modos.png (solo en modo 'ambos')
  - CSV con métricas por frame (IoU, Dice, Acc)
"""

import os
import csv
import math
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import segmentation_models_pytorch as smp

# =================================================================
# 1. CONFIGURACIÓN
# =================================================================

BASE_DIR     = Path(r'C:\Users\migue\Desktop\modelo_predictivo')
DATA_DIR     = BASE_DIR / 'data'

DIR_C2_PREP  = DATA_DIR / 'canal_2'
DIR_C2_DIFF  = DATA_DIR / 'diff'
DIR_C3_PREP  = DATA_DIR / 'canal_3'
DIR_C3_MASK  = DATA_DIR / 'mask'

CKPT_PATH     = BASE_DIR / 'resultados' / 'modelo_segmentacion' / 'checkpoints' / 'best_model.pth'
OUTPUT_DIR    = BASE_DIR / 'resultados' / 'modelo_segmentacion' / 'predicciones'

# Diffs encadenados generados por inferencia_regresion_C2.py
DIR_DIFF_PRED = BASE_DIR / 'resultados' / 'modelo_regresion_C2' / 'predicciones' / 'npy' / 'diff_pred'
# C2 predicho (para usar como C2_prep en frames sin GT)
DIR_PRED_C2   = BASE_DIR / 'resultados' / 'modelo_regresion_C2' / 'predicciones' / 'npy' / 'pred_c2'
# C3 predicho de regresion_C3 (para usar como C3_prep en frames sin GT)
DIR_PRED_C3   = BASE_DIR / 'resultados' / 'modelo_regresion_c3' / 'predicciones' / 'npy' / 'pred'
# Predicciones autoregresivas de C2 (frame_NNN_auto_pred.npy)
DIR_C2_PRED   = BASE_DIR / 'resultados' / 'modelo_regresion_C2' / 'predicciones' / 'npy' / 'pred'

# Subcarpetas de salida
NPY_PRED  = OUTPUT_DIR / 'npy' / 'pred'
NPY_REAL  = OUTPUT_DIR / 'npy' / 'pred_con_gt'
NPY_ERROR = OUTPUT_DIR / 'npy' / 'error'
PNG_PRED  = OUTPUT_DIR / 'png' / 'pred'
PNG_REAL  = OUTPUT_DIR / 'png' / 'pred_con_gt'
PNG_ERROR = OUTPUT_DIR / 'png' / 'error'
FIG_DIR   = OUTPUT_DIR / 'figuras'


# --- MODO DE RECORTE ---
# Debe coincidir con el tamaño de las imagenes
CROP_MODE   = 'cuadrado'   # 'cuadrado' | 'rectangular'
CROP_SIZE   = 80
CROP_WIDTH  = 80
CROP_HEIGHT = 64

THRESHOLD    = 0.5

# Frames a PREDECIR (base 1, orden alfabético)
# PREDICT_FROM = 36 → predice el frame 36 usando el frame 35 como input
# PREDICT_FROM = 41 con 40 archivos disponibles → predicción sin ground truth
PREDICT_FROM   = 36
PREDICT_TO     = 45

# Modo de inferencia
INFERENCE_MODE = 'ambos'   # 'independiente' | 'autoregresiva' | 'ambos'


# Resolver dimensiones
if CROP_MODE == 'cuadrado':
    CROP_W, CROP_H = CROP_SIZE, CROP_SIZE
else:
    CROP_W, CROP_H = CROP_WIDTH, CROP_HEIGHT

# =================================================================
# 2. ÍNDICE POSICIONAL
# =================================================================

def build_file_index(directory, extension='.npy'):
    return sorted([f for f in Path(directory).iterdir() if f.suffix == extension])

def get_file(index_list, pos):
    if pos < 1 or pos > len(index_list):
        raise IndexError(f"Posición {pos} fuera de rango. Disponible: 1–{len(index_list)}.")
    return index_list[pos - 1]

idx_c2_prep   = build_file_index(DIR_C2_PREP)
idx_c2_diff   = build_file_index(DIR_C2_DIFF)
idx_c3_prep   = build_file_index(DIR_C3_PREP)
idx_c3_mask   = build_file_index(DIR_C3_MASK)

# Indice de diffs encadenados (puede estar vacio si no se corrio C2 antes)
idx_diff_pred = build_file_index(DIR_DIFF_PRED) if DIR_DIFF_PRED.exists() else []
# Indices de C2 y C3 predichos
idx_pred_c2   = build_file_index(DIR_PRED_C2)   if DIR_PRED_C2.exists()   else []
idx_pred_c3   = build_file_index(DIR_PRED_C3)   if DIR_PRED_C3.exists()   else []
# Indice de predicciones autoregresivas de C2
idx_c2_pred   = build_file_index(DIR_C2_PRED)   if DIR_C2_PRED.exists()   else []

# =================================================================
# 3. UTILIDADES
# =================================================================

def load_npy(path):
    arr = np.load(str(path)).astype(np.float32)
    mn, mx = arr.min(), arr.max()
    if mx - mn > 1e-8:
        arr = (arr - mn) / (mx - mn)
    return arr

def load_mask(path):
    arr = np.load(str(path)).astype(np.float32)
    return (arr > 127).astype(np.float32)

def build_input_tensor(c2_prep, c2_diff, c3_prep, device):
    x = np.stack([c2_prep, c2_diff, c3_prep], axis=0)
    return torch.from_numpy(x).unsqueeze(0).to(device)

def predict(model, tensor):
    with torch.no_grad():
        logits = model(tensor)
        prob   = torch.sigmoid(logits).squeeze().cpu().numpy()
        binary = (prob > THRESHOLD).astype(np.float32)
    return prob, binary

def compute_metrics(pred_bin, mask_real, eps=1e-6):
    inter = (pred_bin * mask_real).sum()
    union = pred_bin.sum() + mask_real.sum()
    iou   = float(inter / (union - inter + eps))
    dice  = float(2 * inter / (union + eps))
    acc   = float((pred_bin == mask_real).mean())
    return {'iou': iou, 'dice': dice, 'acc': acc}

def save_array_png(arr, path_png, cmap='gray_r', vmin=0, vmax=1):
    plt.imsave(str(path_png), arr, cmap=cmap, vmin=vmin, vmax=vmax)

def save_frame_files(arr, stem, npy_dir, png_dir, cmap='gray_r'):
    np.save(str(npy_dir / (stem + '.npy')), arr)
    save_array_png(arr, png_dir / (stem + '.png'), cmap=cmap)

def load_diff_pred(frame_out):
    """Carga el diff encadenado de C2 para el frame indicado. None si no existe."""
    stem_buscado = f"frame_{frame_out:03d}_sin_gt_diff_pred"
    for f in idx_diff_pred:
        if f.stem == stem_buscado:
            return np.load(str(f)).astype(np.float32)
    return None

def load_pred_c2(frame_out):
    """Carga C2 predicho del frame indicado generado por inferencia_regresion_C2."""
    stem_buscado = f"frame_{frame_out:03d}_sin_gt_pred_c2"
    for f in idx_pred_c2:
        if f.stem == stem_buscado:
            arr = np.load(str(f)).astype(np.float32)
            mn, mx = arr.min(), arr.max()
            if mx - mn > 1e-8: arr = (arr - mn) / (mx - mn)
            return arr
    return None

def load_pred_c3(frame):
    """C3 predicho por inferencia_regresion_C3 en modo sin GT.

    Solo busca archivos '_sin_gt_pred': el fallback a '_indep_pred' haria
    que el paso 0 tomara una prediccion en vez del C3 real disponible.
    None si no existe.
    """
    stem_buscado = f"frame_{frame:03d}_sin_gt_pred"
    for f in idx_pred_c3:
        if f.stem == stem_buscado:
            arr = np.load(str(f)).astype(np.float32)
            mn, mx = arr.min(), arr.max()
            if mx - mn > 1e-8: arr = (arr - mn) / (mx - mn)
            return arr
    return None

def _load_norm(path):
    arr = np.load(str(path)).astype(np.float32)
    mn, mx = arr.min(), arr.max()
    if mx - mn > 1e-8: arr = (arr - mn) / (mx - mn)
    return arr

def load_auto_pred_c2(frame):
    """C2 predicho en modo autoregresivo. C2 nombra su prediccion por el
    frame que contiene, asi que la clave es el frame de input."""
    stem_buscado = f"frame_{frame:03d}_auto_pred"
    for f in idx_c2_pred:
        if f.stem == stem_buscado:
            return _load_norm(f)
    return None

def load_auto_diff_c2(frame_in):
    """Diff encadenado que C2 uso como input en el frame indicado.
    C2 lo guarda bajo el stem del frame TARGET, de ahi el frame_in + 1."""
    stem_buscado = f"frame_{frame_in + 1:03d}_auto_diff"
    for f in idx_diff_pred:
        if f.stem == stem_buscado:
            return np.load(str(f)).astype(np.float32)
    return None

def load_auto_pred_c3(frame):
    """C3 predicho en modo autoregresivo por inferencia_regresion_C3.
    Fase continua — mismo dominio que el canal C3_prep del entrenamiento."""
    stem_buscado = f"frame_{frame:03d}_auto_pred"
    for f in idx_pred_c3:
        if f.stem == stem_buscado:
            return _load_norm(f)
    return None

# =================================================================
# 4. CARGAR MODELO
# =================================================================

def load_model(device):
    model = smp.Unet(
        encoder_name='efficientnet-b0', encoder_weights=None,
        in_channels=3, classes=1, activation=None).to(device)
    ckpt = torch.load(str(CKPT_PATH), map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"Modelo cargado — epoch {ckpt['epoch']} | "
          f"val IoU={ckpt['val_iou']:.4f} | val Dice={ckpt['val_dice']:.4f}\n")
    return model

# =================================================================
# 5. INFERENCIA
# =================================================================

def run_independiente(model, device, positions):
    print("─" * 65)
    print("MODO: Independiente")
    print("─" * 65)
    results = []
    for pos in positions:
        c2_prep    = load_npy(get_file(idx_c2_prep, pos))
        # El diff del frame N esta en la posicion N-1. Usar 'pos' cargaria
        # |C2[N+1] - C2[N]|, el cambio hacia el target — fuga.
        c2_diff    = load_npy(get_file(idx_c2_diff, pos - 1))
        c3_prep    = load_npy(get_file(idx_c3_prep, pos))
        mask_real  = load_mask(get_file(idx_c3_mask, pos + 1))
        c3_prep_real = load_npy(get_file(idx_c3_prep, pos + 1))

        tensor       = build_input_tensor(c2_prep, c2_diff, c3_prep, device)
        prob, binary = predict(model, tensor)
        metrics      = compute_metrics(binary, mask_real)

        fname_in  = get_file(idx_c2_prep, pos).stem
        fname_out = get_file(idx_c3_mask, pos + 1).stem
        print(f"  Pos {pos:>3} → {pos+1:>3} | IoU={metrics['iou']:.4f} "
              f"Dice={metrics['dice']:.4f} Acc={metrics['acc']:.4f}")

        # Guardar NPY y PNG
        stem = f"frame_{pos+1:03d}_indep"
        save_frame_files(binary,                      stem + '_pred',  NPY_PRED,  PNG_PRED,  'gray_r')
        save_frame_files(mask_real,                   stem + '_pred_con_gt',  NPY_REAL,  PNG_REAL,  'gray_r')
        save_frame_files(np.abs(binary - mask_real),  stem + '_error', NPY_ERROR, PNG_ERROR, 'Reds')

        results.append({
            'pos_in': pos, 'pos_out': pos + 1,
            'label_in': fname_in, 'label_out': fname_out,
            'c2_prep': c2_prep, 'c2_diff': c2_diff,
            'c3_prep_input': c3_prep,
            'prob': prob, 'binary': binary,
            'mask_real': mask_real,
            'c3_prep_real': c3_prep_real,
            'metrics': metrics, 'modo': 'independiente',
        })
    return results

def run_autoregresiva(model, device, positions):
    print("─" * 65)
    print("MODO: Autoregresiva (los 3 canales vienen del pipeline)")
    if idx_c2_pred and idx_pred_c3:
        print("  C2 y diff de inferencia_regresion_C2.py, C3 de inferencia_regresion_C3.py")
    else:
        print("  AVISO: faltan predicciones autoregresivas de C2 y/o C3.")
        print("  Los canales que falten caeran a datos REALES y la degradacion")
        print("  medida quedara subestimada. Correr C2 y C3 antes que este script.")
    print("─" * 65)
    results   = []
    prev_pred = None
    for i, pos in enumerate(positions):
        mask_real    = load_mask(get_file(idx_c3_mask, pos + 1))
        c3_prep_real = load_npy(get_file(idx_c3_prep, pos + 1))

        # Paso 0: los 3 canales reales — unico punto de arranque medido.
        # Pasos 1+: los 3 canales provienen del pipeline C2 -> C3 -> Seg.
        # El slot C3_prep lleva FASE CONTINUA (predicha por RC3), no la
        # mascara binaria: es el dominio con el que se entreno el modelo.
        if i == 0:
            c2_prep  = load_npy(get_file(idx_c2_prep, pos))
            c2_diff  = load_npy(get_file(idx_c2_diff, pos - 1))
            c3_input = load_npy(get_file(idx_c3_prep, pos))
            c2_src, diff_src, fuente = 'real', 'real', 'real'
        else:
            c2_auto = load_auto_pred_c2(pos)
            if c2_auto is not None:
                c2_prep, c2_src = c2_auto, 'C2 auto'
            else:
                c2_prep, c2_src = load_npy(get_file(idx_c2_prep, pos)), 'real (fallback)'

            d_auto = load_auto_diff_c2(pos)
            if d_auto is not None:
                c2_diff, diff_src = d_auto, 'encadenado C2'
            else:
                c2_diff, diff_src = load_npy(get_file(idx_c2_diff, pos - 1)), 'real (fallback)'

            c3_auto = load_auto_pred_c3(pos)
            if c3_auto is not None:
                c3_input, fuente = c3_auto, 'C3 auto'
            else:
                c3_input, fuente = load_npy(get_file(idx_c3_prep, pos)), 'real (fallback)'

        tensor       = build_input_tensor(c2_prep, c2_diff, c3_input, device)
        prob, binary = predict(model, tensor)
        metrics      = compute_metrics(binary, mask_real)

        fname_in  = get_file(idx_c2_prep, pos).stem
        fname_out = get_file(idx_c3_mask, pos + 1).stem
        print(f"  Pos {pos:>3} → {pos+1:>3} | IoU={metrics['iou']:.4f} "
              f"Dice={metrics['dice']:.4f} Acc={metrics['acc']:.4f} "
              f"[C2: {c2_src} | diff: {diff_src} | C3: {fuente}]")

        # Guardar NPY y PNG
        stem = f"frame_{pos+1:03d}_auto"
        save_frame_files(binary,                      stem + '_pred',  NPY_PRED,  PNG_PRED,  'gray_r')
        save_frame_files(mask_real,                   stem + '_pred_con_gt',  NPY_REAL,  PNG_REAL,  'gray_r')
        save_frame_files(np.abs(binary - mask_real),  stem + '_error', NPY_ERROR, PNG_ERROR, 'Reds')

        results.append({
            'pos_in': pos, 'pos_out': pos + 1,
            'label_in': fname_in, 'label_out': fname_out,
            'c2_prep': c2_prep, 'c2_diff': c2_diff,
            'c3_prep_input': c3_input,
            'prob': prob, 'binary': binary,
            'mask_real': mask_real,
            'c3_prep_real': c3_prep_real,
            'metrics': metrics, 'modo': 'autoregresiva',
            'c3_fuente': fuente,
            'c2_fuente': c2_src,
            'diff_fuente': diff_src,
        })
        prev_pred = binary
    return results


# =================================================================
# 5b. INFERENCIA SIN GROUND TRUTH — SEGMENTACIÓN
# =================================================================

def run_sin_gt(model, device, positions, n_c2, n_mask, modo):
    """
    Predicción encadenada para frames sin ground truth disponible.
    - Paso 0: usa el último frame real disponible como input.
    - Pasos siguientes: usa la máscara predicha anterior como input C3.
    - C2_diff: usa diff encadenado de inferencia_regresion_C2 si existe,
               si no usa el último diff real disponible.
    """
    print("─" * 65)
    print("MODO: Sin ground truth (predicción encadenada)")
    if idx_diff_pred:
        print("  Usando diffs encadenados de inferencia_regresion_C2.py")
    else:
        print("  AVISO: no se encontraron diffs de C2 — usando último diff real")
        print("  Correr inferencia_regresion_C2.py primero para mejores resultados")
    print("─" * 65)
    results   = []
    prev_pred = None   # máscara binaria predicha en el paso anterior

    for i, pos in enumerate(positions):
        # C2_prep: el slot lleva el frame de INPUT (pos), no el target.
        # En el paso 0 no hay prediccion todavia -> cae al ultimo real,
        # que es justamente el input correcto.
        c2_pred = load_pred_c2(pos)
        if c2_pred is not None:
            c2_prep = c2_pred
            c2p_src = f'C2 predicho frame {pos}'
        else:
            c2_pos  = min(pos, n_c2)
            c2_prep = load_npy(get_file(idx_c2_prep, c2_pos))
            c2p_src = f'C2 real (pos {c2_pos})'

        # C2_diff: C2 lo guarda bajo el stem del frame target aunque el
        # contenido corresponda al frame de input, de ahi la clave pos+1.
        diff_pred = load_diff_pred(pos + 1)
        if diff_pred is not None:
            c2_diff     = diff_pred
            diff_fuente = f'encadenado C2 (input frame {pos})'
        else:
            diff_pos    = min(pos - 1, len(idx_c2_diff))
            c2_diff     = (load_npy(get_file(idx_c2_diff, diff_pos))
                           if diff_pos >= 1
                           else np.zeros((CROP_H, CROP_W), dtype=np.float32))
            diff_fuente = f'real (pos {diff_pos})'

        # C3_prep: fase continua predicha por RC3 — mismo dominio del
        # entrenamiento. No se encadena la mascara binaria propia.
        c3_rc3 = load_pred_c3(pos)
        if c3_rc3 is not None:
            c3_input = c3_rc3
            fuente   = f'C3 predicho RC3 frame {pos}'
        else:
            c3_pos   = min(pos, len(idx_c3_prep))
            c3_input = load_npy(get_file(idx_c3_prep, c3_pos))
            fuente   = f'C3 real (pos {c3_pos})'

        tensor       = build_input_tensor(c2_prep, c2_diff, c3_input, device)
        prob, binary = predict(model, tensor)

        print(f"  Frame {pos+1:>3} (paso {i+1:>2}) | C2: {c2p_src} | C3: {fuente} | diff: {diff_fuente}")

        # Guardar NPY y PNG — prediccion y error vs. pred anterior
        stem = f"frame_{pos+1:03d}_sin_gt"
        save_frame_files(binary, stem + '_pred', NPY_PRED, PNG_PRED, 'gray_r')

        # Error sin GT: |pred_actual - pred_anterior|
        # Para paso 0: prev_pred es None → comparar contra último real disponible
        if prev_pred is not None:
            ref_error = prev_pred
        else:
            ref_pos   = min(pos, n_mask)
            ref_error = load_mask(get_file(idx_c3_mask, ref_pos))
        error_sgt = np.abs(binary - ref_error).astype(np.float32)
        save_frame_files(error_sgt, stem + '_error', NPY_ERROR, PNG_ERROR, 'Reds')

        results.append({
            'pos_in': pos, 'pos_out': pos + 1, 'paso': i + 1,
            'label_in': fuente, 'label_out': f'frame_{pos+1}_predicho',
            'c2_prep': c2_prep, 'c2_diff': c2_diff,
            'c3_prep_input': c3_input,
            'prob': prob, 'binary': binary,
            'mask_real': None, 'c3_prep_real': None,
            'metrics': None, 'modo': f'sin_gt_{modo}',
        })
        prev_pred = binary
    return results

# =================================================================
# 6. VISUALIZACIÓN — 8 PANELES
# =================================================================

def plot_results(results, out_dir, suffix=''):
    """PNG con 3 filas x 4 columnas + PNG resumen.

    Fila 1 — Inputs:
        C2 prep | C2 diff | C3 input | (vacio)
    Fila 2 — Prediccion vs. Real:
        Prediccion (binaria) | Real (mascara Otsu) | |Pred - Real| | Probabilidad
    Fila 3 — Error vs. fase:
        |Pred - C3_prep real| | C3_prep real | (vacio) | (vacio)
    """

    for r in results:
        # C3 input: frame real (twilight) o mascara predicha (gray) en autoregresivo
        # C3 input: siempre fase continua (real o predicha por RC3),
        # nunca una mascara binaria — de ahi el colormap fijo.
        c3_src           = r.get('c3_fuente', 'real')
        c3_input_cmap    = 'twilight'
        c3_input_titulo  = ('C3 input\n(frame ' + str(r['pos_in']) + ' ' + c3_src + ')')

        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        fig.suptitle(
            'Pos ' + str(r['pos_in']) + ' → ' + str(r['pos_out']) +
            '  [' + r['modo'] + ']' +
            '  IoU=' + str(round(r['metrics']['iou'],  4)) +
            '  Dice=' + str(round(r['metrics']['dice'], 4)) +
            '  Acc=' + str(round(r['metrics']['acc'],  4)),
            fontsize=11, fontweight='bold')

        plt.subplots_adjust(hspace=0.35, wspace=0.35)

        # ── Fila 0 — Inputs ──────────────────────────────────────────
        fila0 = [
            (r['c2_prep'],       'C2 prep input\n(amplitud PFM)',  'RdBu_r',        0, 1),
            (r['c2_diff'],       'C2 diff\n(cambio entre frames)', 'hot',           0, 1),
            (r['c3_prep_input'], c3_input_titulo,                   c3_input_cmap,   0, 1),
        ]
        for col, (data, title, cmap, vmin, vmax) in enumerate(fila0):
            im = axes[0, col].imshow(data, cmap=cmap, vmin=vmin, vmax=vmax,
                                     interpolation='nearest')
            axes[0, col].set_title(title, fontsize=8)
            axes[0, col].axis('off')
            plt.colorbar(im, ax=axes[0, col], fraction=0.046, pad=0.04)
        axes[0, 3].axis('off')
        axes[0, 0].set_ylabel('Inputs', fontsize=9, fontweight='bold', color='#888780')

        # ── Fila 1 — Prediccion vs. Real ─────────────────────────────
        fila1 = [
            (r['binary'],                          'Prediccion\n(binaria, umbral 0.5)', 'gray_r',     0, 1),
            (r['mask_real'],                       'Real\n(mascara Otsu)',              'gray_r',     0, 1),
            (r['c3_prep_real'],                    'C3_prep real\n(fase continua, frame ' + str(r['pos_out']) + ')', 'twilight', 0, 1),
            (r['prob'],                            'Probabilidad\n(salida continua)',   'RdBu_r',   0, 1),
        ]
        for col, (data, title, cmap, vmin, vmax) in enumerate(fila1):
            im = axes[1, col].imshow(data, cmap=cmap, vmin=vmin, vmax=vmax,
                                     interpolation='nearest')
            axes[1, col].set_title(title, fontsize=8)
            axes[1, col].axis('off')
            plt.colorbar(im, ax=axes[1, col], fraction=0.046, pad=0.04)
        axes[1, 0].set_ylabel('Prediccion vs. Real', fontsize=9, fontweight='bold', color='#378ADD')

        # ── Fila 2 — Error duro vs. error suave ──────────────────────
        # Ambos contra el MISMO ground truth (mascara Otsu):
        #   binaria vs mascara  -> donde se equivoco tras umbralizar
        #   prob    vs mascara  -> lo mismo pesado por la confianza;
        #                          revela si el error viene de dudas en las
        #                          paredes de dominio o de fallos seguros.
        fila2 = [
            (np.abs(r['binary'] - r['mask_real']), '|Pred − Real|\n(error binario)',              'Reds', 0, 1),
            (np.abs(r['prob']   - r['mask_real']), '|Prob − Real|\n(error continuo, confianza)',  'Reds', 0, 1),
        ]
        for col, (data, title, cmap, vmin, vmax) in enumerate(fila2):
            im = axes[2, col].imshow(data, cmap=cmap, vmin=vmin, vmax=vmax,
                                     interpolation='nearest')
            axes[2, col].set_title(title, fontsize=8)
            axes[2, col].axis('off')
            plt.colorbar(im, ax=axes[2, col], fraction=0.046, pad=0.04)
        axes[2, 2].axis('off')
        axes[2, 3].axis('off')
        axes[2, 0].set_ylabel('Errores', fontsize=9, fontweight='bold', color='#D85A30')

        fname = FIG_DIR / ('pos_' + str(r['pos_in']).zfill(3) + '_' +
                           str(r['pos_out']).zfill(3) + '_' +
                           r['modo'] + suffix + '.png')
        plt.savefig(str(fname), dpi=150, bbox_inches='tight')
        plt.close()

    # PNG resumen — predicción binaria vs. máscara vs. C3_prep
    n   = len(results)
    fig, axes = plt.subplots(3, n, figsize=(4 * n, 12))
    fig.suptitle(f"Resumen — [{results[0]['modo']}]", fontsize=13)

    for i, r in enumerate(results):
        axes[0, i].imshow(r['binary'],      cmap='gray_r')
        axes[0, i].set_title(f"Pred binaria\npos {r['pos_out']}\n"
                              f"IoU={r['metrics']['iou']:.3f}", fontsize=8)
        axes[0, i].axis('off')
        axes[1, i].imshow(r['mask_real'],   cmap='gray_r')
        axes[1, i].set_title(f"Máscara real\npos {r['pos_out']}", fontsize=8)
        axes[1, i].axis('off')
        axes[2, i].imshow(r['c3_prep_real'], cmap='twilight')
        axes[2, i].set_title(f"C3_prep real\npos {r['pos_out']}", fontsize=8)
        axes[2, i].axis('off')

    plt.tight_layout()
    fname = FIG_DIR / f"resumen_{results[0]['modo']}{suffix}.png"
    plt.savefig(str(fname), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Resumen guardado en: {FIG_DIR}")

def plot_comparacion(res_i, res_a, out_dir):
    """PNG comparativo independiente vs. autoregresiva — 6 filas."""
    n   = len(res_i)
    fig, axes = plt.subplots(6, n, figsize=(4 * n, 24))

    labels = ['Pred indep.', 'Pred auto.',
              'Máscara real', 'C3_prep real',
              '|Indep − Máscara|', '|Auto − Máscara|']
    for row, label in enumerate(labels):
        axes[row, 0].set_ylabel(label, fontsize=10, rotation=90, labelpad=10)

    for col, (ri, ra) in enumerate(zip(res_i, res_a)):
        axes[0, col].imshow(ri['binary'],      cmap='gray_r')
        axes[0, col].set_title(f"Pos {ri['pos_out']}\nIoU={ri['metrics']['iou']:.3f}",
                               fontsize=8)
        axes[1, col].imshow(ra['binary'],      cmap='gray_r')
        axes[1, col].set_title(f"Pos {ra['pos_out']}\nIoU={ra['metrics']['iou']:.3f}",
                               fontsize=8)
        axes[2, col].imshow(ri['mask_real'],   cmap='gray_r')
        axes[2, col].set_title(f"Pos {ri['pos_out']}", fontsize=8)
        axes[3, col].imshow(ri['c3_prep_real'], cmap='twilight')
        axes[3, col].set_title(f"Pos {ri['pos_out']}", fontsize=8)
        axes[4, col].imshow(np.abs(ri['binary'] - ri['mask_real']),
                            cmap='Reds', vmin=0, vmax=1)
        axes[5, col].imshow(np.abs(ra['binary'] - ra['mask_real']),
                            cmap='Reds', vmin=0, vmax=1)
        for row in range(6):
            axes[row, col].axis('off')

    plt.suptitle("Comparación: Independiente vs. Autoregresiva", fontsize=14)
    plt.tight_layout()
    plt.savefig(str(FIG_DIR / 'comparacion_modos.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Comparacion guardada: {FIG_DIR / 'comparacion_modos.png'}")


def plot_sin_gt(results, out_dir):
    if not results:
        return

    # Ultimo frame real disponible — fijo para todas las comparativas
    ultimo_pos  = max(1, results[0]['pos_in'])
    ultimo_real = load_mask(get_file(idx_c3_mask, ultimo_pos))
    lbl_ultimo  = "Mascara real frame " + str(ultimo_pos)   # etiqueta fija

    for i, r in enumerate(results):
        pred_ant  = results[i-1]['binary'] if i > 0 else ultimo_real
        label_ant = ("Pred. frame " + str(results[i-1]['pos_out'])) if i > 0 else lbl_ultimo
        pred_val  = r['binary']

        titulo = ("Seg frame " + str(r['pos_out']) +
                  " paso " + str(r['paso']) +
                  " — Sin ground truth  |  Input: " + str(r['label_in']))

        # Figura 3 filas x 3 columnas (C3 input omitido — ver inferencia_regresion_C3)
        fig, axes = plt.subplots(3, 3, figsize=(11, 10))
        fig.suptitle(titulo, fontsize=10, fontweight='bold')

        # Fila 0 — Inputs relevantes + predicción
        row0 = [
            (r['c2_prep'], 'C2 input',       'RdBu_r'),
            (r['c2_diff'], 'C2 diff',        'hot'),
            (pred_val,     'Seg predicha',   'gray_r'),
        ]
        for col, (data, title, cmap) in enumerate(row0):
            im = axes[0, col].imshow(data, cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
            axes[0, col].set_title(title, fontsize=8)
            axes[0, col].axis('off')
            plt.colorbar(im, ax=axes[0, col], fraction=0.046, pad=0.04)
        axes[0, 0].set_ylabel('Inputs / Prediccion', fontsize=8, fontweight='bold')

        # Fila 1 — Pred actual vs. ultimo real (etiqueta siempre fija)
        row1 = [
            (pred_val,                       'Seg predicha',       'gray_r'),
            (ultimo_real,                     lbl_ultimo,           'gray_r'),
            (np.abs(pred_val - ultimo_real), '|Pred - Ultimo real|','Reds'),
        ]
        for col, (data, title, cmap) in enumerate(row1):
            im = axes[1, col].imshow(data, cmap=cmap, vmin=0, vmax=1 if cmap!='Reds' else 0.5,
                                     interpolation='nearest')
            axes[1, col].set_title(title, fontsize=8)
            axes[1, col].axis('off')
            plt.colorbar(im, ax=axes[1, col], fraction=0.046, pad=0.04)
        axes[1, 0].set_ylabel('vs. Ultimo real', fontsize=8, fontweight='bold', color='#D85A30')

        # Fila 2 — Pred actual vs. pred anterior
        row2 = [
            (pred_val,                      'Prediccion actual',   'gray_r'),
            (pred_ant,                       label_ant,            'gray_r'),
            (np.abs(pred_val - pred_ant),   '|Actual - Anterior|', 'Reds'),
        ]
        for col, (data, title, cmap) in enumerate(row2):
            im = axes[2, col].imshow(data, cmap=cmap, vmin=0, vmax=1 if cmap!='Reds' else 0.5,
                                     interpolation='nearest')
            axes[2, col].set_title(title, fontsize=8)
            axes[2, col].axis('off')
            plt.colorbar(im, ax=axes[2, col], fraction=0.046, pad=0.04)
        axes[2, 0].set_ylabel('vs. Pred. anterior', fontsize=8, fontweight='bold', color='#534AB7')

        plt.tight_layout()
        fname = FIG_DIR / ("sin_gt_frame_" + str(r['pos_out']).zfill(3) +
                           "_paso_" + str(r['paso']).zfill(2) + ".png")
        plt.savefig(str(fname), dpi=150, bbox_inches='tight')
        plt.close()

    # PNG resumen
    n = len(results)
    ncols = n + 1
    fig, axes = plt.subplots(3, ncols, figsize=(4*ncols, 12))
    titulo_res = ("Seg encadenada sin GT  frames " +
                  str(results[0]['pos_out']) + " a " + str(results[-1]['pos_out']) +
                  "  |  Col 0 = " + lbl_ultimo + "  Col 1..N = predicciones")
    fig.suptitle(titulo_res, fontsize=10, fontweight='bold')

    axes[0, 0].imshow(ultimo_real, cmap='gray_r', vmin=0, vmax=1)
    axes[0, 0].set_title(lbl_ultimo, fontsize=8)
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')
    axes[2, 0].axis('off')

    for i, r in enumerate(results):
        pred_val = r['binary']
        pred_a   = results[i-1]['binary'] if i > 0 else ultimo_real

        axes[0, i+1].imshow(pred_val, cmap='gray_r', vmin=0, vmax=1)
        axes[0, i+1].set_title("Frame " + str(r['pos_out']) + " paso " + str(r['paso']), fontsize=8)
        axes[0, i+1].axis('off')
        axes[1, i+1].imshow(np.abs(pred_val - ultimo_real), cmap='Reds', vmin=0, vmax=0.5)
        axes[1, i+1].set_title('|Pred - Ultimo real|', fontsize=7)
        axes[1, i+1].axis('off')
        axes[2, i+1].imshow(np.abs(pred_val - pred_a), cmap='Reds', vmin=0, vmax=0.5)
        axes[2, i+1].set_title('|Pred - Anterior|', fontsize=7)
        axes[2, i+1].axis('off')

    for row_i, (lab, col) in enumerate([
        ('Seg predicho',    '#444441'),
        ('vs. Ultimo real',     '#D85A30'),
        ('vs. Pred. anterior',  '#534AB7'),
    ]):
        axes[row_i, 0].set_ylabel(lab, fontsize=9, fontweight='bold', color=col)

    plt.tight_layout()
    plt.savefig(str(FIG_DIR / 'sin_gt_evolucion.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Sin GT Seg: " + str(n) + " frames guardados en " + str(FIG_DIR))
# =================================================================
# 7. GUARDAR MÉTRICAS CSV
# =================================================================

def save_metrics_csv(results, out_dir, filename):
    path = out_dir / filename
    with open(str(path), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['pos_in', 'pos_out', 'archivo_in', 'archivo_out',
                         'modo', 'iou', 'dice', 'acc'])
        for r in results:
            writer.writerow([r['pos_in'], r['pos_out'],
                             r['label_in'], r['label_out'], r['modo'],
                             round(r['metrics']['iou'],  4),
                             round(r['metrics']['dice'], 4),
                             round(r['metrics']['acc'],  4)])
    print(f"  Métricas guardadas: {path}")


def print_mapeo(positions):
    """Muestra qué archivo real corresponde a cada posición antes de inferir."""
    print("─" * 65)
    print("Mapeo posición → archivo")
    print(f"  {'Pos':>4}  {'Rol':<12}  Archivo")
    print("─" * 65)
    for pos in positions:
        f_in  = get_file(idx_c2_prep, pos).stem
        f_dif = get_file(idx_c2_diff, pos - 1).stem
        f_msk = get_file(idx_c3_mask, pos + 1).stem
        f_c3  = get_file(idx_c3_prep, pos + 1).stem
        print(f"  {pos:>4}  input        {f_in}")
        print(f"  {pos-1:>4}  diff         {f_dif}")
        print(f"  {pos+1:>4}  mask target  {f_msk}")
        print(f"  {pos+1:>4}  C3_prep real {f_c3}")
        print()
    print("─" * 65)

# =================================================================
# 8. MAIN
# =================================================================

def main():
    for d in [OUTPUT_DIR, NPY_PRED, NPY_REAL, NPY_ERROR,
              PNG_PRED, PNG_REAL, PNG_ERROR, FIG_DIR]:
        os.makedirs(d, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo : {device}")

    n_files        = len(idx_c2_prep)
    n_diff         = len(idx_c2_diff)
    n_mask         = len(idx_c3_mask)

    # Convertir frames objetivo → posiciones de input (input = objetivo − 1)
    if PREDICT_FROM < 3:
        print(f'ERROR: PREDICT_FROM={PREDICT_FROM} invalido. Mínimo es 3.')
        print('  Para predecir el frame N se usa el input en pos N-1 y el diff')
        print('  en pos N-2. Con N=2 el indice del diff seria 0, fuera de rango.')
        return
    if PREDICT_FROM - 1 > n_files:
        print(f'ERROR: el input requerido (pos {PREDICT_FROM-1}) no existe '
              f'(solo hay {n_files} archivos).')
        return

    # positions = posiciones de INPUT (PREDICT_FROM-1 .. PREDICT_TO-1)
    positions     = list(range(PREDICT_FROM - 1, PREDICT_TO))
    # Separar con y sin ground truth
    pos_con_gt    = [p for p in positions if p + 1 <= n_mask]
    pos_sin_gt    = [p for p in positions if p + 1 >  n_mask]

    model = load_model(device)
    print(f'Archivos disponibles : {n_files} (C2/C3), {n_diff} (diff), {n_mask} (mask)')
    print(f'Frames a predecir    : {PREDICT_FROM}–{PREDICT_TO}')
    print(f'Inputs               : posiciones {positions[0]}–{positions[-1]}')
    if pos_con_gt:
        print(f'Con ground truth     : frames {[p+1 for p in pos_con_gt]}')
    if pos_sin_gt:
        print(f'Sin ground truth     : frames {[p+1 for p in pos_sin_gt]} '
              f'(predicción encadenada)')
    if idx_diff_pred:
        print(f'Diffs encadenados C2 : {len(idx_diff_pred)} archivos disponibles')
    else:
        print('Diffs encadenados C2 : no disponibles — correr inferencia_regresion_C2.py primero')
    n_auto_c2 = len([f for f in idx_c2_pred if f.stem.endswith('_auto_pred')])
    n_auto_c3 = len([f for f in idx_pred_c3 if f.stem.endswith('_auto_pred')])
    print(f'C2 autoregresivo     : {n_auto_c2} archivos'
          + ('' if n_auto_c2 else '  <- el modo auto usara C2 REAL'))
    print(f'C3 autoregresivo     : {n_auto_c3} archivos'
          + ('' if n_auto_c3 else '  <- el modo auto usara C3 REAL'))
    print()
    if pos_con_gt:
        print_mapeo(pos_con_gt)

    def _run_con_gt(fn_run, suffix, csv_name):
        if not pos_con_gt: return []
        res = fn_run(model, device, pos_con_gt)
        plot_results(res, OUTPUT_DIR, suffix=suffix)
        save_metrics_csv(res, OUTPUT_DIR, csv_name)
        return res

    if INFERENCE_MODE == 'independiente':
        _run_con_gt(run_independiente, '', 'metricas_independiente.csv')
        if pos_sin_gt:
            res_p = run_sin_gt(model, device, pos_sin_gt, n_files, n_mask, 'independiente')
            plot_sin_gt(res_p, OUTPUT_DIR)

    elif INFERENCE_MODE == 'autoregresiva':
        _run_con_gt(run_autoregresiva, '', 'metricas_autoregresiva.csv')
        if pos_sin_gt:
            res_p = run_sin_gt(model, device, pos_sin_gt, n_files, n_mask, 'autoregresiva')
            plot_sin_gt(res_p, OUTPUT_DIR)

    elif INFERENCE_MODE == 'ambos':
        res_i = _run_con_gt(run_independiente,  '_indep', 'metricas_independiente.csv')
        print()
        res_a = _run_con_gt(run_autoregresiva,  '_auto',  'metricas_autoregresiva.csv')
        if res_i and res_a:
            plot_comparacion(res_i, res_a, OUTPUT_DIR)
        if pos_sin_gt:
            res_p = run_sin_gt(model, device, pos_sin_gt, n_files, n_mask, 'ambos')
            plot_sin_gt(res_p, OUTPUT_DIR)

    print(f"\nResultados en : {OUTPUT_DIR}")
    print(f"NPY pred      : {NPY_PRED}")
    print(f"NPY pred_con_gt : {NPY_REAL}")
    print(f"NPY error     : {NPY_ERROR}")
    print(f"Figuras       : {FIG_DIR}")

if __name__ == '__main__':
    main()