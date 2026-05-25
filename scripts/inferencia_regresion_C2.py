"""
inferencia_regresion_C2.py — Inferencia del modelo de regresión de amplitud PFM
Tesis: Evolución de los Dominios Ferroeléctricos con Deep Learning

Los archivos se indexan por POSICIÓN en la lista ordenada alfabéticamente.
Posición 1 = primer archivo, posición 2 = segundo, etc.
Al inicio muestra el mapeo explícito de posición → nombre de archivo.

Modos disponibles (INFERENCE_MODE):
  'independiente'  → cada predicción usa datos reales del frame N como input
  'autoregresiva'  → usa la imagen C2 predicha del frame anterior como input
  'ambos'          → corre ambos modos y genera comparación lado a lado

Genera en resultados/modelo_regresion_C2/predicciones/:
  - PNG por frame con 2 filas × 3 columnas
  - PNG resumen: grilla de todas las predicciones
  - PNG comparacion_modos.png (solo en modo 'ambos')
  - CSV con métricas por frame (SSIM, PSNR, MSE)

Carpetas de NPY y PNG guardados:
  predicciones/npy/pred/      ← C2 predicho (con y sin GT)
  predicciones/npy/real/      ← C2 real (solo con GT)
  predicciones/npy/error/     ← |pred - real| (solo con GT)
  predicciones/npy/diff_pred/ ← diff encadenado sin GT (para seg y C3)
  predicciones/png/pred/
  predicciones/png/real/
  predicciones/png/error/
  predicciones/png/diff_pred/
"""

import os
import csv
import math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

CKPT_PATH    = BASE_DIR / 'resultados' / 'modelo_regresion_C2' / 'checkpoints' / 'best_model.pth'
OUTPUT_DIR   = BASE_DIR / 'resultados' / 'modelo_regresion_C2' / 'predicciones'

# Subcarpetas de salida
NPY_PRED      = OUTPUT_DIR / 'npy' / 'pred'
NPY_REAL      = OUTPUT_DIR / 'npy' / 'real'
NPY_ERROR     = OUTPUT_DIR / 'npy' / 'error'
NPY_DIFF_PRED = OUTPUT_DIR / 'npy' / 'diff_pred'
PNG_PRED      = OUTPUT_DIR / 'png' / 'pred'
PNG_REAL      = OUTPUT_DIR / 'png' / 'real'
PNG_ERROR     = OUTPUT_DIR / 'png' / 'error'
PNG_DIFF_PRED = OUTPUT_DIR / 'png' / 'diff_pred'
FIG_DIR       = OUTPUT_DIR / 'figuras'

# --- MODO DE RECORTE ---
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

idx_c2_prep = build_file_index(DIR_C2_PREP)
idx_c2_diff = build_file_index(DIR_C2_DIFF)
idx_c3_prep = build_file_index(DIR_C3_PREP)

# =================================================================
# 3. UTILIDADES
# =================================================================

def load_npy(path):
    arr = np.load(str(path)).astype(np.float32)
    mn, mx = arr.min(), arr.max()
    if mx - mn > 1e-8:
        arr = (arr - mn) / (mx - mn)
    return arr

def build_input_tensor(c2_prep, c2_diff, c3_prep, device):
    x = np.stack([c2_prep, c2_diff, c3_prep], axis=0)
    return torch.from_numpy(x).unsqueeze(0).to(device)

def predict(model, tensor):
    with torch.no_grad():
        out = model(tensor)
        out = torch.sigmoid(out).squeeze().cpu().numpy()
    return out

def compute_metrics(pred, target, eps=1e-8):
    pred_t   = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0)
    target_t = torch.from_numpy(target).unsqueeze(0).unsqueeze(0)
    mse_val  = F.mse_loss(pred_t, target_t).item()
    psnr_val = 10 * math.log10(1.0 / mse_val) if mse_val > eps else 100.0
    mu1, mu2 = pred.mean(), target.mean()
    s1, s2   = pred.std(), target.std()
    s12      = ((pred - mu1) * (target - mu2)).mean()
    C1, C2   = 0.01**2, 0.03**2
    ssim_val = float(((2*mu1*mu2 + C1) * (2*s12 + C2)) /
                     ((mu1**2 + mu2**2 + C1) * (s1**2 + s2**2 + C2)))
    return {'mse': mse_val, 'psnr': round(psnr_val, 2), 'ssim': round(ssim_val, 4)}

def save_array_png(arr, path_png, cmap='RdBu_r', vmin=0, vmax=1):
    """Guarda un array float32 como PNG con el colormap indicado."""
    plt.imsave(str(path_png), arr, cmap=cmap, vmin=vmin, vmax=vmax)

def save_frame_files(arr, stem, npy_dir, png_dir, cmap='RdBu_r'):
    """Guarda NPY y PNG de un array dado un nombre base."""
    np.save(str(npy_dir / (stem + '.npy')), arr)
    save_array_png(arr, png_dir / (stem + '.png'), cmap=cmap)

# =================================================================
# 4. MAPEO DE POSICIONES
# =================================================================

def print_mapeo(positions):
    print("─" * 65)
    print("Mapeo posición → archivo")
    print(f"  {'Pos':>4}  {'Rol':<12}  Archivo")
    print("─" * 65)
    for pos in positions:
        f_in   = get_file(idx_c2_prep, pos).stem
        f_real = get_file(idx_c2_prep, pos + 1).stem
        print(f"  {pos:>4}  input        {f_in}")
        print(f"  {pos+1:>4}  C2 target    {f_real}")
        print()
    print("─" * 65)

# =================================================================
# 5. CARGAR MODELO
# =================================================================

def load_model(device):
    model = smp.Unet(
        encoder_name='efficientnet-b0', encoder_weights=None,
        in_channels=3, classes=1, activation=None).to(device)
    ckpt = torch.load(str(CKPT_PATH), map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"Modelo cargado — epoch {ckpt['epoch']} | "
          f"val SSIM={ckpt['val_ssim']:.4f} | val PSNR={ckpt['val_psnr']:.2f} dB\n")
    return model

# =================================================================
# 6. INFERENCIA CON GROUND TRUTH
# =================================================================

def run_independiente(model, device, positions):
    print("─" * 65)
    print("MODO: Independiente (inputs reales en cada posición)")
    print("─" * 65)
    results = []
    for pos in positions:
        c2_prep   = load_npy(get_file(idx_c2_prep, pos))
        c2_diff   = load_npy(get_file(idx_c2_diff, pos))
        c3_pos    = min(pos, len(idx_c3_prep))
        c3_prep   = load_npy(get_file(idx_c3_prep, c3_pos))
        c2_real   = load_npy(get_file(idx_c2_prep, pos + 1))

        tensor    = build_input_tensor(c2_prep, c2_diff, c3_prep, device)
        pred      = predict(model, tensor)
        metrics   = compute_metrics(pred, c2_real)

        fname_in  = get_file(idx_c2_prep, pos).stem
        fname_out = get_file(idx_c2_prep, pos + 1).stem
        print(f"  Pos {pos:>3} [{fname_in[:30]}]")
        print(f"    → {pos+1:>3} [{fname_out[:30]}]")
        print(f"       SSIM={metrics['ssim']:.4f}  PSNR={metrics['psnr']:.2f} dB  "
              f"MSE={metrics['mse']:.6f}")

        # Guardar NPY y PNG
        stem = f"frame_{pos+1:03d}_indep"
        save_frame_files(pred,                         stem + '_pred',  NPY_PRED,  PNG_PRED,  'RdBu_r')
        save_frame_files(c2_real,                      stem + '_real',  NPY_REAL,  PNG_REAL,  'RdBu_r')
        save_frame_files(np.abs(pred - c2_real),       stem + '_error', NPY_ERROR, PNG_ERROR, 'Reds')

        results.append({
            'pos_in': pos, 'pos_out': pos + 1,
            'label_in': fname_in, 'label_out': fname_out,
            'c2_prep': c2_prep, 'c2_diff': c2_diff, 'c3_prep': c3_prep,
            'pred': pred, 'real': c2_real,
            'metrics': metrics, 'modo': 'independiente',
        })
    return results

def run_autoregresiva(model, device, positions):
    print("─" * 65)
    print("MODO: Autoregresiva (C2 predicho como input siguiente)")
    print("─" * 65)
    results   = []
    prev_pred = None
    for i, pos in enumerate(positions):
        c2_diff  = load_npy(get_file(idx_c2_diff, pos))
        c3_pos   = min(pos, len(idx_c3_prep))  # clamp al último C3 disponible
        c3_prep  = load_npy(get_file(idx_c3_prep, c3_pos))
        c2_real  = load_npy(get_file(idx_c2_prep, pos + 1))

        if i == 0 or prev_pred is None:
            c2_input = load_npy(get_file(idx_c2_prep, pos))
            fuente   = 'real'
        else:
            c2_input = prev_pred
            fuente   = 'predicha'

        tensor    = build_input_tensor(c2_input, c2_diff, c3_prep, device)
        pred      = predict(model, tensor)
        metrics   = compute_metrics(pred, c2_real)

        fname_in  = get_file(idx_c2_prep, pos).stem
        fname_out = get_file(idx_c2_prep, pos + 1).stem
        print(f"  Pos {pos:>3} [{fname_in[:30]}]")
        print(f"    → {pos+1:>3} [{fname_out[:30]}]")
        print(f"       SSIM={metrics['ssim']:.4f}  PSNR={metrics['psnr']:.2f} dB  "
              f"MSE={metrics['mse']:.6f}  [C2: {fuente}]")

        # Guardar NPY y PNG
        stem = f"frame_{pos+1:03d}_auto"
        save_frame_files(pred,                   stem + '_pred',  NPY_PRED,  PNG_PRED,  'RdBu_r')
        save_frame_files(c2_real,                stem + '_real',  NPY_REAL,  PNG_REAL,  'RdBu_r')
        save_frame_files(np.abs(pred - c2_real), stem + '_error', NPY_ERROR, PNG_ERROR, 'Reds')

        results.append({
            'pos_in': pos, 'pos_out': pos + 1,
            'label_in': fname_in, 'label_out': fname_out,
            'c2_prep': c2_input, 'c2_diff': c2_diff, 'c3_prep': c3_prep,
            'pred': pred, 'real': c2_real,
            'metrics': metrics, 'modo': 'autoregresiva',
            'c2_fuente': fuente,
        })
        prev_pred = pred
    return results

# =================================================================
# 6b. INFERENCIA SIN GROUND TRUTH — REGRESIÓN C2
# =================================================================

def run_sin_gt(model, device, positions, n_c2, n_diff):
    """
    Predicción encadenada para frames sin ground truth.
    - Paso 0: C2 input = último frame real. Diff = último diff real.
    - Pasos siguientes:
        C2 input = C2 predicho anterior
        Diff     = |C2_pred[N] - C2_pred[N-1]|  (diff encadenado real)
    Guarda pred, diff_pred en NPY y PNG para uso en inferencia_segmentacion
    e inferencia_regresion_C3.
    """
    print("─" * 65)
    print("MODO: Sin ground truth (predicción encadenada con diff real)")
    print("─" * 65)
    results        = []
    prev_pred      = None   # predicción del paso anterior
    prev_prev_pred = None   # predicción de dos pasos atrás (para el diff)

    for i, pos in enumerate(positions):
        # C2 input: paso 0 = último real, siguientes = pred anterior
        if i == 0:
            c2_pos   = min(pos, n_c2)
            c2_input = load_npy(get_file(idx_c2_prep, c2_pos))
            fuente   = f'real (pos {c2_pos})'
        else:
            c2_input = prev_pred
            fuente   = f'C2 predicho frame {pos}'

        # Diff encadenado:
        #   Paso 0 → último diff real disponible
        #   Paso 1 → |pred[paso 0] - C2_real_ultimo|  (pred anterior vs. input anterior)
        #   Paso 2+ → |pred[N-1] - pred[N-2]|
        if i == 0:
            diff_pos    = min(pos, n_diff)
            c2_diff     = (load_npy(get_file(idx_c2_diff, diff_pos))
                           if diff_pos >= 1
                           else np.zeros((CROP_H, CROP_W), dtype=np.float32))
            diff_fuente = f'real (pos {diff_pos})'
        else:
            # c2_input en este punto YA ES prev_pred, así que usamos prev_prev_pred
            ref_anterior = prev_prev_pred if prev_prev_pred is not None else load_npy(get_file(idx_c2_prep, min(pos-1, n_c2)))
            c2_diff      = np.abs(prev_pred - ref_anterior).astype(np.float32)
            diff_fuente  = f'encadenado |pred[{pos}] - pred[{pos-1}]|'

        # C3_prep: usar el último disponible
        c3_pos  = min(pos, n_diff)  # n_diff aproxima n_c3 en este contexto
        c3_pos  = min(c3_pos, len(idx_c3_prep))
        c3_prep = load_npy(get_file(idx_c3_prep, c3_pos))

        tensor = build_input_tensor(c2_input, c2_diff, c3_prep, device)
        pred   = predict(model, tensor)

        print(f"  Frame {pos+1:>3} (paso {i+1:>2}) | C2: {fuente} | diff: {diff_fuente}")

        # Guardar NPY y PNG — pred y diff encadenado
        stem = f"frame_{pos+1:03d}_sin_gt"
        save_frame_files(pred,    stem + '_pred',      NPY_PRED,      PNG_PRED,      'RdBu_r')
        save_frame_files(c2_diff, stem + '_diff_pred', NPY_DIFF_PRED, PNG_DIFF_PRED, 'hot')

        results.append({
            'pos_in': pos, 'pos_out': pos + 1, 'paso': i + 1,
            'label_in': fuente, 'label_out': f'frame_{pos+1}_predicho',
            'c2_prep': c2_input, 'c2_diff': c2_diff, 'c3_prep': c3_prep,
            'pred': pred, 'real': None,
            'metrics': None, 'modo': 'sin_gt',
        })
        prev_pred = pred
    return results

# =================================================================
# 7. VISUALIZACIÓN
# =================================================================

def plot_results(results, out_dir, suffix=''):
    """2 filas × 3 columnas por frame + PNG resumen."""
    for r in results:
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        fig.suptitle(
            f"C2 — Pos {r['pos_in']} → {r['pos_out']}  [{r['modo']}]\n"
            f"SSIM={r['metrics']['ssim']:.4f}  PSNR={r['metrics']['psnr']:.2f} dB  "
            f"MSE={r['metrics']['mse']:.6f}",
            fontsize=10, fontweight='bold')

        plt.subplots_adjust(hspace=0.35, wspace=0.35)

        # Fila 0 — Inputs
        fila0 = [
            (r['c2_prep'], 'C2 input\n(amplitud PFM)',    'RdBu_r'),
            (r['c2_diff'], 'C2 diff\n(cambio de frame)',  'hot'),
            (r['c3_prep'], 'C3 input\n(fase PFM)',        'twilight'),
        ]
        for col, (data, title, cmap) in enumerate(fila0):
            im = axes[0, col].imshow(data, cmap=cmap, vmin=0, vmax=1,
                                     interpolation='nearest')
            axes[0, col].set_title(title, fontsize=8)
            axes[0, col].axis('off')
            plt.colorbar(im, ax=axes[0, col], fraction=0.046, pad=0.04)
        axes[0, 0].set_ylabel('Inputs', fontsize=9, fontweight='bold', color='#888780')

        # Fila 1 — Predicción vs Real vs Error
        fila1 = [
            (r['pred'],                        f"C2 predicho\n(frame {r['pos_out']})",  'RdBu_r'),
            (r['real'],                        f"C2 real\n(frame {r['pos_out']})",      'RdBu_r'),
            (np.abs(r['pred'] - r['real']),    '|Pred − Real|\n(error)',                'Reds'),
        ]
        for col, (data, title, cmap) in enumerate(fila1):
            vmax = 0.5 if cmap == 'Reds' else 1
            im = axes[1, col].imshow(data, cmap=cmap, vmin=0, vmax=vmax,
                                     interpolation='nearest')
            axes[1, col].set_title(title, fontsize=8)
            axes[1, col].axis('off')
            plt.colorbar(im, ax=axes[1, col], fraction=0.046, pad=0.04)
        axes[1, 0].set_ylabel('Predicción vs. Real', fontsize=9,
                               fontweight='bold', color='#534AB7')

        fname = FIG_DIR / f"pos_{r['pos_in']:03d}_{r['pos_out']:03d}_{r['modo']}{suffix}.png"
        plt.savefig(str(fname), dpi=150, bbox_inches='tight')
        plt.close()

    # PNG resumen
    n = len(results)
    fig, axes = plt.subplots(3, n, figsize=(4*n, 12))
    if n == 1:
        axes = axes.reshape(3, 1)
    fig.suptitle(f"Resumen C2 [{results[0]['modo']}]", fontsize=13, fontweight='bold')
    for i, r in enumerate(results):
        axes[0, i].imshow(r['pred'], cmap='RdBu_r', vmin=0, vmax=1)
        axes[0, i].set_title(f"Pred frame {r['pos_out']}\nSSIM={r['metrics']['ssim']:.3f}",
                              fontsize=8)
        axes[0, i].axis('off')
        axes[1, i].imshow(r['real'], cmap='RdBu_r', vmin=0, vmax=1)
        axes[1, i].set_title(f"Real frame {r['pos_out']}", fontsize=8)
        axes[1, i].axis('off')
        axes[2, i].imshow(np.abs(r['pred'] - r['real']), cmap='Reds', vmin=0, vmax=0.5)
        axes[2, i].set_title(f"|Pred − Real|", fontsize=8)
        axes[2, i].axis('off')
    for row_i, (lab, col) in enumerate([('Predicho','#534AB7'),('Real','#444441'),('Error','#D85A30')]):
        axes[row_i, 0].set_ylabel(lab, fontsize=9, fontweight='bold', color=col)
    plt.tight_layout()
    plt.savefig(str(FIG_DIR / f"resumen_{results[0]['modo']}{suffix}.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Resumen guardado en: {FIG_DIR}")

def plot_comparacion(res_i, res_a, out_dir):
    n = len(res_i)
    fig, axes = plt.subplots(5, n, figsize=(4*n, 20))
    if n == 1:
        axes = axes.reshape(5, 1)
    labels = ['Independiente', 'Autoregresiva', 'Real C2', '|Indep − Real|', '|Auto − Real|']
    for row, label in enumerate(labels):
        axes[row, 0].set_ylabel(label, fontsize=10, rotation=90, labelpad=10)
    for col, (ri, ra) in enumerate(zip(res_i, res_a)):
        axes[0, col].imshow(ri['pred'], cmap='RdBu_r', vmin=0, vmax=1)
        axes[0, col].set_title(f"Frame {ri['pos_out']}\nSSIM={ri['metrics']['ssim']:.3f}", fontsize=7)
        axes[1, col].imshow(ra['pred'], cmap='RdBu_r', vmin=0, vmax=1)
        axes[1, col].set_title(f"SSIM={ra['metrics']['ssim']:.3f}", fontsize=8)
        axes[2, col].imshow(ri['real'], cmap='RdBu_r', vmin=0, vmax=1)
        axes[2, col].set_title(f"Real frame {ri['pos_out']}", fontsize=8)
        axes[3, col].imshow(np.abs(ri['pred'] - ri['real']), cmap='Reds', vmin=0, vmax=0.5)
        axes[4, col].imshow(np.abs(ra['pred'] - ra['real']), cmap='Reds', vmin=0, vmax=0.5)
        for row in range(5):
            axes[row, col].axis('off')
    plt.suptitle("Comparación C2: Independiente vs. Autoregresiva", fontsize=14)
    plt.tight_layout()
    plt.savefig(str(FIG_DIR / 'comparacion_modos.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Comparacion guardada: {FIG_DIR / 'comparacion_modos.png'}")

def plot_sin_gt(results, out_dir):
    if not results:
        return

    ultimo_pos  = max(1, results[0]['pos_in'])
    ultimo_real = load_npy(get_file(idx_c2_prep, ultimo_pos))
    lbl_ultimo  = f"C2 real frame {ultimo_pos}"

    for i, r in enumerate(results):
        pred_ant     = results[i-1]['pred'] if i > 0 else ultimo_real
        frame_ant    = results[i-1]['pos_out'] if i > 0 else ultimo_pos
        frame_actual = r['pos_out']
        frame_input  = r['pos_in']
        pred_val     = r['pred']

        fig, axes = plt.subplots(3, 3, figsize=(11, 10))
        fig.suptitle(
            f"C2 frame {frame_actual} (paso {r['paso']}) — Sin ground truth\n"
            f"Input C2: {r['label_in']}",
            fontsize=10, fontweight='bold')
        plt.subplots_adjust(hspace=0.4, wspace=0.35)

        # Fila 0 — Inputs + predicción
        fila0 = [
            (r['c2_prep'], f"C2 input\n(frame {frame_input} pred.)", 'RdBu_r'),
            (r['c2_diff'], 'C2 diff\n(encadenado)',                   'hot'),
            (pred_val,     f"C2 predicho\n(frame {frame_actual})",   'RdBu_r'),
        ]
        for col, (data, title, cmap) in enumerate(fila0):
            im = axes[0, col].imshow(data, cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
            axes[0, col].set_title(title, fontsize=8)
            axes[0, col].axis('off')
            plt.colorbar(im, ax=axes[0, col], fraction=0.046, pad=0.04)
        axes[0, 0].set_ylabel('Inputs / Pred.', fontsize=8, fontweight='bold')

        # Fila 1 — vs. último real
        fila1 = [
            (pred_val,                      f"C2 predicho\n(frame {frame_actual})",       'RdBu_r'),
            (ultimo_real,                    f"C2 real\n(frame {ultimo_pos})",             'RdBu_r'),
            (np.abs(pred_val-ultimo_real),   f"|Frame {frame_actual} pred. − Frame {ultimo_pos} real|", 'Reds'),
        ]
        for col, (data, title, cmap) in enumerate(fila1):
            vmax = 0.5 if cmap == 'Reds' else 1
            im = axes[1, col].imshow(data, cmap=cmap, vmin=0, vmax=vmax, interpolation='nearest')
            axes[1, col].set_title(title, fontsize=8)
            axes[1, col].axis('off')
            plt.colorbar(im, ax=axes[1, col], fraction=0.046, pad=0.04)
        axes[1, 0].set_ylabel('vs. Ultimo real', fontsize=8,
                               fontweight='bold', color='#D85A30')

        # Fila 2 — vs. pred anterior
        fila2 = [
            (pred_val,                     f"C2 predicho\n(frame {frame_actual})",        'RdBu_r'),
            (pred_ant,                      f"C2 predicho\n(frame {frame_ant})",          'RdBu_r'),
            (np.abs(pred_val-pred_ant),     f"|Frame {frame_actual} − Frame {frame_ant}|", 'Reds'),
        ]
        for col, (data, title, cmap) in enumerate(fila2):
            vmax = 0.5 if cmap == 'Reds' else 1
            im = axes[2, col].imshow(data, cmap=cmap, vmin=0, vmax=vmax, interpolation='nearest')
            axes[2, col].set_title(title, fontsize=8)
            axes[2, col].axis('off')
            plt.colorbar(im, ax=axes[2, col], fraction=0.046, pad=0.04)
        axes[2, 0].set_ylabel('vs. Pred. anterior', fontsize=8,
                               fontweight='bold', color='#534AB7')

        fname = FIG_DIR / f"sin_gt_frame_{r['pos_out']:03d}_paso_{r['paso']:02d}.png"
        plt.savefig(str(fname), dpi=150, bbox_inches='tight')
        plt.close()

    # PNG resumen — 4 filas: pred / pred anterior / |pred-anterior| / |pred-ultimo real|
    n = len(results)
    ncols = n + 1
    fig, axes = plt.subplots(4, ncols, figsize=(4*ncols, 16))
    fig.suptitle(
        f"C2 encadenado sin GT  frames {results[0]['pos_out']}–{results[-1]['pos_out']}\n"
        f"Col 0 = {lbl_ultimo}  |  Col 1..N = predicciones encadenadas",
        fontsize=10, fontweight='bold')

    axes[0, 0].imshow(ultimo_real, cmap='RdBu_r', vmin=0, vmax=1)
    axes[0, 0].set_title(lbl_ultimo, fontsize=8)
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')
    axes[2, 0].axis('off')
    axes[3, 0].axis('off')

    for i, r in enumerate(results):
        pred_a  = results[i-1]['pred'] if i > 0 else ultimo_real
        label_a = f"Frame {results[i-1]['pos_out']} pred." if i > 0 else lbl_ultimo

        # Fila 0 — predicción actual
        axes[0, i+1].imshow(r['pred'], cmap='RdBu_r', vmin=0, vmax=1)
        axes[0, i+1].set_title(f"Frame {r['pos_out']}\nPaso {r['paso']}", fontsize=8)
        axes[0, i+1].axis('off')

        # Fila 1 — predicción anterior (o último real para paso 1)
        axes[1, i+1].imshow(pred_a, cmap='RdBu_r', vmin=0, vmax=1)
        axes[1, i+1].set_title(label_a, fontsize=8)
        axes[1, i+1].axis('off')

        # Fila 2 — error entre pred actual y pred anterior
        axes[2, i+1].imshow(np.abs(r['pred'] - pred_a), cmap='Reds', vmin=0, vmax=0.5)
        axes[2, i+1].set_title('|Pred − Anterior|', fontsize=7)
        axes[2, i+1].axis('off')

        # Fila 3 — deriva acumulada respecto al último real conocido
        axes[3, i+1].imshow(np.abs(r['pred'] - ultimo_real), cmap='Reds', vmin=0, vmax=0.5)
        axes[3, i+1].set_title('|Pred − Último real|', fontsize=7)
        axes[3, i+1].axis('off')

    for row_i, (lab, col) in enumerate([
        ('C2 predicho',        '#534AB7'),
        ('Pred. anterior',     '#185FA5'),
        ('vs. Pred. anterior', '#1D9E75'),
        ('vs. Último real',    '#D85A30'),
    ]):
        axes[row_i, 0].set_ylabel(lab, fontsize=9, fontweight='bold', color=col)
    plt.tight_layout()
    plt.savefig(str(FIG_DIR / 'sin_gt_evolucion.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Sin GT C2: {n} frames guardados en {FIG_DIR}")

# =================================================================
# 8. GUARDAR MÉTRICAS CSV
# =================================================================

def save_metrics_csv(results, out_dir, filename):
    path = out_dir / filename
    with open(str(path), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['pos_in', 'pos_out', 'archivo_in', 'archivo_out',
                         'modo', 'ssim', 'psnr', 'mse'])
        for r in results:
            writer.writerow([r['pos_in'], r['pos_out'],
                             r['label_in'], r['label_out'], r['modo'],
                             r['metrics']['ssim'], r['metrics']['psnr'],
                             round(r['metrics']['mse'], 8)])
    print(f"  Métricas guardadas: {path}")

# =================================================================
# 9. MAIN
# =================================================================

def main():
    # Crear todas las carpetas de salida
    for d in [OUTPUT_DIR, NPY_PRED, NPY_REAL, NPY_ERROR, NPY_DIFF_PRED,
              PNG_PRED, PNG_REAL, PNG_ERROR, PNG_DIFF_PRED, FIG_DIR]:
        os.makedirs(d, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo : {device}")

    n_files = len(idx_c2_prep)
    n_diff  = len(idx_c2_diff)
    n_c3    = len(idx_c3_prep)

    if PREDICT_FROM < 2:
        print(f'ERROR: PREDICT_FROM={PREDICT_FROM} invalido. Mínimo es 2.')
        return
    if PREDICT_FROM - 1 > n_files:
        print(f'ERROR: el input requerido (pos {PREDICT_FROM-1}) no existe '
              f'(solo hay {n_files} archivos).')
        return

    positions  = list(range(PREDICT_FROM - 1, PREDICT_TO))
    pos_con_gt = [p for p in positions if p + 1 <= n_files]
    pos_sin_gt = [p for p in positions if p + 1 >  n_files]

    model = load_model(device)
    print(f'Archivos disponibles : {n_files} (C2), {n_diff} (diff), {n_c3} (C3)')
    print(f'Frames a predecir    : {PREDICT_FROM}–{PREDICT_TO}')
    print(f'Inputs               : posiciones {positions[0]}–{positions[-1]}')
    if pos_con_gt:
        print(f'Con ground truth     : frames {[p+1 for p in pos_con_gt]}')
    if pos_sin_gt:
        print(f'Sin ground truth     : frames {[p+1 for p in pos_sin_gt]} '
              f'(predicción encadenada con diff real)')
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
            res_p = run_sin_gt(model, device, pos_sin_gt, n_files, n_diff)
            plot_sin_gt(res_p, OUTPUT_DIR)

    elif INFERENCE_MODE == 'autoregresiva':
        _run_con_gt(run_autoregresiva, '', 'metricas_autoregresiva.csv')
        if pos_sin_gt:
            res_p = run_sin_gt(model, device, pos_sin_gt, n_files, n_diff)
            plot_sin_gt(res_p, OUTPUT_DIR)

    elif INFERENCE_MODE == 'ambos':
        res_i = _run_con_gt(run_independiente, '_indep', 'metricas_independiente.csv')
        print()
        res_a = _run_con_gt(run_autoregresiva, '_auto',  'metricas_autoregresiva.csv')
        if res_i and res_a:
            plot_comparacion(res_i, res_a, OUTPUT_DIR)
        if pos_sin_gt:
            res_p = run_sin_gt(model, device, pos_sin_gt, n_files, n_diff)
            plot_sin_gt(res_p, OUTPUT_DIR)

    print(f"\nResultados en: {OUTPUT_DIR}")
    print(f"NPY pred    : {NPY_PRED}")
    print(f"NPY diff    : {NPY_DIFF_PRED}  ← usar en seg y C3 para frames sin GT")

if __name__ == '__main__':
    main()