"""
inferencia_regresion_C3.py — Inferencia del modelo de regresión de fase PFM
Tesis: Evolución de los Dominios Ferroeléctricos con Deep Learning

Los archivos se indexan por POSICIÓN en la lista ordenada alfabéticamente.
Posición 1 = primer archivo, posición 2 = segundo, etc.
Al inicio muestra el mapeo explícito de posición → nombre de archivo.

Modos disponibles (INFERENCE_MODE):
  'independiente'  → cada predicción usa datos reales del frame N como input
  'autoregresiva'  → usa la C3_prep predicha del frame anterior como input C3
  'ambos'          → corre ambos modos y genera comparación lado a lado

Para frames sin ground truth (PREDICT_TO > n_archivos):
  - Usa el diff encadenado generado por inferencia_regresion_C2.py si está disponible
  - Si no, usa el último diff real disponible
  - Encadena C3 predicho como C3 input del siguiente paso

Carpetas de salida en resultados/modelo_regresion_c3/predicciones/:
  npy/pred/   <- C3 predicho (con y sin GT)
  npy/real/   <- C3 real (solo con GT)
  npy/error/  <- |pred - real| (solo con GT)
  png/pred/
  png/real/
  png/error/
  figuras/    <- PNGs de visualización por frame y resúmenes
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
# 1. CONFIGURACION
# =================================================================

BASE_DIR     = Path(r'C:\Users\migue\Desktop\modelo_predictivo')
DATA_DIR     = BASE_DIR / 'data'

DIR_C2_PREP  = DATA_DIR / 'canal_2'
DIR_C2_DIFF  = DATA_DIR / 'diff'
DIR_C3_PREP  = DATA_DIR / 'canal_3'

CKPT_PATH    = BASE_DIR / 'resultados' / 'modelo_regresion_c3' / 'checkpoints' / 'best_model.pth'
OUTPUT_DIR   = BASE_DIR / 'resultados' / 'modelo_regresion_c3' / 'predicciones'

# Diffs encadenados generados por inferencia_regresion_C2.py
DIR_DIFF_PRED = BASE_DIR / 'resultados' / 'modelo_regresion_C2' / 'predicciones' / 'npy' / 'diff_pred'

# Subcarpetas de salida
NPY_PRED  = OUTPUT_DIR / 'npy' / 'pred'
NPY_REAL  = OUTPUT_DIR / 'npy' / 'real'
NPY_ERROR = OUTPUT_DIR / 'npy' / 'error'
PNG_PRED  = OUTPUT_DIR / 'png' / 'pred'
PNG_REAL  = OUTPUT_DIR / 'png' / 'real'
PNG_ERROR = OUTPUT_DIR / 'png' / 'error'
FIG_DIR   = OUTPUT_DIR / 'figuras'

# --- MODO DE RECORTE ---
CROP_MODE   = 'cuadrado'   # 'cuadrado' | 'rectangular'
CROP_SIZE   = 80
CROP_WIDTH  = 80
CROP_HEIGHT = 64

THRESHOLD   = 0.5

# Frames a PREDECIR (base 1, orden alfabetico)
# PREDICT_FROM = 36 -> predice el frame 36 usando el frame 35 como input
# PREDICT_FROM = 41 con 40 archivos disponibles -> prediccion sin ground truth
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
# 2. INDICE POSICIONAL
# =================================================================

def build_file_index(directory, extension='.npy'):
    return sorted([f for f in Path(directory).iterdir() if f.suffix == extension])

def get_file(index_list, pos):
    if pos < 1 or pos > len(index_list):
        raise IndexError(f"Posicion {pos} fuera de rango. Disponible: 1-{len(index_list)}.")
    return index_list[pos - 1]

idx_c2_prep   = build_file_index(DIR_C2_PREP)
idx_c2_diff   = build_file_index(DIR_C2_DIFF)
idx_c3_prep   = build_file_index(DIR_C3_PREP)
idx_c3_target = build_file_index(DIR_C3_PREP)

# Indice de diffs encadenados (puede estar vacio si no se corrio C2 antes)
idx_diff_pred = build_file_index(DIR_DIFF_PRED) if DIR_DIFF_PRED.exists() else []

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

def save_array_png(arr, path_png, cmap='twilight', vmin=0, vmax=1):
    plt.imsave(str(path_png), arr, cmap=cmap, vmin=vmin, vmax=vmax)

def save_frame_files(arr, stem, npy_dir, png_dir, cmap='twilight'):
    np.save(str(npy_dir / (stem + '.npy')), arr)
    save_array_png(arr, png_dir / (stem + '.png'), cmap=cmap)

def load_diff_pred(frame_out):
    """Carga el diff encadenado de C2 para el frame indicado. None si no existe."""
    stem_buscado = f"frame_{frame_out:03d}_sin_gt_diff_pred"
    for f in idx_diff_pred:
        if f.stem == stem_buscado:
            return np.load(str(f)).astype(np.float32)
    return None

# =================================================================
# 4. MAPEO DE POSICIONES
# =================================================================

def print_mapeo(positions):
    print("─" * 65)
    print("Mapeo posicion -> archivo")
    print(f"  {'Pos':>4}  {'Rol':<12}  Archivo")
    print("─" * 65)
    for pos in positions:
        f_in   = get_file(idx_c2_prep, pos).stem
        f_real = get_file(idx_c3_prep, pos + 1).stem
        print(f"  {pos:>4}  input        {f_in}")
        print(f"  {pos+1:>4}  C3 target    {f_real}")
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
    print(f"Modelo cargado (C3) - epoch {ckpt['epoch']} | "
          f"val SSIM={ckpt['val_ssim']:.4f} | val PSNR={ckpt['val_psnr']:.2f} dB\n")
    return model

# =================================================================
# 6. INFERENCIA CON GROUND TRUTH
# =================================================================

def run_independiente(model, device, positions):
    print("─" * 65)
    print("MODO: Independiente (inputs reales en cada posicion)")
    print("─" * 65)
    results = []
    for pos in positions:
        c2_prep = load_npy(get_file(idx_c2_prep, pos))
        c2_diff = load_npy(get_file(idx_c2_diff, pos))
        c3_prep = load_npy(get_file(idx_c3_prep, pos))
        c3_real = load_npy(get_file(idx_c3_target, pos + 1))

        tensor  = build_input_tensor(c2_prep, c2_diff, c3_prep, device)
        pred    = predict(model, tensor)
        metrics = compute_metrics(pred, c3_real)

        fname_in  = get_file(idx_c2_prep, pos).stem
        fname_out = get_file(idx_c3_target, pos + 1).stem
        print(f"  Pos {pos:>3} [{fname_in[:30]}]")
        print(f"    -> {pos+1:>3} [{fname_out[:30]}]")
        print(f"       SSIM={metrics['ssim']:.4f}  PSNR={metrics['psnr']:.2f} dB  "
              f"MSE={metrics['mse']:.6f}")

        stem = f"frame_{pos+1:03d}_indep"
        save_frame_files(pred,                   stem + '_pred',  NPY_PRED,  PNG_PRED,  'twilight')
        save_frame_files(c3_real,                stem + '_real',  NPY_REAL,  PNG_REAL,  'twilight')
        save_frame_files(np.abs(pred - c3_real), stem + '_error', NPY_ERROR, PNG_ERROR, 'Reds')

        results.append({
            'pos_in': pos, 'pos_out': pos + 1,
            'label_in': fname_in, 'label_out': fname_out,
            'c2_prep': c2_prep, 'c2_diff': c2_diff, 'c3_prep': c3_prep,
            'pred': pred, 'real': c3_real,
            'metrics': metrics, 'modo': 'independiente',
        })
    return results

def run_autoregresiva(model, device, positions):
    print("─" * 65)
    print("MODO: Autoregresiva (C3 predicho como input siguiente)")
    print("─" * 65)
    results   = []
    prev_pred = None
    for i, pos in enumerate(positions):
        c2_prep = load_npy(get_file(idx_c2_prep, pos))
        c2_diff = load_npy(get_file(idx_c2_diff, pos))
        c3_real = load_npy(get_file(idx_c3_target, pos + 1))

        if i == 0 or prev_pred is None:
            c3_input = load_npy(get_file(idx_c3_prep, pos))
            fuente   = 'real'
        else:
            c3_input = prev_pred
            fuente   = 'predicha'

        tensor  = build_input_tensor(c2_prep, c2_diff, c3_input, device)
        pred    = predict(model, tensor)
        metrics = compute_metrics(pred, c3_real)

        fname_in  = get_file(idx_c2_prep, pos).stem
        fname_out = get_file(idx_c3_target, pos + 1).stem
        print(f"  Pos {pos:>3} [{fname_in[:30]}]")
        print(f"    -> {pos+1:>3} [{fname_out[:30]}]")
        print(f"       SSIM={metrics['ssim']:.4f}  PSNR={metrics['psnr']:.2f} dB  "
              f"MSE={metrics['mse']:.6f}  [C3: {fuente}]")

        stem = f"frame_{pos+1:03d}_auto"
        save_frame_files(pred,                   stem + '_pred',  NPY_PRED,  PNG_PRED,  'twilight')
        save_frame_files(c3_real,                stem + '_real',  NPY_REAL,  PNG_REAL,  'twilight')
        save_frame_files(np.abs(pred - c3_real), stem + '_error', NPY_ERROR, PNG_ERROR, 'Reds')

        results.append({
            'pos_in': pos, 'pos_out': pos + 1,
            'label_in': fname_in, 'label_out': fname_out,
            'c2_prep': c2_prep, 'c2_diff': c2_diff, 'c3_prep': c3_input,
            'pred': pred, 'real': c3_real,
            'metrics': metrics, 'modo': 'autoregresiva',
            'c3_fuente': fuente,
        })
        prev_pred = pred
    return results

# =================================================================
# 6b. INFERENCIA SIN GROUND TRUTH
# =================================================================

def run_sin_gt(model, device, positions, n_c2, n_diff):
    """
    Prediccion encadenada para frames sin ground truth.
    - C2_prep : clamp al ultimo disponible
    - C2_diff : usa diff encadenado de inferencia_regresion_C2 si existe,
                si no usa el ultimo diff real disponible
    - C3 input: paso 0 = ultimo C3 real, pasos siguientes = C3 predicho anterior
    """
    print("─" * 65)
    print("MODO: Sin ground truth (prediccion encadenada)")
    if idx_diff_pred:
        print("  Usando diffs encadenados de inferencia_regresion_C2.py")
    else:
        print("  AVISO: no se encontraron diffs de C2.")
        print("  Correr inferencia_regresion_C2.py primero para mejores resultados.")
    print("─" * 65)

    results   = []
    prev_pred = None

    for i, pos in enumerate(positions):
        # C2_prep: clamp al ultimo disponible
        c2_pos  = min(pos, n_c2)
        c2_prep = load_npy(get_file(idx_c2_prep, c2_pos))

        # C2_diff: diff encadenado de C2 si existe, sino ultimo real
        diff_pred = load_diff_pred(pos + 1)
        if diff_pred is not None:
            c2_diff     = diff_pred
            diff_fuente = f'encadenado C2 frame {pos+1}'
        else:
            diff_pos    = min(pos, n_diff)
            c2_diff     = (load_npy(get_file(idx_c2_diff, diff_pos))
                           if diff_pos >= 1
                           else np.zeros((CROP_H, CROP_W), dtype=np.float32))
            diff_fuente = f'real (pos {diff_pos})'

        # C3 input: paso 0 = ultimo real, siguientes = C3 predicho
        if i == 0:
            c3_pos   = min(pos, len(idx_c3_prep))
            c3_input = load_npy(get_file(idx_c3_prep, c3_pos))
            fuente   = f'real (pos {c3_pos})'
        else:
            c3_input = prev_pred
            fuente   = f'C3 predicho frame {pos}'

        tensor = build_input_tensor(c2_prep, c2_diff, c3_input, device)
        pred   = predict(model, tensor)

        print(f"  Frame {pos+1:>3} (paso {i+1:>2}) | C3: {fuente} | diff: {diff_fuente}")

        stem = f"frame_{pos+1:03d}_sin_gt"
        save_frame_files(pred, stem + '_pred', NPY_PRED, PNG_PRED, 'twilight')

        results.append({
            'pos_in': pos, 'pos_out': pos + 1, 'paso': i + 1,
            'label_in': fuente, 'label_out': f'frame_{pos+1}_predicho',
            'c2_prep': c2_prep, 'c2_diff': c2_diff, 'c3_prep': c3_input,
            'pred': pred, 'real': None,
            'metrics': None, 'modo': 'sin_gt',
        })
        prev_pred = pred
    return results

# =================================================================
# 7. VISUALIZACION
# =================================================================

def plot_results(results, out_dir, suffix=''):
    """2 filas x 3 columnas por frame + PNG resumen."""
    for r in results:
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        fig.suptitle(
            f"C3 - Pos {r['pos_in']} -> {r['pos_out']}  [{r['modo']}]\n"
            f"SSIM={r['metrics']['ssim']:.4f}  PSNR={r['metrics']['psnr']:.2f} dB  "
            f"MSE={r['metrics']['mse']:.6f}",
            fontsize=10, fontweight='bold')
        plt.subplots_adjust(hspace=0.35, wspace=0.35)

        c3_fuente_real = r.get('c3_fuente', 'real') == 'real'
        c3_titulo = (f"C3 input\n(frame {r['pos_in']} real)"
                     if c3_fuente_real
                     else f"C3 input\n(predicho frame {r['pos_in']})")

        fila0 = [
            (r['c2_prep'], 'C2 input\n(amplitud PFM)',   'RdBu_r'),
            (r['c2_diff'], 'C2 diff\n(cambio de frame)', 'hot'),
            (r['c3_prep'], c3_titulo,                    'twilight'),
        ]
        for col, (data, title, cmap) in enumerate(fila0):
            im = axes[0, col].imshow(data, cmap=cmap, vmin=0, vmax=1,
                                     interpolation='nearest')
            axes[0, col].set_title(title, fontsize=8)
            axes[0, col].axis('off')
            plt.colorbar(im, ax=axes[0, col], fraction=0.046, pad=0.04)
        axes[0, 0].set_ylabel('Inputs', fontsize=9, fontweight='bold', color='#888780')

        fila1 = [
            (r['pred'],                      f"C3 predicho\n(frame {r['pos_out']})", 'twilight'),
            (r['real'],                      f"C3 real\n(frame {r['pos_out']})",     'twilight'),
            (np.abs(r['pred'] - r['real']),  '|Pred - Real|\n(error)',               'Reds'),
        ]
        for col, (data, title, cmap) in enumerate(fila1):
            vmax = 0.5 if cmap == 'Reds' else 1
            im = axes[1, col].imshow(data, cmap=cmap, vmin=0, vmax=vmax,
                                     interpolation='nearest')
            axes[1, col].set_title(title, fontsize=8)
            axes[1, col].axis('off')
            plt.colorbar(im, ax=axes[1, col], fraction=0.046, pad=0.04)
        axes[1, 0].set_ylabel('Prediccion vs. Real', fontsize=9,
                               fontweight='bold', color='#0F6E56')

        fname = FIG_DIR / f"pos_{r['pos_in']:03d}_{r['pos_out']:03d}_{r['modo']}{suffix}.png"
        plt.savefig(str(fname), dpi=150, bbox_inches='tight')
        plt.close()

    n = len(results)
    fig, axes = plt.subplots(3, n, figsize=(4*n, 12))
    if n == 1:
        axes = axes.reshape(3, 1)
    fig.suptitle(f"Resumen C3 [{results[0]['modo']}]", fontsize=13, fontweight='bold')
    for i, r in enumerate(results):
        axes[0, i].imshow(r['pred'], cmap='twilight', vmin=0, vmax=1)
        axes[0, i].set_title(f"Pred frame {r['pos_out']}\nSSIM={r['metrics']['ssim']:.3f}",
                              fontsize=8)
        axes[0, i].axis('off')
        axes[1, i].imshow(r['real'], cmap='twilight', vmin=0, vmax=1)
        axes[1, i].set_title(f"Real frame {r['pos_out']}", fontsize=8)
        axes[1, i].axis('off')
        axes[2, i].imshow(np.abs(r['pred'] - r['real']), cmap='Reds', vmin=0, vmax=0.5)
        axes[2, i].set_title('|Pred - Real|', fontsize=8)
        axes[2, i].axis('off')
    for row_i, (lab, col) in enumerate([
        ('Predicho', '#0F6E56'), ('Real', '#444441'), ('Error', '#D85A30')
    ]):
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
    labels = ['Independiente', 'Autoregresiva', 'Real C3', '|Indep - Real|', '|Auto - Real|']
    for row, label in enumerate(labels):
        axes[row, 0].set_ylabel(label, fontsize=10, rotation=90, labelpad=10)
    for col, (ri, ra) in enumerate(zip(res_i, res_a)):
        axes[0, col].imshow(ri['pred'], cmap='twilight', vmin=0, vmax=1)
        axes[0, col].set_title(f"Frame {ri['pos_out']}\nSSIM={ri['metrics']['ssim']:.3f}",
                               fontsize=7)
        axes[1, col].imshow(ra['pred'], cmap='twilight', vmin=0, vmax=1)
        axes[1, col].set_title(f"SSIM={ra['metrics']['ssim']:.3f}", fontsize=8)
        axes[2, col].imshow(ri['real'], cmap='twilight', vmin=0, vmax=1)
        axes[2, col].set_title(f"Real frame {ri['pos_out']}", fontsize=8)
        axes[3, col].imshow(np.abs(ri['pred'] - ri['real']), cmap='Reds', vmin=0, vmax=0.5)
        axes[4, col].imshow(np.abs(ra['pred'] - ra['real']), cmap='Reds', vmin=0, vmax=0.5)
        for row in range(5):
            axes[row, col].axis('off')
    plt.suptitle("Comparacion C3: Independiente vs. Autoregresiva", fontsize=14)
    plt.tight_layout()
    plt.savefig(str(FIG_DIR / 'comparacion_modos.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Comparacion guardada: {FIG_DIR / 'comparacion_modos.png'}")

def plot_sin_gt(results, out_dir):
    if not results:
        return

    ultimo_pos  = max(1, results[0]['pos_in'])
    ultimo_real = load_npy(get_file(idx_c3_prep, min(ultimo_pos, len(idx_c3_prep))))
    lbl_ultimo  = f"C3 real frame {ultimo_pos}"

    for i, r in enumerate(results):
        pred_ant     = results[i-1]['pred'] if i > 0 else ultimo_real
        frame_ant    = results[i-1]['pos_out'] if i > 0 else ultimo_pos
        frame_actual = r['pos_out']
        frame_input  = r['pos_in']
        pred_val     = r['pred']

        fig, axes = plt.subplots(3, 3, figsize=(11, 10))
        fig.suptitle(
            f"C3 frame {frame_actual} (paso {r['paso']}) - Sin ground truth\n"
            f"Input C3: {r['label_in']}",
            fontsize=10, fontweight='bold')
        plt.subplots_adjust(hspace=0.4, wspace=0.35)

        fila0 = [
            (r['c2_prep'], f"C2 input\n(frame {frame_input})",         'RdBu_r'),
            (r['c2_diff'], 'C2 diff\n(encadenado o real)',               'hot'),
            (pred_val,     f"C3 predicho\n(frame {frame_actual})",      'twilight'),
        ]
        for col, (data, title, cmap) in enumerate(fila0):
            im = axes[0, col].imshow(data, cmap=cmap, vmin=0, vmax=1,
                                     interpolation='nearest')
            axes[0, col].set_title(title, fontsize=8)
            axes[0, col].axis('off')
            plt.colorbar(im, ax=axes[0, col], fraction=0.046, pad=0.04)
        axes[0, 0].set_ylabel('Inputs / Pred.', fontsize=8, fontweight='bold')

        fila1 = [
            (pred_val,                      f"C3 predicho\n(frame {frame_actual})",        'twilight'),
            (ultimo_real,                    f"C3 real\n(frame {ultimo_pos})",              'twilight'),
            (np.abs(pred_val-ultimo_real),   f"|Frame {frame_actual} pred. - Frame {ultimo_pos} real|", 'Reds'),
        ]
        for col, (data, title, cmap) in enumerate(fila1):
            vmax = 0.5 if cmap == 'Reds' else 1
            im = axes[1, col].imshow(data, cmap=cmap, vmin=0, vmax=vmax,
                                     interpolation='nearest')
            axes[1, col].set_title(title, fontsize=8)
            axes[1, col].axis('off')
            plt.colorbar(im, ax=axes[1, col], fraction=0.046, pad=0.04)
        axes[1, 0].set_ylabel('vs. Ultimo real', fontsize=8,
                               fontweight='bold', color='#D85A30')

        fila2 = [
            (pred_val,                     f"C3 predicho\n(frame {frame_actual})",        'twilight'),
            (pred_ant,                      f"C3 predicho\n(frame {frame_ant})",          'twilight'),
            (np.abs(pred_val-pred_ant),     f"|Frame {frame_actual} - Frame {frame_ant}|", 'Reds'),
        ]
        for col, (data, title, cmap) in enumerate(fila2):
            vmax = 0.5 if cmap == 'Reds' else 1
            im = axes[2, col].imshow(data, cmap=cmap, vmin=0, vmax=vmax,
                                     interpolation='nearest')
            axes[2, col].set_title(title, fontsize=8)
            axes[2, col].axis('off')
            plt.colorbar(im, ax=axes[2, col], fraction=0.046, pad=0.04)
        axes[2, 0].set_ylabel('vs. Pred. anterior', fontsize=8,
                               fontweight='bold', color='#534AB7')

        fname = FIG_DIR / f"sin_gt_frame_{r['pos_out']:03d}_paso_{r['paso']:02d}.png"
        plt.savefig(str(fname), dpi=150, bbox_inches='tight')
        plt.close()

    n = len(results)
    ncols = n + 1
    fig, axes = plt.subplots(3, ncols, figsize=(4*ncols, 12))
    fig.suptitle(
        f"C3 encadenado sin GT  frames {results[0]['pos_out']}-{results[-1]['pos_out']}\n"
        f"Col 0 = {lbl_ultimo}  |  Col 1..N = predicciones encadenadas",
        fontsize=10, fontweight='bold')
    axes[0, 0].imshow(ultimo_real, cmap='twilight', vmin=0, vmax=1)
    axes[0, 0].set_title(lbl_ultimo, fontsize=8)
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')
    axes[2, 0].axis('off')
    for i, r in enumerate(results):
        pred_a = results[i-1]['pred'] if i > 0 else ultimo_real
        axes[0, i+1].imshow(r['pred'], cmap='twilight', vmin=0, vmax=1)
        axes[0, i+1].set_title(f"Frame {r['pos_out']}\nPaso {r['paso']}", fontsize=8)
        axes[0, i+1].axis('off')
        axes[1, i+1].imshow(np.abs(r['pred']-ultimo_real), cmap='Reds', vmin=0, vmax=0.5)
        axes[1, i+1].set_title('|Pred - Ultimo real|', fontsize=7)
        axes[1, i+1].axis('off')
        axes[2, i+1].imshow(np.abs(r['pred']-pred_a), cmap='Reds', vmin=0, vmax=0.5)
        axes[2, i+1].set_title('|Pred - Anterior|', fontsize=7)
        axes[2, i+1].axis('off')
    for row_i, (lab, col) in enumerate([
        ('C3 predicho',       '#0F6E56'),
        ('vs. Ultimo real',   '#D85A30'),
        ('vs. Pred. anterior','#534AB7'),
    ]):
        axes[row_i, 0].set_ylabel(lab, fontsize=9, fontweight='bold', color=col)
    plt.tight_layout()
    plt.savefig(str(FIG_DIR / 'sin_gt_evolucion.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Sin GT C3: {n} frames guardados en {FIG_DIR}")

# =================================================================
# 8. GUARDAR METRICAS CSV
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
    print(f"  Metricas guardadas: {path}")

# =================================================================
# 9. MAIN
# =================================================================

def main():
    for d in [OUTPUT_DIR, NPY_PRED, NPY_REAL, NPY_ERROR,
              PNG_PRED, PNG_REAL, PNG_ERROR, FIG_DIR]:
        os.makedirs(d, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo : {device}")

    n_files = len(idx_c2_prep)
    n_diff  = len(idx_c2_diff)
    n_c3    = len(idx_c3_prep)

    if PREDICT_FROM < 2:
        print(f'ERROR: PREDICT_FROM={PREDICT_FROM} invalido. Minimo es 2.')
        return
    if PREDICT_FROM - 1 > n_files:
        print(f'ERROR: el input requerido (pos {PREDICT_FROM-1}) no existe '
              f'(solo hay {n_files} archivos).')
        return

    positions  = list(range(PREDICT_FROM - 1, PREDICT_TO))
    pos_con_gt = [p for p in positions if p + 1 <= n_c3]
    pos_sin_gt = [p for p in positions if p + 1 >  n_c3]

    model = load_model(device)
    print(f'Archivos disponibles : {n_files} (C2), {n_diff} (diff), {n_c3} (C3)')
    print(f'Frames a predecir    : {PREDICT_FROM}-{PREDICT_TO}')
    print(f'Inputs               : posiciones {positions[0]}-{positions[-1]}')
    if pos_con_gt:
        print(f'Con ground truth     : frames {[p+1 for p in pos_con_gt]}')
    if pos_sin_gt:
        print(f'Sin ground truth     : frames {[p+1 for p in pos_sin_gt]} '
              f'(prediccion encadenada)')
    if idx_diff_pred:
        print(f'Diffs encadenados C2 : {len(idx_diff_pred)} archivos disponibles')
    else:
        print('Diffs encadenados C2 : no disponibles - correr inferencia_regresion_C2.py primero')
    print()
    if pos_con_gt:
        print_mapeo(pos_con_gt)

    def _run_con_gt(fn_run, suffix, csv_name):
        if not pos_con_gt: return []
        res = fn_run(model, device, pos_con_gt)
        plot_results(res, FIG_DIR, suffix=suffix)
        save_metrics_csv(res, OUTPUT_DIR, csv_name)
        return res

    if INFERENCE_MODE == 'independiente':
        _run_con_gt(run_independiente, '', 'metricas_independiente.csv')
        if pos_sin_gt:
            res_p = run_sin_gt(model, device, pos_sin_gt, n_files, n_diff)
            plot_sin_gt(res_p, FIG_DIR)

    elif INFERENCE_MODE == 'autoregresiva':
        _run_con_gt(run_autoregresiva, '', 'metricas_autoregresiva.csv')
        if pos_sin_gt:
            res_p = run_sin_gt(model, device, pos_sin_gt, n_files, n_diff)
            plot_sin_gt(res_p, FIG_DIR)

    elif INFERENCE_MODE == 'ambos':
        res_i = _run_con_gt(run_independiente, '_indep', 'metricas_independiente.csv')
        print()
        res_a = _run_con_gt(run_autoregresiva, '_auto',  'metricas_autoregresiva.csv')
        if res_i and res_a:
            plot_comparacion(res_i, res_a, FIG_DIR)
        if pos_sin_gt:
            res_p = run_sin_gt(model, device, pos_sin_gt, n_files, n_diff)
            plot_sin_gt(res_p, FIG_DIR)

    print(f"\nResultados en : {OUTPUT_DIR}")
    print(f"NPY pred      : {NPY_PRED}")
    print(f"Figuras       : {FIG_DIR}")

if __name__ == '__main__':
    main()