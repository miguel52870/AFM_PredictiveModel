"""
reporte_visual.py — Reporte visual del pipeline predictivo de dominios ferroeléctricos
Tesis: Evolución de los Dominios Ferroeléctricos con Deep Learning

Lee los resultados ya generados por los scripts de inferencia y los training_logs.
NO vuelve a correr inferencia ni carga modelos.

Flujo esperado:
  1. Correr inferencia_regresion_C2.py
  2. Correr inferencia_regresion_C3.py
  3. Correr inferencia_segmentacion.py
  4. Correr este script → genera reporte_visual.pdf

Fuentes de datos:
  resultados/modelo_segmentacion/
    training_log.csv
    checkpoints/best_model.pth        ← solo para leer epoch y val_iou
    predicciones/npy/pred/            frame_NNN_indep_pred.npy
    predicciones/npy/real/            frame_NNN_indep_real.npy
    predicciones/npy/error/           frame_NNN_indep_error.npy
    predicciones/metricas_independiente.csv
    predicciones/metricas_autoregresiva.csv

  resultados/modelo_regresion_C2/
    (misma estructura, + npy/diff_pred/)
    predicciones/metricas_independiente.csv
    predicciones/metricas_autoregresiva.csv

  resultados/modelo_regresion_c3/
    (misma estructura que segmentación)

Salida: resultados/reporte_visual.pdf
"""

import os
import csv
import math
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import torch

# =================================================================
# 1. CONFIGURACIÓN
# =================================================================

BASE_DIR = Path(r'C:\Users\migue\Desktop\modelo_predictivo')
DATA_DIR = BASE_DIR / 'data'
RES_DIR  = BASE_DIR / 'resultados'

# ── Rutas de training logs ────────────────────────────────────────
SEG_LOG     = RES_DIR / 'modelo_segmentacion'  / 'training_log.csv'
REG_C2_LOG  = RES_DIR / 'modelo_regresion_C2'  / 'training_log.csv'
REG_C3_LOG  = RES_DIR / 'modelo_regresion_c3'  / 'training_log.csv'

# ── Rutas de checkpoints (solo para leer epoch y métrica) ─────────
SEG_CKPT    = RES_DIR / 'modelo_segmentacion'  / 'checkpoints' / 'best_model.pth'
REG_C2_CKPT = RES_DIR / 'modelo_regresion_C2'  / 'checkpoints' / 'best_model.pth'
REG_C3_CKPT = RES_DIR / 'modelo_regresion_c3'  / 'checkpoints' / 'best_model.pth'

# ── Rutas de métricas de inferencia ───────────────────────────────
SEG_MET_I   = RES_DIR / 'modelo_segmentacion'  / 'predicciones' / 'metricas_independiente.csv'
SEG_MET_A   = RES_DIR / 'modelo_segmentacion'  / 'predicciones' / 'metricas_autoregresiva.csv'
RC2_MET_I   = RES_DIR / 'modelo_regresion_C2'  / 'predicciones' / 'metricas_independiente.csv'
RC2_MET_A   = RES_DIR / 'modelo_regresion_C2'  / 'predicciones' / 'metricas_autoregresiva.csv'
RC3_MET_I   = RES_DIR / 'modelo_regresion_c3'  / 'predicciones' / 'metricas_independiente.csv'
RC3_MET_A   = RES_DIR / 'modelo_regresion_c3'  / 'predicciones' / 'metricas_autoregresiva.csv'

# ── Carpetas NPY de predicciones ──────────────────────────────────
SEG_NPY     = RES_DIR / 'modelo_segmentacion'  / 'predicciones' / 'npy'
RC2_NPY     = RES_DIR / 'modelo_regresion_C2'  / 'predicciones' / 'npy'
RC3_NPY     = RES_DIR / 'modelo_regresion_c3'  / 'predicciones' / 'npy'

# ── Salida ────────────────────────────────────────────────────────
OUTPUT_PDF  = RES_DIR / 'reporte_visual.pdf'

# ── Modo preferido para visualización frame a frame ───────────────
# 'independiente' | 'autoregresiva'
MODO_FRAMES = 'independiente'

# ── Colores ───────────────────────────────────────────────────────
BLUE   = '#378ADD'
ORANGE = '#D85A30'
GREEN  = '#1D9E75'
PURPLE = '#534AB7'
TEAL   = '#0F6E56'
GRAY   = '#888780'

# =================================================================
# 2. UTILIDADES DE CARGA
# =================================================================

def read_csv(path):
    """Lee un CSV y retorna lista de dicts con valores numéricos donde corresponde."""
    if not Path(path).exists():
        print(f"  AVISO: CSV no encontrado: {path}")
        return []
    rows = []
    with open(str(path), 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            parsed = {}
            for k, v in row.items():
                try:
                    parsed[k] = float(v)
                except ValueError:
                    parsed[k] = v
            rows.append(parsed)
    return rows

def read_ckpt_info(path):
    """Lee solo epoch y métrica principal del checkpoint sin cargar el modelo."""
    if not Path(path).exists():
        print(f"  AVISO: checkpoint no encontrado: {path}")
        return {}
    ckpt = torch.load(str(path), map_location='cpu')
    # Retornar todo el dict excepto model_state para no ocupar memoria
    return {k: v for k, v in ckpt.items() if k != 'model_state'}

def load_npy(path):
    """Carga NPY y normaliza a [0,1]."""
    arr = np.load(str(path)).astype(np.float32)
    mn, mx = arr.min(), arr.max()
    if mx - mn > 1e-8:
        arr = (arr - mn) / (mx - mn)
    return arr

def get_npy_files(folder, pattern='*_indep_pred.npy'):
    """Lista archivos NPY en una carpeta ordenados alfabéticamente."""
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(folder.glob(pattern))

def avg(lst, key):
    vals = [r[key] for r in lst if r.get(key) is not None and str(r.get(key)) not in ('', 'None')]
    return round(sum(vals) / len(vals), 4) if vals else None

def avg_str(lst, key, fmt='.4f'):
    v = avg(lst, key)
    return f'{v:{fmt}}' if v is not None else 'N/A'

# =================================================================
# 3. CARGAR RESULTADOS DESDE DISCO
# =================================================================

def cargar_frames_modo(seg_npy, rc2_npy, rc3_npy, modo):
    """
    Carga los arrays NPY de pred/real/error de los 3 modelos para un modo dado.
    Incluye frames con GT (sufijo _indep_ / _auto_) y sin GT (sufijo _sin_gt_).
    Retorna lista de dicts por frame, ordenada por número de frame.
    """
    sufijo = 'indep' if modo == 'independiente' else 'auto'

    def _get(folder, pattern):
        p = Path(folder)
        return sorted(p.glob(pattern)) if p.exists() else []

    # Archivos con GT
    pred_seg_gt  = _get(seg_npy/'pred',  f'*_{sufijo}_pred.npy')
    real_seg_gt  = _get(seg_npy/'real',  f'*_{sufijo}_real.npy')
    error_seg_gt = _get(seg_npy/'error', f'*_{sufijo}_error.npy')
    pred_rc2_gt  = _get(rc2_npy/'pred',  f'*_{sufijo}_pred.npy')
    real_rc2_gt  = _get(rc2_npy/'real',  f'*_{sufijo}_real.npy')
    error_rc2_gt = _get(rc2_npy/'error', f'*_{sufijo}_error.npy')
    pred_rc3_gt  = _get(rc3_npy/'pred',  f'*_{sufijo}_pred.npy')
    real_rc3_gt  = _get(rc3_npy/'real',  f'*_{sufijo}_real.npy')
    error_rc3_gt = _get(rc3_npy/'error', f'*_{sufijo}_error.npy')

    # Archivos sin GT (sufijo _sin_gt_pred.npy)
    pred_seg_sgt = _get(seg_npy/'pred',  '*_sin_gt_pred.npy')
    pred_rc2_sgt = _get(rc2_npy/'pred',  '*_sin_gt_pred.npy')
    pred_rc3_sgt = _get(rc3_npy/'pred',  '*_sin_gt_pred.npy')

    def _frame_num(path):
        try: return int(path.stem.split('_')[1])
        except: return 0

    def _load(path): return load_npy(path) if path else None

    frames = {}

    # Registrar frames con GT
    for i in range(min(len(pred_seg_gt), len(pred_rc2_gt), len(pred_rc3_gt))):
        fn = _frame_num(pred_seg_gt[i])
        frames[fn] = {
            'frame': fn, 'modo': modo, 'has_gt': True,
            'seg_pred':  _load(pred_seg_gt[i]),
            'seg_real':  _load(real_seg_gt[i])  if i < len(real_seg_gt)  else None,
            'seg_error': _load(error_seg_gt[i]) if i < len(error_seg_gt) else None,
            'rc2_pred':  _load(pred_rc2_gt[i]),
            'rc2_real':  _load(real_rc2_gt[i])  if i < len(real_rc2_gt)  else None,
            'rc2_error': _load(error_rc2_gt[i]) if i < len(error_rc2_gt) else None,
            'rc3_pred':  _load(pred_rc3_gt[i]),
            'rc3_real':  _load(real_rc3_gt[i])  if i < len(real_rc3_gt)  else None,
            'rc3_error': _load(error_rc3_gt[i]) if i < len(error_rc3_gt) else None,
        }

    # Registrar frames sin GT
    # Para comparación se usa la predicción del frame anterior (N-1 predicho)
    # Columna "real" = pred del frame anterior | Columna "error" = |pred_N - pred_{N-1}|
    for i in range(min(len(pred_seg_sgt), len(pred_rc2_sgt), len(pred_rc3_sgt))):
        fn  = _frame_num(pred_seg_sgt[i])
        fn_prev = fn - 1  # frame predicho anterior

        # Buscar predicción del frame anterior (puede ser sin_gt o indep)
        def _prev_pred(npy_dir, fn_prev):
            for pat in [f'frame_{fn_prev:03d}_sin_gt_pred.npy',
                        f'frame_{fn_prev:03d}_indep_pred.npy',
                        f'frame_{fn_prev:03d}_auto_pred.npy']:
                cands = list(Path(npy_dir / 'pred').glob(pat))
                if cands:
                    return _load(cands[0])
            return None

        seg_prev = _prev_pred(seg_npy, fn_prev)
        rc2_prev = _prev_pred(rc2_npy, fn_prev)
        rc3_prev = _prev_pred(rc3_npy, fn_prev)

        seg_cur  = _load(pred_seg_sgt[i])
        rc2_cur  = _load(pred_rc2_sgt[i])
        rc3_cur  = _load(pred_rc3_sgt[i])

        frames[fn] = {
            'frame': fn, 'modo': 'sin_gt', 'has_gt': False,
            'seg_pred':  seg_cur,
            'seg_real':  seg_prev,  # mascara predicha anterior (binaria)
            'seg_error': np.abs(seg_cur - seg_prev) if seg_cur is not None and seg_prev is not None else None,
            'rc2_pred':  rc2_cur,
            'rc2_real':  rc2_prev,
            'rc2_error': np.abs(rc2_cur - rc2_prev) if rc2_cur is not None and rc2_prev is not None else None,
            'rc3_pred':  rc3_cur,
            'rc3_real':  rc3_prev,
            'rc3_error': np.abs(rc3_cur - rc3_prev) if rc3_cur is not None and rc3_prev is not None else None,
        }

    if not frames:
        print(f"  AVISO: no se encontraron archivos para modo '{modo}'")
        return []

    result = sorted(frames.values(), key=lambda x: x['frame'])
    n_gt   = sum(1 for f in result if f['has_gt'])
    n_sgt  = sum(1 for f in result if not f['has_gt'])
    print(f"  Frames con GT: {n_gt} | Frames sin GT: {n_sgt} | Total: {len(result)}")
    return result

def cargar_inputs_frame(frame_num, data_dir, rc2_npy=None, rc3_npy=None):
    """
    Carga los inputs usados para predecir frame_num (input = frame_num - 1).
    Para frames sin GT: busca predicciones anteriores por nombre de archivo.
    Para frames con GT: usa datos reales de data/.
    Patrones buscados: _sin_gt_pred, _indep_pred, _auto_pred (en ese orden).
    """
    inp = frame_num - 1

    def _find_prev(npy_dir, fn_prev):
        """Busca la predicción del frame fn_prev en npy_dir/pred/."""
        if npy_dir is None:
            return None
        for pat in [f'frame_{fn_prev:03d}_sin_gt_pred.npy',
                    f'frame_{fn_prev:03d}_indep_pred.npy',
                    f'frame_{fn_prev:03d}_auto_pred.npy']:
            cands = list(Path(npy_dir / 'pred').glob(pat))
            if cands:
                return load_npy(cands[0])
        return None

    # ── C2 input ──────────────────────────────────────────────────
    c2_prev = _find_prev(rc2_npy, inp)
    if c2_prev is not None:
        c2_input  = c2_prev
        c2_fuente = f'C2 predicho frame {inp}'
    else:
        idx_c2   = sorted((data_dir / 'canal_2').glob('*.npy'))
        c2_input  = load_npy(idx_c2[inp-1]) if 0 < inp <= len(idx_c2) else None
        c2_fuente = f'C2 real frame {inp}'

    # ── C2_diff input ─────────────────────────────────────────────
    c2_diff = None
    if rc2_npy is not None:
        diff_cands = list(Path(rc2_npy / 'diff_pred').glob(
            f'frame_{frame_num:03d}_sin_gt_diff_pred.npy'))
        if diff_cands:
            c2_diff = load_npy(diff_cands[0])
    if c2_diff is None:
        idx_diff = sorted((data_dir / 'diff').glob('*.npy'))
        pos_diff = min(inp, len(idx_diff))
        c2_diff  = load_npy(idx_diff[pos_diff-1]) if pos_diff >= 1 else None

    # ── C3 input ──────────────────────────────────────────────────
    # Siempre usar C3 predicho continuo (rc3), nunca la máscara binaria
    c3_prev = _find_prev(rc3_npy, inp)
    if c3_prev is not None:
        c3_input  = c3_prev
        c3_fuente = f'C3 pred frame {inp}'
    else:
        idx_c3  = sorted((data_dir / 'canal_3').glob('*.npy'))
        pos_c3  = min(inp, len(idx_c3))
        c3_input  = load_npy(idx_c3[pos_c3-1]) if pos_c3 >= 1 else None
        c3_fuente = f'C3 real frame {inp}'

    return c2_input, c2_diff, c3_input, c2_fuente, c3_fuente, 'twilight'

# =================================================================
# 4. UTILIDADES DE VISUALIZACIÓN
# =================================================================

def style_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_title(title, fontsize=9, pad=5)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.2, linewidth=0.5)

def imshow_cb(ax, data, cmap, title, vmin=0, vmax=1):
    if data is None:
        ax.set_facecolor('#EEEEEE')
        ax.text(0.5, 0.5, 'no disponible', ha='center', va='center',
                fontsize=7, color=GRAY, transform=ax.transAxes)
        ax.set_title(title, fontsize=8)
        ax.axis('off')
        return
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
    ax.set_title(title, fontsize=8)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# =================================================================
# 5. PÁGINAS DEL PDF
# =================================================================

def page_portada(pdf, seg_ckpt, rc2_ckpt, rc3_ckpt):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor('white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor('white'); ax.axis('off')

    ax.text(0.5, 0.88, 'Reporte Visual — Modelos Predictivos',
            ha='center', fontsize=20, fontweight='bold', transform=ax.transAxes)
    ax.text(0.5, 0.80,
            'Estudio de la Evolución de los Dominios Ferroeléctricos al Switching\n'
            'Usando Deep Learning para Aplicación de Memorias de Estado Sólido',
            ha='center', fontsize=12, color='#5F5E5A',
            transform=ax.transAxes, linespacing=1.8)
    ax.axhline(y=0.74, xmin=0.1, xmax=0.9, color='#D3D1C7', linewidth=1)

    for i, (titulo, salida, metrica, color) in enumerate([
        ('Segmentación', 'C3_mask[N+1] — binaria',
         f"IoU val: {seg_ckpt.get('val_iou', 'N/A'):.4f}" if isinstance(seg_ckpt.get('val_iou'), float) else 'IoU val: N/A',
         BLUE),
        ('Regresión C2', 'C2_prep[N+1] — continua',
         f"SSIM val: {rc2_ckpt.get('val_ssim', 'N/A'):.4f}" if isinstance(rc2_ckpt.get('val_ssim'), float) else 'SSIM val: N/A',
         PURPLE),
        ('Regresión C3', 'C3_prep[N+1] — continua',
         f"SSIM val: {rc3_ckpt.get('val_ssim', 'N/A'):.4f}" if isinstance(rc3_ckpt.get('val_ssim'), float) else 'SSIM val: N/A',
         TEAL),
    ]):
        x0 = 0.06 + i * 0.32
        ax.add_patch(plt.Rectangle((x0, 0.44), 0.28, 0.26, transform=ax.transAxes,
                                    facecolor=f'{color}11', edgecolor=color, linewidth=1.5))
        ax.text(x0+0.14, 0.66, titulo,  ha='center', fontsize=11, fontweight='bold',
                color=color, transform=ax.transAxes)
        ax.text(x0+0.14, 0.55, salida,  ha='center', fontsize=9, color='#444441',
                transform=ax.transAxes)
        ax.text(x0+0.14, 0.48, metrica, ha='center', fontsize=9, fontweight='bold',
                color=color, transform=ax.transAxes)

    for i, (k, v) in enumerate([
        ('Inputs',       'C2_prep[N] + C2_diff[N] + C3_prep[N]'),
        ('Arquitectura', 'U-Net + EfficientNet-B0 (ImageNet pretrained)'),
        ('Dataset',      '38 pares train / 6 pares val (split cronológico)'),
        ('Reporte',      f"Modo de visualización: {MODO_FRAMES}"),
    ]):
        y = 0.34 - i * 0.048
        ax.text(0.28, y, k+':', ha='right', fontsize=9, fontweight='bold',
                color='#444441', transform=ax.transAxes)
        ax.text(0.30, y, v, ha='left', fontsize=9, color='#5F5E5A',
                transform=ax.transAxes)

    pdf.savefig(fig, bbox_inches='tight'); plt.close()


def page_curvas(pdf, seg_log, rc2_log, rc3_log):
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('Curvas de entrenamiento — 3 modelos', fontsize=14,
                 fontweight='bold', y=0.98)

    for row_i, (log, color, label, metrics) in enumerate([
        (seg_log, BLUE,   'Segmentación',
         [('train_loss','val_loss','Loss (BCE+Dice)','loss', None),
          ('train_iou', 'val_iou', 'IoU',           'IoU',  (0,1)),
          ('train_dice','val_dice','Dice',           'Dice', (0,1))]),
        (rc2_log, PURPLE, 'Regresión C2',
         [('train_loss','val_loss','Loss (MSE+SSIM)','loss', None),
          ('train_ssim','val_ssim','SSIM',           'SSIM', (0,1)),
          ('train_psnr','val_psnr','PSNR',           'dB',   None)]),
        (rc3_log, TEAL,   'Regresión C3',
         [('train_loss','val_loss','Loss (MSE+SSIM)','loss', None),
          ('train_ssim','val_ssim','SSIM',           'SSIM', (0,1)),
          ('train_psnr','val_psnr','PSNR',           'dB',   None)]),
    ]):
        if not log:
            for ax in axes[row_i]: ax.text(0.5,0.5,'Sin datos',ha='center',transform=ax.transAxes)
            continue
        epochs = [r['epoch'] for r in log]
        for col_i, (tk, vk, title, ylabel, ylim) in enumerate(metrics):
            ax = axes[row_i, col_i]
            ax.plot(epochs, [r.get(tk,0) for r in log], color=color,  lw=1.5, label='train')
            ax.plot(epochs, [r.get(vk,0) for r in log], color=ORANGE, lw=1.5, ls='--', label='val')
            style_ax(ax, f'{label} — {title}', 'epoch', ylabel)
            if ylim: ax.set_ylim(*ylim)
            vals = [r.get(vk,0) for r in log]
            best = max(vals) if 'loss' not in vk else min(vals)
            ax.axhline(y=best, color=ORANGE, ls=':', lw=1, alpha=0.5)
            ax.legend(fontsize=7)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight'); plt.close()


def page_metricas(pdf, seg_i, seg_a, rc2_i, rc2_a, rc3_i, rc3_a):
    """Gráfica de métricas por frame — independiente vs autoregresiva."""
    if not seg_i and not rc2_i:
        return

    frames_seg = [int(r['pos_out']) for r in seg_i]
    frames_rc2 = [int(r['pos_out']) for r in rc2_i]
    frames_rc3 = [int(r['pos_out']) for r in rc3_i]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle('Métricas de inferencia — independiente vs. autoregresiva',
                 fontsize=13, fontweight='bold', y=0.98)

    def plot_metric(ax, frames_i, vals_i, frames_a, vals_a, color, title, ylabel, ylim=None):
        if vals_i:
            ax.plot(frames_i, vals_i, 'o-',  color=color,  lw=2, ms=6, label='independiente')
        if vals_a:
            ax.plot(frames_a, vals_a, 's--', color=ORANGE, lw=2, ms=6, label='autoregresiva')
        ax.set_xlabel('frame predicho', fontsize=8)
        if ylim: ax.set_ylim(*ylim)
        style_ax(ax, title, 'frame predicho', ylabel)
        ax.legend(fontsize=8)

    plot_metric(axes[0,0], frames_seg, [r.get('iou')  for r in seg_i],
                           [int(r['pos_out']) for r in seg_a], [r.get('iou')  for r in seg_a],
                BLUE,   'Segmentación — IoU',  'IoU',  (0.7, 1.0))
    plot_metric(axes[0,1], frames_seg, [r.get('dice') for r in seg_i],
                           [int(r['pos_out']) for r in seg_a], [r.get('dice') for r in seg_a],
                BLUE,   'Segmentación — Dice', 'Dice', (0.8, 1.0))
    plot_metric(axes[0,2], frames_rc2, [r.get('ssim') for r in rc2_i],
                           [int(r['pos_out']) for r in rc2_a], [r.get('ssim') for r in rc2_a],
                PURPLE, 'Regresión C2 — SSIM', 'SSIM', (0.7, 1.0))
    plot_metric(axes[1,0], frames_rc3, [r.get('ssim') for r in rc3_i],
                           [int(r['pos_out']) for r in rc3_a], [r.get('ssim') for r in rc3_a],
                TEAL,   'Regresión C3 — SSIM', 'SSIM', (0.7, 1.0))
    plot_metric(axes[1,1], frames_rc2, [r.get('psnr') for r in rc2_i],
                           [int(r['pos_out']) for r in rc2_a], [r.get('psnr') for r in rc2_a],
                PURPLE, 'Regresión C2 — PSNR', 'dB')
    plot_metric(axes[1,2], frames_rc3, [r.get('psnr') for r in rc3_i],
                           [int(r['pos_out']) for r in rc3_a], [r.get('psnr') for r in rc3_a],
                TEAL,   'Regresión C3 — PSNR', 'dB')

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight'); plt.close()


def page_frame(pdf, frame_data, met_seg, met_rc2, met_rc3):
    """Visualización de un frame: inputs + 3 modelos (pred/real/error)."""
    fn = frame_data['frame']
    has_gt = frame_data['has_gt']
    zeros  = np.zeros_like(frame_data['seg_pred']) if frame_data['seg_pred'] is not None \
             else np.zeros((80, 80), dtype=np.float32)

    # Buscar métricas del frame en los CSVs
    def _met(lst, frame, key):
        for r in lst:
            if int(r.get('pos_out', 0)) == frame:
                v = r.get(key)
                return f'{float(v):.4f}' if v not in (None,'','None') else 'N/A'
        return 'N/A'

    if has_gt:
        iou_s  = _met(met_seg, fn, 'iou')
        dice_s = _met(met_seg, fn, 'dice')
        ssim2  = _met(met_rc2, fn, 'ssim')
        ssim3  = _met(met_rc3, fn, 'ssim')
        titulo = (f'Frame {fn}  [{frame_data["modo"]}]\n'
                  f'Seg IoU={iou_s}  Dice={dice_s}  |  C2 SSIM={ssim2}  |  C3 SSIM={ssim3}')
    else:
        titulo = f'Frame {fn}  [{frame_data["modo"]}]  — prediccion pura (sin ground truth)'

    # Cargar inputs usados por el modelo (real o predicho del frame anterior)
    has_sin_gt = not frame_data['has_gt']
    c2_prep, c2_diff, c3_prep, c2_fuente, c3_fuente, c3_cmap = cargar_inputs_frame(
        fn, DATA_DIR,
        rc2_npy=RC2_NPY if has_sin_gt else None,
        rc3_npy=RC3_NPY if has_sin_gt else None,
    )
    if frame_data['has_gt']:
        c2_fuente = f'C2 real frame {fn-1}'
        c3_fuente = f'C3 real frame {fn-1}'

    fig = plt.figure(figsize=(12, 16))
    fig.suptitle(titulo, fontsize=11, fontweight='bold', y=0.99)
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.5, wspace=0.35)
    first_axes = []

    # Fila 0 — Inputs (real o predicho del frame anterior según corresponda)
    for col, (data, title, cmap) in enumerate([
        (c2_prep, f'C2 prep\n({c2_fuente})',        'RdBu_r'),
        (c2_diff, 'C2 diff\n(cambio entre frames)', 'hot'),
        (c3_prep, f'C3 prep\n({c3_fuente})',        c3_cmap),
    ]):
        ax = fig.add_subplot(gs[0, col])
        if col == 0: first_axes.append(ax)
        imshow_cb(ax, data, cmap, title)

    # Fila 1 — Segmentación
    for col, (data, title, cmap, vmin, vmax) in enumerate([
        (frame_data['seg_pred'],  'Mascara predicha\n(binaria)',
         'gray', 0, 1),
        (frame_data['seg_real'],
         'Mascara real\n(Otsu)' if has_gt else f'Mask pred\nframe {fn-1}',
         'gray', 0, 1),
        (frame_data['seg_error'],
         'Error seg.\n|pred-real|' if has_gt else f'|mask{fn} - mask{fn-1}|',
         'Reds', 0, 1),
    ]):
        ax = fig.add_subplot(gs[1, col])
        if col == 0: first_axes.append(ax)
        imshow_cb(ax, data, cmap, title, vmin, vmax)

    # Fila 2 — Regresión C2
    for col, (data, title, cmap, vmin, vmax) in enumerate([
        (frame_data['rc2_pred'],
         'C2 predicho\n(amplitud)', 'RdBu_r', 0, 1),
        (frame_data['rc2_real'],
         'C2 real' if has_gt else f'C2 pred\nframe {fn-1}', 'RdBu_r', 0, 1),
        (frame_data['rc2_error'],
         'Error C2\n|pred-real|' if has_gt else f'|C2_{fn} - C2_{fn-1}|', 'Reds', 0, 0.5),
    ]):
        ax = fig.add_subplot(gs[2, col])
        if col == 0: first_axes.append(ax)
        imshow_cb(ax, data, cmap, title, vmin, vmax)

    # Fila 3 — Regresión C3
    for col, (data, title, cmap, vmin, vmax) in enumerate([
        (frame_data['rc3_pred'],
         'C3 predicho\n(fase)', 'twilight', 0, 1),
        (frame_data['rc3_real'],
         'C3 real' if has_gt else f'C3 pred\nframe {fn-1}', 'twilight', 0, 1),
        (frame_data['rc3_error'],
         'Error C3\n|pred-real|' if has_gt else f'|C3_{fn} - C3_{fn-1}|', 'Reds', 0, 0.5),
    ]):
        ax = fig.add_subplot(gs[3, col])
        if col == 0: first_axes.append(ax)
        imshow_cb(ax, data, cmap, title, vmin, vmax)

    for ax0, (label, color) in zip(first_axes, [
        (f'Input frame {fn-1}', GRAY),
        ('Segmentacion',          BLUE),
        ('Regresion C2',          PURPLE),
        ('Regresion C3',          TEAL),
    ]):
        ax0.text(-0.25, 0.5, label, ha='right', va='center', fontsize=8,
                 color=color, fontweight='bold', transform=ax0.transAxes, rotation=90)

    pdf.savefig(fig, bbox_inches='tight'); plt.close()


def page_resumen(pdf, frames_data, modelo, cmap, color):
    """Grilla pred/real/error para todos los frames de un modelo."""
    n = len(frames_data)
    if n == 0: return
    key_p = f'{modelo}_pred'
    key_r = f'{modelo}_real'
    key_e = f'{modelo}_error'

    fig, axes = plt.subplots(3, n, figsize=(4*n, 12))
    if n == 1: axes = axes.reshape(3, 1)
    fig.suptitle(f'Resumen {modelo.upper()} — predicho vs. real  [{frames_data[0]["modo"]}]',
                 fontsize=13, fontweight='bold')

    for i, fd in enumerate(frames_data):
        pred  = fd.get(key_p)
        real  = fd.get(key_r)
        error = fd.get(key_e)
        fn    = fd['frame']
        has_gt= fd['has_gt']

        axes[0,i].imshow(pred, cmap=cmap, vmin=0, vmax=1) if pred is not None \
            else axes[0,i].text(0.5,0.5,'N/A',ha='center',transform=axes[0,i].transAxes)
        axes[0,i].set_title(f'Pred frame {fn}', fontsize=8); axes[0,i].axis('off')

        ref_title = f'Real frame {fn}' if has_gt else f'Pred frame {fn-1}'
        if real is not None:
            axes[1,i].imshow(real, cmap=cmap, vmin=0, vmax=1)
            axes[1,i].set_title(ref_title, fontsize=8)
        else:
            axes[1,i].set_facecolor('#EEEEEE')
            axes[1,i].text(0.5,0.5,'no disp.',ha='center',va='center',
                           fontsize=7,color=GRAY,transform=axes[1,i].transAxes)
            axes[1,i].set_title(ref_title, fontsize=8)
        axes[1,i].axis('off')

        err_title = f'Error frame {fn}' if has_gt else f'|pred{fn}-pred{fn-1}|'
        if error is not None:
            axes[2,i].imshow(error, cmap='Reds', vmin=0, vmax=0.5)
            axes[2,i].set_title(err_title, fontsize=8)
        else:
            axes[2,i].set_facecolor('#EEEEEE')
            axes[2,i].text(0.5,0.5,'no disp.',ha='center',va='center',
                           fontsize=7,color=GRAY,transform=axes[2,i].transAxes)
            axes[2,i].set_title(err_title, fontsize=8)
        axes[2,i].axis('off')

    for row, (lab, col) in enumerate([('Prediccion',color),('Real',color),('Error',ORANGE)]):
        axes[row,0].set_ylabel(lab, fontsize=9, fontweight='bold', color=col)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight'); plt.close()


def page_error(pdf, seg_i, seg_a, rc2_i, rc2_a, rc3_i, rc3_a):
    """Análisis de error por frame y degradación autoregresiva."""
    if not seg_i and not rc2_i: return

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Análisis de error — distribución y degradación autoregresiva',
                 fontsize=13, fontweight='bold', y=0.98)

    def frames(lst): return [int(r['pos_out']) for r in lst]
    def vals(lst, k): return [r.get(k) for r in lst]

    # Fila 0 — MSE/error por frame (barras)
    for ax, lst, key, color, title, ylabel in [
        (axes[0,0], seg_i, 'iou',  BLUE,   'Segmentación — IoU por frame',   'IoU'),
        (axes[0,1], rc2_i, 'mse',  PURPLE, 'Regresión C2 — MSE por frame',   'MSE'),
        (axes[0,2], rc3_i, 'mse',  TEAL,   'Regresión C3 — MSE por frame',   'MSE'),
    ]:
        if lst:
            xs = frames(lst); ys = vals(lst, key)
            ax.bar(xs, ys, color=color, alpha=0.8, width=0.6)
            ax.set_xticks(xs)
            style_ax(ax, title, 'frame predicho', ylabel)

    # Fila 1 — degradación autoregresiva
    for ax, lst_i, lst_a, key, color, title, ylabel in [
        (axes[1,0], seg_i, seg_a, 'iou',  BLUE,   'Segmentación — degradación auto.','IoU'),
        (axes[1,1], rc2_i, rc2_a, 'ssim', PURPLE, 'Regresión C2 — degradación auto.','SSIM'),
        (axes[1,2], rc3_i, rc3_a, 'ssim', TEAL,   'Regresión C3 — degradación auto.','SSIM'),
    ]:
        if lst_i and lst_a:
            xi = frames(lst_i); yi = vals(lst_i, key)
            xa = frames(lst_a); ya = vals(lst_a, key)
            ax.plot(xi, yi, 'o-',  color=color,  lw=2, ms=7, label='independiente')
            ax.plot(xa, ya, 's--', color=ORANGE, lw=2, ms=7, label='autoregresiva')
            ax.fill_between(xi, ya[:len(xi)], yi, alpha=0.15, color=color)
            ax.set_ylim(0.7, 1.0)
            ax.set_xticks(xi)
            style_ax(ax, title, 'frame predicho', ylabel)
            ax.legend(fontsize=8)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight'); plt.close()


def page_viabilidad(pdf, seg_i, seg_a, rc2_i, rc2_a, rc3_i, rc3_a, seg_ckpt, rc2_ckpt, rc3_ckpt):
    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor('white')
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis('off')

    ax.text(0.5, 0.96, 'Resumen de viabilidad — 3 modelos predictivos BiFeO3',
            ha='center', fontsize=15, fontweight='bold', transform=ax.transAxes)
    ax.axhline(y=0.91, xmin=0.05, xmax=0.95, color='#D3D1C7', linewidth=0.8)

    def ep(ckpt, key):
        v = ckpt.get(key)
        return f'{float(v):.4f}' if isinstance(v, (int,float)) else 'N/A'

    caida = lambda vi, va, k: (
        f"{abs((avg(va,k) or 0)-(avg(vi,k) or 0))*100:.1f}%"
        if avg(vi,k) is not None and avg(va,k) is not None else 'N/A'
    )

    headers   = ['Métrica','Seg. indep.','Seg. auto.','C2 indep.','C2 auto.','C3 indep.','C3 auto.']
    rows_data = [
        ['IoU promedio',
         avg_str(seg_i,'iou'), avg_str(seg_a,'iou'), '-', '-', '-', '-'],
        ['Dice promedio',
         avg_str(seg_i,'dice'), avg_str(seg_a,'dice'), '-', '-', '-', '-'],
        ['SSIM promedio', '-', '-',
         avg_str(rc2_i,'ssim'), avg_str(rc2_a,'ssim'),
         avg_str(rc3_i,'ssim'), avg_str(rc3_a,'ssim')],
        ['PSNR promedio (dB)', '-', '-',
         avg_str(rc2_i,'psnr','.2f'), avg_str(rc2_a,'psnr','.2f'),
         avg_str(rc3_i,'psnr','.2f'), avg_str(rc3_a,'psnr','.2f')],
        ['Mejor epoch (train)',
         str(int(seg_ckpt.get('epoch',0))) if seg_ckpt.get('epoch') else 'N/A', '-',
         str(int(rc2_ckpt.get('epoch',0))) if rc2_ckpt.get('epoch') else 'N/A', '-',
         str(int(rc3_ckpt.get('epoch',0))) if rc3_ckpt.get('epoch') else 'N/A', '-'],
        ['Val métrica (train)',
         ep(seg_ckpt,'val_iou'), '-',
         ep(rc2_ckpt,'val_ssim'), '-',
         ep(rc3_ckpt,'val_ssim'), '-'],
        ['Caída autoregresiva',
         caida(seg_i,seg_a,'iou'), '-',
         caida(rc2_i,rc2_a,'ssim'), '-',
         caida(rc3_i,rc3_a,'ssim'), '-'],
    ]

    col_x   = [0.03,0.22,0.34,0.46,0.58,0.70,0.82]
    row_y0  = 0.86; row_h = 0.058
    hcolors = ['#444441',BLUE,BLUE,PURPLE,PURPLE,TEAL,TEAL]

    for j,(cx,hc) in enumerate(zip(col_x,hcolors)):
        ax.add_patch(plt.Rectangle((cx,row_y0),0.13,0.05,
                     transform=ax.transAxes,facecolor=hc,zorder=2))
        ax.text(cx+0.065,row_y0+0.025,headers[j],ha='center',va='center',
                fontsize=8,fontweight='bold',color='white',transform=ax.transAxes)

    for i,row in enumerate(rows_data):
        y  = row_y0-(i+1)*row_h
        bg = '#F1EFE8' if i%2==0 else 'white'
        ax.add_patch(plt.Rectangle((0.02,y),0.96,row_h-0.004,
                     transform=ax.transAxes,facecolor=bg,zorder=1))
        colors = ['#444441',BLUE,BLUE,PURPLE,PURPLE,TEAL,TEAL]
        for j,(cx,val,tc) in enumerate(zip(col_x,row,colors)):
            ax.text(cx+0.065,y+row_h/2-0.004,val,ha='center',va='center',
                    fontsize=8.5,color=tc,transform=ax.transAxes,
                    fontweight='bold' if j==0 else 'normal')

    conclusiones = [
        (BLUE,   f"Segmentación: IoU {avg_str(seg_i,'iou')} (indep.) — predice la distribución de dominios con alta precisión espacial."),
        (PURPLE, f"Regresión C2: SSIM {avg_str(rc2_i,'ssim')} (indep.) — reconstruye la amplitud PFM con buena similitud estructural."),
        (TEAL,   f"Regresión C3: SSIM {avg_str(rc3_i,'ssim')} (indep.) — reconstruye la fase PFM continua, complementando la máscara binaria."),
        (ORANGE, f"Degradación autoregresiva: Seg. {caida(seg_i,seg_a,'iou')} | C2 {caida(rc2_i,rc2_a,'ssim')} | C3 {caida(rc3_i,rc3_a,'ssim')} — acumulación de error esperada."),
        (GREEN,  "Hipótesis validada: los patrones de evolución de dominios ferroeléctricos en BiFeO3 son identificables y predecibles mediante deep learning."),
    ]
    ax.text(0.5,0.44,'Conclusiones',ha='center',fontsize=12,fontweight='bold',transform=ax.transAxes)
    for i,(color,texto) in enumerate(conclusiones):
        ax.text(0.03,0.38-i*0.072,f'• {texto}',ha='left',va='top',
                fontsize=9,color=color,transform=ax.transAxes)

    pdf.savefig(fig,bbox_inches='tight'); plt.close()


# =================================================================
# 6. MAIN
# =================================================================

def main():
    print("Reporte visual — leyendo resultados de inferencia...")
    print(f"  No se cargan modelos. No se re-ejecuta inferencia.")

    # ── Verificar que existen los resultados ─────────────────────
    for p, nombre in [
        (SEG_MET_I, 'metricas seg independiente'),
        (RC2_MET_I, 'metricas C2 independiente'),
        (RC3_MET_I, 'metricas C3 independiente'),
    ]:
        if not Path(p).exists():
            print(f"\n  ERROR: {nombre} no encontrado en:\n    {p}")
            print("  Correr primero los scripts de inferencia.")
            return

    # ── Leer training logs ────────────────────────────────────────
    print("\nLeyendo training logs...")
    seg_log  = read_csv(SEG_LOG)
    rc2_log  = read_csv(REG_C2_LOG)
    rc3_log  = read_csv(REG_C3_LOG)
    print(f"  Seg: {len(seg_log)} epochs | C2: {len(rc2_log)} epochs | C3: {len(rc3_log)} epochs")

    # ── Leer info de checkpoints ──────────────────────────────────
    print("\nLeyendo info de checkpoints...")
    seg_ckpt = read_ckpt_info(SEG_CKPT)
    rc2_ckpt = read_ckpt_info(REG_C2_CKPT)
    rc3_ckpt = read_ckpt_info(REG_C3_CKPT)
    print(f"  Seg  epoch={seg_ckpt.get('epoch','?')} val_iou={seg_ckpt.get('val_iou','?')}")
    print(f"  C2   epoch={rc2_ckpt.get('epoch','?')} val_ssim={rc2_ckpt.get('val_ssim','?')}")
    print(f"  C3   epoch={rc3_ckpt.get('epoch','?')} val_ssim={rc3_ckpt.get('val_ssim','?')}")

    # ── Leer métricas de inferencia ───────────────────────────────
    print("\nLeyendo métricas de inferencia...")
    seg_i = read_csv(SEG_MET_I);  seg_a = read_csv(SEG_MET_A)
    rc2_i = read_csv(RC2_MET_I);  rc2_a = read_csv(RC2_MET_A)
    rc3_i = read_csv(RC3_MET_I);  rc3_a = read_csv(RC3_MET_A)
    print(f"  Seg  indep={len(seg_i)} frames | auto={len(seg_a)} frames")
    print(f"  C2   indep={len(rc2_i)} frames | auto={len(rc2_a)} frames")
    print(f"  C3   indep={len(rc3_i)} frames | auto={len(rc3_a)} frames")

    # ── Cargar arrays NPY de predicciones ────────────────────────
    print(f"\nCargando arrays NPY ({MODO_FRAMES})...")
    frames_data = cargar_frames_modo(SEG_NPY, RC2_NPY, RC3_NPY, MODO_FRAMES)
    print(f"  {len(frames_data)} frames cargados")
    if not frames_data:
        print("  ERROR: no se encontraron archivos NPY de predicciones.")
        print(f"  Verificar carpetas en: {SEG_NPY}")
        return

    # ── Generar PDF ───────────────────────────────────────────────
    os.makedirs(str(RES_DIR), exist_ok=True)
    print(f"\nGenerando PDF: {OUTPUT_PDF}")

    with PdfPages(str(OUTPUT_PDF)) as pdf:
        print("  Portada...");        page_portada(pdf, seg_ckpt, rc2_ckpt, rc3_ckpt)
        print("  Curvas...");         page_curvas(pdf, seg_log, rc2_log, rc3_log)
        print("  Métricas...");       page_metricas(pdf, seg_i, seg_a, rc2_i, rc2_a, rc3_i, rc3_a)

        for i, fd in enumerate(frames_data):
            print(f"  Frame {fd['frame']}...")
            page_frame(pdf, fd, seg_i, rc2_i, rc3_i)

        print("  Resumen segmentación...")
        page_resumen(pdf, frames_data, 'seg', 'gray', BLUE)

        print("  Resumen C2...")
        page_resumen(pdf, frames_data, 'rc2', 'RdBu_r', PURPLE)

        print("  Resumen C3...")
        page_resumen(pdf, frames_data, 'rc3', 'twilight', TEAL)

        print("  Análisis de error...")
        page_error(pdf, seg_i, seg_a, rc2_i, rc2_a, rc3_i, rc3_a)

        print("  Tabla de viabilidad...")
        page_viabilidad(pdf, seg_i, seg_a, rc2_i, rc2_a, rc3_i, rc3_a,
                        seg_ckpt, rc2_ckpt, rc3_ckpt)

    print(f"\nReporte generado: {OUTPUT_PDF}")

if __name__ == '__main__':
    main()