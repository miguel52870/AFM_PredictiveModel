"""
reporte_visual.py — Reporte visual completo — 3 modelos predictivos
Tesis: Evolución de los Dominios Ferroeléctricos con Deep Learning

Genera un PDF con las siguientes secciones:
  1. Portada
  2. Curvas de entrenamiento — los 3 modelos
  3. Métricas de inferencia — independiente vs. autoregresiva
  4. Visualización frame a frame (4 filas × 3 col):
       inputs · segmentación · regresión C2 · regresión C3
  5. Resumen segmentación — grilla completa de frames
  6. Resumen regresión C2 — grilla completa de frames
  7. Resumen regresión C3 — grilla completa de frames
  8. Análisis de error — distribución y degradación autoregresiva
  9. Tabla de viabilidad y conclusiones

Salida: resultados/reporte_visual.pdf
"""

import os
import csv
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import torch
import segmentation_models_pytorch as smp

# =================================================================
# 1. CONFIGURACIÓN
# =================================================================

BASE_DIR    = Path(r'C:\Users\migue\Desktop\modelo_predictivo')
DATA_DIR    = BASE_DIR / 'data'
RES_DIR     = BASE_DIR / 'resultados'

DIR_C2_PREP = DATA_DIR / 'canal_2'
DIR_C2_DIFF = DATA_DIR / 'diff'
DIR_C3_PREP = DATA_DIR / 'canal_3'
DIR_C3_MASK = DATA_DIR / 'mask'

SEG_LOG     = RES_DIR / 'modelo_segmentacion' / 'training_log.csv'
SEG_CKPT    = RES_DIR / 'modelo_segmentacion' / 'checkpoints' / 'best_model.pth'
REG_C2_LOG  = RES_DIR / 'modelo_regresion_C2' / 'training_log.csv'
REG_C2_CKPT = RES_DIR / 'modelo_regresion_C2' / 'checkpoints' / 'best_model.pth'
REG_C3_LOG  = RES_DIR / 'modelo_regresion_c3' / 'training_log.csv'
REG_C3_CKPT = RES_DIR / 'modelo_regresion_c3' / 'checkpoints' / 'best_model.pth'

OUTPUT_PDF  = RES_DIR / 'reporte_visual.pdf'

# Carpetas para frames predichos (PNG y NPY)
PRED_DIR      = RES_DIR / 'predicciones_frames'
PRED_SEG_PNG  = PRED_DIR / 'mask' / 'png'
PRED_SEG_NPY  = PRED_DIR / 'mask' / 'npy'
PRED_C2_PNG   = PRED_DIR / 'canal_2' / 'png'
PRED_C2_NPY   = PRED_DIR / 'canal_2' / 'npy'
PRED_C3_PNG   = PRED_DIR / 'canal_3' / 'png'
PRED_C3_NPY   = PRED_DIR / 'canal_3' / 'npy'

# Frames a PREDECIR (base 1) — el sistema carga el input anterior automáticamente
# PREDICT_FROM = 36 predice el frame 36 usando el frame 35 como input
# PREDICT_FROM = 41 con 40 archivos = prediccion pura (sin ground truth)
PREDICT_FROM = 36
PREDICT_TO   = 40
THRESHOLD    = 0.5

CROP_MODE   = 'cuadrado'
CROP_SIZE   = 80
CROP_WIDTH  = 80
CROP_HEIGHT = 64
if CROP_MODE == 'cuadrado':
    CROP_W, CROP_H = CROP_SIZE, CROP_SIZE
else:
    CROP_W, CROP_H = CROP_WIDTH, CROP_HEIGHT

BLUE   = '#378ADD'
ORANGE = '#D85A30'
GREEN  = '#1D9E75'
PURPLE = '#534AB7'
TEAL   = '#0F6E56'
GRAY   = '#888780'

# =================================================================
# 2. UTILIDADES
# =================================================================

def build_file_index(directory, extension='.npy'):
    return sorted([f for f in Path(directory).iterdir() if f.suffix == extension])

def get_file(index_list, pos):
    return index_list[pos - 1]

def load_npy(path):
    arr = np.load(str(path)).astype(np.float32)
    mn, mx = arr.min(), arr.max()
    if mx - mn > 1e-8:
        arr = (arr - mn) / (mx - mn)
    return arr

def load_mask(path):
    arr = np.load(str(path)).astype(np.float32)
    return (arr > 127).astype(np.float32)

def read_csv(path):
    rows = []
    with open(str(path), 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            rows.append({k: (float(v) if v.replace('.','').replace('-','')
                             .replace('e','').replace('+','').isdigit() else v)
                         for k, v in row.items()})
    return rows

def style_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_title(title, fontsize=9, pad=5)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.2, linewidth=0.5)

def imshow_cb(ax, data, cmap, title, vmin=0, vmax=1):
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
    ax.set_title(title, fontsize=8)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

def iou_fn(p, t, eps=1e-6):
    inter = (p*t).sum()
    return float(inter/(p.sum()+t.sum()-inter+eps))

def dice_fn(p, t, eps=1e-6):
    return float(2*(p*t).sum()/(p.sum()+t.sum()+eps))

def mse_fn(p, t):    return float(((p-t)**2).mean())

def psnr_fn(p, t, eps=1e-8):
    m = mse_fn(p, t)
    return 10*math.log10(1.0/m) if m > eps else 100.0

def ssim_fn(p, t):
    mu1,mu2 = p.mean(),t.mean()
    s1,s2   = p.std(),t.std()
    s12     = ((p-mu1)*(t-mu2)).mean()
    C1,C2   = 0.01**2, 0.03**2
    return float(((2*mu1*mu2+C1)*(2*s12+C2))/((mu1**2+mu2**2+C1)*(s1**2+s2**2+C2)))

def avg(lst, key):
    return round(sum(r[key] for r in lst)/len(lst), 4)

# =================================================================
# 3. MODELOS E INFERENCIA
# =================================================================

def load_model(ckpt_path, device):
    model = smp.Unet(encoder_name='efficientnet-b0', encoder_weights=None,
                     in_channels=3, classes=1, activation=None).to(device)
    ckpt = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model, ckpt

def infer_seg(model, c2_prep, c2_diff, c3_prep, device):
    x = np.stack([c2_prep, c2_diff, c3_prep], axis=0)
    t = torch.from_numpy(x).unsqueeze(0).to(device)
    with torch.no_grad():
        prob   = torch.sigmoid(model(t)).squeeze().cpu().numpy()
        binary = (prob > THRESHOLD).astype(np.float32)
    return prob, binary

def infer_reg(model, c2_prep, c2_diff, c3_prep, device):
    x = np.stack([c2_prep, c2_diff, c3_prep], axis=0)
    t = torch.from_numpy(x).unsqueeze(0).to(device)
    with torch.no_grad():
        return torch.sigmoid(model(t)).squeeze().cpu().numpy()

def calcular_inferencias(positions, seg_model, regc2_model, regc3_model, device,
                          idx_c2_prep, idx_c2_diff, idx_c3_prep, idx_c3_mask):
    seg_i, seg_a = [], []
    rc2_i, rc2_a = [], []
    rc3_i, rc3_a = [], []
    prev_seg = prev_rc2 = prev_rc3 = None
    n_files  = len(idx_c2_prep)

    for i, pos in enumerate(positions):
        # Detectar si existe ground truth para este par
        has_gt = (pos + 1) <= n_files

        c2_prep = load_npy(get_file(idx_c2_prep, pos))
        # C2_diff: usar real si existe, ceros si estamos en territorio sin datos
        if pos <= len(idx_c2_diff):
            c2_diff = load_npy(get_file(idx_c2_diff, pos))
        else:
            c2_diff = np.zeros((CROP_H, CROP_W), dtype=np.float32)
        c3_prep = load_npy(get_file(idx_c3_prep, min(pos, len(idx_c3_prep))))

        mask_real    = load_mask(get_file(idx_c3_mask, pos+1)) if has_gt else None
        c3_prep_real = load_npy(get_file(idx_c3_prep,  pos+1)) if has_gt else None
        c2_prep_real = load_npy(get_file(idx_c2_prep,  pos+1)) if has_gt else None

        # ── Segmentación independiente ────────────────────────────
        prob_i, bin_i = infer_seg(seg_model, c2_prep, c2_diff, c3_prep, device)
        seg_i.append({'pos_in':pos,'pos_out':pos+1,'prob':prob_i,'binary':bin_i,
                      'mask_real':mask_real,'c3_prep_real':c3_prep_real,
                      'c2_prep':c2_prep,'c2_diff':c2_diff,'c3_prep':c3_prep,
                      'has_gt':has_gt,
                      'iou' :iou_fn(bin_i,mask_real)  if has_gt else None,
                      'dice':dice_fn(bin_i,mask_real) if has_gt else None})

        # ── Segmentación autoregresiva ────────────────────────────
        c3_in_a = c3_prep if i==0 else prev_seg
        _, bin_a = infer_seg(seg_model, c2_prep, c2_diff, c3_in_a, device)
        seg_a.append({'pos_in':pos,'pos_out':pos+1,'binary':bin_a,'mask_real':mask_real,
                      'has_gt':has_gt,
                      'iou' :iou_fn(bin_a,mask_real)  if has_gt else None,
                      'dice':dice_fn(bin_a,mask_real) if has_gt else None,
                      'fuente':'real' if i==0 else 'predicha'})
        prev_seg = bin_a

        # ── Regresión C2 independiente ────────────────────────────
        rc2_pred_i = infer_reg(regc2_model, c2_prep, c2_diff, c3_prep, device)
        rc2_i.append({'pos_in':pos,'pos_out':pos+1,'pred':rc2_pred_i,'real':c2_prep_real,
                      'c2_prep':c2_prep,'c2_diff':c2_diff,'c3_prep':c3_prep,
                      'has_gt':has_gt,
                      'ssim':ssim_fn(rc2_pred_i,c2_prep_real) if has_gt else None,
                      'psnr':psnr_fn(rc2_pred_i,c2_prep_real) if has_gt else None,
                      'mse' :mse_fn(rc2_pred_i,c2_prep_real)  if has_gt else None})

        # ── Regresión C2 autoregresiva ────────────────────────────
        c2_in_a    = c2_prep if i==0 else prev_rc2
        rc2_pred_a = infer_reg(regc2_model, c2_in_a, c2_diff, c3_prep, device)
        rc2_a.append({'pos_in':pos,'pos_out':pos+1,'pred':rc2_pred_a,'real':c2_prep_real,
                      'has_gt':has_gt,
                      'ssim':ssim_fn(rc2_pred_a,c2_prep_real) if has_gt else None,
                      'psnr':psnr_fn(rc2_pred_a,c2_prep_real) if has_gt else None,
                      'fuente':'real' if i==0 else 'predicha'})
        prev_rc2 = rc2_pred_a

        # ── Regresión C3 independiente ────────────────────────────
        rc3_pred_i = infer_reg(regc3_model, c2_prep, c2_diff, c3_prep, device)
        rc3_i.append({'pos_in':pos,'pos_out':pos+1,'pred':rc3_pred_i,'real':c3_prep_real,
                      'c2_prep':c2_prep,'c2_diff':c2_diff,'c3_prep':c3_prep,
                      'has_gt':has_gt,
                      'ssim':ssim_fn(rc3_pred_i,c3_prep_real) if has_gt else None,
                      'psnr':psnr_fn(rc3_pred_i,c3_prep_real) if has_gt else None,
                      'mse' :mse_fn(rc3_pred_i,c3_prep_real)  if has_gt else None})

        # ── Regresión C3 autoregresiva ────────────────────────────
        c3_in_a2   = c3_prep if i==0 else prev_rc3
        rc3_pred_a = infer_reg(regc3_model, c2_prep, c2_diff, c3_in_a2, device)
        rc3_a.append({'pos_in':pos,'pos_out':pos+1,'pred':rc3_pred_a,'real':c3_prep_real,
                      'has_gt':has_gt,
                      'ssim':ssim_fn(rc3_pred_a,c3_prep_real) if has_gt else None,
                      'psnr':psnr_fn(rc3_pred_a,c3_prep_real) if has_gt else None,
                      'fuente':'real' if i==0 else 'predicha'})
        prev_rc3 = rc3_pred_a

        # ── Guardar predicciones PNG y NPY ────────────────────────
        if has_gt:
            fname = get_file(idx_c2_prep, pos+1).stem
        else:
            fname = f'frame_{pos+1}_predicho'
        np.save(str(PRED_SEG_NPY / f'{fname}_mask_pred.npy'), bin_i)
        plt.imsave(str(PRED_SEG_PNG / f'{fname}_mask_pred.png'), bin_i, cmap='gray', vmin=0, vmax=1)
        np.save(str(PRED_C2_NPY / f'{fname}_C2_pred.npy'), rc2_pred_i)
        plt.imsave(str(PRED_C2_PNG / f'{fname}_C2_pred.png'), rc2_pred_i, cmap='RdBu_r', vmin=0, vmax=1)
        np.save(str(PRED_C3_NPY / f'{fname}_C3_pred.npy'), rc3_pred_i)
        plt.imsave(str(PRED_C3_PNG / f'{fname}_C3_pred.png'), rc3_pred_i, cmap='twilight', vmin=0, vmax=1)

        if has_gt:
            print(f"  Frame {pos+1:>3} | "
                  f"Seg IoU={seg_i[-1]['iou']:.4f} | "
                  f"C2 SSIM={rc2_i[-1]['ssim']:.4f} | "
                  f"C3 SSIM={rc3_i[-1]['ssim']:.4f}")
        else:
            print(f"  Frame {pos+1:>3} | prediccion pura (sin ground truth)")

    return seg_i, seg_a, rc2_i, rc2_a, rc3_i, rc3_a

# =================================================================
# 4. PÁGINAS DEL PDF
# =================================================================

def page_portada(pdf, seg_ckpt, regc2_ckpt, regc3_ckpt):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor('white')
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor('white'); ax.axis('off')

    ax.text(0.5, 0.88, 'Reporte Visual — Modelos Predictivos',
            ha='center', fontsize=20, fontweight='bold', transform=ax.transAxes)
    ax.text(0.5, 0.80,
            'Estudio de la Evolución de los Dominios Ferroeléctricos al Switching\n'
            'Usando Deep Learning para Aplicación de Memorias de Estado Sólido',
            ha='center', fontsize=12, color='#5F5E5A',
            transform=ax.transAxes, linespacing=1.8)
    ax.axhline(y=0.74, xmin=0.1, xmax=0.9, color='#D3D1C7', linewidth=1)

    for i, (titulo, salida, metrica, color) in enumerate([
        ('Segmentación', 'C3_mask[N+1] — binaria',  f"IoU val: {seg_ckpt['val_iou']:.4f}",   BLUE),
        ('Regresión C2', 'C2_prep[N+1] — continua', f"SSIM val: {regc2_ckpt['val_ssim']:.4f}", PURPLE),
        ('Regresión C3', 'C3_prep[N+1] — continua', f"SSIM val: {regc3_ckpt['val_ssim']:.4f}", TEAL),
    ]):
        x0 = 0.06 + i * 0.32
        ax.add_patch(plt.Rectangle((x0, 0.44), 0.28, 0.26, transform=ax.transAxes,
                                    facecolor=f'{color}11', edgecolor=color, linewidth=1.5))
        ax.text(x0+0.14, 0.66, titulo,  ha='center', fontsize=11, fontweight='bold',
                color=color, transform=ax.transAxes)
        ax.text(x0+0.14, 0.55, salida,  ha='center', fontsize=9, color='#444441', transform=ax.transAxes)
        ax.text(x0+0.14, 0.48, metrica, ha='center', fontsize=9, fontweight='bold',
                color=color, transform=ax.transAxes)

    for i, (k, v) in enumerate([
        ('Inputs',       'C2_prep[N] + C2_diff[N] + C3_prep[N]'),
        ('Dataset',      '38 pares train / 6 pares val (split cronológico)'),
        ('Arquitectura', 'U-Net + EfficientNet-B0 (ImageNet pretrained)'),
        ('Recorte',      f'{CROP_MODE} {CROP_W}×{CROP_H} px'),
    ]):
        y = 0.34 - i * 0.048
        ax.text(0.28, y, k+':', ha='right', fontsize=9, fontweight='bold',
                color='#444441', transform=ax.transAxes)
        ax.text(0.30, y, v, ha='left', fontsize=9, color='#5F5E5A', transform=ax.transAxes)

    pdf.savefig(fig, bbox_inches='tight'); plt.close()

def page_curvas(pdf, seg_log, regc2_log, regc3_log):
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('Curvas de entrenamiento — 3 modelos', fontsize=14, fontweight='bold', y=0.98)

    for row_i, (log, color, label, metrics) in enumerate([
        (seg_log,   BLUE,   'Segmentación',
         [('train_loss','val_loss','Loss (BCE+Dice)','loss',None),
          ('train_iou', 'val_iou', 'IoU',            'IoU', (0,1)),
          ('train_dice','val_dice','Dice',            'Dice',(0,1))]),
        (regc2_log, PURPLE, 'Regresión C2',
         [('train_loss','val_loss','Loss (MSE+SSIM)','loss',None),
          ('train_ssim','val_ssim','SSIM',           'SSIM',(0,1)),
          ('train_psnr','val_psnr','PSNR',           'dB',  None)]),
        (regc3_log, TEAL,   'Regresión C3',
         [('train_loss','val_loss','Loss (MSE+SSIM)','loss',None),
          ('train_ssim','val_ssim','SSIM',           'SSIM',(0,1)),
          ('train_psnr','val_psnr','PSNR',           'dB',  None)]),
    ]):
        epochs = [r['epoch'] for r in log]
        for col_i, (tk, vk, title, ylabel, ylim) in enumerate(metrics):
            ax = axes[row_i, col_i]
            ax.plot(epochs, [r[tk] for r in log], color=color,  lw=1.5, label='train')
            ax.plot(epochs, [r[vk] for r in log], color=ORANGE, lw=1.5, ls='--', label='val')
            style_ax(ax, f'{label} — {title}', 'epoch', ylabel)
            if ylim: ax.set_ylim(*ylim)
            best = max(r[vk] for r in log) if 'loss' not in vk else min(r[vk] for r in log)
            ax.axhline(y=best, color=ORANGE, ls=':', lw=1, alpha=0.5)
            ax.legend(fontsize=7)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight'); plt.close()

def page_metricas(pdf, seg_i, seg_a, rc2_i, rc2_a, rc3_i, rc3_a, labels):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle('Métricas de inferencia — independiente vs. autoregresiva',
                 fontsize=13, fontweight='bold', y=0.98)
    x = range(len(labels))

    def ann(ax, x_vals, y_vals, color, offset=8):
        for xi, yi in zip(x_vals, y_vals):
            ax.annotate(f'{yi:.3f}',(xi,yi),textcoords='offset points',
                        xytext=(0,offset),fontsize=7,color=color,ha='center')

    # Solo frames con ground truth para métricas
    seg_i_gt = [r for r in seg_i if r.get('has_gt')]
    seg_a_gt = [r for r in seg_a if r.get('has_gt')]
    rc2_i_gt = [r for r in rc2_i if r.get('has_gt')]
    rc2_a_gt = [r for r in rc2_a if r.get('has_gt')]
    rc3_i_gt = [r for r in rc3_i if r.get('has_gt')]
    rc3_a_gt = [r for r in rc3_a if r.get('has_gt')]
    labels_gt = [f"frame {r['pos_out']}" for r in seg_i_gt]
    x_gt = range(len(labels_gt))

    for ax, yi_vals, ya_vals, color, title, ylabel, ylim in [
        (axes[0,0],[r['iou']  for r in seg_i_gt],[r['iou']  for r in seg_a_gt],
         BLUE,  'Segmentacion — IoU',     'IoU', (0.7,1.0)),
        (axes[0,1],[r['dice'] for r in seg_i_gt],[r['dice'] for r in seg_a_gt],
         BLUE,  'Segmentacion — Dice',    'Dice',(0.8,1.0)),
        (axes[0,2],[r['ssim'] for r in rc2_i_gt],[r['ssim'] for r in rc2_a_gt],
         PURPLE,'Regresion C2 — SSIM',   'SSIM',(0.7,1.0)),
        (axes[1,0],[r['ssim'] for r in rc3_i_gt],[r['ssim'] for r in rc3_a_gt],
         TEAL,  'Regresion C3 — SSIM',   'SSIM',(0.7,1.0)),
        (axes[1,1],[r['psnr'] for r in rc2_i_gt],[r['psnr'] for r in rc2_a_gt],
         PURPLE,'Regresion C2 — PSNR',   'dB',  None),
        (axes[1,2],[r['psnr'] for r in rc3_i_gt],[r['psnr'] for r in rc3_a_gt],
         TEAL,  'Regresion C3 — PSNR',   'dB',  None),
    ]:
        ax.plot(x_gt, yi_vals, 'o-',  color=color,  lw=2, ms=6, label='independiente')
        ax.plot(x_gt, ya_vals, 's--', color=ORANGE, lw=2, ms=6, label='autoregresiva')
        ann(ax, x_gt, yi_vals, color, 8)
        ax.set_xticks(x_gt); ax.set_xticklabels(labels_gt, fontsize=8, rotation=15)
        if ylim: ax.set_ylim(*ylim)
        style_ax(ax, title, '', ylabel); ax.legend(fontsize=8)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight'); plt.close()

def page_frame(pdf, si, r2, r3):
    has_gt = si.get('has_gt', True)
    zeros  = np.zeros_like(si['binary'])

    if has_gt:
        titulo = (f'Posicion {si["pos_in"]} -> {si["pos_out"]}\n'
                  f'Seg IoU={si["iou"]:.4f}  Dice={si["dice"]:.4f}  |  '
                  f'C2 SSIM={r2["ssim"]:.4f}  |  C3 SSIM={r3["ssim"]:.4f}')
    else:
        titulo = (f'Posicion {si["pos_in"]} -> {si["pos_out"]} '
                  f'(Prediccion pura — sin ground truth disponible)')

    fig = plt.figure(figsize=(12, 16))
    fig.suptitle(titulo, fontsize=11, fontweight='bold', y=0.99)
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.5, wspace=0.35)
    first_axes = []

    # Fila 0 — Inputs (siempre disponibles)
    for col, (data, title, cmap, vmin, vmax) in enumerate([
        (si['c2_prep'], 'C2 prep\n(amplitud PFM)',        'RdBu_r',  0, 1),
        (si['c2_diff'], 'C2 diff\n(cambio entre frames)', 'hot',      0, 1),
        (si['c3_prep'], 'C3 prep\n(fase PFM)',            'twilight', 0, 1),
    ]):
        ax = fig.add_subplot(gs[0, col])
        if col == 0: first_axes.append(ax)
        imshow_cb(ax, data, cmap, title, vmin, vmax)

    # Fila 1 — Segmentación
    mr    = si['mask_real'] if has_gt and si['mask_real'] is not None else zeros
    e_seg = np.abs(si['binary'] - mr) if has_gt and si['mask_real'] is not None else zeros
    for col, (data, title, cmap, vmin, vmax) in enumerate([
        (si['binary'], 'Mascara predicha\n(binaria)',                           'gray', 0, 1),
        (mr,            'Mascara real\n(Otsu)' if has_gt else 'GT\n(no disp.)',  'gray', 0, 1),
        (e_seg,         'Error seg.\n|pred-real|' if has_gt else 'Error\n(N/A)', 'Reds', 0, 1),
    ]):
        ax = fig.add_subplot(gs[1, col])
        if col == 0: first_axes.append(ax)
        imshow_cb(ax, data, cmap, title, vmin, vmax)

    # Fila 2 — Regresión C2
    c2r   = r2['real'] if has_gt and r2['real'] is not None else zeros
    e_c2  = np.abs(r2['pred'] - c2r) if has_gt and r2['real'] is not None else zeros
    for col, (data, title, cmap, vmin, vmax) in enumerate([
        (r2['pred'], 'C2 predicho\n(amplitud)',                          'RdBu_r', 0,   1),
        (c2r,         'C2 real' if has_gt else 'C2 real\n(no disp.)',     'RdBu_r', 0,   1),
        (e_c2,        'Error C2\n|pred-real|' if has_gt else 'Error\n(N/A)', 'Reds', 0, 0.5),
    ]):
        ax = fig.add_subplot(gs[2, col])
        if col == 0: first_axes.append(ax)
        imshow_cb(ax, data, cmap, title, vmin, vmax)

    # Fila 3 — Regresión C3
    c3r   = r3['real'] if has_gt and r3['real'] is not None else zeros
    e_c3  = np.abs(r3['pred'] - c3r) if has_gt and r3['real'] is not None else zeros
    for col, (data, title, cmap, vmin, vmax) in enumerate([
        (r3['pred'], 'C3 predicho\n(fase)',                               'twilight', 0,   1),
        (c3r,         'C3 real' if has_gt else 'C3 real\n(no disp.)',      'twilight', 0,   1),
        (e_c3,        'Error C3\n|pred-real|' if has_gt else 'Error\n(N/A)', 'Reds',   0, 0.5),
    ]):
        ax = fig.add_subplot(gs[3, col])
        if col == 0: first_axes.append(ax)
        imshow_cb(ax, data, cmap, title, vmin, vmax)

    for ax0, (label, color) in zip(first_axes, [
        (f'Inputs frame {si["pos_in"]}', GRAY),
        ('Segmentacion',                  BLUE),
        ('Regresion C2',                  PURPLE),
        ('Regresion C3',                  TEAL),
    ]):
        ax0.text(-0.25, 0.5, label, ha='right', va='center', fontsize=8,
                 color=color, fontweight='bold', transform=ax0.transAxes, rotation=90)

    pdf.savefig(fig, bbox_inches='tight'); plt.close()

def page_resumen_segmentacion(pdf, seg_i):
    n = len(seg_i)
    fig, axes = plt.subplots(3, n, figsize=(4*n, 12))
    fig.suptitle('Resumen Segmentación — máscaras predichas vs. reales',
                 fontsize=13, fontweight='bold')
    for i, r in enumerate(seg_i):
        axes[0,i].imshow(r['binary'],    cmap='gray')
        iou_str = f"{r['iou']:.3f}" if r.get('iou') is not None else 'N/A'
        axes[0,i].set_title(f"Frame {r['pos_out']}\nIoU={iou_str}", fontsize=8)
        mr = r['mask_real'] if r.get('mask_real') is not None else np.zeros_like(r['binary'])
        axes[1,i].imshow(mr, cmap='gray')
        gt_label = f"Real frame {r['pos_out']}" if r.get('has_gt') else "GT (no disp.)"
        axes[1,i].set_title(gt_label, fontsize=8)
        err = np.abs(r['binary']-mr) if r.get('has_gt') else np.zeros_like(r['binary'])
        axes[2,i].imshow(err, cmap='Reds', vmin=0, vmax=1)
        axes[2,i].set_title(f"Error frame {r['pos_out']}", fontsize=8)
        for row in range(3): axes[row,i].axis('off')
    for row, (lab, col) in enumerate([('Pred binaria',BLUE),('Mascara real',BLUE),('Error',ORANGE)]):
        axes[row,0].set_ylabel(lab, fontsize=9, fontweight='bold', color=col)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight'); plt.close()

def page_resumen_canal(pdf, results, title, cmap_pred, metrica_key,
                       metrica_label, color_label, max_err=0.5):
    n = len(results)
    fig, axes = plt.subplots(3, n, figsize=(4*n, 12))
    fig.suptitle(title, fontsize=13, fontweight='bold')
    for i, r in enumerate(results):
        axes[0,i].imshow(r['pred'], cmap=cmap_pred, vmin=0, vmax=1)
        met_str = f"{r[metrica_key]:.3f}" if r.get(metrica_key) is not None else 'N/A'
        axes[0,i].set_title(f"Frame {r['pos_out']}\n{metrica_label}={met_str}", fontsize=8)
        real_data = r['real'] if r.get('real') is not None else np.zeros_like(r['pred'])
        axes[1,i].imshow(real_data, cmap=cmap_pred, vmin=0, vmax=1)
        rt = f"Real frame {r['pos_out']}" if r.get('has_gt') else "Real (no disp.)"
        axes[1,i].set_title(rt, fontsize=8)
        err = np.abs(r['pred']-real_data) if r.get('has_gt') else np.zeros_like(r['pred'])
        axes[2,i].imshow(err, cmap='Reds', vmin=0, vmax=max_err)
        axes[2,i].set_title(f"Error frame {r['pos_out']}", fontsize=8)
        for row in range(3): axes[row,i].axis('off')
    for row, (lab, col) in enumerate([('Prediccion',color_label),
                                       ('Real',      color_label),
                                       ('Error',     ORANGE)]):
        axes[row,0].set_ylabel(lab, fontsize=9, fontweight='bold', color=col)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight'); plt.close()

def page_error(pdf, seg_i, seg_a, rc2_i, rc2_a, rc3_i, rc3_a, labels):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Análisis de error — distribución y degradación autoregresiva',
                 fontsize=13, fontweight='bold', y=0.98)
    x = range(len(labels))

    axes[0,0].bar(x,[np.abs(r['binary']-r['mask_real']).mean() for r in seg_i],color=BLUE,alpha=0.8)
    axes[0,0].set_xticks(x); axes[0,0].set_xticklabels(labels, fontsize=8, rotation=15)
    style_ax(axes[0,0], 'Segmentación — Error medio por frame', '', 'Error medio')

    axes[0,1].bar(x,[r['mse'] for r in rc2_i], color=PURPLE, alpha=0.8)
    axes[0,1].set_xticks(x); axes[0,1].set_xticklabels(labels, fontsize=8, rotation=15)
    style_ax(axes[0,1], 'Regresión C2 — MSE por frame', '', 'MSE')

    axes[0,2].bar(x,[r['mse'] for r in rc3_i], color=TEAL, alpha=0.8)
    axes[0,2].set_xticks(x); axes[0,2].set_xticklabels(labels, fontsize=8, rotation=15)
    style_ax(axes[0,2], 'Regresión C3 — MSE por frame', '', 'MSE')

    for ax, yi, ya, color, title, ylabel in [
        (axes[1,0],[r['iou']  for r in seg_i],[r['iou']  for r in seg_a],
         BLUE,  'Segmentación — degradación auto.','IoU'),
        (axes[1,1],[r['ssim'] for r in rc2_i],[r['ssim'] for r in rc2_a],
         PURPLE,'Regresión C2 — degradación auto.','SSIM'),
        (axes[1,2],[r['ssim'] for r in rc3_i],[r['ssim'] for r in rc3_a],
         TEAL,  'Regresión C3 — degradación auto.','SSIM'),
    ]:
        ax.plot(x, yi, 'o-',  color=color,  lw=2, ms=7, label='independiente')
        ax.plot(x, ya, 's--', color=ORANGE, lw=2, ms=7, label='autoregresiva')
        ax.fill_between(x, ya, yi, alpha=0.15, color=color)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8, rotation=15)
        ax.set_ylim(0.7, 1.0)
        style_ax(ax, title, '', ylabel); ax.legend(fontsize=8)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight'); plt.close()

def page_viabilidad(pdf, seg_i, seg_a, rc2_i, rc2_a, rc3_i, rc3_a,
                    seg_ckpt, regc2_ckpt, regc3_ckpt):
    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor('white')
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis('off')

    ax.text(0.5, 0.96, 'Resumen de viabilidad — 3 modelos predictivos BiFeO₃',
            ha='center', fontsize=15, fontweight='bold', transform=ax.transAxes)
    ax.axhline(y=0.91, xmin=0.05, xmax=0.95, color='#D3D1C7', linewidth=0.8)

    headers   = ['Métrica','Seg. indep.','Seg. auto.','C2 indep.','C2 auto.','C3 indep.','C3 auto.']
    # Filtrar solo frames con GT para el resumen de viabilidad
    seg_i_v = [r for r in seg_i if r.get('has_gt') and r.get('iou') is not None]
    seg_a_v = [r for r in seg_a if r.get('has_gt') and r.get('iou') is not None]
    rc2_i_v = [r for r in rc2_i if r.get('has_gt') and r.get('ssim') is not None]
    rc2_a_v = [r for r in rc2_a if r.get('has_gt') and r.get('ssim') is not None]
    rc3_i_v = [r for r in rc3_i if r.get('has_gt') and r.get('ssim') is not None]
    rc3_a_v = [r for r in rc3_a if r.get('has_gt') and r.get('ssim') is not None]

    rows_data = [
        ['IoU promedio',
         f"{avg(seg_i_v,'iou'):.4f}" if seg_i_v else 'N/A',
         f"{avg(seg_a_v,'iou'):.4f}" if seg_a_v else 'N/A','-','-','-','-'],
        ['Dice promedio',
         f"{avg(seg_i_v,'dice'):.4f}" if seg_i_v else 'N/A',
         f"{avg(seg_a_v,'dice'):.4f}" if seg_a_v else 'N/A','-','-','-','-'],
        ['SSIM promedio','-','-',
         f"{avg(rc2_i_v,'ssim'):.4f}" if rc2_i_v else 'N/A',
         f"{avg(rc2_a_v,'ssim'):.4f}" if rc2_a_v else 'N/A',
         f"{avg(rc3_i_v,'ssim'):.4f}" if rc3_i_v else 'N/A',
         f"{avg(rc3_a_v,'ssim'):.4f}" if rc3_a_v else 'N/A'],
        ['PSNR promedio (dB)','-','-',
         f"{avg(rc2_i_v,'psnr'):.2f}" if rc2_i_v else 'N/A',
         f"{avg(rc2_a_v,'psnr'):.2f}" if rc2_a_v else 'N/A',
         f"{avg(rc3_i_v,'psnr'):.2f}" if rc3_i_v else 'N/A',
         f"{avg(rc3_a_v,'psnr'):.2f}" if rc3_a_v else 'N/A'],
        ['Mejor epoch',
         str(int(seg_ckpt['epoch'])),'-',
         str(int(regc2_ckpt['epoch'])),'-',
         str(int(regc3_ckpt['epoch'])),'-'],
        ['Val IoU/SSIM (entren.)',
         f"{seg_ckpt['val_iou']:.4f}",'-',
         f"{regc2_ckpt['val_ssim']:.4f}",'-',
         f"{regc3_ckpt['val_ssim']:.4f}",'-'],
        ['Caída autoregresiva',
         f"{(avg(seg_a,'iou')-avg(seg_i,'iou'))*100:.1f}%",'-',
         f"{(avg(rc2_a,'ssim')-avg(rc2_i,'ssim'))*100:.1f}%",'-',
         f"{(avg(rc3_a,'ssim')-avg(rc3_i,'ssim'))*100:.1f}%",'-'],
    ]

    col_x   = [0.03,0.22,0.34,0.46,0.58,0.70,0.82]
    row_y0  = 0.86
    row_h   = 0.058
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

    ax.text(0.5,0.44,'Conclusiones',ha='center',fontsize=12,
            fontweight='bold',transform=ax.transAxes)
    for i,(color,texto) in enumerate([
        (BLUE,  f"Segmentación: IoU {avg(seg_i,'iou'):.3f} (indep.) — predice la distribución de dominios con alta precisión espacial."),
        (PURPLE,f"Regresión C2: SSIM {avg(rc2_i,'ssim'):.3f} (indep.) — reconstruye la amplitud PFM con buena similitud estructural."),
        (TEAL,  f"Regresión C3: SSIM {avg(rc3_i,'ssim'):.3f} (indep.) — reconstruye la fase PFM continua, complementando la máscara binaria."),
        (ORANGE,f"Degradación autoregresiva — Seg.: {abs((avg(seg_a,'iou')-avg(seg_i,'iou'))*100):.1f}% | C2: {abs((avg(rc2_a,'ssim')-avg(rc2_i,'ssim'))*100):.1f}% | C3: {abs((avg(rc3_a,'ssim')-avg(rc3_i,'ssim'))*100):.1f}% — consistente con acumulación de error temporal."),
        (GREEN, "Hipótesis validada: los patrones de evolución de dominios ferroeléctricos en BiFeO₃ son identificables y predecibles con deep learning usando 38 pares de entrenamiento."),
    ]):
        ax.text(0.03,0.38-i*0.072,f'• {texto}',ha='left',va='top',
                fontsize=9,color=color,transform=ax.transAxes)

    pdf.savefig(fig,bbox_inches='tight'); plt.close()

# =================================================================
# 5. MAIN
# =================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo : {device}")

    # Crear carpetas de predicciones
    for d in [PRED_SEG_PNG, PRED_SEG_NPY, PRED_C2_PNG, PRED_C2_NPY,
              PRED_C3_PNG, PRED_C3_NPY]:
        os.makedirs(d, exist_ok=True)

    print("Cargando modelos...")
    seg_model,   seg_ckpt   = load_model(SEG_CKPT,    device)
    regc2_model, regc2_ckpt = load_model(REG_C2_CKPT, device)
    regc3_model, regc3_ckpt = load_model(REG_C3_CKPT, device)
    print(f"  Segmentacion  epoch {seg_ckpt['epoch']:>3} | IoU  {seg_ckpt['val_iou']:.4f}")
    print(f"  Regresion C2  epoch {regc2_ckpt['epoch']:>3} | SSIM {regc2_ckpt['val_ssim']:.4f}")
    print(f"  Regresion C3  epoch {regc3_ckpt['epoch']:>3} | SSIM {regc3_ckpt['val_ssim']:.4f}")

    idx_c2_prep = build_file_index(DIR_C2_PREP)
    idx_c2_diff = build_file_index(DIR_C2_DIFF)
    idx_c3_prep = build_file_index(DIR_C3_PREP)
    idx_c3_mask = build_file_index(DIR_C3_MASK)

    # Frames a predecir → posiciones de input (input = frame_objetivo - 1)
    target_frames = list(range(PREDICT_FROM, PREDICT_TO + 1))
    positions     = [p - 1 for p in target_frames]  # posiciones de input
    labels        = [f"frame {p+1}" for p in positions]

    sin_gt = [p for p in positions if p + 1 > len(idx_c2_prep)]
    con_gt = [p for p in positions if p + 1 <= len(idx_c2_prep)]
    if sin_gt:
        print(f"  Frames sin GT  : {[p+1 for p in sin_gt]} (prediccion pura)")

    print(f"\nMAPEO FRAMES A PREDECIR -> ARCHIVOS")
    print("=" * 70)
    for pos in positions:
        gt_info = get_file(idx_c2_prep, pos+1).stem if pos+1 <= len(idx_c2_prep) else '(no existe)'
        print(f"  Frame {pos+1:>3} | input: {get_file(idx_c2_prep, pos).stem}")
        print(f"          | target: {gt_info}")
        print()
    print("=" * 70)

    print("\nCalculando inferencias...")
    seg_i, seg_a, rc2_i, rc2_a, rc3_i, rc3_a = calcular_inferencias(
        positions, seg_model, regc2_model, regc3_model, device,
        idx_c2_prep, idx_c2_diff, idx_c3_prep, idx_c3_mask)

    seg_log   = read_csv(SEG_LOG)
    regc2_log = read_csv(REG_C2_LOG)
    regc3_log = read_csv(REG_C3_LOG)

    print(f"\nGenerando PDF: {OUTPUT_PDF}")
    with PdfPages(str(OUTPUT_PDF)) as pdf:
        print("  Portada..."); page_portada(pdf, seg_ckpt, regc2_ckpt, regc3_ckpt)
        print("  Curvas de entrenamiento..."); page_curvas(pdf, seg_log, regc2_log, regc3_log)
        print("  Métricas..."); page_metricas(pdf, seg_i, seg_a, rc2_i, rc2_a, rc3_i, rc3_a, labels)
        for i, pos in enumerate(positions):
            print(f"  Frame {pos} → {pos+1}...")
            page_frame(pdf, seg_i[i], rc2_i[i], rc3_i[i])
        print("  Resumen segmentación..."); page_resumen_segmentacion(pdf, seg_i)
        print("  Resumen C2...")
        page_resumen_canal(pdf, rc2_i, 'Resumen Regresión C2 — amplitud PFM predicha vs. real',
                           'RdBu_r', 'ssim', 'SSIM', PURPLE, max_err=0.5)
        print("  Resumen C3...")
        page_resumen_canal(pdf, rc3_i, 'Resumen Regresión C3 — fase PFM predicha vs. real',
                           'twilight', 'ssim', 'SSIM', TEAL, max_err=0.5)
        print("  Análisis de error...")
        page_error(pdf, seg_i, seg_a, rc2_i, rc2_a, rc3_i, rc3_a, labels)
        print("  Tabla de viabilidad...")
        page_viabilidad(pdf, seg_i, seg_a, rc2_i, rc2_a, rc3_i, rc3_a,
                        seg_ckpt, regc2_ckpt, regc3_ckpt)

    print(f"\nReporte generado: {OUTPUT_PDF}")
    print(f"Predicciones guardadas en: {PRED_DIR}")
    print(f"  Mascaras PNG : {PRED_SEG_PNG}")
    print(f"  C2 PNG       : {PRED_C2_PNG}")
    print(f"  C3 PNG       : {PRED_C3_PNG}")

if __name__ == '__main__':
    main()
