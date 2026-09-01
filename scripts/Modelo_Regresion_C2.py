"""
modelo_regresion.py — Modelo de regresión para predicción de amplitud PFM
Tesis: Evolución de los Dominios Ferroeléctricos con Deep Learning

Arquitectura : U-Net con encoder EfficientNet-B0 preentrenado (ImageNet)
Framework    : PyTorch + segmentation_models_pytorch
Inputs       : C2_prep[N], C2_diff[N], C3_prep[N]  → 3 canales apilados 80×80
Output       : C2_prep[N+1]                         → imagen continua 80×80 [0,1]
Loss         : MSE + SSIM combinados
Métricas     : SSIM, PSNR, MSE
"""

import os
import csv
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
import segmentation_models_pytorch as smp
import math

# =================================================================
# 1. CONFIGURACIÓN
# =================================================================

# --- RUTAS ---
BASE_DIR     = Path(r'C:\Users\migue\Desktop\modelo_predictivo')
DATA_DIR     = BASE_DIR / 'data'

DIR_C2_PREP  = DATA_DIR / 'canal_2'
DIR_C2_DIFF  = DATA_DIR / 'diff'
DIR_C3_PREP  = DATA_DIR / 'canal_3'

OUTPUT_DIR   = BASE_DIR / 'resultados' / 'modelo_regresion_C2'
CKPT_DIR     = OUTPUT_DIR / 'checkpoints'
LOG_CSV      = OUTPUT_DIR / 'training_log.csv'

# --- DATASET ---
# Los archivos se indexan por POSICIÓN en lista ordenada alfabéticamente
# Posición 1 = primer archivo, posición 2 = segundo, etc.
#
# Par válido: input en posición N, target C2_prep en posición N+1
# El diff del frame N está en idx_c2_diff[N-1]: la carpeta diff/ tiene 39
# archivos (frames 2..40) porque el frame 1 no tiene frame anterior, mientras
# canal_2/ tiene 40. Por eso la posición 1 no puede ser input.
# Con 40 archivos → 38 pares posibles (posiciones 2..39)
# Split: posiciones 2-33 train (32 pares), 34-39 val (6 pares)
# Posición 40 solo se usa como target del par 39
#
TOTAL_FILES  = 40
VAL_FRAMES   = 6

# --- MODO DE RECORTE ---
# 'cuadrado'    → usa CROP_SIZE para ancho y alto
# 'rectangular' → usa CROP_WIDTH y CROP_HEIGHT de forma independiente
CROP_MODE   = 'cuadrado'

# Parámetros cuadrado
CROP_SIZE   = 80

# Parámetros rectangular (solo si CROP_MODE = 'rectangular')
# Ambos valores deben ser divisibles por 32 (requisito U-Net)
CROP_WIDTH  = 80    # ancho en px (eje X)
CROP_HEIGHT = 64    # alto  en px (eje Y)

# --- ENTRENAMIENTO ---
BATCH_SIZE   = 4
EPOCHS       = 100
LR           = 1e-4
WEIGHT_DECAY = 1e-4

# Peso relativo MSE vs SSIM en la loss
# loss = ALPHA * MSE + (1 - ALPHA) * (1 - SSIM)
ALPHA        = 0.5

# --- REPRODUCIBILIDAD ---
SEED         = 42

# =================================================================
# 2. REPRODUCIBILIDAD
# =================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(SEED)

# =================================================================
# 3. ÍNDICE POSICIONAL DE ARCHIVOS
# =================================================================

def build_file_index(directory, extension='.npy'):
    """Lista ordenada alfabéticamente de archivos con la extensión dada."""
    return sorted([
        f for f in Path(directory).iterdir()
        if f.suffix == extension
    ])

idx_c2_prep = build_file_index(DIR_C2_PREP)
idx_c2_diff = build_file_index(DIR_C2_DIFF)
idx_c3_prep = build_file_index(DIR_C3_PREP)

def get_file(index_list, pos):
    """Retorna archivo en posición 'pos' (base 1)."""
    if pos < 1 or pos > len(index_list):
        raise IndexError(
            f"Posición {pos} fuera de rango. "
            f"Disponible: 1–{len(index_list)}."
        )
    return index_list[pos - 1]


# Resolver dimensiones según modo seleccionado
if CROP_MODE == 'cuadrado':
    CROP_W = CROP_SIZE
    CROP_H = CROP_SIZE
else:
    CROP_W = CROP_WIDTH
    CROP_H = CROP_HEIGHT
    if CROP_W % 32 != 0 or CROP_H % 32 != 0:
        print(f"ADVERTENCIA: CROP_WIDTH={CROP_W} o CROP_HEIGHT={CROP_H} "
              f"no son divisibles por 32 (requisito U-Net).")
HALF_W = CROP_W // 2
HALF_H = CROP_H // 2

# =================================================================
# 4. DATASET
# =================================================================

class RegressionDataset(Dataset):
    """
    Par (posición_N, posición_N+1):
      Input  → [C2_prep_N, C2_diff_N, C3_prep_N]  tensor (3, 80, 80) float32
      Target → C2_prep_{N+1}                        tensor (1, 80, 80) float32 [0,1]

    Augmentation (solo train):
      - Flip horizontal y vertical aleatorio
      - Rotación aleatoria 90°/180°/270°
      La misma transformación se aplica a input y target.
    """

    def __init__(self, positions, augment=False):
        """
        positions: lista de enteros N (base 1) donde el par (N, N+1) es válido
        """
        self.positions = positions
        self.augment   = augment

    def __len__(self):
        return len(self.positions)

    def _load_npy(self, path):
        """Carga NPY y normaliza a [0, 1] por min-max."""
        arr = np.load(str(path)).astype(np.float32)
        mn, mx = arr.min(), arr.max()
        if mx - mn > 1e-8:
            arr = (arr - mn) / (mx - mn)
        return arr

    def _augment(self, x, y):
        """Aplica la misma transformacion a input (C,H,W) y target (H,W).
        Flips: siempre disponibles — funcionan con cualquier forma.
        Rotaciones 90/270: solo si recorte cuadrado (CROP_W == CROP_H).
        """
        if random.random() > 0.5:
            x = np.flip(x, axis=2).copy()
            y = np.flip(y, axis=1).copy()
        if random.random() > 0.5:
            x = np.flip(x, axis=1).copy()
            y = np.flip(y, axis=0).copy()
        if CROP_W == CROP_H:
            k = random.randint(0, 3)
            if k > 0:
                x = np.rot90(x, k, axes=(1, 2)).copy()
                y = np.rot90(y, k, axes=(0, 1)).copy()
        return x, y

    def __getitem__(self, idx):
        pos = self.positions[idx]

        c2_prep = self._load_npy(get_file(idx_c2_prep, pos))
        # El diff del frame N esta en la posicion N-1: diff/ arranca un frame
        # despues que canal_2/ porque el primer frame no tiene frame anterior.
        # Usar 'pos' aqui cargaria |C2[N+1] - C2[N]|, es decir el cambio hacia
        # el target — fuga de informacion.
        c2_diff = self._load_npy(get_file(idx_c2_diff, pos - 1))
        c3_prep = self._load_npy(get_file(idx_c3_prep, pos))

        # Target: C2_prep del frame siguiente
        c2_next = self._load_npy(get_file(idx_c2_prep, pos + 1))

        x = np.stack([c2_prep, c2_diff, c3_prep], axis=0)  # (3, 80, 80)
        y = c2_next                                          # (80, 80)

        if self.augment:
            x, y = self._augment(x, y)

        return torch.from_numpy(x), torch.from_numpy(y).unsqueeze(0)

# =================================================================
# 5. SSIM
# =================================================================

def gaussian_kernel(size=11, sigma=1.5):
    """Genera kernel gaussiano 2D para SSIM."""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g      = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = g.unsqueeze(0) * g.unsqueeze(1)
    return (kernel / kernel.sum()).unsqueeze(0).unsqueeze(0)

class SSIMLoss(nn.Module):
    """
    SSIM como función de pérdida: loss = 1 - SSIM
    Valores de SSIM cercanos a 1 indican imágenes estructuralmente similares.
    """
    def __init__(self, window_size=11, sigma=1.5, C1=0.01**2, C2=0.03**2):
        super().__init__()
        self.window_size = window_size
        self.C1          = C1
        self.C2          = C2
        kernel           = gaussian_kernel(window_size, sigma)
        self.register_buffer('kernel', kernel)

    def forward(self, pred, target):
        kernel  = self.kernel.expand(pred.shape[1], 1, -1, -1)
        pad     = self.window_size // 2

        mu1     = F.conv2d(pred,   kernel, padding=pad, groups=pred.shape[1])
        mu2     = F.conv2d(target, kernel, padding=pad, groups=pred.shape[1])
        mu1_sq  = mu1 ** 2
        mu2_sq  = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(pred   * pred,   kernel, padding=pad, groups=pred.shape[1]) - mu1_sq
        sigma2_sq = F.conv2d(target * target, kernel, padding=pad, groups=pred.shape[1]) - mu2_sq
        sigma12   = F.conv2d(pred   * target, kernel, padding=pad, groups=pred.shape[1]) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)) / \
                   ((mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2))

        return 1 - ssim_map.mean()

# =================================================================
# 6. LOSS COMBINADA — MSE + SSIM
# =================================================================

class MSESSIMLoss(nn.Module):
    """
    loss = alpha * MSE + (1 - alpha) * (1 - SSIM)
    MSE penaliza diferencias píxel a píxel.
    SSIM penaliza diferencias en estructura y textura.
    """
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.mse   = nn.MSELoss()
        self.ssim  = SSIMLoss()

    def forward(self, pred, target):
        return self.alpha * self.mse(pred, target) + \
               (1 - self.alpha) * self.ssim(pred, target)

# =================================================================
# 7. MÉTRICAS
# =================================================================

def compute_ssim(pred, target):
    """SSIM promedio del batch (valor real, no loss)."""
    # Asegurar que la instancia de SSIMLoss (y su kernel) esté en el
    # mismo dispositivo que las tensores de entrada.
    ssim_loss = SSIMLoss().to(pred.device)
    with torch.no_grad():
        return 1 - ssim_loss(pred, target).item()

def compute_psnr(pred, target, eps=1e-8):
    """PSNR en dB. Asume imágenes normalizadas a [0,1]."""
    mse = F.mse_loss(pred, target).item()
    if mse < eps:
        return 100.0
    return 10 * math.log10(1.0 / mse)

def compute_metrics(pred, target):
    mse  = F.mse_loss(pred, target).item()
    ssim = compute_ssim(pred, target)
    psnr = compute_psnr(pred, target)
    return {'mse': mse, 'ssim': ssim, 'psnr': psnr}

# =================================================================
# 8. LOOP DE ENTRENAMIENTO
# =================================================================

def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    metrics    = {'mse': 0.0, 'ssim': 0.0, 'psnr': 0.0}

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if is_train:
                optimizer.zero_grad()

            pred = model(x)
            # Sigmoid para mantener output en [0,1]
            pred = torch.sigmoid(pred)
            loss = criterion(pred, y)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            m = compute_metrics(pred.detach(), y)
            for k in metrics:
                metrics[k] += m[k]

    n = len(loader)
    return total_loss / n, {k: v / n for k, v in metrics.items()}

# =================================================================
# 9. MAIN
# =================================================================

def main():

    os.makedirs(CKPT_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo : {device}")
    if device.type == 'cuda':
        print(f"GPU         : {torch.cuda.get_device_name(0)}")
        print(f"VRAM        : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Verificar archivos disponibles
    n_c2 = len(idx_c2_prep)
    n_diff = len(idx_c2_diff)
    n_c3 = len(idx_c3_prep)
    print(f"\nArchivos C2_prep : {n_c2}")
    print(f"Archivos C2_diff : {n_diff}")
    print(f"Archivos C3_prep : {n_c3}")

    # Pares válidos: posición N como input, N+1 como target
    # El diff del frame N está en idx_c2_diff[N-1], por lo que la posición 1
    # no tiene diff previo y queda excluida.
    # Con 40 archivos → posiciones 2-39 como input (38 pares)
    all_positions  = list(range(2, n_c2))        # 2..39
    train_positions = all_positions[:-VAL_FRAMES] # 2..33
    val_positions   = all_positions[-VAL_FRAMES:] # 34..39

    crop_label = f"{CROP_W}px" if CROP_MODE == "cuadrado" else f"{CROP_W}x{CROP_H}px"
    print(f"Modo recorte  : {CROP_MODE} ({crop_label})")
    print(f"\nPares totales : {len(all_positions)}")
    print(f"\nTrain : {len(train_positions)} pares")
    print(f"  Posición {train_positions[0]:>3} → archivo: {get_file(idx_c2_diff, train_positions[0] - 1).stem}")
    print(f"  Posición {train_positions[-1]:>3} → archivo: {get_file(idx_c2_diff, train_positions[-1] - 1).stem}")
    print(f"Val   : {len(val_positions)} pares")
    print(f"  Posición {val_positions[0]:>3} → archivo: {get_file(idx_c2_diff, val_positions[0] - 1).stem}")
    print(f"  Posición {val_positions[-1]:>3} → archivo: {get_file(idx_c2_diff, val_positions[-1] - 1).stem}\n")

    # --- DataLoaders ---
    train_loader = DataLoader(
        RegressionDataset(train_positions, augment=True),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(
        RegressionDataset(val_positions, augment=False),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # --- Modelo U-Net para regresión ---
    # activation=None → aplicamos sigmoid manualmente en run_epoch
    model = smp.Unet(
        encoder_name    = 'efficientnet-b0',
        encoder_weights = 'imagenet',
        in_channels     = 3,
        classes         = 1,       # imagen continua de un canal
        activation      = None,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parámetros  : {n_params:,}\n")

    # Crear la loss y moverla al mismo dispositivo que el modelo
    criterion = MSESSIMLoss(alpha=ALPHA).to(device)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # --- CSV log ---
    with open(LOG_CSV, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow([
            'epoch', 'lr',
            'train_loss', 'train_mse', 'train_ssim', 'train_psnr',
            'val_loss',   'val_mse',   'val_ssim',   'val_psnr'])

    best_ssim  = 0.0
    best_epoch = 0

    print(f"{'Ep':>4} {'T_loss':>8} {'T_SSIM':>7} {'T_PSNR':>7} │ "
          f"{'V_loss':>8} {'V_SSIM':>7} {'V_PSNR':>7}")
    print("─" * 65)

    for epoch in range(1, EPOCHS + 1):

        train_loss, train_m = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss,   val_m   = run_epoch(model, val_loader,   criterion, device)
        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        # Guardar mejor modelo según SSIM de validación
        if val_m['ssim'] > best_ssim:
            best_ssim  = val_m['ssim']
            best_epoch = epoch
            torch.save({
                'epoch'      : epoch,
                'model_state': model.state_dict(),
                'val_ssim'   : best_ssim,
                'val_psnr'   : val_m['psnr'],
                'val_mse'    : val_m['mse'],
            }, CKPT_DIR / 'best_model.pth')

        # Checkpoint cada 10 epochs
        if epoch % 10 == 0:
            torch.save({
                'epoch'      : epoch,
                'model_state': model.state_dict(),
                'val_ssim'   : val_m['ssim'],
            }, CKPT_DIR / f'checkpoint_ep{epoch:03d}.pth')

        # Log
        with open(LOG_CSV, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([
                epoch, round(lr, 8),
                round(train_loss, 6), round(train_m['mse'], 6),
                round(train_m['ssim'], 4), round(train_m['psnr'], 2),
                round(val_loss, 6), round(val_m['mse'], 6),
                round(val_m['ssim'], 4), round(val_m['psnr'], 2)])

        mejor = ' ← mejor' if epoch == best_epoch else ''
        print(f"{epoch:>4} {train_loss:>8.4f} {train_m['ssim']:>7.4f} "
              f"{train_m['psnr']:>7.2f} │ "
              f"{val_loss:>8.4f} {val_m['ssim']:>7.4f} "
              f"{val_m['psnr']:>7.2f}{mejor}")

    print(f"\n{'─'*65}")
    print(f"Mejor val SSIM : {best_ssim:.4f}  (epoch {best_epoch})")
    print(f"Modelo         : {CKPT_DIR / 'best_model.pth'}")
    print(f"Log            : {LOG_CSV}")

if __name__ == '__main__':
    main()