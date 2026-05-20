"""
modelo_segmentacion.py — Modelo predictivo de dominios ferroeléctricos
Tesis: Evolución de los Dominios Ferroeléctricos al Switching Usando Deep Learning

Arquitectura : U-Net con encoder EfficientNet-B0 preentrenado (ImageNet)
Framework    : PyTorch + segmentation_models_pytorch
Inputs       : C2_prep[N], C2_diff[N], C3_prep[N]  → 3 canales apilados 80×80
Output       : C3_mask[N+1]                         → máscara binaria 80×80
Loss         : BCE + Dice combinados
Métricas     : IoU, Dice coefficient, Pixel Accuracy

Indexación   : los archivos se ordenan alfabéticamente y se acceden por
               POSICIÓN (base 1), independientemente del nombre del archivo.
               Posición 1 = primer archivo, posición 2 = segundo, etc.

Salida:
  resultados/modelo_segmentacion/checkpoints/best_model.pth
  resultados/modelo_segmentacion/checkpoints/checkpoint_epXXX.pth
  resultados/modelo_segmentacion/training_log.csv
"""

import os
import csv
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
import segmentation_models_pytorch as smp

# =================================================================
# 1. CONFIGURACIÓN
# =================================================================

# --- RUTAS ---
BASE_DIR     = Path(r'C:\Users\migue\Desktop\modelo_predictivo')
DATA_DIR     = BASE_DIR / 'data'

DIR_C2_PREP  = DATA_DIR / 'canal_2'
DIR_C2_DIFF  = DATA_DIR / 'diff'
DIR_C3_PREP  = DATA_DIR / 'canal_3'
DIR_C3_MASK  = DATA_DIR / 'mask'

OUTPUT_DIR   = BASE_DIR / 'resultados' / 'modelo_segmentacion'
CKPT_DIR     = OUTPUT_DIR / 'checkpoints'
LOG_CSV      = OUTPUT_DIR / 'training_log.csv'

# --- DATASET ---
# Los archivos se indexan por POSICIÓN en lista ordenada alfabéticamente (base 1).
# Un par válido es (posición N como input, posición N+1 como target).
# idx_c2_diff determina el número de pares disponibles (tiene menos archivos que mask).
# Con 39 archivos diff y 40 mask → 39 pares posibles (pos 1..39)
# Split: posiciones 1..33 train, 34..39 val → 33 train / 6 val
#

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
THRESHOLD    = 0.5

# --- SPLIT CRONOLÓGICO ---
VAL_FRAMES   = 6

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

def get_file(index_list, pos):
    """Retorna archivo en posición 'pos' (base 1)."""
    if pos < 1 or pos > len(index_list):
        raise IndexError(
            f"Posición {pos} fuera de rango. "
            f"Disponible: 1–{len(index_list)}."
        )
    return index_list[pos - 1]

# Construir índices al inicio — disponibles globalmente para Dataset y main
idx_c2_prep = build_file_index(DIR_C2_PREP)
idx_c2_diff = build_file_index(DIR_C2_DIFF)
idx_c3_prep = build_file_index(DIR_C3_PREP)
idx_c3_mask = build_file_index(DIR_C3_MASK)


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

class FerroelectricDataset(Dataset):
    """
    Par (posición_N, posición_N+1):
      Input  → [C2_prep_N, C2_diff_N, C3_prep_N]  tensor (3, 80, 80) float32
      Target → C3_mask_{N+1}                        tensor (1, 80, 80) float32 {0,1}

    Augmentation (solo train):
      - Flip horizontal y vertical aleatorio
      - Rotación aleatoria 90°/180°/270°
      La misma transformación se aplica a input y target.
    """

    def __init__(self, positions, augment=False):
        """
        positions: lista de enteros N (base 1) donde el par (N, N+1) es válido.
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

    def _load_mask(self, path):
        """Carga máscara binaria (0 o 255) → {0.0, 1.0}."""
        arr = np.load(str(path)).astype(np.float32)
        return (arr > 127).astype(np.float32)

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
        c2_diff = self._load_npy(get_file(idx_c2_diff, pos))
        c3_prep = self._load_npy(get_file(idx_c3_prep, pos))
        mask    = self._load_mask(get_file(idx_c3_mask, pos + 1))

        x = np.stack([c2_prep, c2_diff, c3_prep], axis=0)  # (3, 80, 80)
        y = mask                                             # (80, 80)

        if self.augment:
            x, y = self._augment(x, y)

        return torch.from_numpy(x), torch.from_numpy(y).unsqueeze(0)

# =================================================================
# 5. LOSS — BCE + Dice
# =================================================================

class BCEDiceLoss(nn.Module):
    """
    loss = alpha * BCE + (1 - alpha) * Dice
    alpha=0.5 da peso igual a ambas componentes.
    """
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.bce   = nn.BCEWithLogitsLoss()

    def dice_loss(self, logits, targets, eps=1e-6):
        probs = torch.sigmoid(logits)
        num   = 2 * (probs * targets).sum(dim=(1, 2, 3))
        den   = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + eps
        return 1 - (num / den).mean()

    def forward(self, logits, targets):
        return self.alpha * self.bce(logits, targets) + \
               (1 - self.alpha) * self.dice_loss(logits, targets)

# =================================================================
# 6. MÉTRICAS
# =================================================================

def compute_metrics(logits, targets, threshold=THRESHOLD, eps=1e-6):
    preds   = (torch.sigmoid(logits) > threshold).float()
    targets = targets.float()
    inter   = (preds * targets).sum(dim=(1, 2, 3))
    union   = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    iou     = (inter / (union - inter + eps)).mean().item()
    dice    = (2 * inter / (union + eps)).mean().item()
    acc     = (preds == targets).float().mean().item()
    return {'iou': iou, 'dice': dice, 'acc': acc}

# =================================================================
# 7. LOOP DE ENTRENAMIENTO
# =================================================================

def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    metrics    = {'iou': 0.0, 'dice': 0.0, 'acc': 0.0}

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if is_train:
                optimizer.zero_grad()
            logits = model(x)
            loss   = criterion(logits, y)
            if is_train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            m = compute_metrics(logits.detach(), y)
            for k in metrics:
                metrics[k] += m[k]

    n = len(loader)
    return total_loss / n, {k: v / n for k, v in metrics.items()}

# =================================================================
# 8. MAIN
# =================================================================

def main():

    os.makedirs(CKPT_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo : {device}")
    if device.type == 'cuda':
        print(f"GPU         : {torch.cuda.get_device_name(0)}")
        print(f"VRAM        : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # --- Verificar archivos disponibles ---
    n_diff = len(idx_c2_diff)   # diff tiene menos archivos que mask → limita los pares
    n_mask = len(idx_c3_mask)

    print(f"\nArchivos C2_diff : {n_diff}")
    print(f"Archivos C3_mask : {n_mask}")

    # Pares válidos: posición N como input, N+1 como target de mask
    # idx_c2_diff va de pos 1..n_diff
    # idx_c3_mask va de pos 1..n_mask, necesitamos pos+1 → hasta n_mask
    all_positions  = list(range(1, n_diff + 1))   # 1..n_diff
    # Verificar que pos+1 existe en mask para todos los pares
    all_positions  = [p for p in all_positions if p + 1 <= n_mask]

    train_positions = all_positions[:-VAL_FRAMES]
    val_positions   = all_positions[-VAL_FRAMES:]

    crop_label = f"{CROP_W}px" if CROP_MODE == "cuadrado" else f"{CROP_W}x{CROP_H}px"
    print(f"Modo recorte     : {CROP_MODE} ({crop_label})")
    print(f"Pares válidos    : {len(all_positions)}")
    print(f"\nTrain : {len(train_positions)} pares")
    print(f"  Posición {train_positions[0]:>3} → archivo: {get_file(idx_c2_diff, train_positions[0]).stem}")
    print(f"  Posición {train_positions[-1]:>3} → archivo: {get_file(idx_c2_diff, train_positions[-1]).stem}")
    print(f"Val   : {len(val_positions)} pares")
    print(f"  Posición {val_positions[0]:>3} → archivo: {get_file(idx_c2_diff, val_positions[0]).stem}")
    print(f"  Posición {val_positions[-1]:>3} → archivo: {get_file(idx_c2_diff, val_positions[-1]).stem}\n")

    # --- DataLoaders ---
    train_loader = DataLoader(
        FerroelectricDataset(train_positions, augment=True),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(
        FerroelectricDataset(val_positions, augment=False),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # --- Modelo ---
    model = smp.Unet(
        encoder_name    = 'efficientnet-b0',
        encoder_weights = 'imagenet',
        in_channels     = 3,
        classes         = 1,
        activation      = None,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parámetros  : {n_params:,}\n")

    criterion = BCEDiceLoss(alpha=0.5)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # --- CSV log ---
    with open(LOG_CSV, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow([
            'epoch', 'lr',
            'train_loss', 'train_iou', 'train_dice', 'train_acc',
            'val_loss',   'val_iou',   'val_dice',   'val_acc'])

    best_iou   = 0.0
    best_epoch = 0

    print(f"{'Ep':>4} {'T_loss':>8} {'T_IoU':>7} {'T_Dice':>7} │ "
          f"{'V_loss':>8} {'V_IoU':>7} {'V_Dice':>7}")
    print("─" * 65)

    for epoch in range(1, EPOCHS + 1):

        train_loss, train_m = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss,   val_m   = run_epoch(model, val_loader,   criterion, device)
        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        if val_m['iou'] > best_iou:
            best_iou   = val_m['iou']
            best_epoch = epoch
            torch.save({
                'epoch'      : epoch,
                'model_state': model.state_dict(),
                'val_iou'    : best_iou,
                'val_dice'   : val_m['dice'],
            }, CKPT_DIR / 'best_model.pth')

        if epoch % 10 == 0:
            torch.save({
                'epoch'      : epoch,
                'model_state': model.state_dict(),
                'val_iou'    : val_m['iou'],
            }, CKPT_DIR / f'checkpoint_ep{epoch:03d}.pth')

        with open(LOG_CSV, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([
                epoch, round(lr, 8),
                round(train_loss, 6), round(train_m['iou'], 4),
                round(train_m['dice'], 4), round(train_m['acc'], 4),
                round(val_loss, 6), round(val_m['iou'], 4),
                round(val_m['dice'], 4), round(val_m['acc'], 4)])

        mejor = ' ← mejor' if epoch == best_epoch else ''
        print(f"{epoch:>4} {train_loss:>8.4f} {train_m['iou']:>7.4f} "
              f"{train_m['dice']:>7.4f} │ "
              f"{val_loss:>8.4f} {val_m['iou']:>7.4f} "
              f"{val_m['dice']:>7.4f}{mejor}")

    print(f"\n{'─'*65}")
    print(f"Mejor val IoU  : {best_iou:.4f}  (epoch {best_epoch})")
    print(f"Modelo         : {CKPT_DIR / 'best_model.pth'}")
    print(f"Log            : {LOG_CSV}")

if __name__ == '__main__':
    main()