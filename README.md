# Modelo Predictivo de Dominios Ferroeléctricos

**Tesis:** Estudio de la Evolución de los Dominios Ferroeléctricos al Switching Usando Deep Learning para Aplicación de Memorias de Estado Sólido

**Instituto Tecnológico de Querétaro — Maestría en Ciencia de Datos**
**Autor:** Miguel Angel Castro Medina

---

## Descripción general

Este proyecto implementa un pipeline completo de deep learning para predecir la evolución temporal de dominios ferroeléctricos en películas delgadas de materiales ferroelectricos a partir de imágenes de Microscopía de Fuerza Piezoeléctrica (PFM) obtenidas con AFM. El sistema aprende a anticipar el estado del material en el siguiente ciclo de switching eléctrico, con aplicación directa al diseño de memorias de estado sólido ferroeléctricas.

El pipeline opera sobre tres canales de imagen AFM y produce tres tipos de predicción independientes:

| Modelo | Input | Output | Métrica |
|---|---|---|---|
| `Modelo_Segmentacion` | C2_prep[N] + C2_diff[N] + C3_prep[N] | C3_mask[N+1] — distribución binaria de dominios | IoU, Dice |
| `Modelo_Regresion_C2` | C2_prep[N] + C2_diff[N] + C3_prep[N] | C2_prep[N+1] — amplitud PFM continua | SSIM, PSNR |
| `Modelo_Regresion_C3` | C2_prep[N] + C2_diff[N] + C3_prep[N] | C3_prep[N+1] — fase PFM continua | SSIM, PSNR |

---

## Estructura del proyecto

```
modelo_predictivo/
│
├── data/
│   ├── canal_1/              # NPYs Canal 1 preprocesado (topografía AFM)
│   ├── canal_2/              # NPYs Canal 2 preprocesado (amplitud PFM)
│   ├── canal_3/              # NPYs Canal 3 preprocesado (fase PFM)
│   ├── diff/                 # NPYs diferencia |C2[N] − C2[N-1]|
│   └── mask/                 # NPYs máscaras binarias Otsu de C3
│
├── resultados/
│   ├── modelo_segmentacion/
│   │   ├── checkpoints/      # best_model.pth + checkpoint_epXXX.pth
│   │   ├── predicciones/
│   │   │   ├── npy/pred/     # frame_NNN_indep_pred.npy | frame_NNN_sin_gt_pred.npy
│   │   │   ├── npy/real/     # frame_NNN_indep_real.npy
│   │   │   ├── npy/error/    # frame_NNN_indep_error.npy
│   │   │   ├── png/...       # idem en PNG con colormap gray
│   │   │   ├── figuras/      # PNGs comparativos y resúmenes
│   │   │   ├── metricas_independiente.csv
│   │   │   └── metricas_autoregresiva.csv
│   │   └── training_log.csv
│   │
│   ├── modelo_regresion_C2/
│   │   ├── checkpoints/
│   │   ├── predicciones/
│   │   │   ├── npy/pred/     # frame_NNN_indep_pred.npy | frame_NNN_sin_gt_pred.npy
│   │   │   ├── npy/real/
│   │   │   ├── npy/error/
│   │   │   ├── npy/diff_pred/ # frame_NNN_sin_gt_diff_pred.npy (diffs encadenados)
│   │   │   ├── png/...       # colormap RdBu_r
│   │   │   ├── figuras/
│   │   │   ├── metricas_independiente.csv
│   │   │   └── metricas_autoregresiva.csv
│   │   └── training_log.csv
│   │
│   ├── modelo_regresion_c3/
│   │   ├── checkpoints/
│   │   ├── predicciones/
│   │   │   ├── npy/pred/     # frame_NNN_indep_pred.npy | frame_NNN_sin_gt_pred.npy
│   │   │   ├── npy/real/
│   │   │   ├── npy/error/
│   │   │   ├── png/...       # colormap twilight
│   │   │   ├── figuras/
│   │   │   ├── metricas_independiente.csv
│   │   │   └── metricas_autoregresiva.csv
│   │   └── training_log.csv
│   │
│   └── reporte_visual.pdf
│
├── Modelo_Segmentacion.py
├── Modelo_Regresion_C2.py
├── Modelo_Regresion_C3.py
├── Entrenamiento_pipeline.py     # ← entrena los 3 modelos con un solo script
├── inferencia_segmentacion.py
├── inferencia_regresion_C2.py
├── inferencia_regresion_C3.py
├── Inferencia_pipeline.py        # ← corre las 3 inferencias con un solo script
├── reporte_visual.py
├── reporte_visual.ipynb
└── requirements.txt
```

> **Nota:** Los scripts de adquisición y preprocesamiento (`AFM_ToolKit`, `AFM_Preprocessing`, `AFM_ROI_Toolkit`) son repositorios separados que generan los archivos de la carpeta `data/`.

---

## Física del problema

Las imágenes PFM de BiFeO₃ contienen tres canales con información complementaria:

- **Canal 1 — Topografía:** morfología superficial de la muestra. Sirve como referencia espacial para la detección del Target Area.
- **Canal 2 — Amplitud PFM:** magnitud de la deformación piezoeléctrica en cada punto. Valores altos indican dominios activos; valores bajos indican zonas de pared de dominio o dominios fatigados.
- **Canal 3 — Fase PFM:** ángulo de desfase entre la señal de excitación y la respuesta piezoeléctrica. Toma exactamente dos valores (0° o 180°) correspondientes a las dos orientaciones de polarización espontánea: dominio ↑ y dominio ↓.

El **switching ferroeléctrico** ocurre al aplicar un campo eléctrico que invierte la polarización en zonas del material. Con cada ciclo de switching, los dominios evolucionan — este proyecto predice esa evolución frame a frame.

---

## Prerrequisitos

### Hardware recomendado
- GPU NVIDIA con CUDA 11.8+ (probado en RTX 3050 Laptop, 4.3 GB VRAM)
- 16 GB RAM mínimo
- 10 GB espacio en disco

### Software
- Python 3.10+
- CUDA Toolkit 11.8 o superior

---

## Instalación

```powershell
cd C:\Users\migue\Desktop\modelo_predictivo

python -m venv env
env\Scripts\activate

# PyTorch con CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
```

---

## Pipeline completo

```
Archivos .000 (AFM raw)
      ↓  AFM_ToolKit (Gwyddion + pygwy)
  .npy + .png por canal
      ↓  AFM_Preprocessing
  C2_prep, C3_prep, C2_diff, C3_mask  (preprocesados y normalizados)
      ↓  AFM_ROI_Toolkit (YOLO + registro)
  Recortes 80×80 px alineados por canal
      ↓
  Entrenamiento_pipeline.py   ← entrena los 3 modelos
      ↓
  Inferencia_pipeline.py      ← corre las 3 inferencias
      ↓
  reporte_visual.py / reporte_visual.ipynb   ← lee resultados y genera PDF
```

---

## Etapa 4 — Entrenamiento

### Opción A — Pipeline unificado (recomendado)

```powershell
python Entrenamiento_pipeline.py
```

Entrena los 3 modelos secuencialmente con la misma configuración. Si ya existe un `best_model.pth` para algún modelo, lo omite automáticamente.

```python
# Parámetros clave en Entrenamiento_pipeline.py
EPOCHS       = 100
BATCH_SIZE   = 4
LR           = 1e-4
VAL_FRAMES   = 6      # últimos N pares para validación (split cronológico)

ENTRENAR_SEG = True   # activar/desactivar por modelo
ENTRENAR_C2  = True
ENTRENAR_C3  = True

FORZAR_SEG   = False  # True = reentrenar aunque exista checkpoint
FORZAR_C2    = False
FORZAR_C3    = False
```

### Opción B — Scripts individuales

```powershell
python Modelo_Segmentacion.py
python Modelo_Regresion_C2.py
python Modelo_Regresion_C3.py
```

### Indexación posicional

Todos los scripts indexan los archivos por **posición** en la lista ordenada alfabéticamente (base 1), independientemente del nombre del archivo. Al iniciar cada entrenamiento se imprime el mapeo explícito posición → nombre de archivo para verificación.

### Arquitectura común

- **U-Net** con encoder **EfficientNet-B0** preentrenado en ImageNet
- **Transfer learning:** con 38 pares de entrenamiento es necesario partir de representaciones pre-aprendidas; el fine-tuning completo re-especializa el encoder hacia imágenes AFM
- **Split cronológico:** últimos `VAL_FRAMES` pares como validación — evita data leakage en series temporales

| Modelo | Loss | Métrica principal |
|---|---|---|
| Segmentación | BCE + Dice (α=0.5) | IoU |
| Regresión C2 | MSE + SSIM (α=0.5) | SSIM |
| Regresión C3 | MSE + SSIM (α=0.5) | SSIM |

---

## Etapa 5 — Inferencia

### Opción A — Pipeline unificado (recomendado)

```powershell
python Inferencia_pipeline.py
```

Ejecuta los 3 modelos en el orden correcto: **C2 → C3 → Seg**. El orden importa porque C2 genera los `diff_pred` encadenados que C3 y segmentación necesitan para frames sin ground truth.

```python
# Parámetros en Inferencia_pipeline.py
PREDICT_FROM   = 36    # primer frame a PREDECIR (base 1)
PREDICT_TO     = 40    # último frame a predecir
INFERENCE_MODE = 'ambos'  # 'independiente' | 'autoregresiva' | 'ambos'

CORRER_C2  = True
CORRER_C3  = True
CORRER_SEG = True
```

### Opción B — Scripts individuales

```powershell
python inferencia_regresion_C2.py
python inferencia_regresion_C3.py
python inferencia_segmentacion.py
```

### Semántica de PREDICT_FROM / PREDICT_TO

`PREDICT_FROM` y `PREDICT_TO` referencian los **frames a predecir** (el output), no los inputs. El sistema calcula automáticamente que para predecir el frame P necesita el frame P−1 como input:

```
PREDICT_FROM = 36, PREDICT_TO = 40
→ predice frames 36, 37, 38, 39, 40
→ inputs utilizados: frames 35, 36, 37, 38, 39
```

El valor mínimo válido es `PREDICT_FROM = 2` (para predecir el frame 2 se necesita el frame 1 como input).

### Modos de inferencia

| Modo | Descripción | Ground truth |
|---|---|---|
| `independiente` | Cada predicción usa datos reales del frame N | Sí — calcula métricas |
| `autoregresiva` | La predicción anterior se usa como input siguiente | Sí — calcula métricas |
| `ambos` | Corre ambos modos y genera comparación | Sí |

### Frames sin ground truth (predicción más allá del dataset)

Si `PREDICT_TO` supera el número de archivos disponibles, el sistema detecta automáticamente qué frames no tienen ground truth y los procesa en **modo encadenado sin métricas**:

```
Frame 40 (con GT) → input: frame 39 real          → métricas calculadas
Frame 41 (sin GT) → input: frame 40 real           → solo predicción
Frame 42 (sin GT) → input: frame 41 predicho (C2/C3 encadenados)
Frame 43 (sin GT) → input: frame 42 predicho
...
```

### Estructura de salidas por modelo

```
resultados/modelo_regresion_C2/predicciones/
  npy/
    pred/       frame_036_indep_pred.npy   frame_041_sin_gt_pred.npy
    real/       frame_036_indep_real.npy   (solo frames con GT)
    error/      frame_036_indep_error.npy  (solo frames con GT)
    diff_pred/  frame_041_sin_gt_diff_pred.npy  (diffs encadenados para sin GT)
  png/          (idem, con colormap RdBu_r)
  figuras/      comparaciones, resúmenes, evolución sin GT
  metricas_independiente.csv
  metricas_autoregresiva.csv

resultados/modelo_regresion_c3/predicciones/
  npy/pred/     frame_036_indep_pred.npy   frame_041_sin_gt_pred.npy
  ...           (colormap twilight)

resultados/modelo_segmentacion/predicciones/
  npy/pred/     frame_036_indep_pred.npy   frame_041_sin_gt_pred.npy
  ...           (colormap gray)
```

**Patrón de nombres:**
- Con GT: `frame_{NNN}_{indep|auto}_{pred|real|error}.npy`
- Sin GT: `frame_{NNN}_sin_gt_pred.npy` y `frame_{NNN}_sin_gt_diff_pred.npy`

---

## Etapa 6 — Reporte visual

El reporte **lee los resultados ya generados** por los scripts de inferencia. No vuelve a correr inferencia ni carga modelos.

```powershell
# Flujo correcto:
python Inferencia_pipeline.py   # 1. generar resultados
python reporte_visual.py        # 2. generar PDF

# O interactivo:
jupyter notebook reporte_visual.ipynb
```

```python
# Parámetro principal en reporte_visual.py
MODO_FRAMES = 'independiente'  # modo a visualizar frame a frame
```

### Contenido del reporte

1. Portada con métricas de validación de los 3 modelos
2. Curvas de entrenamiento — loss, IoU/Dice (seg) y SSIM/PSNR (regresión)
3. Métricas de inferencia — independiente vs. autoregresiva por frame
4. Visualización frame a frame — 4 filas × 3 columnas:
   - **Fila 1 — Inputs:** C2_prep | C2_diff | C3_prep (real o predicho anterior)
   - **Fila 2 — Segmentación:** máscara pred | máscara real o pred anterior | error/diff
   - **Fila 3 — Regresión C2:** C2 pred | C2 real o pred anterior | error/diff
   - **Fila 4 — Regresión C3:** C3 pred | C3 real o pred anterior | error/diff
5. Resúmenes por canal — grilla completa pred vs. referencia
6. Análisis de error — distribución por frame y degradación autoregresiva
7. Tabla de viabilidad con conclusiones

Para frames sin ground truth, las columnas de "real" muestran la predicción del frame anterior como referencia de evolución.

---

## Resultados obtenidos

Dataset: 40 imágenes AFM de BiFeO₃ · Recorte: 80×80 px · Split: 33 train / 6 val

| Modelo | Métrica val (entrenamiento) | Métrica indep. (inferencia) | Época |
|---|---|---|---|
| Segmentación | IoU = 0.8305 | IoU ≈ 0.83–0.85 | 67 |
| Regresión C2 | SSIM = 0.7930 | SSIM ≈ 0.89–0.92 | 96 |
| Regresión C3 | SSIM = 0.7778 | SSIM ≈ 0.88–0.93 | 100 |

La degradación en modo autoregresivo es gradual y consistente con la acumulación de error esperada en predicción temporal encadenada.

---

## Consideraciones para datasets más grandes

| Parámetro | ~40 imgs | 100–500 | 1k–10k | 10k–30k+ |
|---|---|---|---|---|
| Encoder | EfficientNet-B0 | B0/B2 | B4/ResNet50 | B7/ResNet101 |
| Batch size | 4 | 8–16 | 16–32 | 32–64 |
| Learning rate | 1e-4 | 1e-4–3e-4 | 3e-4–1e-3 | 1e-3–3e-3 |
| Early stopping | No | Opcional | Recomendado | Imprescindible |
| Augmentation | flips + rot90 | flips + rot + ruido | Moderada | Mínima |

---

## Notas técnicas

**Semántica PREDICT_FROM/TO:** referencian los frames a predecir, no los inputs. Internamente se calcula `posición_input = PREDICT_FROM − 1`.

**Orden del pipeline de inferencia:** siempre C2 → C3 → Seg. C2 genera los `diff_pred` encadenados que C3 y segmentación necesitan para frames sin GT.

**Augmentation y recortes rectangulares:** las rotaciones de 90°/270° solo se aplican si `CROP_W == CROP_H`. Con recortes rectangulares se usan exclusivamente flips.

**Divisibilidad por 32:** la arquitectura U-Net con EfficientNet-B0 realiza 5 niveles de downsampling (×32). Las dimensiones del recorte deben ser divisibles por 32.

**Kernel SSIM en GPU:** el kernel gaussiano de SSIMLoss se mueve explícitamente con `.to(pred.device)` antes de cada convolución.

**Indexación posicional:** base 1 sobre listas ordenadas alfabéticamente. El mapeo posición → archivo se imprime al inicio de cada ejecución.

**Reporte visual sin re-inferencia:** `reporte_visual.py` y `reporte_visual.ipynb` leen únicamente CSVs de métricas, checkpoints (solo epoch y métrica, sin cargar pesos) y NPYs de predicciones. No requieren GPU.

---

## Licencia

Proyecto académico — Instituto Tecnológico de Querétaro. Todos los derechos reservados.