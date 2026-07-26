# Optimización de Hiperparámetros (Optuna)

Guía de la búsqueda de hiperparámetros del clasificador de edad. Pensada para que **el equipo**
(Ale + compañero) y **el profesor** entiendan qué se optimiza, cómo, y de dónde salen los
valores que usan los entrenamientos.

## TL;DR / flujo de trabajo

1. Se corre Optuna **una sola vez** por framework, sobre el dataset combinado (`all`).
2. Optuna guarda el mejor resultado en un JSON dentro de `models/` (**versionado en git**).
3. Los entrenamientos (`train_pytorch.py`, `train_tensorflow.py`) **leen ese JSON por defecto**
   (`load_hparams`); si falta, caen a un fallback hardcodeado. **No hace falta re-correr Optuna
   en cada entrenamiento** (es lo más lento del pipeline).
4. Los valores hallados sobre `all` se **reutilizan** para las corridas por etnia.

```bash
# Búsqueda (lenta, ~40-60 min/framework en CPU). Correr sólo cuando se quiera re-optimizar:
python pytorch/optuna_pytorch.py --trials 20 --epochs 5
python tensorflow/optuna_tensorflow.py --trials 20 --epochs 5

# Entrenamiento (usa el JSON automáticamente):
python run_pipeline.py
```

## Archivos

| Archivo | Rol |
|---------|-----|
| [metrics.py](metrics.py) | Definición **única y compartida** de las métricas (F1, recall, AUC, accuracy). La usan ambos frameworks y debería usarla también el modelo de etnia del compañero. |
| [pytorch/optuna_pytorch.py](pytorch/optuna_pytorch.py) · [tensorflow/optuna_tensorflow.py](tensorflow/optuna_tensorflow.py) | Scripts de búsqueda. |
| `models/best_hparams_pytorch.json` · `models/best_hparams_keras.json` | Mejores hiperparámetros (los lee el entrenamiento). **Versionados en git.** |
| `models/optuna_pareto_*.json` | Frente de Pareto completo (sólo en modo multiobjetivo), para documentación. **Versionados en git.** |
| `HPARAMS_FALLBACK` en cada `train_*.py` | Respaldo hardcodeado (`# origen: optuna`) por si falta el JSON. |

## Métrica objetivo — por qué NO accuracy

Accuracy es **engañosa** en problemas con clases desbalanceadas: si una clase es rara, el modelo
puede ignorarla y aun así tener accuracy alta (acierta las clases frecuentes). Aunque acá
balanceamos por undersampling, el desempeño *por clase* sigue importando, así que optimizamos:

- **F1 (macro)**: media armónica de precision y recall, promediada por clase con igual peso.
  Es la métrica más balanceada y la **primaria** del proyecto.
- **Recall (macro)**: de cada clase real, cuánto se detecta. Importa si no queremos "perder" casos.
- **AUC-ROC (macro, One-vs-Rest)**: capacidad de ranking independiente del umbral.

`macro` = promedio simple entre clases (todas pesan igual), a diferencia de `weighted` (pesa por
frecuencia) o `micro` (≈ accuracy cuando las clases están balanceadas).

### Flag `--metrica`

Ambos scripts de Optuna aceptan `--metrica`:

| Valor | Qué hace |
|-------|----------|
| `multi` **(default)** | Optimiza **F1, recall y AUC a la vez** (multiobjetivo → frente de Pareto). |
| `f1` / `recall` / `auc` | Optimiza **una sola** métrica (objetivo único, con pruning). |
| `accuracy` | Disponible por completitud, pero **no recomendada** (ver arriba). |

## Qué hace Optuna cuando hay trade-off entre métricas (multiobjetivo)

Con **un solo** objetivo, Optuna busca el trial con el valor más alto: hay un único ganador.

Con **varios** objetivos a la vez (F1, recall, AUC) aparece el trade-off: un set de
hiperparámetros puede ganar en recall pero perder en AUC frente a otro. Ya no hay un único
"mejor". Optuna usa **dominancia de Pareto**:

- Un trial **A domina a B** si A es *mejor o igual en todas* las métricas y *estrictamente mejor
  en al menos una*. Si A domina a B, B se descarta.
- Los trials que **nadie domina** forman el **frente de Pareto**: cada uno es un compromiso
  distinto (uno prioriza recall, otro AUC, otro un punto intermedio). Optuna los expone en
  `study.best_trials` (plural) — **no elige por vos**.

Como el entrenamiento necesita **un** set de hiperparámetros, nuestros scripts eligen del frente
el trial con **mejor F1** (métrica primaria) como representante, y lo escriben en
`best_hparams_*.json`. El **frente completo** se guarda en `optuna_pareto_*.json` para poder
mostrar y discutir los trade-offs (útil para la interrogación oral). Si se quisiera priorizar
otro criterio (p.ej. recall), se cambia la línea `elegido = max(frente, key=lambda t: t.values[0])`
en el `optuna_*.py` (índice 0 = F1, 1 = recall, 2 = AUC, según `METRICAS_MULTIOBJETIVO`).

## Espacio de búsqueda

Idéntico en PyTorch y Keras (misma arquitectura conceptual):

| Hiperparámetro   | Tipo                  | Rango / valores | Notas |
|------------------|-----------------------|-----------------|-------|
| `n_conv_layers`  | entero                | 2 – 5           | N° de bloques conv (Conv → BatchNorm → ReLU → MaxPool). |
| `base_channels`  | categórico            | {16, 32}        | Canales del primer bloque. |
| `lr`             | flotante (escala log) | 1e-4 – 1e-2     | Learning rate de Adam. |

**Derivación de canales**: crecen ×2 por bloque desde `base_channels`, tope 256.
Ej.: `n_conv_layers=3`, `base_channels=16` → `[16, 32, 64]`.

### Fijos (no se buscan)

| Hiperparámetro | Valor | Dónde |
|----------------|-------|-------|
| `batch_size`   | 32    | `BATCH_SIZE` en data/dataset.py |
| `dropout`      | 0.3   | capa antes de la salida |
| `IMG_SIZE`     | 64    | data/dataset.py |
| optimizador    | Adam  | train_*.py |
| épocas (final) | 10    | `EPOCHS` en train_*.py |
| augmentation   | flip horizontal + rotación ~10° + jitter/contraste | train_*.py |

## Metodología

- **Dataset**: sólo `all`. Los hiperparámetros se reutilizan para las etnias (asunción: la
  arquitectura óptima no cambia sustancialmente entre subsets del mismo problema).
- **Épocas por trial**: menos que el final (`--epochs`, típico 5) para acelerar la búsqueda; el
  entrenamiento final usa `EPOCHS = 10`.
- **Pruning**: `MedianPruner` corta trials malos temprano. **Sólo en objetivo único** — Optuna no
  soporta pruners en multiobjetivo, así que en `multi` todos los trials corren completos.
- **Split**: mismo `get_splits` (70/15/15 estratificado) que el entrenamiento. La búsqueda usa
  train + val; test queda intacto para la evaluación final.

## Valores vigentes (hardcodeados como fallback / última búsqueda)

> Nota: los `best_hparams_*.json` versionados provienen de una búsqueda previa (objetivo
> accuracy). Al correr Optuna con el default actual (`multi`), estos valores se actualizan.
> Los `HPARAMS_FALLBACK` en los `train_*.py` reflejan estos mismos valores.

### PyTorch (`age_cnn_pytorch`)

| Hiperparámetro | Valor | Canales |
|----------------|-------|---------|
| `n_conv_layers` | 3 | `[16, 32, 64]` |
| `base_channels` | 16 | |
| `lr` | 0.00028898883555273204 | |

### TensorFlow / Keras (`age_cnn_keras`)

| Hiperparámetro | Valor | Canales |
|----------------|-------|---------|
| `n_conv_layers` | 2 | `[16, 32]` |
| `base_channels` | 16 | |
| `lr` | 0.0002880345768063458 | |

**Lectura**: ambos frameworks convergieron a learning rates bajos y parecidos (~2.9e-4, por
debajo del 1e-3 default previo) y a arquitecturas chicas (2–3 bloques). Coherente con haber
reducido el problema a **3 clases** (`menor`/`adulto·a`/`anciano·a`): menos clases → menos
capacidad necesaria → redes chicas generalizan igual y entrenan más rápido.

## Para el compañero (modelo de etnia)

- Reutilizá [metrics.py](metrics.py) para que las métricas se calculen igual en ambos modelos.
- Mismo patrón: corré Optuna una vez, versioná tu `best_hparams_*.json`, y que tu entrenamiento
  lo lea por defecto (copiá la función `load_hparams`).
- Documentá acá (o en un `.md` propio enlazado desde este) tu espacio de búsqueda y tus valores.

## Cómo re-hacer la búsqueda

```bash
python pytorch/optuna_pytorch.py --trials 20 --epochs 5              # multiobjetivo (default)
python tensorflow/optuna_tensorflow.py --trials 20 --epochs 5
# o con objetivo único:
python pytorch/optuna_pytorch.py --trials 20 --epochs 5 --metrica f1
```

Cada corrida reescribe `models/best_hparams_*.json` (y `optuna_pareto_*.json` en modo `multi`).
El entrenamiento toma los nuevos valores automáticamente en la próxima corrida.
