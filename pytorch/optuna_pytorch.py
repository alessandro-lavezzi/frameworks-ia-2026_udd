"""Búsqueda de hiperparámetros con Optuna para el clasificador de edad en PyTorch.

Corre una sola vez sobre el dataset combinado (etnia='all'; ver data/dataset.py): los
hiperparámetros encontrados se reusan para las corridas por etnia individual en
pytorch/train_pytorch.py, en vez de repetir la búsqueda una vez por etnia (5x más caro).

Espacio de búsqueda: arquitectura (N° de bloques convolucionales y sus canales) + learning
rate. Batch size y dropout quedan fijos en los valores de train_pytorch.py.

Métrica objetivo (--metrica): por defecto 'multi', que optimiza F1, recall y AUC-ROC a la vez
(multiobjetivo -> frente de Pareto; ver HIPERPARAMETROS.md). También puede ser una sola métrica
(f1/recall/auc/accuracy). NO se usa accuracy por defecto porque es engañosa con clases
desbalanceadas: promedia parejo y esconde el mal desempeño en clases minoritarias.

Uso:
    python pytorch/optuna_pytorch.py --trials 20 --epochs 5                 # multiobjetivo (default)
    python pytorch/optuna_pytorch.py --trials 20 --epochs 5 --metrica f1    # objetivo único
"""
import os
import sys
import json
import argparse
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import BATCH_SIZE, build_dataframe, get_splits
from metrics import METRICAS_INDIVIDUALES, METRICAS_MULTIOBJETIVO, compute_metric
# Import directo del sibling (no vía paquete "pytorch."): Python ya agrega la carpeta del
# script a sys.path al correrlo directamente, y evita depender de que "pytorch" no colisione
# con nada instalado (la librería real se importa como "torch", no "pytorch").
# AgeCNN es la MISMA clase que usa el entrenamiento: así la arquitectura que Optuna evalúa es
# exactamente la que después se entrena y se guarda (una sola fuente de verdad).
from train_pytorch import (
    UTKFaceDataset, transform, train_transform, run_epoch, device,
    AgeCNN, channels_from_hparams, HPARAMS_PATH,
)

# Frente de Pareto (modo multiobjetivo) se guarda aparte, para documentación/interrogación oral.
PARETO_PATH = os.path.join(os.path.dirname(HPARAMS_PATH), "optuna_pareto_pytorch.json")


def eval_probs(model, loader):
    """Probabilidades (softmax) y etiquetas reales sobre un loader. Base para todas las métricas."""
    model.eval()
    probs, y_true = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            p = F.softmax(model(imgs.to(device)), dim=1)
            probs.extend(p.cpu().numpy())
            y_true.extend(labels.numpy())
    return np.array(probs), np.array(y_true)


def make_objective(train_df, val_df, epochs, metrica):
    train_loader = DataLoader(UTKFaceDataset(train_df, train_transform), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(UTKFaceDataset(val_df, transform), batch_size=BATCH_SIZE, shuffle=False)
    multi = metrica == "multi"

    def objective(trial):
        hp = {
            "n_conv_layers": trial.suggest_int("n_conv_layers", 2, 5),
            "base_channels": trial.suggest_categorical("base_channels", [16, 32]),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        }
        model = AgeCNN(channels=channels_from_hparams(hp)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=hp["lr"])
        criterion = nn.CrossEntropyLoss()

        probs, y_true = None, None
        for epoch in range(epochs):
            run_epoch(model, train_loader, optimizer, criterion, train=True)
            probs, y_true = eval_probs(model, val_loader)
            # Pruning (cortar trials malos temprano) solo tiene sentido con objetivo único:
            # Optuna no soporta pruners en multiobjetivo.
            if not multi:
                trial.report(compute_metric(metrica, y_true, probs), epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        # Se reusan las probabilidades de la última época (no hay pasada extra).
        if multi:
            return tuple(compute_metric(m, y_true, probs) for m in METRICAS_MULTIOBJETIVO)
        return compute_metric(metrica, y_true, probs)

    return objective


def main(n_trials, epochs, metrica):
    print("Device:", device, "| Métrica objetivo:", metrica)
    train_df, val_df, _ = get_splits(build_dataframe(etnia="all"))
    print(f"Train: {len(train_df)} | Val: {len(val_df)}")
    multi = metrica == "multi"

    if multi:
        # Un 'direction' por métrica; sin pruner (Optuna no lo soporta en multiobjetivo).
        study = optuna.create_study(directions=["maximize"] * len(METRICAS_MULTIOBJETIVO))
    else:
        study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(make_objective(train_df, val_df, epochs, metrica), n_trials=n_trials)

    if multi:
        # No hay un único mejor: hay un frente de Pareto (trials no dominados). Se elige como
        # representante el de mejor F1 (primera métrica), y se guarda el frente completo aparte.
        frente = study.best_trials
        print(f"\nFrente de Pareto: {len(frente)} trials no dominados "
              f"(métricas: {METRICAS_MULTIOBJETIVO})")
        for t in frente:
            valores = {m: round(v, 4) for m, v in zip(METRICAS_MULTIOBJETIVO, t.values)}
            print(f"  trial {t.number}: {valores} | params: {t.params}")
        elegido = max(frente, key=lambda t: t.values[0])  # mejor F1 del frente
        best_params = elegido.params
        print(f"\nRepresentante elegido (mejor F1 del frente): trial {elegido.number} -> {best_params}")

        os.makedirs(os.path.dirname(PARETO_PATH), exist_ok=True)
        with open(PARETO_PATH, "w") as f:
            json.dump([{"number": t.number,
                        "metricas": dict(zip(METRICAS_MULTIOBJETIVO, t.values)),
                        "params": t.params} for t in frente], f, indent=2)
        print(f"Frente de Pareto guardado en: {PARETO_PATH}")
    else:
        print(f"\nMejor trial ({metrica}): {study.best_value:.4f} | params: {study.best_params}")
        best_params = study.best_params

    os.makedirs(os.path.dirname(HPARAMS_PATH), exist_ok=True)
    with open(HPARAMS_PATH, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"Mejores hiperparámetros guardados en: {HPARAMS_PATH}")
    print("train_pytorch.py los usará automáticamente en la próxima corrida (lee este JSON).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Búsqueda de hiperparámetros (Optuna) para el modelo PyTorch, sobre el dataset combinado ('all')."
    )
    parser.add_argument("--trials", type=int, default=20, help="N° de trials de Optuna (default: 20).")
    parser.add_argument("--epochs", type=int, default=5,
                         help="Épocas por trial, menos que el entrenamiento final para que la búsqueda sea rápida (default: 5).")
    parser.add_argument("--metrica", default="multi", choices=METRICAS_INDIVIDUALES + ["multi"],
                         help="Métrica objetivo. 'multi' (default) optimiza F1+recall+AUC a la vez (Pareto).")
    args = parser.parse_args()
    main(n_trials=args.trials, epochs=args.epochs, metrica=args.metrica)
