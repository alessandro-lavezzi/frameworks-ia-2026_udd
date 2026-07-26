"""Búsqueda de hiperparámetros con Optuna para el clasificador de edad en TensorFlow/Keras.

Corre una sola vez sobre el dataset combinado (etnia='all'; ver data/dataset.py): los
hiperparámetros encontrados se reusan para las corridas por etnia individual en
tensorflow/train_tensorflow.py, en vez de repetir la búsqueda una vez por etnia (5x más caro).

Espacio de búsqueda: arquitectura (N° de bloques convolucionales y sus canales) + learning
rate. Batch size y dropout quedan fijos en los valores de train_tensorflow.py.

Métrica objetivo (--metrica): por defecto 'multi', que optimiza F1, recall y AUC-ROC a la vez
(multiobjetivo -> frente de Pareto; ver HIPERPARAMETROS.md). También puede ser una sola métrica
(f1/recall/auc/accuracy). NO se usa accuracy por defecto porque es engañosa con clases
desbalanceadas: promedia parejo y esconde el mal desempeño en clases minoritarias.

Uso:
    python tensorflow/optuna_tensorflow.py --trials 20 --epochs 5                # multiobjetivo (default)
    python tensorflow/optuna_tensorflow.py --trials 20 --epochs 5 --metrica f1   # objetivo único
"""
import os
import sys
import json
import argparse
import numpy as np
import optuna

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import IMG_SIZE, build_dataframe, get_splits
from metrics import METRICAS_INDIVIDUALES, METRICAS_MULTIOBJETIVO, compute_metric
# Import directo (no "tensorflow.train_tensorflow"): esta carpeta se llama igual que la librería
# tensorflow, así que un import con paquete "tensorflow." resolvería contra esa librería, no
# contra esta carpeta. Python ya agrega la carpeta del script a sys.path al correrlo directamente,
# así que el import directo del sibling funciona (y train_tensorflow importa tensorflow por dentro).
# build_model es el MISMO builder que usa el entrenamiento: la arquitectura que Optuna evalúa es
# exactamente la que después se entrena y se guarda (una sola fuente de verdad).
from train_tensorflow import make_tf_dataset, build_model, channels_from_hparams, HPARAMS_PATH

# Frente de Pareto (modo multiobjetivo) se guarda aparte, para documentación/interrogación oral.
PARETO_PATH = os.path.join(os.path.dirname(HPARAMS_PATH), "optuna_pareto_keras.json")


def make_objective(train_ds, val_ds, y_val, epochs, metrica):
    multi = metrica == "multi"

    def objective(trial):
        hp = {
            "n_conv_layers": trial.suggest_int("n_conv_layers", 2, 5),
            "base_channels": trial.suggest_categorical("base_channels", [16, 32]),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        }
        # n_conv_layers de MaxPooling2D no puede reducir el mapa de features a 0.
        if IMG_SIZE // (2 ** hp["n_conv_layers"]) < 1:
            raise optuna.TrialPruned()

        model = build_model(channels=channels_from_hparams(hp), lr=hp["lr"])
        model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=0)
        # val_ds no se baraja, así que las probabilidades de predict quedan alineadas con y_val.
        probs = model.predict(val_ds, verbose=0)

        if multi:
            return tuple(compute_metric(m, y_val, probs) for m in METRICAS_MULTIOBJETIVO)
        return compute_metric(metrica, y_val, probs)

    return objective


def main(n_trials, epochs, metrica):
    print("Métrica objetivo:", metrica)
    train_df, val_df, _ = get_splits(build_dataframe(etnia="all"))
    print(f"Train: {len(train_df)} | Val: {len(val_df)}")
    multi = metrica == "multi"

    train_ds = make_tf_dataset(train_df, shuffle=True, augment=True)
    val_ds = make_tf_dataset(val_df)
    # Etiquetas reales de validación en el mismo orden que entrega val_ds (sin shuffle).
    y_val = np.concatenate([y.numpy() for _, y in val_ds])

    if multi:
        study = optuna.create_study(directions=["maximize"] * len(METRICAS_MULTIOBJETIVO))
    else:
        study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(make_objective(train_ds, val_ds, y_val, epochs, metrica), n_trials=n_trials)

    if multi:
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
    print("train_tensorflow.py los usará automáticamente en la próxima corrida (lee este JSON).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Búsqueda de hiperparámetros (Optuna) para el modelo Keras, sobre el dataset combinado ('all')."
    )
    parser.add_argument("--trials", type=int, default=20, help="N° de trials de Optuna (default: 20).")
    parser.add_argument("--epochs", type=int, default=5,
                         help="Épocas por trial, menos que el entrenamiento final para que la búsqueda sea rápida (default: 5).")
    parser.add_argument("--metrica", default="multi", choices=METRICAS_INDIVIDUALES + ["multi"],
                         help="Métrica objetivo. 'multi' (default) optimiza F1+recall+AUC a la vez (Pareto).")
    args = parser.parse_args()
    main(n_trials=args.trials, epochs=args.epochs, metrica=args.metrica)
