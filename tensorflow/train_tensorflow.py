"""Entrenamiento y evaluación del clasificador de edad en TensorFlow/Keras. Guarda el modelo para inferencia.

--etnia selecciona el subset de entrenamiento (white/black/asian/indian/all). El undersampling
sigue siendo por sexo en todos los casos (ver data/dataset.py); la etnia solo filtra qué imágenes
entran. Con 'all' (default) se entrena con todas las etnias juntas, para comparar contra las
corridas por etnia individual.
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import IMG_SIZE, BATCH_SIZE, SEED, CLASS_NAMES, ETNIAS_DISPONIBLES, FIGURAS_DIR, build_dataframe, get_splits

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
HPARAMS_PATH = os.path.join(MODELS_DIR, "best_hparams_keras.json")
EPOCHS = 10

# Hiperparámetros: el entrenamiento lee por defecto models/best_hparams_keras.json (lo que deja
# Optuna, versionado en git). Si el JSON no existe, cae a estos valores hardcodeados de respaldo,
# que provienen de la última búsqueda de Optuna (ver HIPERPARAMETROS.md). Así el flujo normal usa
# siempre el resultado de Optuna, pero el script no se rompe si falta el archivo.
HPARAMS_FALLBACK = {
    "n_conv_layers": 2,               # origen: optuna
    "base_channels": 16,              # origen: optuna
    "lr": 0.0002880345768063458,      # origen: optuna
}


def load_hparams(verbose: bool = False) -> dict:
    """Hiperparámetros a usar: el JSON de Optuna si existe, si no el fallback hardcodeado."""
    hp = dict(HPARAMS_FALLBACK)
    if os.path.exists(HPARAMS_PATH):
        with open(HPARAMS_PATH) as f:
            hp.update(json.load(f))
        if verbose:
            print(f"Hiperparámetros desde Optuna ({os.path.basename(HPARAMS_PATH)}): {hp}")
    elif verbose:
        print(f"Sin JSON de Optuna; uso fallback hardcodeado: {hp}")
    return hp


def channels_from_hparams(hp: dict) -> list:
    """Deriva la lista de canales por bloque conv: crecen x2 desde base_channels, tope 256."""
    return [min(hp["base_channels"] * (2 ** i), 256) for i in range(hp["n_conv_layers"])]


def get_model_path(etnia: str) -> str:
    # 'all' conserva el nombre histórico (age_cnn_keras.keras) para no romper demo/infer.py,
    # que por defecto usa el modelo combinado. Las corridas por etnia van con sufijo aparte.
    if etnia.lower() == "all":
        return os.path.join(MODELS_DIR, "age_cnn_keras.keras")
    return os.path.join(MODELS_DIR, f"age_cnn_keras_{etnia.lower()}.keras")


def load_and_preprocess(filepath, label):
    img = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return img, label


# Augmentation definida ANTES de armar el dataset generator (ver "Lecciones aprendidas" en
# CLAUDE.md): flip horizontal y jitter leve tienen sentido en rostros; rotación acotada (~10°)
# para no distorsionar rasgos de los que depende la edad. Solo se aplica al set de entrenamiento.
augmentacion = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.03),
    tf.keras.layers.RandomContrast(0.2),
], name="augmentacion")


def make_tf_dataset(dataframe, shuffle=False, augment=False):
    ds = tf.data.Dataset.from_tensor_slices((dataframe["filepath"].values, dataframe["label"].values))
    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe), seed=SEED)
    ds = ds.batch(BATCH_SIZE)
    if augment:
        ds = ds.map(lambda x, y: (augmentacion(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def build_model(channels=None, lr=1e-3, dropout=0.3):
    """CNN con N bloques conv configurables por Optuna (canales por bloque) + learning rate.
    Sin argumentos reproduce la arquitectura original (16,32,64,128) con lr=1e-3.
    """
    channels = channels if channels is not None else [16, 32, 64, 128]
    if IMG_SIZE // (2 ** len(channels)) < 1:
        raise ValueError(f"Demasiados bloques conv para IMG_SIZE={IMG_SIZE}: el mapa de features llega a 0.")

    model = tf.keras.Sequential([tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))])
    for out_ch in channels:
        model.add(tf.keras.layers.Conv2D(out_ch, 3, padding="same", use_bias=False))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation("relu"))
        model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax"))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main(etnia: str = "all"):
    train_df, val_df, test_df = get_splits(build_dataframe(etnia=etnia))
    print(f"Etnia: {etnia} | Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    train_ds = make_tf_dataset(train_df, shuffle=True, augment=True)
    val_ds = make_tf_dataset(val_df)
    test_ds = make_tf_dataset(test_df)

    hp = load_hparams(verbose=True)
    model = build_model(channels=channels_from_hparams(hp), lr=hp["lr"])
    model.summary()

    t0 = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
    print(f"\nTiempo total de entrenamiento (Keras): {time.time() - t0:.2f} segundos")

    os.makedirs(FIGURAS_DIR, exist_ok=True)
    # Curvas de entrenamiento.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy"); axes[0].legend()
    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss"); axes[1].legend()
    plt.tight_layout()
    # plt.show()  # desactivado en esta corrida: se guarda en figuras/ para no bloquear el pipeline
    plt.savefig(os.path.join(FIGURAS_DIR, f"keras_curvas_{etnia}.png"), dpi=120)
    plt.close()

    # --- Evaluación en test (el modelo ya entrega softmax) ---
    probs, y_true = [], []
    for imgs, labels in test_ds:
        probs.extend(model.predict(imgs, verbose=0))
        y_true.extend(labels.numpy())
    probs, y_true = np.array(probs), np.array(y_true)
    y_pred = probs.argmax(1)

    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=CLASS_NAMES).plot()
    plt.title(f"Matriz de confusión — TensorFlow/Keras (etnia: {etnia})")
    plt.tight_layout()
    # plt.show()  # desactivado en esta corrida: se guarda en figuras/ para no bloquear el pipeline
    plt.savefig(os.path.join(FIGURAS_DIR, f"keras_confusion_{etnia}.png"), dpi=120)
    plt.close()

    # Curva ROC multiclase One-vs-Rest + micro-promedio.
    n_classes = len(CLASS_NAMES)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    plt.figure(figsize=(8, 7))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        plt.plot(fpr, tpr, lw=1.8, label=f"{CLASS_NAMES[i]} (AUC = {auc(fpr, tpr):.3f})")
    fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), probs.ravel())
    plt.plot(fpr_micro, tpr_micro, "k--", lw=2.2, label=f"micro-promedio (AUC = {auc(fpr_micro, tpr_micro):.3f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle=":")
    plt.xlim([0, 1]); plt.ylim([0, 1.02])
    plt.xlabel("Tasa de falsos positivos (FPR)")
    plt.ylabel("Tasa de verdaderos positivos (TPR)")
    plt.title(f"Curva ROC multiclase — TensorFlow/Keras (etnia: {etnia})")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    # plt.show()  # desactivado en esta corrida: se guarda en figuras/ para no bloquear el pipeline
    plt.savefig(os.path.join(FIGURAS_DIR, f"keras_roc_{etnia}.png"), dpi=120)
    plt.close()

    model_path = get_model_path(etnia)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"Modelo guardado en: {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrena el clasificador de edad (Keras) para una etnia (o todas).")
    parser.add_argument("--etnia", default="all", choices=[e.lower() for e in ETNIAS_DISPONIBLES],
                         help="Etnia a usar, o 'all' para todas juntas (default: all).")
    args = parser.parse_args()
    main(etnia=args.etnia)
