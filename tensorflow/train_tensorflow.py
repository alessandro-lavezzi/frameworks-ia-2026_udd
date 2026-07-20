"""Entrenamiento y evaluación del clasificador de edad en TensorFlow/Keras. Guarda el modelo para inferencia."""
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import IMG_SIZE, BATCH_SIZE, SEED, CLASS_NAMES, build_dataframe, get_splits

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "age_cnn_keras.keras")
EPOCHS = 10


def load_and_preprocess(filepath, label):
    img = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return img, label


def make_tf_dataset(dataframe, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((dataframe["filepath"].values, dataframe["label"].values))
    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe), seed=SEED)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    train_df, val_df, test_df = get_splits(build_dataframe())
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    train_ds = make_tf_dataset(train_df, shuffle=True)
    val_ds = make_tf_dataset(val_df)
    test_ds = make_tf_dataset(test_df)

    model = build_model()
    model.summary()

    t0 = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
    print(f"\nTiempo total de entrenamiento (Keras): {time.time() - t0:.2f} segundos")

    # Curvas de entrenamiento.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy"); axes[0].legend()
    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss"); axes[1].legend()
    plt.show()

    # --- Evaluación en test (el modelo ya entrega softmax) ---
    probs, y_true = [], []
    for imgs, labels in test_ds:
        probs.extend(model.predict(imgs, verbose=0))
        y_true.extend(labels.numpy())
    probs, y_true = np.array(probs), np.array(y_true)
    y_pred = probs.argmax(1)

    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=CLASS_NAMES).plot()
    plt.title("Matriz de confusión — TensorFlow/Keras")
    plt.show()

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
    plt.title("Curva ROC multiclase — TensorFlow/Keras")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.show()

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Modelo guardado en: {MODEL_PATH}")


if __name__ == "__main__":
    main()
