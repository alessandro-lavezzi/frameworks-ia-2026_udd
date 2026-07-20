"""Consumo/inferencia independiente: clasifica fotos propias con los modelos ya entrenados.

Requiere haber corrido antes pytorch/train_pytorch.py y/o tensorflow/train_tensorflow.py
(cada uno guarda su modelo en models/). Usa los modelos que estén disponibles.
"""
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from data.dataset import IMG_SIZE, CLASS_NAMES

FOTOS_PRUEBA = os.path.join(ROOT, "fotos_prueba")
PT_MODEL_PATH = os.path.join(ROOT, "models", "age_cnn_pytorch.pt")
TF_MODEL_PATH = os.path.join(ROOT, "models", "age_cnn_keras.keras")


def cargar_modelo_pt():
    """Carga el modelo PyTorch si su archivo existe; devuelve una función de predicción o None."""
    if not os.path.exists(PT_MODEL_PATH):
        return None
    import torch
    import torchvision.transforms as T
    from pytorch.train_pytorch import AgeCNN, transform  # misma arquitectura y preprocesamiento

    model = AgeCNN(num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load(PT_MODEL_PATH, map_location="cpu"))
    model.eval()

    def predecir(img_pil):
        x = transform(img_pil).unsqueeze(0)
        with torch.no_grad():
            return torch.softmax(model(x), dim=1)[0].numpy()
    return predecir


def cargar_modelo_tf():
    """Carga el modelo Keras si su archivo existe; devuelve una función de predicción o None."""
    if not os.path.exists(TF_MODEL_PATH):
        return None
    import tensorflow as tf

    model = tf.keras.models.load_model(TF_MODEL_PATH)

    def predecir(img_pil):
        arr = np.asarray(img_pil.resize((IMG_SIZE, IMG_SIZE)), dtype="float32") / 255.0
        return model.predict(arr[None, ...], verbose=0)[0]
    return predecir


def formatear_todas(probs):
    """5 clases ordenadas de mayor a menor probabilidad (para consola)."""
    orden = np.argsort(probs)[::-1]
    return "\n".join(f"  {CLASS_NAMES[i]:<12} {probs[i]*100:5.1f}%" for i in orden)


def formatear_top(probs):
    """Solo la categoría elegida (para el título de la foto)."""
    i = int(np.argmax(probs))
    return f"{CLASS_NAMES[i]} ({probs[i]*100:.1f}%)"


def main():
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    rutas = []
    for e in exts:
        rutas += glob.glob(os.path.join(FOTOS_PRUEBA, e))
        rutas += glob.glob(os.path.join(FOTOS_PRUEBA, e.upper()))
    rutas = sorted(set(rutas))
    if not rutas:
        raise FileNotFoundError(f"No encontré imágenes en '{FOTOS_PRUEBA}' (una cara por imagen, jpg o png).")
    print(f"Fotos encontradas: {len(rutas)}")

    pred_pt = cargar_modelo_pt()
    pred_tf = cargar_modelo_tf()
    if pred_pt is None and pred_tf is None:
        raise FileNotFoundError("No hay modelos entrenados. Corre primero los scripts de entrenamiento.")

    # Desglose completo en consola.
    for ruta in rutas:
        img = Image.open(ruta).convert("RGB")
        print("=" * 40)
        print(os.path.basename(ruta))
        if pred_pt:
            print("PyTorch:\n" + formatear_todas(pred_pt(img)))
        if pred_tf:
            print("Keras:\n" + formatear_todas(pred_tf(img)))
    print("=" * 40)

    # Grilla con la clase elegida por cada modelo sobre cada foto.
    n = len(rutas)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.4, rows * 4.6))
    axes = np.array(axes).reshape(-1)
    for ax, ruta in zip(axes, rutas):
        img = Image.open(ruta).convert("RGB")
        lineas = [os.path.basename(ruta)]
        if pred_pt:
            lineas.append(f"PyTorch: {formatear_top(pred_pt(img))}")
        if pred_tf:
            lineas.append(f"Keras: {formatear_top(pred_tf(img))}")
        ax.imshow(img)
        ax.axis("off")
        ax.set_title("\n".join(lineas), fontsize=8, family="monospace", loc="left")
    for ax in axes[n:]:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
