"""Consumo/inferencia independiente: clasifica fotos propias con los modelos ya entrenados.

Requiere haber corrido antes pytorch/train_pytorch.py y/o tensorflow/train_tensorflow.py
(cada uno guarda su modelo en models/). Usa los modelos que estén disponibles.

Por defecto, además de la predicción muestra el heatmap Grad-CAM (ver heatmap_gradcam.ipynb):
la zona de la cara en la que se fijó cada modelo para decidir el rango etario.
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
    """Carga el modelo PyTorch si su archivo existe.

    Devuelve (predecir, gradcam) o (None, None). `predecir` recibe una imagen PIL y
    entrega las probabilidades; `gradcam` recibe una imagen PIL y entrega (heatmap, clase_predicha).
    """
    if not os.path.exists(PT_MODEL_PATH):
        return None, None
    import torch
    import torch.nn.functional as F
    from pytorch.train_pytorch import AgeCNN, transform  # misma arquitectura y preprocesamiento

    model = AgeCNN(num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load(PT_MODEL_PATH, map_location="cpu"))
    model.eval()

    def predecir(img_pil):
        x = transform(img_pil).unsqueeze(0)
        with torch.no_grad():
            return torch.softmax(model(x), dim=1)[0].numpy()

    # Grad-CAM engancha (hooks) la ULTIMA capa convolucional (conv4): usa los gradientes de la
    # clase predicha que llegan a esa capa para ponderar sus mapas de activación.
    activations, gradients = {}, {}

    def _save_activation(module, inp, out):
        activations["value"] = out.detach()

    def _save_gradient(module, grad_in, grad_out):
        gradients["value"] = grad_out[0].detach()

    model.conv4.register_forward_hook(_save_activation)
    model.conv4.register_full_backward_hook(_save_gradient)

    def gradcam(img_pil):
        x = transform(img_pil).unsqueeze(0)
        model.zero_grad()
        logits = model(x)
        class_idx = int(logits.argmax(dim=1).item())
        logits[0, class_idx].backward()
        weights = gradients["value"].mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * activations["value"]).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx

    return predecir, gradcam


def cargar_modelo_tf():
    """Carga el modelo Keras si su archivo existe.

    Devuelve (predecir, gradcam) o (None, None), con la misma interfaz que `cargar_modelo_pt`.
    """
    if not os.path.exists(TF_MODEL_PATH):
        return None, None
    import tensorflow as tf

    model = tf.keras.models.load_model(TF_MODEL_PATH)

    def preprocesar(img_pil):
        return np.asarray(img_pil.resize((IMG_SIZE, IMG_SIZE)), dtype="float32") / 255.0

    def predecir(img_pil):
        arr = preprocesar(img_pil)
        return model.predict(arr[None, ...], verbose=0)[0]

    # Reconstruimos el forward de forma funcional (compatible con Keras 3) para poder capturar
    # la salida de la ULTIMA capa Conv2D.
    inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = inp
    conv_output = None
    for layer in model.layers:
        x = layer(x)
        if isinstance(layer, tf.keras.layers.Conv2D):
            conv_output = x
    grad_model = tf.keras.models.Model(inp, [conv_output, x])

    def gradcam(img_pil):
        arr = preprocesar(img_pil)
        x = tf.convert_to_tensor(arr[None, ...], dtype=tf.float32)
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(x)
            class_idx = int(tf.argmax(preds[0]))
            loss = preds[:, class_idx]
        grads = tape.gradient(loss, conv_out)
        weights = tf.reduce_mean(grads, axis=(0, 1, 2))
        cam = tf.reduce_sum(conv_out[0] * weights, axis=-1)
        cam = tf.nn.relu(cam)
        cam = tf.image.resize(cam[..., tf.newaxis], (IMG_SIZE, IMG_SIZE)).numpy().squeeze()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx

    return predecir, gradcam


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

    pred_pt, cam_pt = cargar_modelo_pt()
    pred_tf, cam_tf = cargar_modelo_tf()
    if pred_pt is None and pred_tf is None:
        raise FileNotFoundError("No hay modelos entrenados. Corre primero los scripts de entrenamiento.")

    modelos = []
    if pred_pt:
        modelos.append(("PyTorch", pred_pt, cam_pt))
    if pred_tf:
        modelos.append(("Keras", pred_tf, cam_tf))

    # Desglose completo en consola.
    for ruta in rutas:
        img = Image.open(ruta).convert("RGB")
        print("=" * 40)
        print(os.path.basename(ruta))
        for nombre, predecir, _ in modelos:
            print(f"{nombre}:\n" + formatear_todas(predecir(img)))
    print("=" * 40)

    # Grilla: por cada foto, la imagen original + el heatmap Grad-CAM de cada modelo disponible
    # (zona de la cara en la que se fijó el modelo para decidir el rango etario).
    n = len(rutas)
    cols = 1 + len(modelos)
    fig, axes = plt.subplots(n, cols, figsize=(cols * 3.2, n * 3.6))
    axes = np.array(axes).reshape(n, cols)
    for i, ruta in enumerate(rutas):
        img = Image.open(ruta).convert("RGB")
        img_resized = img.resize((IMG_SIZE, IMG_SIZE))

        axes[i, 0].imshow(img_resized)
        axes[i, 0].set_title(os.path.basename(ruta), fontsize=8, loc="left")
        axes[i, 0].axis("off")

        for j, (nombre, _, gradcam) in enumerate(modelos, start=1):
            cam, clase_idx = gradcam(img)
            axes[i, j].imshow(img_resized)
            axes[i, j].imshow(cam, cmap="jet", alpha=0.5)
            axes[i, j].set_title(f"{nombre}: {CLASS_NAMES[clase_idx]}", fontsize=8, loc="left")
            axes[i, j].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
