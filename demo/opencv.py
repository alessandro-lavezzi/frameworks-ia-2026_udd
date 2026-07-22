"""Demo en vivo: clasifica el rango etario desde la cámara en tiempo real, con OpenCV.

Complementa a infer.py (que trabaja sobre fotos fijas en fotos_prueba/): aquí la fuente
de la imagen es la webcam, y se muestra un panel por modelo disponible (PyTorch / Keras)
con el heatmap Grad-CAM superpuesto en vivo, para ver en qué se está fijando cada uno.

Reutiliza la carga de modelos y las funciones de predicción/Grad-CAM de infer.py, no las
duplica: solo cambia el origen de la imagen (cámara en vez de archivos).

Requiere haber corrido antes los scripts de entrenamiento (models/age_cnn_pytorch.pt y/o
models/age_cnn_keras.keras).

Controles:
  q / ESC  -> salir
"""
import os
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from infer import cargar_modelo_pt, cargar_modelo_tf, CLASS_NAMES

# Grad-CAM implica un forward + backward por modelo; correrlo cada frame haría caer los FPS
# de forma notoria, así que solo se recalcula cada N frames (el video se sigue viendo fluido).
FRAMES_POR_INFERENCIA = 5
TAMANO_PANEL = 320
ALPHA_HEATMAP = 0.45


def frame_a_pil(frame_bgr):
    """Convierte un frame de OpenCV (BGR) a imagen PIL RGB, el formato que esperan los modelos."""
    return Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))


def dibujar_panel(frame_bgr, nombre_modelo, cam, texto_prediccion):
    """Redimensiona el frame, superpone el heatmap Grad-CAM (si ya hay uno calculado) y el texto."""
    panel = cv2.resize(frame_bgr, (TAMANO_PANEL, TAMANO_PANEL))
    if cam is not None:
        heatmap = cv2.applyColorMap((cam * 255).astype("uint8"), cv2.COLORMAP_JET)
        heatmap = cv2.resize(heatmap, (TAMANO_PANEL, TAMANO_PANEL))
        panel = cv2.addWeighted(panel, 1 - ALPHA_HEATMAP, heatmap, ALPHA_HEATMAP, 0)
    cv2.rectangle(panel, (0, 0), (TAMANO_PANEL, 28), (0, 0, 0), -1)
    cv2.putText(panel, f"{nombre_modelo}: {texto_prediccion}", (6, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def main():
    pred_pt, cam_pt = cargar_modelo_pt()
    pred_tf, cam_tf = cargar_modelo_tf()
    if pred_pt is None and pred_tf is None:
        raise FileNotFoundError("No hay modelos entrenados. Corre primero los scripts de entrenamiento.")

    modelos = []
    if pred_pt:
        modelos.append(("PyTorch", cam_pt))
    if pred_tf:
        modelos.append(("Keras", cam_tf))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara (verifica que esté conectada y no esté en uso por otra app).")

    print("Cámara en vivo. Presiona 'q' o ESC para salir.")
    resultados = {nombre: (None, "...") for nombre, _ in modelos}
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("No se pudo leer un frame de la cámara.")
                break

            if frame_idx % FRAMES_POR_INFERENCIA == 0:
                img_pil = frame_a_pil(frame)
                for nombre, gradcam in modelos:
                    cam, clase_idx = gradcam(img_pil)
                    resultados[nombre] = (cam, CLASS_NAMES[clase_idx])
            frame_idx += 1

            paneles = [dibujar_panel(frame, nombre, *resultados[nombre]) for nombre, _ in modelos]
            vista = np.hstack(paneles) if len(paneles) > 1 else paneles[0]
            cv2.imshow("Clasificador de edad - en vivo (q/ESC para salir)", vista)

            tecla = cv2.waitKey(1) & 0xFF
            if tecla in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
