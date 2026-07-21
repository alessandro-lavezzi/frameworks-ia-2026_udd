"""Corre el pipeline completo del proyecto en orden: dataset -> entrenamiento -> inferencia.

1. data/dataset.py           - construye y verifica el dataset balanceado (gráficos de distribución)
2. pytorch/train_pytorch.py  - entrena y evalúa el modelo en PyTorch, guarda models/age_cnn_pytorch.pt
3. tensorflow/train_tensorflow.py - entrena y evalúa el modelo en Keras, guarda models/age_cnn_keras.keras
4. demo/infer.py             - inferencia + Grad-CAM sobre las fotos en fotos_prueba/

Cada paso corre como proceso independiente (mismo comportamiento que ejecutarlo a mano).
Los gráficos de matplotlib de cada script se muestran igual que por separado: hay que
cerrar cada ventana para que el pipeline avance al siguiente gráfico o paso.
Si un paso falla, el pipeline se detiene ahí (no sigue con los siguientes).
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

PASOS = [
    ("1/4 Dataset (verificación de balanceo)", ROOT / "data" / "dataset.py"),
    ("2/4 Entrenamiento PyTorch", ROOT / "pytorch" / "train_pytorch.py"),
    ("3/4 Entrenamiento TensorFlow/Keras", ROOT / "tensorflow" / "train_tensorflow.py"),
    ("4/4 Inferencia + Grad-CAM (demo)", ROOT / "demo" / "infer.py"),
]


def main():
    for nombre, script in PASOS:
        print("\n" + "=" * 60)
        print(f"PASO {nombre}  ({script.relative_to(ROOT)})")
        print("=" * 60)
        resultado = subprocess.run([sys.executable, str(script)], cwd=ROOT)
        if resultado.returncode != 0:
            print(f"\n'{nombre}' terminó con error (código {resultado.returncode}). Pipeline detenido.")
            sys.exit(resultado.returncode)
    print("\nPipeline completo: dataset -> PyTorch -> Keras -> inferencia/demo.")


if __name__ == "__main__":
    main()
