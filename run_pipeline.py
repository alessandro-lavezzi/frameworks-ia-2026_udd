"""Corre el pipeline completo del proyecto en orden: dataset -> entrenamiento -> inferencia.

1. data/dataset.py           - construye y verifica el dataset balanceado (gráficos de distribución)
2. pytorch/train_pytorch.py  - entrena y evalúa, por etnia, guarda models/age_cnn_pytorch[_<etnia>].pt
3. tensorflow/train_tensorflow.py - ídem en Keras, models/age_cnn_keras[_<etnia>].keras
4. demo/infer.py             - inferencia + Grad-CAM sobre las fotos en fotos_prueba/ (usa el modelo 'all')

Los pasos 2 y 3 se repiten una vez por etnia (ETNIAS_DISPONIBLES en data/dataset.py: cada etnia
individual + 'all' con todas juntas, para comparar). El undersampling dentro de cada corrida es
por sexo, no por etnia -- ver data/dataset.py.

Cada paso corre como proceso independiente (mismo comportamiento que ejecutarlo a mano).
Los gráficos de matplotlib de cada script se muestran igual que por separado: hay que
cerrar cada ventana para que el pipeline avance al siguiente gráfico o paso.
Si un paso falla, el pipeline se detiene ahí (no sigue con los siguientes).
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from data.dataset import ETNIAS_DISPONIBLES

PASOS_FIJOS_INICIO = [
    ("Dataset (verificación de balanceo)", ROOT / "data" / "dataset.py", []),
]
PASOS_FIJOS_FIN = [
    ("Inferencia + Grad-CAM (demo)", ROOT / "demo" / "infer.py", []),
]


def pasos_de_entrenamiento():
    pasos = []
    for etnia in ETNIAS_DISPONIBLES:
        pasos.append((f"Entrenamiento PyTorch (etnia={etnia})", ROOT / "pytorch" / "train_pytorch.py",
                       ["--etnia", etnia.lower()]))
        pasos.append((f"Entrenamiento TensorFlow/Keras (etnia={etnia})", ROOT / "tensorflow" / "train_tensorflow.py",
                       ["--etnia", etnia.lower()]))
    return pasos


def main():
    pasos = PASOS_FIJOS_INICIO + pasos_de_entrenamiento() + PASOS_FIJOS_FIN
    total = len(pasos)
    for i, (nombre, script, args) in enumerate(pasos, start=1):
        print("\n" + "=" * 60)
        print(f"PASO {i}/{total}: {nombre}  ({script.relative_to(ROOT)})")
        print("=" * 60)
        resultado = subprocess.run([sys.executable, str(script), *args], cwd=ROOT)
        if resultado.returncode != 0:
            print(f"\n'{nombre}' terminó con error (código {resultado.returncode}). Pipeline detenido.")
            sys.exit(resultado.returncode)
    print(f"\nPipeline completo: dataset -> PyTorch/Keras x {len(ETNIAS_DISPONIBLES)} etnias -> inferencia/demo.")


if __name__ == "__main__":
    main()
