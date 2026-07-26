"""Corre el pipeline completo del proyecto en orden: dataset -> entrenamiento -> inferencia.

1. data/dataset.py           - construye y verifica el dataset balanceado (gráficos de distribución)
2. pytorch/train_pytorch.py  - entrena y evalúa, por etnia, guarda models/age_cnn_pytorch[_<etnia>].pt
3. tensorflow/train_tensorflow.py - ídem en Keras, models/age_cnn_keras[_<etnia>].keras
4. demo/infer.py             - inferencia + Grad-CAM sobre las fotos en fotos_prueba/ (usa el modelo 'all')

Los entrenamientos usan hiperparámetros HARDCODEADOS (origen: optuna) definidos en cada train_*.py;
por eso el pipeline NO corre Optuna: la búsqueda es lo más lento y ya se hizo una vez (ver
HIPERPARAMETROS.md). Para re-hacer la búsqueda, correr a mano pytorch/optuna_pytorch.py y
tensorflow/optuna_tensorflow.py, o descomentar PASOS_OPTUNA más abajo.

Los pasos 2 y 3 se repiten una vez por etnia (ETNIAS_DISPONIBLES en data/dataset.py: cada etnia
individual + 'all' con todas juntas, para comparar). El undersampling dentro de cada corrida es
por sexo, no por etnia -- ver data/dataset.py.

Cada paso corre como proceso independiente (mismo comportamiento que ejecutarlo a mano).
En esta corrida los scripts guardan los gráficos en figuras/ (savefig) en vez de abrir ventanas,
así el pipeline avanza sin intervención. Si un paso falla, el pipeline se detiene ahí.
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
# Optuna queda FUERA del pipeline por defecto (los hparams ya están hardcodeados en los train_*.py).
# Para volver a incluir la búsqueda, agregá PASOS_OPTUNA al armado de `pasos` en main() y ajustá
# el presupuesto (--trials/--epochs). Es lento: ~40-60 min por framework en CPU.
PASOS_OPTUNA = [
    ("Optuna PyTorch (búsqueda de hiperparámetros)", ROOT / "pytorch" / "optuna_pytorch.py",
     ["--trials", "20", "--epochs", "5"]),
    ("Optuna TensorFlow/Keras (búsqueda de hiperparámetros)", ROOT / "tensorflow" / "optuna_tensorflow.py",
     ["--trials", "20", "--epochs", "5"]),
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
    # Optuna NO se incluye por defecto (hparams ya hardcodeados). Para re-buscar, insertá
    # `+ PASOS_OPTUNA` después de PASOS_FIJOS_INICIO.
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
    print(f"\nPipeline completo: dataset -> PyTorch/Keras x {len(ETNIAS_DISPONIBLES)} etnias "
          "-> inferencia/demo. Gráficos guardados en figuras/.")


if __name__ == "__main__":
    main()
