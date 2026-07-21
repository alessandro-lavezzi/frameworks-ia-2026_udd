"""Preparación del dataset UTKFace: parseo de nombres, balanceo por edad y etnia, y splits.

Fuente única: carpeta con todos los .jpg (sin subcarpetas por clase). El segmento etario
y la etnia se extraen del nombre de archivo '[edad]_[genero]_[raza]_[fecha].jpg'.
"""
import os
import re
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# --- Configuración general (compartida por todos los scripts) ---
# Absoluta y relativa a este archivo: así funciona sin importar desde qué carpeta se ejecute
# (por ejemplo, cuando pytorch/train_pytorch.py importa DATA_DIR corriendo desde la raíz).
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "UTKFace_data", "UTKFace")
IMG_SIZE = 64
BATCH_SIZE = 32
SEED = 42

CLASS_NAMES = ["niños", "jóvenes", "adulto joven", "adulto", "adulto mayor"]
# 'Others' (código 4) se excluye: agrupa etnias muy distintas y no aporta una categoría consistente.
RACE_NAMES = {0: "White", 1: "Black", 2: "Asian", 3: "Indian"}

np.random.seed(SEED)


def age_to_segment(age: int) -> int:
    """Edad entera -> índice de segmento (0-4)."""
    if age <= 18:
        return 0
    elif age <= 30:
        return 1
    elif age <= 45:
        return 2
    elif age <= 60:
        return 3
    return 4


def parse_age_from_filename(filename: str):
    match = re.match(r"^(\d+)_\d+_\d+_.+\.jpg$", os.path.basename(filename))
    return int(match.group(1)) if match else None


def parse_race_from_filename(filename: str):
    """Código de etnia, o None si el formato no calza o la etnia es 'Others' (excluida)."""
    parts = os.path.basename(filename).split("_")
    try:
        race = int(parts[2])
        return race if race in RACE_NAMES else None
    except (ValueError, IndexError):
        return None


def build_dataframe(verbose: bool = True) -> pd.DataFrame:
    """DataFrame balanceado por (segmento, etnia): mismo N° de imágenes por combinación (cap global)."""
    all_fps = glob.glob(os.path.join(DATA_DIR, "*.jpg"))
    if not all_fps:
        raise FileNotFoundError(
            f"No hay imágenes en '{DATA_DIR}'. Revisa que la carpeta UTKFace exista con los .jpg adentro."
        )

    records_by_bucket = {(s, r): [] for s in range(len(CLASS_NAMES)) for r in RACE_NAMES}
    skipped = 0
    for fp in all_fps:
        age = parse_age_from_filename(fp)
        race = parse_race_from_filename(fp)
        if age is None or race is None:
            skipped += 1
            continue
        records_by_bucket[(age_to_segment(age), race)].append(fp)

    # Cap global = mínimo entre las 20 combinaciones -> balanceo doble (edad y etnia).
    cap = min(len(fps) for fps in records_by_bucket.values())
    rng = np.random.RandomState(SEED)
    records = []
    for (label_idx, race), fps in records_by_bucket.items():
        for fp in rng.choice(fps, size=cap, replace=False):
            records.append({"filepath": fp, "label": label_idx, "race": race})

    df = pd.DataFrame(records, columns=["filepath", "label", "race"])
    if verbose:
        print(f"Totales: {len(all_fps)} | descartadas: {skipped} | cap global: {cap} | usadas: {len(df)}")
        print(df["label"].value_counts().sort_index().rename(index=lambda i: CLASS_NAMES[i]))
    return df


def get_splits(df: pd.DataFrame):
    """Split estratificado por clase: 70% train, 15% val, 15% test."""
    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=SEED)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=SEED)
    return train_df, val_df, test_df


if __name__ == "__main__":
    # Ejecutar directamente muestra las distribuciones (equivalente a las celdas de análisis del notebook).
    import matplotlib.pyplot as plt

    df = build_dataframe()

    # Distribución de clases (balanceada -> barras del mismo alto).
    counts = df["label"].value_counts().reindex(range(len(CLASS_NAMES)), fill_value=0)
    plt.figure(figsize=(6, 4))
    plt.bar(CLASS_NAMES, counts.values)
    plt.title("Distribución de clases (balanceada por edad y etnia)")
    plt.ylabel("N° de imágenes")
    plt.show()

    # Composición étnica por segmento (debe salir perfectamente pareja).
    tab = (
        df.assign(race_name=df["race"].map(RACE_NAMES))
          .pivot_table(index="label", columns="race_name", aggfunc="size", fill_value=0)
          .reindex(range(len(CLASS_NAMES)))
    )
    tab.index = [CLASS_NAMES[i] for i in tab.index]
    tab = tab.reindex(columns=list(RACE_NAMES.values()), fill_value=0)
    print(tab.to_string())
    ax = tab.plot(kind="bar", stacked=True, figsize=(8, 5), colormap="tab10")
    ax.set_title("Composición étnica por segmento etario (balanceada, sin 'Others')")
    ax.set_ylabel("N° de imágenes")
    ax.set_xlabel("Segmento etario")
    ax.legend(title="Etnia", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    plt.show()
