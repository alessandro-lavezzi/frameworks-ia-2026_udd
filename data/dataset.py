"""Preparación del dataset UTKFace: parseo de nombres, filtrado por etnia y balanceo por
edad y sexo (undersampling por sexo), y splits.

Fuente única: carpeta con todos los .jpg (sin subcarpetas por clase). El segmento etario,
el sexo y la etnia se extraen del nombre de archivo '[edad]_[genero]_[raza]_[fecha].jpg'.

La etnia es un FILTRO (selecciona el subset con el que se entrena, ej. solo "white"), no
una dimensión de balanceo: el undersampling se aplica por sexo dentro de cada segmento
etario. Con etnia=None/"all" se usa el dataset completo (todas las etnias juntas), para
comparar contra las corridas por etnia individual.
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

CLASS_NAMES = ["menor", "adulto/a", "anciano/a"]  # 0-17, 18-60, 61+ (3 clases)
# 'Others' (código 4) se excluye: agrupa etnias muy distintas y no aporta una categoría consistente.
RACE_NAMES = {0: "White", 1: "Black", 2: "Asian", 3: "Indian"}
RACE_CODES = {name.lower(): code for code, name in RACE_NAMES.items()}
GENDER_NAMES = {0: "Hombre", 1: "Mujer"}
ETNIAS_DISPONIBLES = list(RACE_NAMES.values()) + ["all"]

np.random.seed(SEED)


def age_to_segment(age: int) -> int:
    """Edad entera -> índice de segmento (0-4)."""
    if age <= 17:
        return 0
    elif age <= 60:
        return 1
    return 2


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


def parse_gender_from_filename(filename: str):
    """Código de sexo (0=hombre, 1=mujer), o None si el formato no calza."""
    parts = os.path.basename(filename).split("_")
    try:
        gender = int(parts[1])
        return gender if gender in GENDER_NAMES else None
    except (ValueError, IndexError):
        return None


def resolve_etnia(etnia):
    """None/'all' -> sin filtro (todas las etnias). Nombre ('white') o código (0) -> código de raza."""
    if etnia is None or str(etnia).lower() == "all":
        return None
    if isinstance(etnia, str) and etnia.lower() in RACE_CODES:
        return RACE_CODES[etnia.lower()]
    code = int(etnia)
    if code not in RACE_NAMES:
        raise ValueError(f"Etnia inválida: {etnia!r}. Opciones: {ETNIAS_DISPONIBLES}.")
    return code


def build_dataframe(etnia=None, verbose: bool = True) -> pd.DataFrame:
    """DataFrame filtrado por etnia (o todas, si etnia=None/'all') y balanceado por
    (segmento, sexo): mismo N° de imágenes por combinación (cap global). El undersampling
    es por sexo, no por etnia -- la etnia solo selecciona el subset de entrada.
    """
    race_filter = resolve_etnia(etnia)
    all_fps = glob.glob(os.path.join(DATA_DIR, "*.jpg"))
    if not all_fps:
        raise FileNotFoundError(
            f"No hay imágenes en '{DATA_DIR}'. Revisa que la carpeta UTKFace exista con los .jpg adentro."
        )

    records_by_bucket = {(s, g): [] for s in range(len(CLASS_NAMES)) for g in GENDER_NAMES}
    skipped = 0
    for fp in all_fps:
        age = parse_age_from_filename(fp)
        race = parse_race_from_filename(fp)
        gender = parse_gender_from_filename(fp)
        if age is None or race is None or gender is None:
            skipped += 1
            continue
        if race_filter is not None and race != race_filter:
            continue
        records_by_bucket[(age_to_segment(age), gender)].append((fp, race))

    # Cap global = mínimo entre las 10 combinaciones (segmento x sexo) -> balanceo doble (edad y sexo).
    cap = min(len(entries) for entries in records_by_bucket.values())
    if cap == 0:
        raise ValueError(
            f"No hay suficientes imágenes para etnia={etnia!r}: alguna combinación (segmento, sexo) "
            "quedó vacía. Prueba con 'all' o revisa el dataset."
        )
    rng = np.random.RandomState(SEED)
    records = []
    for (label_idx, gender), entries in records_by_bucket.items():
        for i in rng.choice(len(entries), size=cap, replace=False):
            fp, race = entries[i]
            records.append({"filepath": fp, "label": label_idx, "race": race, "gender": gender})

    df = pd.DataFrame(records, columns=["filepath", "label", "race", "gender"])
    if verbose:
        etnia_label = RACE_NAMES.get(race_filter, "todas") if race_filter is not None else "todas"
        print(f"Etnia: {etnia_label} | Totales: {len(all_fps)} | descartadas: {skipped} | "
              f"cap global: {cap} | usadas: {len(df)}")
        print(df["label"].value_counts().sort_index().rename(index=lambda i: CLASS_NAMES[i]))
    return df


def get_splits(df: pd.DataFrame):
    """Split estratificado por clase: 70% train, 15% val, 15% test."""
    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=SEED)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=SEED)
    return train_df, val_df, test_df


if __name__ == "__main__":
    # Ejecutar directamente muestra las distribuciones (equivalente a las celdas de análisis del notebook).
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Verifica el balanceo del dataset para una etnia (o todas).")
    parser.add_argument("--etnia", default="all", choices=[e.lower() for e in ETNIAS_DISPONIBLES],
                         help="Etnia a usar, o 'all' para todas juntas (default: all).")
    args = parser.parse_args()

    df = build_dataframe(etnia=args.etnia)

    # Distribución de clases (balanceada -> barras del mismo alto).
    counts = df["label"].value_counts().reindex(range(len(CLASS_NAMES)), fill_value=0)
    plt.figure(figsize=(6, 4))
    plt.bar(CLASS_NAMES, counts.values)
    plt.title(f"Distribución de clases (balanceada por edad y sexo) — etnia: {args.etnia}")
    plt.ylabel("N° de imágenes")
    plt.show()

    # Composición por sexo por segmento (debe salir perfectamente pareja: es la dimensión balanceada).
    tab = (
        df.assign(gender_name=df["gender"].map(GENDER_NAMES))
          .pivot_table(index="label", columns="gender_name", aggfunc="size", fill_value=0)
          .reindex(range(len(CLASS_NAMES)))
    )
    tab.index = [CLASS_NAMES[i] for i in tab.index]
    tab = tab.reindex(columns=list(GENDER_NAMES.values()), fill_value=0)
    print(tab.to_string())
    ax = tab.plot(kind="bar", stacked=True, figsize=(8, 5), colormap="tab10")
    ax.set_title(f"Composición por sexo por segmento etario — etnia: {args.etnia} (undersampling por sexo)")
    ax.set_ylabel("N° de imágenes")
    ax.set_xlabel("Segmento etario")
    ax.legend(title="Sexo", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    plt.show()
