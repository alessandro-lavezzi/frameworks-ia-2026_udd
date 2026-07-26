"""Entrenamiento y evaluación del clasificador de edad en PyTorch. Guarda el modelo para inferencia.

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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import IMG_SIZE, BATCH_SIZE, CLASS_NAMES, ETNIAS_DISPONIBLES, FIGURAS_DIR, build_dataframe, get_splits

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
HPARAMS_PATH = os.path.join(MODELS_DIR, "best_hparams_pytorch.json")
EPOCHS = 10

# Hiperparámetros: el entrenamiento lee por defecto models/best_hparams_pytorch.json (lo que deja
# Optuna, versionado en git). Si el JSON no existe, cae a estos valores hardcodeados de respaldo,
# que provienen de la última búsqueda de Optuna (ver HIPERPARAMETROS.md). Así el flujo normal usa
# siempre el resultado de Optuna, pero el script no se rompe si falta el archivo.
HPARAMS_FALLBACK = {
    "n_conv_layers": 3,               # origen: optuna
    "base_channels": 16,              # origen: optuna
    "lr": 0.00028898883555273204,     # origen: optuna
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
    # 'all' conserva el nombre histórico (age_cnn_pytorch.pt) para no romper demo/infer.py,
    # que por defecto usa el modelo combinado. Las corridas por etnia van con sufijo aparte.
    if etnia.lower() == "all":
        return os.path.join(MODELS_DIR, "age_cnn_pytorch.pt")
    return os.path.join(MODELS_DIR, f"age_cnn_pytorch_{etnia.lower()}.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# `transform`: preprocesamiento base, sin augmentation. Se usa en val/test y también es el que
# importan infer.py y opencv.py para inferencia (ahí no queremos aleatoriedad).
transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# `train_transform`: solo para entrenamiento. El augmentation va ANTES de ToTensor/Normalize
# para que las transformaciones geométricas y de color actúen sobre la imagen PIL original.
# Flip horizontal y jitter leve tienen sentido en rostros; rotación se deja acotada (10°) para
# no distorsionar rasgos de los que depende la edad (arrugas, mentón, etc.).
train_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=10),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


class UTKFaceDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["filepath"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, row["label"]


class AgeCNN(nn.Module):
    """CNN con N bloques conv configurables (canales por bloque). La arquitectura la determinan
    los hiperparámetros de Optuna vía `channels`; sin argumentos reproduce la original (16,32,64,128).

    `self.last_conv` apunta a la última capa Conv2d, para que Grad-CAM (demo/infer.py) enganche
    ahí sin depender de un nombre fijo como 'conv4'.
    """

    def __init__(self, num_classes=len(CLASS_NAMES), channels=None, dropout=0.3):
        super().__init__()
        channels = channels if channels is not None else [16, 32, 64, 128]
        blocks = []
        self.last_conv = None
        in_ch = 3
        for out_ch in channels:
            conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
            self.last_conv = conv
            blocks.append(nn.Sequential(
                conv,
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
            ))
            in_ch = out_ch
        self.blocks = nn.ModuleList(blocks)

        spatial = IMG_SIZE // (2 ** len(channels))
        if spatial < 1:
            raise ValueError(f"Demasiados bloques conv para IMG_SIZE={IMG_SIZE}: el mapa de features llega a 0.")
        self.fc1 = nn.Linear(in_ch * spatial * spatial, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = x.flatten(1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


def run_epoch(model, loader, optimizer, criterion, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += imgs.size(0)
    return total_loss / total, correct / total


def main(etnia: str = "all"):
    print("Device:", device)
    train_df, val_df, test_df = get_splits(build_dataframe(etnia=etnia))
    print(f"Etnia: {etnia} | Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    train_loader = DataLoader(UTKFaceDataset(train_df, train_transform), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(UTKFaceDataset(val_df, transform), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(UTKFaceDataset(test_df, transform), batch_size=BATCH_SIZE, shuffle=False)

    hp = load_hparams(verbose=True)
    model = AgeCNN(num_classes=len(CLASS_NAMES), channels=channels_from_hparams(hp)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hp["lr"])

    t0 = time.time()
    for epoch in range(EPOCHS):
        tr_loss, tr_acc = run_epoch(model, train_loader, optimizer, criterion, train=True)
        va_loss, va_acc = run_epoch(model, val_loader, optimizer, criterion, train=False)
        print(f"Epoch {epoch+1}/{EPOCHS} | train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
              f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}")
    print(f"\nTiempo total de entrenamiento (PyTorch): {time.time() - t0:.2f} segundos")

    # --- Evaluación en test: probabilidades (softmax) para reporte, matriz de confusión y ROC ---
    model.eval()
    probs, y_true = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            p = F.softmax(model(imgs.to(device)), dim=1)
            probs.extend(p.cpu().numpy())
            y_true.extend(labels.numpy())
    probs, y_true = np.array(probs), np.array(y_true)
    y_pred = probs.argmax(1)

    os.makedirs(FIGURAS_DIR, exist_ok=True)
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=CLASS_NAMES).plot()
    plt.title(f"Matriz de confusión — PyTorch (etnia: {etnia})")
    plt.tight_layout()
    # plt.show()  # desactivado en esta corrida: se guarda en figuras/ para no bloquear el pipeline
    plt.savefig(os.path.join(FIGURAS_DIR, f"pytorch_confusion_{etnia}.png"), dpi=120)
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
    plt.title(f"Curva ROC multiclase — PyTorch (etnia: {etnia})")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    # plt.show()  # desactivado en esta corrida: se guarda en figuras/ para no bloquear el pipeline
    plt.savefig(os.path.join(FIGURAS_DIR, f"pytorch_roc_{etnia}.png"), dpi=120)
    plt.close()

    model_path = get_model_path(etnia)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Modelo guardado en: {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrena el clasificador de edad (PyTorch) para una etnia (o todas).")
    parser.add_argument("--etnia", default="all", choices=[e.lower() for e in ETNIAS_DISPONIBLES],
                         help="Etnia a usar, o 'all' para todas juntas (default: all).")
    args = parser.parse_args()
    main(etnia=args.etnia)
