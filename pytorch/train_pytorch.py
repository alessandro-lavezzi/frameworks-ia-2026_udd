"""Entrenamiento y evaluación del clasificador de edad en PyTorch. Guarda el modelo para inferencia."""
import os
import sys
import time
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
from data.dataset import IMG_SIZE, BATCH_SIZE, CLASS_NAMES, build_dataframe, get_splits

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "age_cnn_pytorch.pt")
EPOCHS = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mismo preprocesamiento en entrenamiento e inferencia.
transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
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
    def __init__(self, num_classes=5):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # IMG_SIZE=64 -> tras 3 poolings /2: 64 -> 32 -> 16 -> 8
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
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


def main():
    print("Device:", device)
    train_df, val_df, test_df = get_splits(build_dataframe())
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    train_loader = DataLoader(UTKFaceDataset(train_df, transform), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(UTKFaceDataset(val_df, transform), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(UTKFaceDataset(test_df, transform), batch_size=BATCH_SIZE, shuffle=False)

    model = AgeCNN(num_classes=len(CLASS_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

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

    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=CLASS_NAMES).plot()
    plt.title("Matriz de confusión — PyTorch")
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
    plt.title("Curva ROC multiclase — PyTorch")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.show()

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Modelo guardado en: {MODEL_PATH}")


if __name__ == "__main__":
    main()
