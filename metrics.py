"""Métricas compartidas para la búsqueda de hiperparámetros (Optuna) y la evaluación.

Un único lugar donde se define CÓMO se calcula cada métrica, para que PyTorch y TensorFlow
(y el modelo de etnia del compañero) usen exactamente la misma definición. Todas las métricas
multiclase se calculan con `average="macro"`: promedian las clases con el mismo peso, así una
clase minoritaria no queda tapada por las mayoritarias (a diferencia de accuracy, que es
engañosa justamente en ese caso).
"""
from sklearn.metrics import f1_score, recall_score, roc_auc_score, accuracy_score

# Métricas que se pueden elegir como objetivo ÚNICO de Optuna (--metrica).
METRICAS_INDIVIDUALES = ["f1", "recall", "auc", "accuracy"]

# Métricas que se optimizan JUNTAS en modo multiobjetivo (--metrica multi, el default).
# El orden importa: define el orden de los valores en el frente de Pareto (values[0] = f1, etc.).
METRICAS_MULTIOBJETIVO = ["f1", "recall", "auc"]


def compute_metric(nombre: str, y_true, probs) -> float:
    """Calcula una métrica a partir de las etiquetas reales y las probabilidades (softmax).

    `probs` tiene forma (n_muestras, n_clases). AUC necesita las probabilidades; el resto usa
    la clase predicha (argmax).
    """
    y_pred = probs.argmax(1)
    if nombre == "f1":
        return f1_score(y_true, y_pred, average="macro")
    if nombre == "recall":
        return recall_score(y_true, y_pred, average="macro")
    if nombre == "auc":
        # One-vs-Rest: AUC de cada clase contra el resto, promediado macro. Independiente del umbral.
        return roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
    if nombre == "accuracy":
        return accuracy_score(y_true, y_pred)
    raise ValueError(f"Métrica desconocida: {nombre!r}. Opciones: {METRICAS_INDIVIDUALES}.")


def compute_all_metrics(y_true, probs) -> dict:
    """Todas las métricas de una, para reportes (consola, .md, interrogación oral)."""
    return {nombre: compute_metric(nombre, y_true, probs) for nombre in METRICAS_INDIVIDUALES}
