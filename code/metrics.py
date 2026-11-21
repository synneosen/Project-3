import numpy as np

def accuracy(y_true, y_pred):
    return np.mean(y_pred == y_true)


import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
    roc_auc_score,
)


def accuracy(y_true, y_pred):
    return np.mean(y_pred == y_true)


def accuracy_se(y_true, y_pred):
    p = accuracy(y_true, y_pred)
    return np.sqrt(p * (1 - p) / len(y_true))


def per_class_scores(y_true, y_pred, n_classes=10):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(n_classes), zero_division=0
    )
    return precision, recall, f1


def confusion(y_true, y_pred, n_classes=10):
    return confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))


def roc_one_vs_rest(y_true, y_proba, n_classes=10):
    curves = {}
    aucs = np.zeros(n_classes)
    for k in range(n_classes):
        y_k = (y_true == k).astype(int)
        fpr, tpr, _ = roc_curve(y_k, y_proba[:, k])
        curves[k] = (fpr, tpr)
        aucs[k] = roc_auc_score(y_k, y_proba[:, k])
    return curves, aucs