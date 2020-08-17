import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score, confusion_matrix, classification_report
from typing import NamedTuple


class ItemsForMetricsComputation(NamedTuple):
    """
    Evaluation output (always contains labels), to be used
    to compute metrics.
    """

    predictions: np.ndarray
    label_ids: np.ndarray


def simple_accuracy(preds: np.ndarray, labels: np.ndarray):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    confusion_matrix_ = confusion_matrix(y_true=labels, y_pred=preds, labels=[0, 1])
    tp = int(confusion_matrix_[1][1])
    tn = int(confusion_matrix_[0][0])
    fp = int(confusion_matrix_[0][1])
    fn = int(confusion_matrix_[1][0])
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,

    }


def acc_and_f1_muti(preds, labels):
    acc = accuracy_score(y_true=labels, y_pred=preds)
    macro_f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
    micro_f1 = f1_score(y_true=labels, y_pred=preds, average='micro')
    confusion_matrix_ = confusion_matrix(y_true=labels, y_pred=preds)
    report = classification_report(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        'confusion_matrix': confusion_matrix_
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }