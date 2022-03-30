from functools import partial
from typing import List

import numpy as np
import seqeval.metrics as seq_metric
import sklearn.metrics as cls_metric
from seqeval.scheme import IOB2
from snorkel.utils import probs_to_preds

# Maybe add metric, one versus all, multi-class auc.

def metric_to_direction(metric: str) -> str:
    if metric in ["acc", "f1_binary", "f1_micro", "f1_macro", "f1_weighted", "auc", "ap", "f1_max", "mcc"]:
        return "maximize"
    if metric in ["logloss", "brier", "ece"]:
        return "minimize"
    if metric in SEQ_METRIC:
        return "maximize"
    raise NotImplementedError(f"cannot automatically decide the direction for {metric}!")


def max_metric(probs, true_labels, metric, num=100, return_curve=False):
    p = probs[:,0]
    max_p = max(p)
    min_p = min(p)
    c_range = np.linspace(min_p, max_p, num)


    metric_vals = []
    for c in c_range:
        preds = np.zeros_like(p)
        preds[p < c] = 1
        metric_vals.append(metric(true_labels, preds))
    c_max_index = np.argmax(metric_vals)
    c_max = c_range[c_max_index]
    print(f'Max cutoff {c_max}')

    if return_curve:
        return metric_vals[c_max_index], c_range, metric_vals
    return metric_vals[c_max_index]

def f1_max(y_true: np.ndarray, y_proba: np.ndarray, **kwargs):
    return max_metric(y_proba, y_true, metric=cls_metric.f1_score, **kwargs)

def brier_score_loss(
    y_true: np.ndarray,
    y_proba: np.ndarray,
):
    r = len(np.unique(y_true))
    return np.mean((np.eye(r)[y_true] - y_proba) ** 2)


def accuracy_score_(y_true: np.ndarray, y_proba: np.ndarray, **kwargs):
    y_pred = probs_to_preds(y_proba, **kwargs)
    return cls_metric.accuracy_score(y_true, y_pred)


def mcc(y_true: np.ndarray, y_proba: np.ndarray, **kwargs):
    y_pred = probs_to_preds(y_proba, **kwargs)
    return cls_metric.matthews_corrcoef(y_true, y_pred)


def f1_score_(y_true: np.ndarray, y_proba: np.ndarray, average: str, **kwargs):
    if average == "binary" and len(np.unique(y_true)) > 2:
        return 0.0
    y_pred = probs_to_preds(y_proba, **kwargs)
    return cls_metric.f1_score(y_true, y_pred, average=average)


def recall_score_(y_true: np.ndarray, y_proba: np.ndarray, average: str, **kwargs):
    if average == "binary" and len(np.unique(y_true)) > 2:
        return 0.0
    y_pred = probs_to_preds(y_proba, **kwargs)
    return cls_metric.recall_score(y_true, y_pred, average=average)


def precision_score_(y_true: np.ndarray, y_proba: np.ndarray, average: str, **kwargs):
    if average == "binary" and len(np.unique(y_true)) > 2:
        return 0.0
    y_pred = probs_to_preds(y_proba, **kwargs)
    return cls_metric.precision_score(y_true, y_pred, average=average)


def average_precision_score_(y_true: np.ndarray, y_proba: np.ndarray, **kwargs):
    if len(np.unique(y_true)) > 2:
        return 0.0
    return cls_metric.average_precision_score(y_true, y_proba[:, 1])


def auc_score_(y_true: np.ndarray, y_proba: np.ndarray, **kwargs):
    if len(np.unique(y_true)) > 2:
        return 0.0
    fpr, tpr, thresholds = cls_metric.roc_curve(y_true, y_proba[:, 1], pos_label=1, **kwargs)
    return cls_metric.auc(fpr, tpr)


def f1_score_seq(y_true: List[List], y_pred: List[List], id2label: dict, strict=True):
    y_true = [[id2label[x] for x in y] for y in y_true]
    y_pred = [[id2label[x] for x in y] for y in y_pred]
    kwargs = {}
    if strict:
        kwargs["mode"] = "strict"
        kwargs["scheme"] = IOB2
    return seq_metric.f1_score(y_true, y_pred, **kwargs)


def precision_seq(y_true: List[List], y_pred: List[List], id2label: dict, strict=True):
    y_true = [[id2label[x] for x in y] for y in y_true]
    y_pred = [[id2label[x] for x in y] for y in y_pred]
    kwargs = {}
    if strict:
        kwargs["mode"] = "strict"
        kwargs["scheme"] = IOB2
    return seq_metric.precision_score(y_true, y_pred, **kwargs)


def recall_seq(y_true: List[List], y_pred: List[List], id2label: dict, strict=True):
    y_true = [[id2label[x] for x in y] for y in y_true]
    y_pred = [[id2label[x] for x in y] for y in y_pred]
    kwargs = {}
    if strict:
        kwargs["mode"] = "strict"
        kwargs["scheme"] = IOB2
    return seq_metric.recall_score(y_true, y_pred, **kwargs)

def ece(y_true, y_proba, bin_count=5, eps=0.01):
    n_data = len(y_true)
    indices = np.arange(n_data)
    y_pred = probs_to_preds(y_proba)
    pred_conf = np.array([probs[p] for probs, p in zip(y_proba, y_pred)])
    bin_ids = np.zeros(n_data)

    min_conf = np.min(pred_conf)
    max_conf = np.max(pred_conf)
    print(f'min_conf: {min_conf}, max_conf: {max_conf}')
    bins = np.linspace(min_conf, max_conf + eps, bin_count + 1)

    for i in range(bin_count):
        # Bin i contains all predictions with i/bin_count <= confidence < (i+1)/bin_count (u)
        conf_lower = pred_conf >= bins[i]
        conf_upper = pred_conf < bins[i+1]
        in_bin = conf_lower & conf_upper
        bin_ids[in_bin] = i
    
    bin_correct = np.zeros(bin_count)

    # for bin_id, pred, true in zip(bin_ids, y_pred, y_true):
    #     bin_sizes[bin_id] += 1
    #     if pred == true:
    #         bin_correct[bin_id] += 1

    bin_sizes = np.array([sum(bin_ids == i) for i in range(bin_count)])
    correct = y_pred == y_true
    bin_correct = np.array([sum(correct[bin_ids == i]) for i in range(bin_count)])
    
    bin_accs = []
    for n, corr in zip(bin_sizes, bin_correct):
        if n > 0:
            bin_accs.append(corr/n)
        else:
            bin_accs.append(0.0)

    bin_accs = np.array(bin_accs)
    cmean = lambda x: 0 if len(x) == 0 else np.mean(x)
    bin_conf = np.array([cmean(pred_conf[bin_ids == i]) for i in range(bin_count)])
    diffs = np.abs(bin_accs - bin_conf)

    ece = np.sum(bin_sizes*diffs)/np.sum(bin_sizes)
    return ece


    
    



METRIC = {
    "acc": accuracy_score_,
    "auc": auc_score_,
    "f1_binary": partial(f1_score_, average="binary"),
    "f1_micro": partial(f1_score_, average="micro"),
    "f1_macro": partial(f1_score_, average="macro"),
    "f1_max": f1_max,
    "f1_weighted": partial(f1_score_, average="weighted"),
    "recall_binary": partial(recall_score_, average="binary"),
    "recall_micro": partial(recall_score_, average="micro"),
    "recall_macro": partial(recall_score_, average="macro"),
    "recall_weighted": partial(recall_score_, average="weighted"),
    "precision_binary": partial(precision_score_, average="binary"),
    "precision_micro": partial(precision_score_, average="micro"),
    "precision_macro": partial(precision_score_, average="macro"),
    "precision_weighted": partial(precision_score_, average="weighted"),
    "logloss": cls_metric.log_loss,
    "brier": brier_score_loss,
    "ap": average_precision_score_,
    "mcc": mcc,
    "ece": ece
}

SEQ_METRIC = {
    "f1_seq": partial(f1_score_seq),
    "precision_seq": partial(precision_seq),
    "recall_seq": partial(recall_seq),
}


class AverageMeter:
    def __init__(self, names: List[str]):
        self.named_dict = {n: [] for n in names}

    def update(self, **kwargs):
        for k, v in kwargs.items():
            self.named_dict[k].append(v)

    def get_results(self):
        results = {
            n:  (np.mean(l), np.std(l))
            for n, l in self.named_dict.items()
            if len(l) > 0
        }
        return results
