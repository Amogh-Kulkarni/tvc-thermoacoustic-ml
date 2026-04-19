import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


def per_class_metrics(y_true, y_pred, class_names):
    labels = list(range(len(class_names)))
    p, r, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    rows = []
    for i, name in enumerate(class_names):
        rows.append({
            'class': name,
            'precision': round(p[i], 4),
            'recall': round(r[i], 4),
            'f1': round(f1[i], 4),
            'support': int(sup[i]),
        })
    p_m, r_m, f1_m, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average='macro', zero_division=0)
    rows.append({
        'class': 'MACRO',
        'precision': round(p_m, 4),
        'recall': round(r_m, 4),
        'f1': round(f1_m, 4),
        'support': int(sum(sup)),
    })
    return rows


def bootstrap_accuracy_ci(y_true, y_pred, n_boot=1000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    accs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        accs.append((y_true[idx] == y_pred[idx]).mean())
    accs = np.array(accs)
    return {
        'accuracy': (y_true == y_pred).mean(),
        'ci_lower': np.percentile(accs, 2.5),
        'ci_upper': np.percentile(accs, 97.5),
        'ci_width': np.percentile(accs, 97.5) - np.percentile(accs, 2.5),
    }
