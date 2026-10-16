import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             f1_score, mean_absolute_error, mean_squared_error, r2_score,
                             roc_auc_score)


def regression_metrics(y_log_true, y_log_pred):
    """Metrics in minutes, after undoing the log1p transform."""
    t, p = np.expm1(y_log_true), np.expm1(y_log_pred)
    return {
        "mae_min": float(mean_absolute_error(t, p)),
        "rmse_min": float(np.sqrt(mean_squared_error(t, p))),
        "r2": float(r2_score(t, p)),
        "median_abs_pct_err": float(np.median(np.abs(p - t) / t) * 100),
    }


def classification_metrics(y, prob):
    pred = (prob >= 0.5).astype(int)
    return {
        "auroc": float(roc_auc_score(y, prob)),
        "avg_precision": float(average_precision_score(y, prob)),
        "brier": float(brier_score_loss(y, prob)),
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred)),
        "positive_rate": float(np.mean(y)),
    }


def evaluate(task, y, pred):
    return regression_metrics(y, pred) if task == "duration" else classification_metrics(y, pred)
