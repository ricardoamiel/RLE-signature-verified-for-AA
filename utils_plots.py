# utils_plots.py
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc

def plot_roc(y_true, y_prob, out_png):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("Curva ROC (validación)")
    plt.legend(loc="lower right")
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(out_png); plt.close()
    return float(roc_auc)

def plot_pr(y_true, y_prob, out_png):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Curva Precision–Recall (validación)")
    plt.legend(loc="lower left")
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(out_png); plt.close()
    return float(pr_auc)

def plot_loss_curve(history, out_png):
    # history: list of dicts con keys 'epoch', 'train_loss'
    xs = [h["epoch"] for h in history]
    ys = [h["train_loss"] for h in history]
    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Época")
    plt.ylabel("Loss (train)")
    plt.title("Curva de pérdida (train)")
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(out_png); plt.close()