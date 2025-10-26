"""
plot_per_writer.py
Versión simplificada con paleta azul (skyblue / steelblue)
Genera:
  - per-writer_precision_recall_f1.png
  - per-writer_fpr_fnr.png
  - per-writer_confusion_counts.png
  - (opcional) matriz de confusión 2x2 por writer
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Paleta de dos tonos
COLOR_LIGHT = "#87CEEB"   # skyblue
COLOR_DARK  = "#4682B4"   # steelblue

def _ensure_out(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

def plot_prf(df, out_png, topk=None):
    dfp = df.copy().sort_values("f1", ascending=False)
    if topk: dfp = dfp.head(topk)
    x = np.arange(len(dfp))
    w = 0.28

    plt.figure(figsize=(max(10, len(dfp)*0.35), 5))
    plt.bar(x - w, dfp["precision"], width=w, color=COLOR_DARK, label="Precision")
    plt.bar(x,       dfp["recall"],    width=w, color=COLOR_LIGHT, label="Recall")
    plt.bar(x + w, dfp["f1"],         width=w, color="#1E90FF", label="F1")  # dodgerblue
    plt.xticks(x, dfp["writer_id"].astype(str), rotation=90)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Precision / Recall / F1 por writer (ordenado por F1)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300); plt.close()

def plot_fpr_fnr(df, out_png, topk=None):
    dfp = df.copy().sort_values("fpr", ascending=False)
    if topk: dfp = dfp.head(topk)
    x = np.arange(len(dfp)); w = 0.35

    plt.figure(figsize=(max(10, len(dfp)*0.35), 5))
    plt.bar(x - w/2, dfp["fpr"], width=w, color=COLOR_LIGHT, label="FPR")
    plt.bar(x + w/2, dfp["fnr"], width=w, color=COLOR_DARK, label="FNR")
    plt.xticks(x, dfp["writer_id"].astype(str), rotation=90)
    plt.ylim(0, 1.05)
    plt.ylabel("Tasa")
    plt.title("FPR / FNR por writer (ordenado por FPR)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300); plt.close()

def plot_confusion_counts(df, out_png, topk=None):
    dfp = df.copy().sort_values("n_pairs", ascending=False)
    if topk: dfp = dfp.head(topk)
    x = np.arange(len(dfp))

    tp, fp, tn, fn = dfp["tp"], dfp["fp"], dfp["tn"], dfp["fn"]
    plt.figure(figsize=(max(10, len(dfp)*0.35), 6))
    p1 = plt.bar(x, tp, label="TP", color=COLOR_DARK)
    p2 = plt.bar(x, fp, bottom=tp, label="FP", color=COLOR_LIGHT)
    plt.xticks(x, dfp["writer_id"].astype(str), rotation=90)
    plt.ylabel("Conteo")
    plt.title("TP / FP por writer")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300); plt.close()

def plot_writer_matrix(df, writer_id, out_png):
    row = df[df["writer_id"] == writer_id]
    if row.empty:
        print(f"[WARN] writer_id {writer_id} no encontrado")
        return
    r = row.iloc[0]
    mat = np.array([[r["tn"], r["fp"]],
                    [r["fn"], r["tp"]]], dtype=float)
    plt.figure(figsize=(4,4))
    im = plt.imshow(mat, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks([0,1], ["Pred 0","Pred 1"])
    plt.yticks([0,1], ["Real 0","Real 1"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, int(mat[i,j]), ha="center", va="center", color="white" if mat[i,j]>mat.max()/2 else "black")
    plt.title(f"Matriz de confusión – writer {int(writer_id)}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--topk", type=int, default=None)
    ap.add_argument("--writer_example", type=int, default=None)
    args = ap.parse_args()

    out_dir = Path(args.out)
    _ensure_out(out_dir)

    df = pd.read_csv(args.csv)

    plot_prf(df, out_dir/"per-writer_precision_recall_f1.png", topk=args.topk)
    plot_fpr_fnr(df, out_dir/"per-writer_fpr_fnr.png", topk=args.topk)
    plot_confusion_counts(df, out_dir/"per-writer_confusion_counts.png", topk=args.topk)

    if args.writer_example is not None:
        plot_writer_matrix(df, args.writer_example, out_dir/f"confusion_writer_{args.writer_example}.png")

    print(f"[OK] Plots guardados en: {out_dir}")

if __name__ == "__main__":
    main()
