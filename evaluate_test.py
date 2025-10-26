"""
evaluate_test.py
Evalúa el modelo siamés en split=='test' y genera:
- Métricas globales (AUC, EER, Prec/Rec/F1, Accuracy, FPR, FNR)
- STP proxy (proporción de pares auto-validados >= threshold)
- Plots ROC y Precision-Recall
- CSV con predicciones por par

Ejemplo:
python evaluate_test.py \
  --data_csv out/signatures_index_with_split.csv \
  --norm_csv out/normalization_stats.csv \
  --ckpt runs/siamese_split/best.pt \
  --calib runs/siamese_split/calibration.json \
  --out_dir runs/siamese_split

"""

import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc

from dataloader import PairSampler, ToGrayscale, Resize, ToTensor, Normalize, Compose
from model import SiameseBin
from utils_plots import plot_roc, plot_pr

# -----------------------------
# Helpers
# -----------------------------
def load_norm(norm_csv):
    s = pd.read_csv(norm_csv).iloc[0]
    return float(s["mean_gray"]), float(s["std_gray"])

def build_eval_loader(data_csv, norm_csv, img_size=224, pairs_per_writer=100, batch_size=32):
    df = pd.read_csv(data_csv)
    mean_g, std_g = load_norm(norm_csv)
    tfm = Compose([ToGrayscale(), Resize((img_size,img_size)), ToTensor(), Normalize([mean_g],[std_g])])
    test_set = PairSampler(df, split="test", transform=tfm, n_pairs_per_writer=pairs_per_writer)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    return test_loader, df

def load_model(ckpt_path, emb_dim=256, device="cpu"):
    model = SiameseBin(emb_dim=emb_dim).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

def load_threshold(calib_path, fallback=0.85):
    if calib_path and Path(calib_path).exists():
        with open(calib_path) as f:
            d = json.load(f)
        return float(d.get("threshold", fallback)), d
    return float(fallback), None

def confusion_at_threshold(y_true, y_prob, thr):
    y_pred = (y_prob >= thr).astype(np.int32)
    tp = int(((y_pred==1)&(y_true==1)).sum())
    tn = int(((y_pred==0)&(y_true==0)).sum())
    fp = int(((y_pred==1)&(y_true==0)).sum())
    fn = int(((y_pred==0)&(y_true==1)).sum())

    prec = tp / max(tp+fp, 1)
    rec  = tp / max(tp+fn, 1)
    f1   = 2*prec*rec / max(prec+rec, 1e-8)
    acc  = (tp+tn) / max(tp+tn+fp+fn, 1)
    fpr  = fp / max(fp+tn, 1)
    fnr  = fn / max(fn+tp, 1)
    stp_proxy = (y_pred==1).mean()  # % auto-validados (pares con score >= thr)

    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "accuracy": float(acc),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "stp_proxy": float(stp_proxy)
    }

@torch.no_grad()
def collect_scores(model, loader, device):
    y_true, y_prob, writers = [], [], []
    for x1,x2,y,w in loader:                  # <- ahora también leemos w
        x1, x2 = x1.to(device), x2.to(device)
        logit,_ = model(x1,x2)
        prob = torch.sigmoid(logit).cpu().numpy()
        y_prob.append(prob)
        y_true.append(y.numpy())
        writers.append(np.array(w))
    y_true  = np.concatenate(y_true)
    y_prob  = np.concatenate(y_prob)
    writers = np.concatenate(writers).astype(int)
    return y_true, y_prob, writers

def per_writer_metrics(y_true, y_prob, writers, thr):
    rows = []
    for wid in np.unique(writers):
        m = (writers == wid)
        yt = y_true[m]; yp = y_prob[m]
        y_pred = (yp >= thr).astype(np.int32)

        tp = int(((y_pred==1)&(yt==1)).sum())
        tn = int(((y_pred==0)&(yt==0)).sum())
        fp = int(((y_pred==1)&(yt==0)).sum())
        fn = int(((y_pred==0)&(yt==1)).sum())

        prec = tp / max(tp+fp, 1)
        rec  = tp / max(tp+fn, 1)
        f1   = 2*prec*rec / max(prec+rec, 1e-8)
        acc  = (tp+tn) / max(tp+tn+fp+fn, 1)
        fpr  = fp / max(fp+tn, 1)
        fnr  = fn / max(fn+tp, 1)
        stp  = (y_pred==1).mean()

        rows.append({
            "writer_id": int(wid),
            "n_pairs": int(m.sum()),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "accuracy": float(acc),
            "fpr": float(fpr),
            "fnr": float(fnr),
            "stp_proxy_writer": float(stp)
        })
    return rows


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", type=str, default="out/signatures_index_with_split.csv")
    ap.add_argument("--norm_csv", type=str, default="out/normalization_stats.csv")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--calib", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default="runs/siamese_split")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--pairs_per_writer", type=int, default=120)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--emb_dim", type=int, default=256)
    ap.add_argument("--default_threshold", type=float, default=0.85)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "test_plots").mkdir(parents=True, exist_ok=True)

    device = "cpu" if (args.cpu or not torch.cuda.is_available()) else "cuda"

    # Loader de test
    test_loader, df = build_eval_loader(
        args.data_csv, args.norm_csv,
        img_size=args.img_size,
        pairs_per_writer=args.pairs_per_writer,
        batch_size=args.batch_size
    )

    # Modelo + threshold
    model = load_model(args.ckpt, emb_dim=args.emb_dim, device=device)
    thr, calib_meta = load_threshold(args.calib, fallback=args.default_threshold)

    # Scores
    y_true, y_prob, writers = collect_scores(model, test_loader, device)

    # AUC / EER
    auc_val = roc_auc_score(y_true, y_prob) if len(set(y_true))>1 else float("nan")
    # EER por barrido sencillo
    thresholds = np.linspace(0,1,201)
    best_gap, eer_val = 1.0, None
    for t in thresholds:
        y_pred = (y_prob >= t).astype(np.int32)
        tp = ((y_pred==1)&(y_true==1)).sum()
        tn = ((y_pred==0)&(y_true==0)).sum()
        fp = ((y_pred==1)&(y_true==0)).sum()
        fn = ((y_pred==0)&(y_true==1)).sum()
        fpr = fp / max(fp+tn,1)
        fnr = fn / max(fn+tp,1)
        gap = abs(fpr-fnr)
        if gap < best_gap:
            best_gap = gap
            eer_val = 0.5*(fpr+fnr)
    eer_val = float(eer_val if eer_val is not None else 0.0)

    # Confusión y métricas al threshold calibrado
    conf = confusion_at_threshold(y_true, y_prob, thr)
    rows_writer = per_writer_metrics(y_true, y_prob, writers, thr)
    per_writer_csv = out_dir / "test_per_writer.csv"
    pd.DataFrame(rows_writer).to_csv(per_writer_csv, index=False)
    print(f"Saved: {per_writer_csv.name}")

    # Plots
    roc_png = out_dir / "test_plots" / "roc_test.png"
    pr_png  = out_dir / "test_plots" / "pr_test.png"
    plot_roc(y_true, y_prob, roc_png)
    plot_pr(y_true, y_prob, pr_png)

    # CSV de predicciones (útil para auditoría / dashboard offline)
    pred_csv = out_dir / "test_predictions.csv"
    pd.DataFrame({
        "y_true": y_true.astype(int),
        "y_prob": y_prob
    }).to_csv(pred_csv, index=False)

    # JSON de resultados
    summary = {
        "timestamp": datetime.utcnow().isoformat(),
        "checkpoint": str(args.ckpt),
        "calibration_file": str(args.calib) if args.calib else None,
        "threshold_used": thr,
        "test_auc": float(auc_val),
        "test_eer": float(eer_val),
        "metrics_at_threshold": conf,
        "pairs_per_writer": args.pairs_per_writer,
        "batch_size": args.batch_size,
        "img_size": args.img_size,
        "emb_dim": args.emb_dim
    }
    with open(out_dir / "test_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print corto
    print("\n=== TEST SUMMARY ===")
    print(f"AUC: {auc_val:.4f} | EER: {eer_val:.4f}")
    print(f"Threshold (calibration): {thr:.4f}")
    print(f"Precision: {conf['precision']:.3f} | Recall: {conf['recall']:.3f} | F1: {conf['f1']:.3f} | Acc: {conf['accuracy']:.3f}")
    print(f"FPR: {conf['fpr']:.3f} | FNR: {conf['fnr']:.3f} | STP_proxy: {conf['stp_proxy']:.3f}")
    print(f"Saved: {roc_png.name}, {pr_png.name}, test_predictions.csv, test_summary.json")

if __name__ == "__main__":
    main()