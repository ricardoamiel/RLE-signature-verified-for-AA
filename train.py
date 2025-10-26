# train.py
import os, random, json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_recall_curve

from dataloader import get_loaders
from model import SiameseBin
from utils_plots import plot_roc, plot_pr, plot_loss_curve

# -------------------- Config --------------------
TARGET_PRECISION = 0.85
EPOCHS           = 20
LR               = 1e-3
OUT_DIR          = Path("runs/siamese_split")
DATA_CSV         = "out/signatures_index_with_split.csv"
NORM_CSV         = "out/normalization_stats.csv"

USE_AMP          = True
GRAD_ACCUM       = 1
PRINT_GPU_EVERY  = 50

# -------------------- Seeds --------------------
def set_seed_all(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -------------------- Métricas --------------------
def compute_metrics(y_true, y_prob):
    auc = roc_auc_score(y_true, y_prob) if len(set(y_true))>1 else float("nan")
    # EER por barrido simple
    thresholds = np.linspace(0,1,201)
    best_gap, eer_val = 1.0, None
    for t in thresholds:
        y_pred = (y_prob >= t).astype(np.int32)
        tp = ((y_pred==1)&(y_true==1)).sum()
        tn = ((y_pred==0)&(y_true==0)).sum()
        fp = ((y_pred==1)&(y_true==0)).sum()
        fn = ((y_pred==0)&(y_true==1)).sum()
        fpr = fp / max(fp+tn,1); fnr = fn / max(fn+tp,1)
        gap = abs(fpr-fnr)
        if gap < best_gap:
            best_gap = gap
            eer_val = 0.5*(fpr+fnr)
    return {"auc": float(auc), "eer": float(eer_val if eer_val is not None else 0.0)}

def calibrate_threshold_for_precision(y_true, y_prob, target_precision=0.85):
    precisions, recalls, ths = precision_recall_curve(y_true, y_prob)
    thr = 0.5
    for p, t in zip(precisions[:-1], ths):
        if p >= target_precision:
            thr = float(t); break
    y_pred = (y_prob >= thr).astype(int)
    tp = ((y_pred==1)&(y_true==1)).sum()
    fp = ((y_pred==1)&(y_true==0)).sum()
    fn = ((y_pred==0)&(y_true==1)).sum()
    prec = tp / max(tp+fp,1); rec = tp / max(tp+fn,1)
    return thr, float(prec), float(rec)

# -------------------- Train/Eval --------------------
def train_one_epoch(model, loader, opt, device, scaler=None, epoch=1):
    model.train(); total_loss = 0.0
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    opt.zero_grad(set_to_none=True)

    for step, (x1,x2,y,_) in enumerate(loader, start=1):
        x1,x2,y = x1.to(device), x2.to(device), y.to(device)
        with torch.cuda.amp.autocast(enabled=USE_AMP, dtype=torch.float16):
            logit,_ = model(x1,x2)
            loss = F.binary_cross_entropy_with_logits(logit, y) / GRAD_ACCUM

        if scaler: scaler.scale(loss).backward()
        else: loss.backward()

        if step % GRAD_ACCUM == 0:
            if scaler: scaler.step(opt); scaler.update()
            else: opt.step()
            opt.zero_grad(set_to_none=True)

        total_loss += loss.item() * y.size(0) * GRAD_ACCUM

        if torch.cuda.is_available() and (step % PRINT_GPU_EVERY == 0):
            used = torch.cuda.memory_allocated()/1024**2
            reserved = torch.cuda.memory_reserved()/1024**2
            peak = torch.cuda.max_memory_allocated()/1024**2
            print(f"[GPU][ep {epoch} step {step}] used={used:.0f}MB reserved={reserved:.0f}MB peak={peak:.0f}MB")

    epoch_loss = total_loss / len(loader.dataset)
    peak = torch.cuda.max_memory_allocated()/1024**2 if torch.cuda.is_available() else 0.0
    return epoch_loss, float(peak)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); y_true,y_prob = [],[]; total_loss=0.0; n=0
    for x1,x2,y,_ in loader:
        x1,x2,y = x1.to(device),x2.to(device),y.to(device)
        logit,_ = model(x1,x2)
        prob = torch.sigmoid(logit).cpu().numpy()
        batch_loss = F.binary_cross_entropy_with_logits(logit, y).item()
        total_loss += batch_loss * y.size(0); n += y.size(0)
        y_true.append(y.cpu().numpy()); y_prob.append(prob)
    y_true = np.concatenate(y_true); y_prob = np.concatenate(y_prob)
    metrics = compute_metrics(y_true, y_prob)
    val_loss = total_loss / max(n,1)
    return metrics, y_true, y_prob, val_loss

# -------------------- Main --------------------
def main():
    set_seed_all(42)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR/"plots").mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, _ = get_loaders(DATA_CSV, NORM_CSV)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SiameseBin().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP and torch.cuda.is_available())

    log_rows = []
    # Criterio: mayor AUC, luego menor EER, luego menor val_loss
    best_key = (-1.0, 1.0, 1e9)

    for epoch in range(1, EPOCHS+1):
        # (opcional) re-muestreo determinista por época si tu PairSampler lo soporta
        if hasattr(train_loader, "dataset") and hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch, n_pairs_per_writer=None)

        train_loss, peak_vram = train_one_epoch(model, train_loader, opt, device, scaler, epoch)
        val_metrics, y_true, y_prob, val_loss = evaluate(model, val_loader, device)
        auc, eer = val_metrics["auc"], val_metrics["eer"]

        row = {"epoch": epoch, "train_loss": round(train_loss,6), "val_loss": round(val_loss,6),
               "val_auc": round(auc,6), "val_eer": round(eer,6), "gpu_peak_mb": round(peak_vram,1)}
        log_rows.append(row)
        print(f"[{epoch:02d}] loss={train_loss:.4f} val_loss={val_loss:.4f} auc={auc:.4f} eer={eer:.4f} peakVRAM={peak_vram:.0f}MB")

        current_key = (auc, -eer, -val_loss)
        if current_key > best_key:
            best_key = current_key
            ckpt_path = OUT_DIR / "best.pt"
            torch.save({"model": model.state_dict()}, ckpt_path)

            thr, prec, rec = calibrate_threshold_for_precision(y_true, y_prob, TARGET_PRECISION)
            calib = {
                "threshold": round(float(thr), 6),
                "target_precision": TARGET_PRECISION,
                "val_auc": round(float(auc), 6),
                "val_eer": round(float(eer), 6),
                "val_loss": round(float(val_loss), 6),
                "val_precision_at_threshold": round(prec, 6),
                "val_recall_at_threshold": round(rec, 6),
                "model_checkpoint": str(ckpt_path),
                "created_at": datetime.utcnow().isoformat()
            }
            with open(OUT_DIR / "calibration.json", "w") as f:
                json.dump(calib, f, indent=2)

            plot_roc(y_true, y_prob, OUT_DIR / "plots" / f"roc_epoch{epoch:02d}.png")
            plot_pr (y_true, y_prob, OUT_DIR / "plots" / f"pr_epoch{epoch:02d}.png")
            print(f"  -> Guardado: {ckpt_path.name}, calibration.json y plots")

        # “last”
        torch.save({"model": model.state_dict()}, OUT_DIR / "last.pt")
        pd.DataFrame(log_rows).to_csv(OUT_DIR / "train_log.csv", index=False)

    plot_loss_curve(log_rows, OUT_DIR / "plots" / "loss_curve.png")
    print(f"Entrenamiento finalizado. Mejor clave={best_key}")

if __name__ == "__main__":
    main()
