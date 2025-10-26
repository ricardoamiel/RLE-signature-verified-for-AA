# cv_train.py
import os, random, json
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve

from dataloader import get_loaders
from model import SiameseBin
from utils_plots import plot_roc, plot_pr, plot_loss_curve

# ======================
# Configuración general
# ======================
DATA_CSV   = "out/signatures_index.csv"
NORM_CSV   = "out/normalization_stats.csv"
OUT_DIR    = Path("runs/siamese_cv")
EPOCHS     = 20
LR         = 1e-3
FOLDS      = 5
TARGET_PRECISION = 0.85
USE_AMP    = True

def set_seed_all(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ======================
# Funciones auxiliares
# ======================
def compute_metrics(y_true, y_prob):
    auc = roc_auc_score(y_true, y_prob) if len(set(y_true))>1 else float("nan")
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
        if gap < best_gap: best_gap, eer_val = gap, 0.5*(fpr+fnr)
    return {"auc": float(auc), "eer": float(eer_val or 0.0)}

def calibrate_threshold(y_true, y_prob, target=0.85):
    precisions, recalls, ths = precision_recall_curve(y_true, y_prob)
    thr = 0.5
    for p, t in zip(precisions[:-1], ths):
        if p >= target: thr = float(t); break
    y_pred = (y_prob >= thr).astype(int)
    tp = ((y_pred==1)&(y_true==1)).sum(); fp = ((y_pred==1)&(y_true==0)).sum(); fn = ((y_pred==0)&(y_true==1)).sum()
    prec = tp / max(tp+fp,1); rec = tp / max(tp+fn,1)
    return thr, float(prec), float(rec)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); y_true,y_prob,total_loss,n= [],[],0.0,0
    for x1,x2,y,_ in loader:
        x1,x2,y = x1.to(device),x2.to(device),y.to(device)
        logit,_ = model(x1,x2)
        prob = torch.sigmoid(logit).cpu().numpy()
        total_loss += F.binary_cross_entropy_with_logits(logit, y).item() * y.size(0)
        n += y.size(0)
        y_true.append(y.cpu().numpy()); y_prob.append(prob)
    y_true, y_prob = np.concatenate(y_true), np.concatenate(y_prob)
    metrics = compute_metrics(y_true, y_prob)
    return metrics, y_true, y_prob, total_loss/max(n,1)

def train_one_epoch(model, loader, opt, device, scaler=None):
    model.train(); tot=0.0
    for x1,x2,y,_ in loader:
        x1,x2,y = x1.to(device),x2.to(device),y.to(device)
        with torch.cuda.amp.autocast(enabled=USE_AMP, dtype=torch.float16):
            logit,_ = model(x1,x2)
            loss = F.binary_cross_entropy_with_logits(logit, y)
        if scaler: scaler.scale(loss).backward()
        else: loss.backward()
        if scaler:
            scaler.step(opt); scaler.update()
        else: opt.step()
        opt.zero_grad(set_to_none=True)
        tot += loss.item() * y.size(0)
    return tot/len(loader.dataset)

# ======================
# Entrenamiento por fold
# ======================
def train_fold(fold, train_idx, val_idx, df_all):
    out_fold = OUT_DIR / f"fold_{fold}"
    out_fold.mkdir(parents=True, exist_ok=True)

    # Generar CSV temporal con split=train/val
    df_all["split"] = "train"
    df_all.loc[val_idx, "split"] = "val"
    tmp_csv = out_fold / f"split_{fold}.csv"
    df_all.to_csv(tmp_csv, index=False)

    train_loader, val_loader, _ = get_loaders(str(tmp_csv), NORM_CSV)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseBin().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP and torch.cuda.is_available())

    log_rows = []
    best_key = (-1.0, 1.0, 1e9)

    for epoch in range(1,EPOCHS+1):
        train_loss = train_one_epoch(model, train_loader, opt, device, scaler)
        val_metrics, y_true, y_prob, val_loss = evaluate(model, val_loader, device)
        auc, eer = val_metrics["auc"], val_metrics["eer"]
        log_rows.append({"epoch":epoch,"train_loss":train_loss,"val_loss":val_loss,"auc":auc,"eer":eer})

        print(f"[Fold {fold} | Ep {epoch:02d}] loss={train_loss:.4f} val_loss={val_loss:.4f} auc={auc:.4f} eer={eer:.4f}")

        key = (auc, -eer, -val_loss)
        if key > best_key:
            best_key = key
            ckpt = out_fold/"best.pt"
            torch.save({"model":model.state_dict()}, ckpt)
            thr, prec, rec = calibrate_threshold(y_true,y_prob,TARGET_PRECISION)
            calib = {"threshold":thr,"auc":auc,"eer":eer,"prec":prec,"rec":rec,"created_at":datetime.utcnow().isoformat()}
            with open(out_fold/"calibration.json","w") as f: json.dump(calib,f,indent=2)
            plot_roc(y_true,y_prob,out_fold/f"roc_ep{epoch:02d}.png")
            plot_pr (y_true,y_prob,out_fold/f"pr_ep{epoch:02d}.png")

    pd.DataFrame(log_rows).to_csv(out_fold/"train_log.csv",index=False)
    plot_loss_curve(log_rows, out_fold/"loss_curve.png")
    return best_key

# ======================
# Main CV Loop
# ======================
def main(folds=FOLDS):
    set_seed_all(42)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_CSV)
    groups = df["writer_id"].values
    gkf = GroupKFold(n_splits=folds)

    results=[]
    for f,(tr,vl) in enumerate(gkf.split(df, groups=groups),start=1):
        key = train_fold(f,tr,vl,df.copy())
        results.append({"fold":f,"auc":key[0],"eer":-key[1],"val_loss":-key[2]})
    df_res = pd.DataFrame(results)
    df_res.to_csv(OUT_DIR/"summary.csv",index=False)

    best_fold = df_res.sort_values(by=["auc","eer","val_loss"],ascending=[False,True,True]).iloc[0]["fold"]
    src = OUT_DIR/f"fold_{int(best_fold)}"
    dst = Path("runs/siamese_split")
    dst.mkdir(parents=True, exist_ok=True)
    for f in ["best.pt","calibration.json"]:
        if (src/f).exists(): 
            torch.save(torch.load(src/f, map_location="cpu"), dst/f)
    print(f"\n✅ Mejor fold: {best_fold} copiado a {dst}")

if __name__=="__main__":
    main()
