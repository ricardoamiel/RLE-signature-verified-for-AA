# recalibrate_thresholds.py
import argparse, json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from dataloader import PairSampler, ToGrayscale, Resize, ToTensor, Normalize, Compose
from model import SiameseBin

def load_norm(norm_csv):
    s = pd.read_csv(norm_csv).iloc[0]
    return float(s["mean_gray"]), float(s["std_gray"])

def build_val_loader(data_csv, norm_csv, img_size=224, pairs_per_writer=120, batch_size=32):
    df = pd.read_csv(data_csv)
    mean_g, std_g = load_norm(norm_csv)
    tfm = Compose([ToGrayscale(), Resize((img_size, img_size)), ToTensor(), Normalize([mean_g],[std_g])])
    val_set = PairSampler(df, split="val", transform=tfm, n_pairs_per_writer=pairs_per_writer, seed=42)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    return val_loader, df

def load_model(ckpt_path, emb_dim=256, device="cpu"):
    model = SiameseBin(emb_dim=emb_dim).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

def precision_recall_thresholds(y_true, y_prob):
    from sklearn.metrics import precision_recall_curve
    p, r, t = precision_recall_curve(y_true, y_prob)
    return p, r, t

def calibrate_threshold_max_stp(y_true, y_prob, min_precision=0.85, eps=1e-6):
    p, r, ths = precision_recall_thresholds(y_true, y_prob)
    best = {"t": 0.5, "stp": -1.0, "p": 0.0, "r": 0.0}
    for pi, ri, ti in zip(p[:-1], r[:-1], ths):
        if (pi + eps) >= min_precision:
            stp = float((y_prob >= ti).mean())
            if (stp > best["stp"] + 1e-12) or (abs(stp - best["stp"]) <= 1e-12 and ti < best["t"]):
                best = {"t": float(ti), "stp": stp, "p": float(pi), "r": float(ri)}
    if best["stp"] < 0:
        # fallback conservador
        ti = float(np.percentile(y_prob, 95))
        y_pred = (y_prob >= ti).astype(int)
        tp = ((y_pred==1)&(y_true==1)).sum(); fp = ((y_pred==1)&(y_true==0)).sum()
        pi = tp / max(tp+fp, 1)
        ri = tp / max((y_true==1).sum(), 1)
        stp = float((y_prob >= ti).mean())
        best = {"t": ti, "stp": stp, "p": float(pi), "r": float(ri)}
    return best["t"], best["p"], best["r"], best["stp"]

def per_writer_thresholds(y_true, y_prob, writers, min_precision_list=(0.85, 0.80), writer_min_pairs=60, eps=1e-6):
    out = {}
    for wid in np.unique(writers):
        m = (writers == wid)
        if m.sum() < writer_min_pairs:
            continue
        entry = {"n_pairs": int(m.sum())}
        for mp in min_precision_list:
            t, p, r, stp = calibrate_threshold_max_stp(y_true[m], y_prob[m], min_precision=mp, eps=eps)
            key = f"p{int(round(mp*100))}"
            entry[key] = {
                "threshold": float(t),
                "precision": float(p),
                "recall": float(r),
                "stp_proxy": float(stp)
            }
        out[int(wid)] = entry
    return out

@torch.no_grad()
def collect_logits_and_meta(model, loader, device):
    logits, y_true, writers = [], [], []
    for x1, x2, y, w in loader:
        x1, x2 = x1.to(device), x2.to(device)
        logit, _ = model(x1, x2)
        logits.append(logit.cpu().numpy())
        y_true.append(y.numpy())
        writers.append(np.array(w))
    logits = np.concatenate(logits)
    y_true = np.concatenate(y_true)
    writers = np.concatenate(writers).astype(int)
    return logits, y_true, writers

def fit_temperature(logits_val, y_true):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logits = torch.tensor(logits_val, dtype=torch.float32, device=device)
    y = torch.tensor(y_true, dtype=torch.float32, device=device)
    T = torch.nn.Parameter(torch.ones(1, device=device))
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=50)
    bce = nn.BCEWithLogitsLoss()
    def closure():
        opt.zero_grad()
        loss = bce(logits / T.clamp_min(1e-3), y)
        loss.backward()
        return loss
    opt.step(closure)
    return float(T.detach().clamp_min(1e-3).cpu().item())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", type=str, default="out/signatures_index_with_split.csv")
    ap.add_argument("--norm_csv", type=str, default="out/normalization_stats.csv")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="runs/siamese_split")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--pairs_per_writer", type=int, default=120)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--emb_dim", type=int, default=256)
    ap.add_argument("--per_writer", type=str, default="true")
    ap.add_argument("--writer_min_pairs", type=int, default=60)
    ap.add_argument("--temperature", type=str, default="false")  # por defecto off
    ap.add_argument("--save_npz", type=str, default="false")
    args = ap.parse_args()

    per_writer_flag = args.per_writer.lower() == "true"
    temp_flag       = args.temperature.lower() == "true"
    save_npz_flag   = args.save_npz.lower() == "true"

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = load_model(args.ckpt, emb_dim=args.emb_dim, device=device)
    val_loader, _ = build_val_loader(args.data_csv, args.norm_csv, img_size=args.img_size,
                                     pairs_per_writer=args.pairs_per_writer, batch_size=args.batch_size)

    logits, y_true, writers = collect_logits_and_meta(model, val_loader, device)
    T = 1.0
    if temp_flag:
        T = fit_temperature(logits, y_true)
        print(f"[Calibración] Temperature scaling T={T:.4f}")
    y_prob = 1.0 / (1.0 + np.exp(-logits / T))

    if save_npz_flag:
        np.savez(out_dir / "val_logits_probs.npz", logits=logits, y_true=y_true, writers=writers, T=np.array([T], dtype=np.float32))

    # Dos políticas: p85 y p80
    thresholds = {}
    for mp in (0.85, 0.80):
        t, p, r, stp = calibrate_threshold_max_stp(y_true, y_prob, min_precision=mp)
        key = f"p{int(round(mp*100))}"
        thresholds[key] = {
            "threshold": round(float(t), 6),
            "precision": round(float(p), 6),
            "recall": round(float(r), 6),
            "stp_proxy": round(float(stp), 6)
        }

    calib = {
        "model_checkpoint": str(args.ckpt),
        "temperature": round(float(T), 6),
        "thresholds": thresholds,
        "created_at": datetime.utcnow().isoformat()
    }
    with open(out_dir / "calibration_maxstp.json", "w") as f:
        json.dump(calib, f, indent=2)
    print(f"[OK] Guardado global → {out_dir/'calibration_maxstp.json'}")

    if per_writer_flag:
        tw = per_writer_thresholds(y_true, y_prob, writers,
                                   min_precision_list=(0.85, 0.80),
                                   writer_min_pairs=args.writer_min_pairs)
        with open(out_dir / "thresholds_by_writer.json", "w") as f:
            json.dump({str(k): v for k, v in tw.items()}, f, indent=2)
        print(f"[OK] Guardado por writer → {out_dir/'thresholds_by_writer.json'} (writers calibrados: {len(tw)})")

    print("\nResumen global:")
    for k, v in thresholds.items():
        print(f"  {k}: thr={v['threshold']:.6f} | P={v['precision']:.3f} | R={v['recall']:.3f} | STP={v['stp_proxy']:.3f} | T={T:.3f}")
    print("Listo.")
    
if __name__ == "__main__":
    main()
