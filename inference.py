"""
inference.py
Inferencia para verificación de firmas con el modelo siamés entrenado.

Modos:
1) Par único:
   python inference.py --ckpt runs/siamese_split/best.pt \
                       --img1 signatures/full_org/original_3_10.png \
                       --img2 signatures/full_forg/forgeries_3_10.png \
                       --norm_csv out/normalization_stats.csv \
                       --calib runs/siamese_split/calibration.json

2) Batch por CSV (con columnas: img1,img2):
   python inference.py --ckpt runs/siamese_split/best.pt \
                       --pairs_csv data/pairs_eval.csv \
                       --out_csv data/pairs_scores.csv \
                       --norm_csv out/normalization_stats.csv \
                       --calib runs/siamese_split/calibration.json
"""

import json, argparse, os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from model import SiameseBin

# Transforms mínimos (independientes de torchvision)
class ToGrayscale:
    def __call__(self, img): return img.convert("L")

class Resize:
    def __init__(self, size): self.size = size if isinstance(size, tuple) else (size, size)
    def __call__(self, img): return img.resize(self.size, Image.BILINEAR)

class ToTensor:
    def __call__(self, img):
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = arr[None, ...] if arr.ndim == 2 else arr.transpose(2,0,1)
        return torch.from_numpy(arr)

class Normalize:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean)[:, None, None]
        self.std  = torch.tensor(std)[:, None, None]
    def __call__(self, x): return (x - self.mean) / (self.std + 1e-8)

class Compose:
    def __init__(self, tfms): self.tfms = tfms
    def __call__(self, x):
        for t in self.tfms: x = t(x)
        return x

def load_norm(norm_csv):
    s = pd.read_csv(norm_csv).iloc[0]
    return float(s["mean_gray"]), float(s["std_gray"])

def load_threshold(calib_path, fallback=0.85):
    calib = {"threshold": fallback, "target_precision": None, "val_auc": None,
             "created_at": None, "model_checkpoint": None}
    p = Path(calib_path) if calib_path else None
    if p and p.exists():
        try:
            with open(p) as f:
                d = json.load(f)
            calib.update(d)
        except Exception:
            pass
    return calib

def build_transform(img_size, norm_csv):
    mean_g, std_g = load_norm(norm_csv)
    return Compose([ToGrayscale(), Resize((img_size, img_size)), ToTensor(), Normalize([mean_g],[std_g])])

def load_model(ckpt_path, emb_dim=256, device="cpu"):
    model = SiameseBin(emb_dim=emb_dim).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

@torch.no_grad()
def score_pair(model, tfm, img1_path, img2_path, device="cpu"):
    x1 = tfm(Image.open(img1_path).convert("L")).unsqueeze(0).to(device)
    x2 = tfm(Image.open(img2_path).convert("L")).unsqueeze(0).to(device)
    logit, _ = model(x1, x2)
    prob = torch.sigmoid(logit).item()
    return float(prob)

def decide(prob, threshold):
    return "auto_validado" if prob >= threshold else "revision_manual"

def run_pair(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    model = load_model(args.ckpt, emb_dim=args.emb_dim, device=device)
    tfm = build_transform(args.img_size, args.norm_csv)
    calib = load_threshold(args.calib, fallback=args.default_threshold)
    thr = calib["threshold"]

    prob = score_pair(model, tfm, args.img1, args.img2, device=device)
    decision = decide(prob, thr)

    print(f"score={prob:.4f} | threshold={thr:.4f} | decision={decision}")
    # Salida JSON opcional
    if args.out_json:
        out = {
            "score": round(prob, 6),
            "threshold": thr,
            "decision": decision,
            "model_checkpoint": str(args.ckpt),
            "calibration": calib,
            "img1": args.img1,
            "img2": args.img2,
            "created_at": datetime.utcnow().isoformat()
        }
        with open(args.out_json, "w") as f: json.dump(out, f, indent=2)
        print(f"[OK] Guardado: {args.out_json}")

def run_batch(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    model = load_model(args.ckpt, emb_dim=args.emb_dim, device=device)
    tfm = build_transform(args.img_size, args.norm_csv)
    calib = load_threshold(args.calib, fallback=args.default_threshold)
    thr = calib["threshold"]

    df = pd.read_csv(args.pairs_csv)
    assert {"img1","img2"}.issubset(df.columns), "El CSV debe tener columnas: img1,img2"
    scores, decisions = [], []
    for _, row in df.iterrows():
        p = score_pair(model, tfm, row["img1"], row["img2"], device=device)
        scores.append(p); decisions.append(decide(p, thr))
    df_out = df.copy()
    df_out["score"] = scores
    df_out["threshold"] = thr
    df_out["decision"] = decisions
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.out_csv, index=False)
    print(f"[OK] Guardado: {args.out_csv} ({len(df_out)} filas)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="Ruta a best.pt")
    ap.add_argument("--norm_csv", type=str, default="out/normalization_stats.csv")
    ap.add_argument("--calib", type=str, default=None, help="Ruta a calibration.json (opcional)")
    ap.add_argument("--emb_dim", type=int, default=256)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--default_threshold", type=float, default=0.85)
    ap.add_argument("--cpu", action="store_true", help="Forzar CPU")

    sub = ap.add_subparsers(dest="mode", required=True)

    sp1 = sub.add_parser("pair", help="Inferir un par de imágenes")
    sp1.add_argument("--img1", type=str, required=True)
    sp1.add_argument("--img2", type=str, required=True)
    sp1.add_argument("--out_json", type=str, default=None)

    sp2 = sub.add_parser("batch", help="Inferir un CSV de pares (img1,img2)")
    sp2.add_argument("--pairs_csv", type=str, required=True)
    sp2.add_argument("--out_csv", type=str, required=True)

    args = ap.parse_args()
    if args.mode == "pair":
        run_pair(args)
    else:
        run_batch(args)

if __name__ == "__main__":
    main()
