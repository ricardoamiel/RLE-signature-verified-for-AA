# threshold_sweep_test.py
# Barrido de umbrales en TEST para ver trade-off Precision vs STP
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.metrics import precision_recall_curve, roc_auc_score
from dataloader import PairSampler, ToGrayscale, Resize, ToTensor, Normalize, Compose
from model import SiameseBin

def load_norm(norm_csv):
    s = pd.read_csv(norm_csv).iloc[0]
    return float(s["mean_gray"]), float(s["std_gray"])

def build_test_loader(data_csv, norm_csv, img_size=224, pairs_per_writer=120, batch_size=32):
    df = pd.read_csv(data_csv)
    mean_g, std_g = load_norm(norm_csv)
    tfm = Compose([ToGrayscale(), Resize((img_size,img_size)), ToTensor(), Normalize([mean_g],[std_g])])
    ds = PairSampler(df, split="test", transform=tfm, n_pairs_per_writer=pairs_per_writer, seed=123)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return dl

def load_model(ckpt, emb_dim=256, device="cpu"):
    m = SiameseBin(emb_dim=emb_dim).to(device)
    state = torch.load(ckpt, map_location=device)
    state = state["model"] if isinstance(state, dict) and "model" in state else state
    m.load_state_dict(state, strict=True); m.eval()
    return m

@torch.no_grad()
def collect(model, loader, device):
    ys, ps = [], []
    for x1,x2,y,_ in loader:
        x1,x2 = x1.to(device), x2.to(device)
        logit,_ = model(x1,x2)
        p = torch.sigmoid(logit).cpu().numpy()
        ys.append(y.numpy()); ps.append(p)
    return np.concatenate(ys), np.concatenate(ps)

def metrics_at(y_true, y_prob, thr):
    y_pred = (y_prob >= thr).astype(np.int32)
    tp = ((y_pred==1)&(y_true==1)).sum()
    fp = ((y_pred==1)&(y_true==0)).sum()
    fn = ((y_pred==0)&(y_true==1)).sum()
    prec = tp / max(tp+fp,1)
    rec  = tp / max(tp+fn,1)
    stp  = (y_pred==1).mean()
    return prec, rec, stp

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", default="out/signatures_index_with_split.csv")
    ap.add_argument("--norm_csv", default="out/normalization_stats.csv")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pairs_per_writer", type=int, default=200)
    ap.add_argument("--img_size", type=int, default=224)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dl = build_test_loader(args.data_csv, args.norm_csv, args.img_size, args.pairs_per_writer)
    model = load_model(args.ckpt, device=device)
    y_true, y_prob = collect(model, dl, device)
    print(f"AUC(test) = {roc_auc_score(y_true, y_prob):.4f}")

    MIN_PREC = 0.85
    EPS = 1e-6  # tolerancia de precisión

    grid = np.linspace(0.0, 0.025, 401)  # ajusta el rango si tus scores son muy bajos
    best = None
    print("thr\tprecision\trecall\tstp")
    for t in grid:
        p,r,s = metrics_at(y_true, y_prob, t)
        print(f"{t:.4f}\t{p:.3f}\t\t{r:.3f}\t{s:.3f}")
        if p + EPS >= MIN_PREC:
            # Criterio: maximizar STP; si empata en STP, preferir el umbral más bajo (más cobertura)
            if (best is None) or (s > best[2] + 1e-12) or (abs(s - best[2]) <= 1e-12 and t < best[0]):
                best = (t,p,r,s)
    if best:
        print(f"\n>>> Mejor STP con precision>=0.85: thr={best[0]:.4f} | prec={best[1]:.3f} | rec={best[2]:.3f} | STP={best[3]:.3f}")
    else:
        print("\nNo hay umbral que cumpla precision>=0.85 en test (raro si AUC~1).")

if __name__ == "__main__":
    main()
