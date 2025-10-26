"""
EDA de firmas (genuinas vs falsificadas)
- Estructura esperada:
  signatures/
    full_org/original_{writerId}_{sampleId}.png
    full_forg/forgeries_{writerId}_{sampleId}.png

Salida:
- /out/signatures_index.csv
- /out/normalization_stats.csv
- /out/eda_scatter_sizes.png
- /out/eda_hist_intensity.png
- /out/signatures_index_with_split.csv   (si enable_split=True)
"""

import os, re, glob, random
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# ============== Configuración ==============
DATA_DIR = Path("signatures")         # Cambia si tu dataset está en otra ruta
OUT_DIR  = Path("out")                # Carpeta de salida para CSVs y gráficos
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Control de split por writer (identidad)
ENABLE_SPLIT = True
N_TRAIN_WRITERS = 35
N_VAL_WRITERS   = 10
RANDOM_SEED     = 42

# ============== Utilidades ==============
def parse_info_from_name(fname: str):
    """
    Devuelve (writer_id, sample_id, label)
    label ∈ {"genuine", "forgery"} según el nombre del archivo.
    """
    bname = os.path.basename(fname)
    m_org = re.match(r"original_(\d+)_(\d+)\.png$", bname, flags=re.IGNORECASE)
    m_forg = re.match(r"forgeries_(\d+)_(\d+)\.png$", bname, flags=re.IGNORECASE)
    if m_org:
        return int(m_org.group(1)), int(m_org.group(2)), "genuine"
    if m_forg:
        return int(m_forg.group(1)), int(m_forg.group(2)), "forgery"
    return None, None, None

def scan_dataset(base_dir: Path) -> pd.DataFrame:
    """
    Recorre full_org y full_forg y arma el índice de imágenes.
    """
    rows = []
    org_dir  = base_dir / "full_org"
    forg_dir = base_dir / "full_forg"

    for p in glob.glob(str(org_dir / "*.png")):
        w, s, lab = parse_info_from_name(p)
        if lab: rows.append((p, w, s, lab))

    for p in glob.glob(str(forg_dir / "*.png")):
        w, s, lab = parse_info_from_name(p)
        if lab: rows.append((p, w, s, lab))

    df = pd.DataFrame(rows, columns=["path", "writer_id", "sample_id", "label"])
    return df.sort_values(["writer_id", "label", "sample_id"]).reset_index(drop=True)

def get_image_size(path: str):
    try:
        with Image.open(path) as im:
            return im.size  # (width, height)
    except Exception:
        return None

def plot_size_scatter(paths, out_png):
    """
    Dispersión de tamaños (w vs h) en una muestra.
    """
    sizes = [get_image_size(p) for p in paths]
    sizes = [s for s in sizes if s is not None]
    if not sizes:
        return None
    ws, hs = zip(*sizes)
    plt.figure()
    plt.scatter(ws, hs, s=14)
    plt.xlabel("Ancho (px)")
    plt.ylabel("Alto (px)")
    plt.title("Dispersión de tamaños de imagen")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    return out_png

def plot_intensity_hist(paths, out_png, max_imgs=300):
    """
    Histograma de intensidades (0-255) en escala de grises.
    """
    vals = []
    for p in paths[:max_imgs]:
        try:
            with Image.open(p) as im:
                g = np.array(im.convert("L"), dtype=np.uint8).flatten()
                vals.append(g)
        except Exception:
            pass
    if not vals:
        return None
    arr = np.concatenate(vals)
    plt.figure()
    plt.hist(arr, bins=32)
    plt.xlabel("Intensidad (0-255)")
    plt.ylabel("Frecuencia")
    plt.title("Histograma de intensidades (muestra)")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    return out_png

def compute_mean_std(paths, max_imgs=1000):
    """
    Mean y std en [0,1] de imágenes en escala de grises.
    """
    means, stds = [], []
    for p in paths[:max_imgs]:
        try:
            with Image.open(p) as im:
                g = np.array(im.convert("L"), dtype=np.float32) / 255.0
                means.append(g.mean())
                stds.append(g.std())
        except Exception:
            pass
    if not means:
        return 0.5, 0.5
    return float(np.mean(means)), float(np.mean(stds))

def split_by_writer(df: pd.DataFrame, seed=42, n_train=35, n_val=10):
    """
    Split por identidad: writers únicos → train/val/test.
    No mezcla firmas del mismo writer entre splits (evita fuga).
    """
    writers = sorted(df["writer_id"].unique().tolist())
    rng = random.Random(seed)
    rng.shuffle(writers)

    if len(writers) < (n_train + n_val):
        n_train = max(1, len(writers)//2)
        n_val = max(0, (len(writers)-n_train)//2)

    train_w = set(writers[:n_train])
    val_w   = set(writers[n_train:n_train+n_val])
    test_w  = set(writers[n_train+n_val:])

    def which_split(w):
        if w in train_w: return "train"
        if w in val_w:   return "val"
        return "test"

    df2 = df.copy()
    df2["split"] = df2["writer_id"].apply(which_split)
    return df2, train_w, val_w, test_w

# ============== EDA principal ==============
def main():
    if not (DATA_DIR / "full_org").exists() or not (DATA_DIR / "full_forg").exists():
        print(f"[WARN] No se encontró {DATA_DIR}/full_org o full_forg. "
              "Asegúrate de que la estructura del dataset es correcta.")
    df = scan_dataset(DATA_DIR)
    if df.empty:
        raise SystemExit("[ERROR] No se encontraron imágenes .png con el patrón esperado.")

    # 1) Guardar índice completo
    index_csv = OUT_DIR / "signatures_index.csv"
    df.to_csv(index_csv, index=False)
    print(f"[OK] Índice guardado en: {index_csv}")

    # 2) Conteos
    print("\nConteo por etiqueta:")
    print(df["label"].value_counts())
    print("\nWriters:", df["writer_id"].nunique())

    # 3) Gráficos
    paths = df["path"].tolist()
    scatter_png = plot_size_scatter(paths, OUT_DIR / "eda_scatter_sizes.png")
    hist_png    = plot_intensity_hist(paths, OUT_DIR / "eda_hist_intensity.png", max_imgs=300)
    print(f"[OK] Gráfico tamaños: {scatter_png}")
    print(f"[OK] Histograma intensidades: {hist_png}")

    # 4) Mean/Std para normalización
    mean_g, std_g = compute_mean_std(paths, max_imgs=1000)
    norm_csv = OUT_DIR / "normalization_stats.csv"
    pd.DataFrame([{"mean_gray": round(mean_g,6), "std_gray": round(std_g,6)}]).to_csv(norm_csv, index=False)
    print(f"[OK] Normalización (mean,std) → guardado en: {norm_csv} | mean={mean_g:.4f}, std={std_g:.4f}")

    # 5) Split por writer (opcional pero recomendado para verificación)
    if ENABLE_SPLIT:
        df_split, train_w, val_w, test_w = split_by_writer(
            df, seed=RANDOM_SEED, n_train=N_TRAIN_WRITERS, n_val=N_VAL_WRITERS
        )
        split_csv = OUT_DIR / "signatures_index_with_split.csv"
        df_split.to_csv(split_csv, index=False)
        print(f"[OK] Split por writer guardado en: {split_csv}")
        print(f"   - train writers: {len(train_w)} | val writers: {len(val_w)} | test writers: {len(test_w)}")

if __name__ == "__main__":
    main()
