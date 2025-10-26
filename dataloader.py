import os
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader

# ======== TRANSFORMS SIN TORCHVISION ======== #
class ToGrayscale:
    def __call__(self, img): return img.convert("L")

class Resize:
    def __init__(self, size): self.size = size if isinstance(size, tuple) else (size, size)
    def __call__(self, img): return img.resize(self.size, Image.BILINEAR)

class RandomRotate:
    """Rotación leve, reproducible con RNG propio."""
    def __init__(self, degrees=3, rng=None):
        self.degrees = float(degrees)
        # rng debe ser np.random.RandomState o Generator compatible; si no, crea uno local
        self.rng = rng if rng is not None else np.random.RandomState(123)

    def __call__(self, img):
        angle = self.rng.uniform(-self.degrees, self.degrees)
        # fillcolor=255 mantiene fondo blanco (Pillow>=5.2)
        return img.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=255)

class ToTensor:
    def __call__(self, img):
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = arr[None, ...] if arr.ndim == 2 else arr.transpose(2,0,1)
        return torch.from_numpy(arr)

class Normalize:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean)[:, None, None]
        self.std = torch.tensor(std)[:, None, None]
    def __call__(self, x): return (x - self.mean) / (self.std + 1e-8)

class Compose:
    def __init__(self, tfms): self.tfms = tfms
    def __call__(self, x):
        for t in self.tfms: x = t(x)
        return x

# ======== DATASET Y SAMPLER ======== #
def build_writer_pool(df):
    pool = {}
    for _, r in df.iterrows():
        w = int(r["writer_id"]); lab = r["label"]
        pool.setdefault(w, {"genuine": [], "forgery": []})
        pool[w][lab].append(r["path"])
    return pool

class PairSampler(Dataset):
    """
    Dataset de pares (img1, img2, label).
    - Usa RNG propio (np.random.RandomState) para reproducibilidad.
    - Permite re-muestrear pares por época con set_epoch(epoch).
    """
    def __init__(self, df_split, split, transform, n_pairs_per_writer=200, seed=42):
        self.transform = transform
        self.base_seed = int(seed)
        self.split = split

        df = df_split[df_split["split"] == split].reset_index(drop=True)
        self.pool = build_writer_pool(df)
        self.writers = list(self.pool.keys())

        # RNG interno
        self.rng = np.random.RandomState(self.base_seed)
        # Construye los pares iniciales
        self.pairs = self._make_pairs(n_pairs_per_writer)

    def _make_pairs(self, n):
        pairs = []
        rng = self.rng  # alias
        for w in self.writers:
            org = self.pool[w]["genuine"]; forg = self.pool[w]["forgery"]
            # positivos (genuina-genuina del mismo writer)
            for _ in range(max(1, n//2)):
                if len(org) >= 2:
                    a, b = rng.choice(org, size=2, replace=False)
                    pairs.append((a, b, 1, w))
            # negativos (genuina-forgery del mismo writer)
            for _ in range(max(1, n//2)):
                if len(org)>=1 and len(forg)>=1:
                    a = rng.choice(org); b = rng.choice(forg)
                    pairs.append((a, b, 0, w))
        rng.shuffle(pairs)
        return pairs

    def set_epoch(self, epoch, n_pairs_per_writer=None):
        """
        Re-muestrea pares de forma determinista para la época dada.
        Si no pasas n_pairs_per_writer, mantiene el mismo tamaño.
        """
        # seed combinada para esta época
        seed_ep = self.base_seed + int(epoch)
        self.rng = np.random.RandomState(seed_ep)
        if n_pairs_per_writer is None:
            # mantener longitud previa / writers
            n_pairs_per_writer = max(2, len(self.pairs) // max(1, len(self.writers)))
        self.pairs = self._make_pairs(n_pairs_per_writer)

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        p1, p2, y, w = self.pairs[idx]
        im1, im2 = Image.open(p1).convert("L"), Image.open(p2).convert("L")
        if self.transform:
            im1, im2 = self.transform(im1), self.transform(im2)
        return im1, im2, torch.tensor(y, dtype=torch.float32), int(w)

# ======== FUNCION CARGA GLOBAL ======== #
def get_loaders(data_csv, norm_csv, img_size=224, pairs_per_writer=200, batch_size=32, seed=42):
    df = pd.read_csv(data_csv)
    norm = pd.read_csv(norm_csv).iloc[0].to_dict()
    mean_g, std_g = [float(norm["mean_gray"])], [float(norm["std_gray"])]

    # RNGs por split para transformar y muestrear de forma reproducible
    rng_train = np.random.RandomState(seed)
    rng_val   = np.random.RandomState(seed + 999)

    train_tfms = Compose([
        ToGrayscale(),
        Resize((img_size,img_size)),
        RandomRotate(3, rng=rng_train),   # rotación reproducible
        ToTensor(),
        Normalize(mean_g,std_g)
    ])
    eval_tfms  = Compose([
        ToGrayscale(),
        Resize((img_size,img_size)),
        ToTensor(),
        Normalize(mean_g,std_g)
    ])

    train_set = PairSampler(df, split="train", transform=train_tfms,
                            n_pairs_per_writer=pairs_per_writer, seed=seed)
    val_set   = PairSampler(df, split="val",   transform=eval_tfms,
                            n_pairs_per_writer=max(60, pairs_per_writer//2), seed=seed+999)

    # Nota: si quieres determinismo completo, pasa generator y worker_init_fn desde train.py
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, df
