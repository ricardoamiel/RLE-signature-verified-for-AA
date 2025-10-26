# app.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from datetime import datetime
import tempfile, shutil, os, time
import pandas as pd
import torch

from model import SiameseBin
from inference import build_transform, load_model, load_threshold, score_pair

# ==================================================
# CONFIGURACIÓN GLOBAL
# ==================================================
CKPT_PATH = "runs/siamese_split/best.pt"
NORM_CSV  = "out/normalization_stats.csv"
CALIB     = "runs/siamese_split/calibration.json"
IMG_SIZE  = 224
EMB_DIM   = 256
LOG_CSV   = "runs/ops_logs.csv"
REF_INDEX_CSV = "out/signatures_index.csv"

DEFAULT_THRESHOLD_HIGH = 0.85
DEFAULT_THRESHOLD_LOW  = 0.80
USE_CALIBRATION = False

# ==================================================
# APP FASTAPI
# ==================================================
app = FastAPI(title="Signature Verification API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static folder
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Carga del modelo
device = "cuda" if torch.cuda.is_available() else "cpu"
model  = load_model(CKPT_PATH, emb_dim=EMB_DIM, device=device)
tfm    = build_transform(IMG_SIZE, NORM_CSV)

if USE_CALIBRATION:
    thr = load_threshold(CALIB, fallback=DEFAULT_THRESHOLD_HIGH)["threshold"]
else:
    thr = DEFAULT_THRESHOLD_HIGH

MODEL_VERSION = Path(CKPT_PATH).name
CALIB_VERSION = Path(CALIB).name if Path(CALIB).exists() else "default"

# ==================================================
# FUNCIONES AUXILIARES
# ==================================================
def append_log(row: dict):
    df = pd.DataFrame([row])
    Path(LOG_CSV).parent.mkdir(parents=True, exist_ok=True)
    header = not Path(LOG_CSV).exists()
    df.to_csv(LOG_CSV, mode="a", header=header, index=False)

def pick_threshold(override_threshold: float | None):
    if override_threshold is not None:
        return float(override_threshold)
    if USE_CALIBRATION:
        return load_threshold(CALIB, fallback=DEFAULT_THRESHOLD_HIGH)["threshold"]
    return DEFAULT_THRESHOLD_HIGH

def decide_band(prob: float, high: float, low: float):
    if prob >= high:
        return "auto_validado"
    elif prob >= low:
        return "revision_rapida"
    else:
        return "revision_manual"

def resolve_reference_path(writer_id: int, ref_sample_id: int | None = 1):
    if not Path(REF_INDEX_CSV).exists():
        raise FileNotFoundError(f"No existe {REF_INDEX_CSV}.")
    df = pd.read_csv(REF_INDEX_CSV)
    df = df[(df["writer_id"] == int(writer_id)) & (df["label"] == "genuine")]
    if df.empty:
        raise ValueError(f"No hay genuinas para writer_id={writer_id}")
    if ref_sample_id is not None:
        row = df[df["sample_id"] == int(ref_sample_id)]
        if not row.empty:
            return row.iloc[0]["path"]
    return df.iloc[0]["path"]

# ==================================================
# ENDPOINTS
# ==================================================
@app.get("/", response_class=HTMLResponse)
def index_page():
    idx = static_dir / "index.html"
    if idx.exists():
        return idx.read_text(encoding="utf-8")
    return HTMLResponse("<h3>Sube <code>static/index.html</code></h3>")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": device,
        "checkpoint": CKPT_PATH,
        "threshold_policy": "override > default(0.85)",
        "model_version": MODEL_VERSION,
        "calibration_version": CALIB_VERSION if USE_CALIBRATION else "ignored",
    }

# ---------- /score ----------
@app.post("/score")
async def score_endpoint(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...),
    override_threshold: float | None = Form(None),
    low_threshold: float | None = Form(None),
    doc_id: str | None = Form(None),
    client_id: str | None = Form(None),
    writer_id: str | None = Form(None),
):
    t0 = time.time()
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file1.filename).suffix) as f1, \
         tempfile.NamedTemporaryFile(delete=False, suffix=Path(file2.filename).suffix) as f2:
        shutil.copyfileobj(file1.file, f1)
        shutil.copyfileobj(file2.file, f2)
        p1, p2 = f1.name, f2.name

    high_thr = pick_threshold(override_threshold)
    low_thr  = float(low_threshold) if low_threshold is not None else DEFAULT_THRESHOLD_LOW

    try:
        score = score_pair(model, tfm, p1, p2, device=device)
        decision = decide_band(score, high_thr, low_thr)
        latency_ms = int((time.time() - t0) * 1000)
        append_log({
            "timestamp": datetime.utcnow().isoformat(),
            "doc_id": doc_id, "client_id": client_id, "writer_id": writer_id,
            "ref_path": file1.filename, "probe_path": file2.filename,
            "score": round(score,6), "threshold_high": high_thr, "threshold_low": low_thr,
            "decision": decision,
            "model_version": MODEL_VERSION, "calibration_version": CALIB_VERSION,
            "latency_ms": latency_ms, "status": "ok"
        })
        return JSONResponse({
            "score": round(score,6),
            "threshold": high_thr,
            "threshold_low": low_thr,
            "decision": decision,
            "latency_ms": latency_ms,
            "model_version": MODEL_VERSION,
            "calibration_version": CALIB_VERSION
        })
    finally:
        for p in (p1, p2):
            try: os.remove(p)
            except: pass

# ---------- /score_csv_upload ----------
@app.post("/score_csv_upload")
async def score_csv_upload(
    csv_file: UploadFile = File(...),
    override_threshold: float | None = Form(None),
    low_threshold: float | None = Form(None),
    client_id: str | None = Form(None),
):
    high_thr = pick_threshold(override_threshold)
    low_thr  = float(low_threshold) if low_threshold is not None else DEFAULT_THRESHOLD_LOW
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tf:
        shutil.copyfileobj(csv_file.file, tf)
        tmp_csv = tf.name
    df = pd.read_csv(tmp_csv)
    assert {"img1","img2"}.issubset(df.columns), "CSV debe tener columnas: img1,img2"

    scores, decisions, latencies = [], [], []
    for _, row in df.iterrows():
        t0 = time.time()
        s = score_pair(model, tfm, row["img1"], row["img2"], device=device)
        d = decide_band(s, high_thr, low_thr)
        scores.append(s); decisions.append(d)
        latencies.append(int((time.time()-t0)*1000))
        append_log({
            "timestamp": datetime.utcnow().isoformat(),
            "doc_id": row.get("doc_id"), "client_id": client_id or row.get("client_id"),
            "writer_id": row.get("writer_id"),
            "ref_path": row["img1"], "probe_path": row["img2"],
            "score": round(float(s),6),
            "threshold_high": high_thr, "threshold_low": low_thr,
            "decision": d,
            "model_version": MODEL_VERSION, "calibration_version": CALIB_VERSION,
            "latency_ms": latencies[-1], "status":"ok"
        })

    df_out = df.copy()
    df_out["score"] = [round(float(s),6) for s in scores]
    df_out["threshold_high"] = high_thr
    df_out["threshold_low"]  = low_thr
    df_out["decision"] = decisions
    out_csv = f"out/pairs_scores_upload_{int(time.time())}.csv"
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False)
    try: os.remove(tmp_csv)
    except: pass
    return {"saved": out_csv, "rows": len(df_out), "avg_latency_ms": sum(latencies)//max(len(latencies),1)}

# ---------- /score_writer ----------
@app.post("/score_writer")
async def score_writer(
    probe: UploadFile = File(...),
    writer_id: int = Form(...),
    ref_sample_id: int | None = Form(1),
    override_threshold: float | None = Form(None),
    low_threshold: float | None = Form(None),
    doc_id: str | None = Form(None),
    client_id: str | None = Form(None),
):
    high_thr = pick_threshold(override_threshold)
    low_thr  = float(low_threshold) if low_threshold is not None else DEFAULT_THRESHOLD_LOW
    ref_path = resolve_reference_path(writer_id, ref_sample_id)
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(probe.filename).suffix) as f2:
        shutil.copyfileobj(probe.file, f2)
        p2 = f2.name
    try:
        s = score_pair(model, tfm, ref_path, p2, device=device)
        d = decide_band(s, high_thr, low_thr)
        append_log({
            "timestamp": datetime.utcnow().isoformat(),
            "doc_id": doc_id, "client_id": client_id, "writer_id": writer_id,
            "ref_path": ref_path, "probe_path": probe.filename,
            "score": round(float(s),6),
            "threshold_high": high_thr, "threshold_low": low_thr,
            "decision": d,
            "model_version": MODEL_VERSION, "calibration_version": CALIB_VERSION,
            "latency_ms": 0, "status":"ok"
        })
        return JSONResponse({
            "score": round(float(s),6),
            "threshold": high_thr,
            "threshold_low": low_thr,
            "decision": d,
            "ref_path_used": ref_path
        })
    finally:
        try: os.remove(p2)
        except: pass
