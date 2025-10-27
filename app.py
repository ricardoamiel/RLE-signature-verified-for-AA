# app.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from datetime import datetime
import tempfile, shutil, os, time, json
import pandas as pd
import torch

from inference import build_transform, load_model, score_pair

# ==================================================
# CONFIGURACIÓN (ajusta estas rutas según el experimento elegido)
# ==================================================
# SUGERIDO: usar el mejor checkpoint (ej. runs_first_approach) y su calibración MAX-STP
MODEL_ROOT = Path("runs_first_approach/siamese_split")
# Si prefieres el otro experimento, cambia a Path("runs/siamese_split")

CKPT_PATH            = str(MODEL_ROOT / "best.pt")
CALIB_MAXSTP_JSON    = str(MODEL_ROOT / "calibration_maxstp.json")      # global p85 y p80
CALIB_BY_WRITER_JSON = str(MODEL_ROOT / "thresholds_by_writer.json")    # opcional por writer
NORM_CSV             = "out/normalization_stats.csv"
REF_INDEX_CSV        = "out/signatures_index.csv"                       # índice con rutas genuinas

IMG_SIZE  = 224
EMB_DIM   = 256
LOG_CSV   = "runs/ops_logs.csv"

# Umbrales por defecto si NO hay calibración:
DEFAULT_THRESHOLD_HIGH = 0.85   # auto-validado
DEFAULT_THRESHOLD_LOW  = 0.80   # revisión rápida

# Usar calibración (si existe) por defecto
USE_CALIBRATION = True

# ==================================================
# APP FASTAPI
# ==================================================
app = FastAPI(title="Signature Verification API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Static (frontend)
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# ==================================================
# CARGA DE MODELO & TRANSFORMS
# ==================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
model  = load_model(CKPT_PATH, emb_dim=EMB_DIM, device=device)
tfm    = build_transform(IMG_SIZE, NORM_CSV)

MODEL_VERSION = Path(CKPT_PATH).name

# ==================================================
# CALIBRACIÓN: GLOBAL (p85/p80) + POR WRITER (opcional)
# ==================================================
def _safe_read_json(path: str):
    p = Path(path)
    if not p.exists(): return None
    try:
        with open(p, "r") as f:
            return json.load(f)
    except Exception:
        return None

def load_global_thresholds(calib_maxstp_json: str):
    """
    Espera estructura:
    {
      "thresholds": {
        "p85": {"threshold": float, "precision": ..., "recall": ..., "stp_proxy": ...},
        "p80": {"threshold": float, "precision": ..., "recall": ..., "stp_proxy": ...}
      },
      ...
    }
    """
    data = _safe_read_json(calib_maxstp_json)
    if not data or "thresholds" not in data:
        return {
            "p85": {"threshold": DEFAULT_THRESHOLD_HIGH},
            "p80": {"threshold": DEFAULT_THRESHOLD_LOW},
        }
    th = data["thresholds"]
    # Fallback si faltan claves
    p85_thr = th.get("p85", {}).get("threshold", DEFAULT_THRESHOLD_HIGH)
    p80_thr = th.get("p80", {}).get("threshold", DEFAULT_THRESHOLD_LOW)
    return {"p85": {"threshold": float(p85_thr)}, "p80": {"threshold": float(p80_thr)}}

def load_writer_thresholds(calib_by_writer_json: str):
    """
    Estructura esperada por writer_id (string o int):
    {
      "9": {
        "n_pairs": 120,
        "p85": {"threshold": ..., "precision": ..., "recall": ..., "stp_proxy": ...},
        "p80": {"threshold": ..., "precision": ..., "recall": ..., "stp_proxy": ...}
      },
      ...
    }
    """
    data = _safe_read_json(calib_by_writer_json)
    return data or {}

GLOBAL_CALIB = load_global_thresholds(CALIB_MAXSTP_JSON) if USE_CALIBRATION else {
    "p85": {"threshold": DEFAULT_THRESHOLD_HIGH}, "p80": {"threshold": DEFAULT_THRESHOLD_LOW}
}
WRITER_CALIB = load_writer_thresholds(CALIB_BY_WRITER_JSON) if USE_CALIBRATION else {}

CALIB_VERSION = Path(CALIB_MAXSTP_JSON).name if Path(CALIB_MAXSTP_JSON).exists() else "default"

# ==================================================
# AUXILIARES
# ==================================================
def append_log(row: dict):
    df = pd.DataFrame([row])
    Path(LOG_CSV).parent.mkdir(parents=True, exist_ok=True)
    header = not Path(LOG_CSV).exists()
    df.to_csv(LOG_CSV, mode="a", header=header, index=False)

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

def pick_thresholds(policy: str | None, override_high: float | None, override_low: float | None):
    """
    Devuelve (thr_high, thr_low, policy_used).
    - policy in {"dual", "p85_only"}; default "dual".
    - overrides mandan.
    - si USE_CALIBRATION=False → usa DEFAULTS.
    """
    policy = (policy or "dual").lower()
    if override_high is not None or override_low is not None:
        h = float(override_high) if override_high is not None else DEFAULT_THRESHOLD_HIGH
        l = float(override_low)  if override_low  is not None else DEFAULT_THRESHOLD_LOW
        return h, l, f"override({policy})"

    if USE_CALIBRATION:
        g_p85 = float(GLOBAL_CALIB["p85"]["threshold"])
        g_p80 = float(GLOBAL_CALIB["p80"]["threshold"])
        if policy == "p85_only":
            return g_p85, g_p80, "calib(p85_only)"
        # dual por defecto
        return g_p85, g_p80, "calib(dual)"

    # sin calibración → defaults
    if policy == "p85_only":
        return DEFAULT_THRESHOLD_HIGH, DEFAULT_THRESHOLD_LOW, "default(p85_only)"
    return DEFAULT_THRESHOLD_HIGH, DEFAULT_THRESHOLD_LOW, "default(dual)"

def maybe_writer_thresholds(writer_id: int | None, policy: str, thr_high: float, thr_low: float):
    """
    Si hay calibración por writer y policy es dual/p85_only, reemplaza thresholds globales por los del writer.
    """
    if not USE_CALIBRATION or not WRITER_CALIB or writer_id is None:
        return thr_high, thr_low, False
    wkey = str(int(writer_id))
    if wkey not in WRITER_CALIB:
        return thr_high, thr_low, False
    wrec = WRITER_CALIB[wkey]
    w_p85 = wrec.get("p85", {}).get("threshold", thr_high)
    w_p80 = wrec.get("p80", {}).get("threshold", thr_low)
    if policy == "p85_only":
        return float(w_p85), float(w_p80), True
    return float(w_p85), float(w_p80), True

def decide_dual(prob: float, thr_high: float, thr_low: float):
    if prob >= thr_high:
        return "auto_validado"
    elif prob >= thr_low:
        return "revision_rapida"
    else:
        return "revision_manual"

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
        "model_version": MODEL_VERSION,
        "calibration_version": CALIB_VERSION if USE_CALIBRATION else "ignored",
        "use_calibration": USE_CALIBRATION,
        "global_thresholds": GLOBAL_CALIB,
        "writer_thresholds_loaded": bool(WRITER_CALIB),
    }

# ---------- /score (dos imágenes) ----------
@app.post("/score")
async def score_endpoint(
    file1: UploadFile = File(..., description="Firma de referencia (genuina registrada)"),
    file2: UploadFile = File(..., description="Firma a validar (documento nuevo)"),
    override_threshold_high: float | None = Form(None),
    override_threshold_low:  float | None = Form(None),
    policy: str | None = Form("dual"),        # "dual" (p85/p80) o "p85_only"
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

    try:
        score = score_pair(model, tfm, p1, p2, device=device)
        thr_high, thr_low, policy_used = pick_thresholds(policy, override_threshold_high, override_threshold_low)
        # no hay writer-id real aquí, así que usamos global (o el que nos dio override)
        decision = decide_dual(score, thr_high, thr_low)
        latency_ms = int((time.time() - t0) * 1000)

        append_log({
            "timestamp": datetime.utcnow().isoformat(),
            "doc_id": doc_id, "client_id": client_id, "writer_id": writer_id,
            "ref_path": file1.filename, "probe_path": file2.filename,
            "score": round(score,6),
            "policy": policy_used,
            "threshold_high": thr_high, "threshold_low": thr_low,
            "decision": decision,
            "model_version": MODEL_VERSION, "calibration_version": CALIB_VERSION,
            "latency_ms": latency_ms, "status": "ok"
        })
        return JSONResponse({
            "score": round(score,6),
            "policy": policy_used,
            "threshold_high": thr_high,
            "threshold_low": thr_low,
            "decision": decision,
            "latency_ms": latency_ms,
            "model_version": MODEL_VERSION,
            "calibration_version": CALIB_VERSION
        })
    finally:
        for p in (p1, p2):
            try: os.remove(p)
            except: pass

# ---------- /score_csv_upload (CSV con columnas img1,img2, opcionales doc_id,writer_id,client_id) ----------
@app.post("/score_csv_upload")
async def score_csv_upload(
    csv_file: UploadFile = File(...),
    override_threshold_high: float | None = Form(None),
    override_threshold_low:  float | None = Form(None),
    policy: str | None = Form("dual"),
    client_id: str | None = Form(None),
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tf:
        shutil.copyfileobj(csv_file.file, tf)
        tmp_csv = tf.name
    df = pd.read_csv(tmp_csv)
    assert {"img1","img2"}.issubset(df.columns), "CSV debe tener columnas: img1,img2"

    thr_high, thr_low, policy_used = pick_thresholds(policy, override_threshold_high, override_threshold_low)
    scores, decisions, latencies = [], [], []

    for _, row in df.iterrows():
        t0 = time.time()
        s = score_pair(model, tfm, row["img1"], row["img2"], device=device)
        d = decide_dual(s, thr_high, thr_low)
        scores.append(s); decisions.append(d)
        latencies.append(int((time.time()-t0)*1000))

        append_log({
            "timestamp": datetime.utcnow().isoformat(),
            "doc_id": row.get("doc_id"), "client_id": client_id or row.get("client_id"),
            "writer_id": row.get("writer_id"),
            "ref_path": row["img1"], "probe_path": row["img2"],
            "score": round(float(s),6),
            "policy": policy_used,
            "threshold_high": thr_high, "threshold_low": thr_low,
            "decision": d,
            "model_version": MODEL_VERSION, "calibration_version": CALIB_VERSION,
            "latency_ms": latencies[-1], "status":"ok"
        })

    df_out = df.copy()
    df_out["score"] = [round(float(s),6) for s in scores]
    df_out["threshold_high"] = thr_high
    df_out["threshold_low"]  = thr_low
    df_out["policy"] = policy_used
    df_out["decision"] = decisions
    out_csv = f"out/pairs_scores_upload_{int(time.time())}.csv"
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False)
    try: os.remove(tmp_csv)
    except: pass
    return {"saved": out_csv, "rows": len(df_out), "avg_latency_ms": sum(latencies)//max(len(latencies),1)}

# ---------- /score_writer (una imagen + writer_id) ----------
@app.post("/score_writer")
async def score_writer(
    probe: UploadFile = File(...),
    writer_id: int = Form(...),
    ref_sample_id: int | None = Form(1),
    override_threshold_high: float | None = Form(None),
    override_threshold_low:  float | None = Form(None),
    policy: str | None = Form("dual"),
    doc_id: str | None = Form(None),
    client_id: str | None = Form(None),
):
    # 1) thresholds base (global u override)
    thr_high, thr_low, policy_used = pick_thresholds(policy, override_threshold_high, override_threshold_low)
    # 2) si hay calibración por writer, la aplicamos (salvo override)
    if override_threshold_high is None and override_threshold_low is None:
        wh, wl, used_writer = maybe_writer_thresholds(writer_id, policy, thr_high, thr_low)
        if used_writer:
            thr_high, thr_low = wh, wl
            policy_used = f"{policy_used}+writer"

    ref_path = resolve_reference_path(writer_id, ref_sample_id)
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(probe.filename).suffix) as f2:
        shutil.copyfileobj(probe.file, f2)
        p2 = f2.name
    try:
        s = score_pair(model, tfm, ref_path, p2, device=device)
        d = decide_dual(s, thr_high, thr_low)
        append_log({
            "timestamp": datetime.utcnow().isoformat(),
            "doc_id": doc_id, "client_id": client_id, "writer_id": writer_id,
            "ref_path": ref_path, "probe_path": probe.filename,
            "score": round(float(s),6),
            "policy": policy_used,
            "threshold_high": thr_high, "threshold_low": thr_low,
            "decision": d,
            "model_version": MODEL_VERSION, "calibration_version": CALIB_VERSION,
            "latency_ms": 0, "status":"ok"
        })
        return JSONResponse({
            "score": round(float(s),6),
            "policy": policy_used,
            "threshold_high": thr_high,
            "threshold_low": thr_low,
            "decision": d,
            "ref_path_used": ref_path
        })
    finally:
        try: os.remove(p2)
        except: pass
