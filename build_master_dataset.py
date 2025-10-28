#!/usr/bin/env python3
# build_master_dataset.py
# Crea tablas maestras para Power BI a partir de logs y JSONs.
#
# MODO AA (sin inputs):
#   - Usa run_build_master_default()
#
# MODO CLI:
#   python build_master_dataset.py --dispatch_csv logs\dispatch_results.csv --ops_csv runs_first_approach\ops_logs.csv --calib_dir runs_first_approach\siamese_split --out_dir logs

import argparse, json, csv, os
from pathlib import Path
from datetime import datetime

# =========================
# Utilidades
# =========================

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def safe_json_loads(s: str):
    """Intenta decodificar JSON (y si viene doble-serializado, hace segunda pasada)."""
    if not s:
        return {}
    try:
        obj = json.loads(s)
        if isinstance(obj, str) and (obj.strip().startswith("{") or obj.strip().startswith("[")):
            try:
                return json.loads(obj)
            except Exception:
                return {"_raw": obj}
        return obj
    except Exception:
        return {}

def write_csv(path: Path, rows, fieldnames):
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in fieldnames})

def load_csv_rows(path: Path):
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        return list(rd)

# =========================
# Aplanado de dispatch
# =========================

def flatten_dispatch_row(row):
    """Aplana una fila del CSV de dispatch_results."""
    resp = safe_json_loads(row.get("response","") or "{}")
    out = {
        "ts": row.get("timestamp") or row.get("ts") or "",
        "ingest_id": row.get("ingest_id",""),
        "mode": row.get("mode",""),
        "status_code": row.get("status_code",""),
        "latency_ms": row.get("latency_ms",""),
        "json_path": row.get("json_path",""),
        "error": row.get("error",""),
        "score": resp.get("score",""),
        "decision": resp.get("decision",""),
        "threshold_high": resp.get("threshold_high","") or resp.get("threshold",""),
        "threshold_low": resp.get("threshold_low",""),
        "policy": resp.get("policy",""),
        "model_version": resp.get("model_version",""),
        "calibration_version": resp.get("calibration_version",""),
        "ref_path_used": resp.get("ref_path_used",""),
        "saved": resp.get("saved",""),
        "rows": resp.get("rows",""),
        "avg_latency_ms": resp.get("avg_latency_ms",""),
    }
    return out

# =========================
# Normalización thresholds_by_writer
# =========================

def _extract_p_block(p):
    """Devuelve tuple (thr, prec, rec, stp) a partir de:
       - número → (num, "", "", "")
       - dict   → (threshold, precision, recall, stp_proxy)
       - otro   → ("", "", "", "")
    """
    if isinstance(p, (int, float)):
        return (p, "", "", "")
    if isinstance(p, dict):
        return (
            p.get("threshold",""),
            p.get("precision",""),
            p.get("recall",""),
            p.get("stp_proxy",""),
        )
    return ("", "", "", "")

def normalize_thresholds_by_writer(tbw_raw):
    """Convierte cualquier estructura a una lista uniforme por writer con p85/p80 desglosados."""
    rows = []

    # Caso 1: {"writers": [...]}
    if isinstance(tbw_raw, dict) and "writers" in tbw_raw:
        tbw_raw = tbw_raw["writers"]

    # Caso 2: dict por ID
    if isinstance(tbw_raw, dict):
        tbw_raw = [{"writer_id": k, **(v if isinstance(v, dict) else {})} for k, v in tbw_raw.items()]

    # Caso 3: lista heterogénea
    if isinstance(tbw_raw, list):
        for w in tbw_raw:
            if isinstance(w, dict):
                wid = w.get("writer_id") or w.get("writer") or w.get("id") or ""
                p85 = w.get("p85")
                p80 = w.get("p80")
                p85_thr, p85_prec, p85_rec, p85_stp = _extract_p_block(p85)
                p80_thr, p80_prec, p80_rec, p80_stp = _extract_p_block(p80)
                rows.append({
                    "writer_id": str(wid),
                    "thr_p85_thr": p85_thr,
                    "thr_p85_prec": p85_prec,
                    "thr_p85_rec": p85_rec,
                    "thr_p85_stp": p85_stp,
                    "thr_p80_thr": p80_thr,
                    "thr_p80_prec": p80_prec,
                    "thr_p80_rec": p80_rec,
                    "thr_p80_stp": p80_stp,
                })
            else:
                # string o número suelto
                rows.append({
                    "writer_id": str(w),
                    "thr_p85_thr": "", "thr_p85_prec": "", "thr_p85_rec": "", "thr_p85_stp": "",
                    "thr_p80_thr": "", "thr_p80_prec": "", "thr_p80_rec": "", "thr_p80_stp": "",
                })
    return rows

# =========================
# Núcleo reutilizable
# =========================

def run_build_master(dispatch_csv: str = r"C:\Users\israe\OneDrive\Documentos\PythonProjects\RLE-signature-verified-for-AA\logs\dispatch_results.csv",
                     ops_csv: str = r"C:\Users\israe\OneDrive\Documentos\PythonProjects\RLE-signature-verified-for-AA\runs_first_approach\ops_logs.csv",
                     calib_dir: str = r"C:\Users\israe\OneDrive\Documentos\PythonProjects\RLE-signature-verified-for-AA\runs_first_approach\siamese_split",
                     out_dir: str = r"C:\Users\israe\OneDrive\Documentos\PythonProjects\RLE-signature-verified-for-AA\logs"):
    """Genera los 4 CSV maestros y devuelve un resumen (dict)."""

    dispatch_csv = Path(dispatch_csv)
    ops_csv = Path(ops_csv)
    calib_dir = Path(calib_dir)
    out_dir = Path(out_dir)

    # 1) master_events.csv
    dispatch_rows = load_csv_rows(dispatch_csv)
    events = [flatten_dispatch_row(r) for r in dispatch_rows]
    write_csv(
        out_dir / "master_events.csv",
        events,
        [
            "ts","ingest_id","mode","status_code","latency_ms","json_path","error",
            "score","decision","threshold_high","threshold_low","policy",
            "model_version","calibration_version","ref_path_used","saved","rows","avg_latency_ms"
        ],
    )

    # 2) master_ops.csv
    ops = load_csv_rows(ops_csv)
    if not ops:
        write_csv(out_dir / "master_ops.csv", [], [
            "timestamp","doc_id","client_id","writer_id","ref_path","probe_path",
            "score","threshold_high","threshold_low","decision",
            "model_version","calibration_version","latency_ms","status"
        ])
    else:
        cols = list({k for r in ops for k in r.keys()})
        write_csv(out_dir / "master_ops.csv", ops, cols)

    # 3) thresholds_by_writer.json -> master_thresholds_by_writer.csv
    tbw_path = calib_dir / "thresholds_by_writer.json"
    tbw_rows = []
    if tbw_path.exists():
        tbw_raw = json.loads(tbw_path.read_text(encoding="utf-8"))
        tbw_rows = normalize_thresholds_by_writer(tbw_raw)

    write_csv(
        out_dir / "master_thresholds_by_writer.csv",
        tbw_rows,
        ["writer_id",
         "thr_p85_thr","thr_p85_prec","thr_p85_rec","thr_p85_stp",
         "thr_p80_thr","thr_p80_prec","thr_p80_rec","thr_p80_stp"]
    )

    # 4) test_summary.json + calibration_maxstp.json -> master_test_summary.csv
    ts_path = calib_dir / "test_summary.json"
    cm_path = calib_dir / "calibration_maxstp.json"
    rows = []
    row = {"source_dir": str(calib_dir)}

    if ts_path.exists():
        ts = json.loads(ts_path.read_text(encoding="utf-8"))
        row.update({
            "model_checkpoint": ts.get("model_checkpoint",""),
            "test_auc": ts.get("test_auc") or ts.get("auc",""),
            "test_eer": ts.get("test_eer") or ts.get("eer",""),
            "metrics_at_p85_stp": ts.get("metrics_at_p85",{}).get("stp_proxy",""),
            "metrics_at_p80_stp": ts.get("metrics_at_p80",{}).get("stp_proxy",""),
            "metrics_at_p85_precision": ts.get("metrics_at_p85",{}).get("precision",""),
            "metrics_at_p80_precision": ts.get("metrics_at_p80",{}).get("precision",""),
        })
    if cm_path.exists():
        cm = json.loads(cm_path.read_text(encoding="utf-8"))
        th = (cm.get("thresholds") or {})
        p85 = th.get("p85") or {}
        p80 = th.get("p80") or {}
        row.update({
            "calib_p85_thr": p85.get("threshold",""),
            "calib_p85_prec": p85.get("precision",""),
            "calib_p85_rec": p85.get("recall",""),
            "calib_p85_stp": p85.get("stp_proxy",""),
            "calib_p80_thr": p80.get("threshold",""),
            "calib_p80_prec": p80.get("precision",""),
            "calib_p80_rec": p80.get("recall",""),
            "calib_p80_stp": p80.get("stp_proxy",""),
        })
    if any(v for k,v in row.items() if k!="source_dir"):
        rows.append(row)

    write_csv(out_dir / "master_test_summary.csv", rows, [
        "source_dir","model_checkpoint",
        "test_auc","test_eer",
        "metrics_at_p85_stp","metrics_at_p80_stp",
        "metrics_at_p85_precision","metrics_at_p80_precision",
        "calib_p85_thr","calib_p85_prec","calib_p85_rec","calib_p85_stp",
        "calib_p80_thr","calib_p80_prec","calib_p80_rec","calib_p80_stp"
    ])

    return {
        "ok": True,
        "out_dir": str(out_dir),
        "generated": [
            str(out_dir / "master_events.csv"),
            str(out_dir / "master_ops.csv"),
            str(out_dir / "master_thresholds_by_writer.csv"),
            str(out_dir / "master_test_summary.csv"),
        ]
    }

# =========================
# Entradas por defecto (AA)
# =========================

# Puedes ajustar estas rutas una sola vez:
DEFAULT_DISPATCH = r"logs\dispatch_results.csv"
DEFAULT_OPS      = r"runs_first_approach\ops_logs.csv"
DEFAULT_CALIB    = r"runs_first_approach\siamese_split"
DEFAULT_OUT      = r"logs"

def run_build_master_default():
    """Pensado para 'Execute function' en Automation Anywhere (sin inputs)."""
    base = Path(__file__).resolve().parent
    return run_build_master(
        dispatch_csv=str((base / DEFAULT_DISPATCH).resolve()),
        ops_csv=str((base / DEFAULT_OPS).resolve()),
        calib_dir=str((base / DEFAULT_CALIB).resolve()),
        out_dir=str((base / DEFAULT_OUT).resolve()),
    )

# =========================
# CLI
# =========================

def main():
    """ap = argparse.ArgumentParser()
    ap.add_argument("--dispatch_csv", required=True)
    ap.add_argument("--ops_csv", required=False, default="runs/ops_logs.csv")
    ap.add_argument("--calib_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    res = run_build_master(args.dispatch_csv, args.ops_csv, args.calib_dir, args.out_dir)"""
    run_build_master(dispatch_csv =r"C:\Users\israe\OneDrive\Documentos\PythonProjects\RLE-signature-verified-for-AA\logs\dispatch_results.csv",
                     ops_csv =r"C:\Users\israe\OneDrive\Documentos\PythonProjects\RLE-signature-verified-for-AA\runs_first_approach\ops_logs.csv",
                     calib_dir =r"C:\Users\israe\OneDrive\Documentos\PythonProjects\RLE-signature-verified-for-AA\runs_first_approach\siamese_split",
                     out_dir =r"C:\Users\israe\OneDrive\Documentos\PythonProjects\RLE-signature-verified-for-AA\logs")
    #print("[OK] Generado dataset para Power BI en:", res["out_dir"])

if __name__ == "__main__":
    main()
