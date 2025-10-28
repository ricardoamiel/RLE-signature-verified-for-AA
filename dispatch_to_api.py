#!/usr/bin/env python3
# dispatch_to_api.py
# Lee JSONs de la cola y llama al endpoint correcto de la API:
#   A -> /score (file1 + file2)
#   B -> /score_csv_upload (csv_file)
#   C -> /score_writer (probe + writer_id)
#
# Uso:
# python dispatch_to_api.py --api_base http://127.0.0.1:8000 --queue queue --results_csv logs/dispatch_results.csv --archive_ok queue/archive_ok --archive_err queue/archive_err

import argparse, json, csv, os, shutil
from pathlib import Path
from datetime import datetime
import requests
import subprocess, sys, shlex

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def log_row(csv_path: Path, row: dict):
    ensure_dir(csv_path.parent)
    hdr = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=row.keys())
        if hdr: wr.writeheader()
        wr.writerow(row)

def post_score(api_base, file1, file2, policy, thr_high, thr_low, doc_id=None, client_id=None, writer_id=None):
    url = f"{api_base.rstrip('/')}/score"
    files = {
        "file1": (Path(file1).name, open(file1, "rb")),
        "file2": (Path(file2).name, open(file2, "rb")),
    }
    data = {
        "override_threshold": str(thr_high),
        "low_threshold": str(thr_low),
        "doc_id": doc_id or "",
        "client_id": client_id or "",
        "writer_id": str(writer_id) if writer_id is not None else "",
        "policy": policy,
    }
    try:
        r = requests.post(url, files=files, data=data, timeout=60)
        return r.status_code, r.json()
    finally:
        for fh in files.values():
            try: fh[1].close()
            except: pass

def post_csv(api_base, csv_path, policy, thr_high, thr_low):
    url = f"{api_base.rstrip('/')}/score_csv_upload"
    files = { "csv_file": (Path(csv_path).name, open(csv_path, "rb")) }
    data = {
        "override_threshold": str(thr_high),
        "low_threshold": str(thr_low),
        "policy": policy
    }
    try:
        r = requests.post(url, files=files, data=data, timeout=120)
        return r.status_code, r.json()
    finally:
        try: files["csv_file"][1].close()
        except: pass

def post_score_writer(api_base, probe_path, writer_id, policy, thr_high, thr_low, ref_sample_id=None, doc_id=None, client_id=None):
    url = f"{api_base.rstrip('/')}/score_writer"
    files = { "probe": (Path(probe_path).name, open(probe_path, "rb")) }
    data = {
        "writer_id": str(writer_id),
        "override_threshold": str(thr_high),
        "low_threshold": str(thr_low),
        "policy": policy
    }
    if ref_sample_id is not None:
        data["ref_sample_id"] = str(ref_sample_id)
    if doc_id: data["doc_id"] = doc_id
    if client_id: data["client_id"] = client_id
    try:
        r = requests.post(url, files=files, data=data, timeout=60)
        return r.status_code, r.json()
    finally:
        try: files["probe"][1].close()
        except: pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api_base", required=True)
    ap.add_argument("--queue", required=True)
    ap.add_argument("--results_csv", required=True)
    ap.add_argument("--archive_ok", required=True)
    ap.add_argument("--archive_err", required=True)
    args = ap.parse_args()

    api_base = args.api_base
    qdir = Path(args.queue)
    results_csv = Path(args.results_csv)
    arch_ok = Path(args.archive_ok)
    arch_err = Path(args.archive_err)
    ensure_dir(arch_ok); ensure_dir(arch_err); ensure_dir(results_csv.parent)

    jobs = sorted([p for p in qdir.iterdir() if p.suffix.lower()==".json"])
    if not jobs:
        print("No hay sobres en cola.")
        return

    for job in jobs:
        try:
            payload = json.loads(job.read_text(encoding="utf-8"))
            mode = payload.get("mode")
            body = payload.get("body", {})
            policy = body.get("policy", "dual")
            thr = body.get("thresholds", {})
            thr_high = float(thr.get("p85", 0.85))
            thr_low  = float(thr.get("p80", 0.80))

            if mode == "A":
                status, out = post_score(
                    api_base,
                    body["file1"], body["file2"],
                    policy, thr_high, thr_low,
                    doc_id=body.get("doc_id"),
                    client_id=body.get("client_id"),
                    writer_id=body.get("writer_id"),
                )
            elif mode == "B":
                status, out = post_csv(
                    api_base,
                    body["csv_path"],
                    policy, thr_high, thr_low
                )
            elif mode == "C":
                status, out = post_score_writer(
                    api_base,
                    body["probe_path"],
                    body["writer_id"],
                    policy, thr_high, thr_low,
                    ref_sample_id=body.get("ref_sample_id"),
                    doc_id=body.get("doc_id"),
                    client_id=body.get("client_id"),
                )
            else:
                raise ValueError(f"Modo no soportado: {mode}")

            ok = 200 <= status < 300
            log_row(results_csv, {
                "ts": datetime.utcnow().isoformat(),
                "ingest_id": payload.get("ingest_id"),
                "mode": mode,
                "status_code": status,
                "response": json.dumps(out, ensure_ascii=False),
            })
            dest = arch_ok if ok else arch_err
            shutil.move(str(job), str(dest / job.name))
            print(f"[{mode}] {job.name} -> {status} ({'OK' if ok else 'ERR'})")

        except Exception as e:
            log_row(results_csv, {
                "ts": datetime.utcnow().isoformat(),
                "ingest_id": "",
                "mode": "",
                "status_code": 0,
                "response": f"ERROR: {e}"
            })
            try: shutil.move(str(job), str(arch_err / job.name))
            except: pass
            print(f"[ERR] {job.name}: {e}")

def run_dispatch(
    api_base: str = "http://127.0.0.1:8000",
    queue: str = r"C:\Users\israe\OneDrive\Documentos\PythonProjects\RLE-signature-verified-for-AA\queue",
    results_csv: str = r"C:\Users\israe\OneDrive\Documentos\PythonProjects\RLE-signature-verified-for-AA\logs\dispatch_results.csv",
    archive_ok: str = r"C:\Users\israe\OneDrive\Documentos\PythonProjects\RLE-signature-verified-for-AA\queue\archive_ok",
    archive_err: str = r"C:\Users\israe\OneDrive\Documentos\PythonProjects\RLE-signature-verified-for-AA\queue\archive_err",
    debug: bool = True,
):
    qdir = Path(queue)
    results_csv_p = Path(results_csv)
    arch_ok = Path(archive_ok)
    arch_err = Path(archive_err)
    ensure_dir(arch_ok); ensure_dir(arch_err); ensure_dir(results_csv_p.parent)

    if not qdir.exists():
        raise FileNotFoundError(f"Queue no existe: {qdir}")

    jobs = sorted([p for p in qdir.iterdir() if p.suffix.lower()==".json"])
    if debug:
        print(f"[DISPATCH] cola={qdir} | jobs={len(jobs)}")
        for j in jobs[:20]:
            print("  -", j.name)

    if not jobs:
        return {"processed":0, "ok":0, "err":0, "reason":"cola vacía"}

    ok = err = 0
    for job in jobs:
        try:
            payload = json.loads(job.read_text(encoding="utf-8"))
            mode = payload.get("mode")
            body = payload.get("body", {})
            policy = body.get("policy", "dual")
            thr = body.get("thresholds", {})
            thr_high = float(thr.get("p85", 0.85))
            thr_low  = float(thr.get("p80", 0.80))

            if mode == "A":
                status, out = post_score(
                    api_base,
                    body["file1"], body["file2"],
                    policy, thr_high, thr_low,
                    doc_id=body.get("doc_id"),
                    client_id=body.get("client_id"),
                    writer_id=body.get("writer_id"),
                )
            elif mode == "B":
                status, out = post_csv(api_base, body["csv_path"], policy, thr_high, thr_low)
            elif mode == "C":
                status, out = post_score_writer(
                    api_base,
                    body["probe_path"],
                    body["writer_id"],
                    policy, thr_high, thr_low,
                    ref_sample_id=body.get("ref_sample_id"),
                    doc_id=body.get("doc_id"),
                    client_id=body.get("client_id"),
                )
            else:
                raise ValueError(f"Modo no soportado: {mode}")

            ok_flag = 200 <= status < 300
            log_row(results_csv_p, {
                "ts": datetime.utcnow().isoformat(),
                "ingest_id": payload.get("ingest_id"),
                "mode": mode,
                "status_code": status,
                "response": json.dumps(out, ensure_ascii=False),
                "json_path": str(job),
            })
            dest = arch_ok if ok_flag else arch_err
            shutil.move(str(job), str(dest / job.name))
            ok += int(ok_flag)
            err += int(not ok_flag)
            if debug:
                print(f"[{mode}] {job.name} -> {status} ({'OK' if ok_flag else 'ERR'})")

        except Exception as e:
            log_row(results_csv_p, {
                "ts": datetime.utcnow().isoformat(),
                "ingest_id": "",
                "mode": "",
                "status_code": 0,
                "response": f"ERROR: {e}",
                "json_path": str(job),
            })
            try: shutil.move(str(job), str(arch_err / job.name))
            except: pass
            err += 1
            if debug:
                print(f"[ERR] {job.name}: {e}")

    return {"processed": ok+err, "ok": ok, "err": err}


def main():
    """ap = argparse.ArgumentParser()
    ap.add_argument("--api_base", required=True)
    ap.add_argument("--queue", required=True)
    ap.add_argument("--results_csv", required=True)
    ap.add_argument("--archive_ok", required=True)
    ap.add_argument("--archive_err", required=True)
    args = ap.parse_args()
    run_dispatch(args.api_base, args.queue, args.results_csv, args.archive_ok, args.archive_err)"""
    run_dispatch(
    api_base="http://127.0.0.1:8000",
    queue=r"C:\Users\israe\OneDrive\Documentos\PythonProjects\RLE-signature-verified-for-AA\queue",
    results_csv=r"C:\Users\israe\OneDrive\Documentos\PythonProjects\RLE-signature-verified-for-AA\logs\dispatch_results.csv",
    archive_ok=r"C:\Users\israe\OneDrive\Documentos\PythonProjects\RLE-signature-verified-for-AA\queue\archive_ok",
    archive_err=r"C:\Users\israe\OneDrive\Documentos\PythonProjects\RLE-signature-verified-for-AA\queue\archive_err",
)

if __name__ == "__main__":
    main()
