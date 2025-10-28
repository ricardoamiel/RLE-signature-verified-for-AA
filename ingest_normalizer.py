#!/usr/bin/env python3
# ingest_normalizer.py
# Escanea inbox, normaliza a landing, y encola JSONs en out_queue para modos A/B/C.
# Uso:
# python ingest_normalizer.py --inbox inbox --landing landing --out_queue queue --log_csv logs/ingest_logs.csv [--force_mode A|B|C]

import argparse, json, hashlib, shutil, os, sys, csv, re
from pathlib import Path
from datetime import datetime
import subprocess, sys, shlex

RE_ORG  = re.compile(r"^original_(\d+)_(\d+)\.(png|jpg|jpeg)$", re.I)
RE_FORG = re.compile(r"^forgeries_(\d+)_(\d+)\.(png|jpg|jpeg)$", re.I)

def sha256_of_paths(paths):
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(Path(p).name.encode("utf-8"))
    return h.hexdigest()

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def is_pair_candidate(name: str):
    return RE_ORG.match(name) or RE_FORG.match(name)

def parse_name(name: str):
    m1 = RE_ORG.match(name)
    if m1:
        return ("original", int(m1.group(1)), int(m1.group(2)))
    m2 = RE_FORG.match(name)
    if m2:
        return ("forgeries", int(m2.group(1)), int(m2.group(2)))
    return (None, None, None)

def is_csv_pairs(path: Path):
    if path.suffix.lower() != ".csv":
        return False
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            rd = csv.DictReader(f)
            cols = [c.strip().lower() for c in rd.fieldnames or []]
            return "img1" in cols and "img2" in cols
    except Exception:
        return False

def mk_ingest_id():
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = str(datetime.now().microsecond)[:3]
    return f"ING-{ts}-{suffix}"

def log_line(log_csv: Path, row: dict):
    ensure_dir(log_csv.parent)
    hdr = not log_csv.exists()
    with log_csv.open("a", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=row.keys())
        if hdr: wr.writeheader()
        wr.writerow(row)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inbox", required=True)
    ap.add_argument("--landing", required=True)
    ap.add_argument("--out_queue", required=True)
    ap.add_argument("--log_csv", required=True)
    ap.add_argument("--policy", default="dual")  # dual | p85_only
    ap.add_argument("--p85", type=float, default=0.85)
    ap.add_argument("--p80", type=float, default=0.80)
    ap.add_argument("--force_mode", choices=["A","B","C"], default=None)
    args = ap.parse_args()

    inbox = Path(args.inbox)
    landing_root = Path(args.landing)
    out_queue = Path(args.out_queue)
    log_csv = Path(args.log_csv)
    ensure_dir(landing_root); ensure_dir(out_queue); ensure_dir(log_csv.parent)

    # 1) Listar archivos del inbox (solo nivel actual)
    entries = [p for p in inbox.iterdir() if p.is_file()]
    if not entries:
        print("Inbox vacío.")
        return

    # 2) Clasificar: imágenes válidas y CSVs de pares
    images = [p for p in entries if is_pair_candidate(p.name)]
    csvs   = [p for p in entries if is_csv_pairs(p)]

    # 3) Construir grupos por (writer_id, sample_id)
    groups = {}  # (writer, sample) -> dict{ 'original':Path?, 'forgeries':Path? }
    for img in images:
        role, wid, sid = parse_name(img.name)
        if role is None: 
            continue
        key = (wid, sid)
        if key not in groups:
            groups[key] = {"original": None, "forgeries": None}
        groups[key][role] = img

    # 4) Si hay CSVs y se fuerza B, procesar B primero
    if args.force_mode == "B" and csvs:
        for csvp in csvs:
            make_envelope_B(csvp, landing_root, out_queue, log_csv, args.policy, args.p85, args.p80)
            try: csvp.unlink()
            except: pass
        print("Hecho (B forzado).")
        return

    # 5) Si hay pares completos y (force A o no hay force y conviene A), generar A
    produced = False
    for (wid, sid), pair in groups.items():
        if pair["original"] is not None and pair["forgeries"] is not None:
            if args.force_mode in (None, "A"):
                make_envelope_A(pair["original"], pair["forgeries"], wid, sid,
                                landing_root, out_queue, log_csv, args.policy, args.p85, args.p80)
                # borrar originales del inbox
                for p in (pair["original"], pair["forgeries"]):
                    try: p.unlink()
                    except: pass
                produced = True

    # 6) CSVs (modo B) sin force, si hay
    if args.force_mode in (None, "B"):
        for csvp in csvs:
            make_envelope_B(csvp, landing_root, out_queue, log_csv, args.policy, args.p85, args.p80)
            try: csvp.unlink()
            except: pass
            produced = True

    # 7) Restos sueltos (modo C): un solo archivo original_*_* o forgeries_*_*, sin su par
    if args.force_mode in (None, "C"):
        for (wid, sid), pair in groups.items():
            # Si ya se procesó como A, saltar
            # Si quedó uno suelto (original o forgeries), encolar C
            solo = pair["original"] or pair["forgeries"]
            if solo and not (pair["original"] and pair["forgeries"]):
                make_envelope_C(solo, wid, sid, landing_root, out_queue, log_csv, args.policy, args.p85, args.p80)
                try: solo.unlink()
                except: pass
                produced = True

    if not produced:
        print("No se generaron sobres (nada que encolar con las reglas actuales).")

def make_envelope_A(org_path: Path, forg_path: Path, wid: int, sid: int,
                    landing_root: Path, out_queue: Path, log_csv: Path,
                    policy: str, p85: float, p80: float):
    ingest_id = mk_ingest_id()
    dest_dir = landing_root / datetime.now().strftime("%Y/%m/%d") / ingest_id
    ensure_dir(dest_dir)
    org_dest = dest_dir / org_path.name
    forg_dest = dest_dir / forg_path.name
    shutil.copy2(org_path, org_dest)
    shutil.copy2(forg_path, forg_dest)

    attachments = [str(org_dest), str(forg_dest)]
    payload = {
        "schema_version": "1.0",
        "ingest_id": ingest_id,
        "source": "watchfolder",
        "received_at": datetime.utcnow().isoformat(),
        "priority": "High",
        "attachments": attachments,
        "hash": sha256_of_paths(attachments),
        "mode": "A",
        "body": {
            "doc_id": forg_path.stem,            # p.ej. forgeries_1_3
            "client_id": f"Cliente-{wid}",
            "writer_id": wid,
            "file1": str(org_dest),
            "file2": str(forg_dest),
            "policy": policy,
            "thresholds": { "p85": p85, "p80": p80 }
        }
    }
    out_json = out_queue / f"{ingest_id}.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    log_line(log_csv, {
        "ts": datetime.utcnow().isoformat(),
        "ingest_id": ingest_id,
        "mode": "A",
        "writer_id": wid,
        "sample_id": sid,
        "attachments": ";".join(attachments)
    })
    print(f"[A] Encolado: {out_json.name}")

def make_envelope_B(csv_path: Path, landing_root: Path, out_queue: Path, log_csv: Path,
                    policy: str, p85: float, p80: float):
    ingest_id = mk_ingest_id()
    dest_dir = landing_root / datetime.now().strftime("%Y/%m/%d") / ingest_id
    ensure_dir(dest_dir)
    csv_dest = dest_dir / csv_path.name
    shutil.copy2(csv_path, csv_dest)

    attachments = [str(csv_dest)]
    payload = {
        "schema_version": "1.0",
        "ingest_id": ingest_id,
        "source": "watchfolder",
        "received_at": datetime.utcnow().isoformat(),
        "priority": "High",
        "attachments": attachments,
        "hash": sha256_of_paths(attachments),
        "mode": "B",
        "body": {
            "csv_path": str(csv_dest),
            "policy": policy,
            "thresholds": { "p85": p85, "p80": p80 }
        }
    }
    out_json = out_queue / f"{ingest_id}.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    log_line(log_csv, {
        "ts": datetime.utcnow().isoformat(),
        "ingest_id": ingest_id,
        "mode": "B",
        "writer_id": "",
        "sample_id": "",
        "attachments": ";".join(attachments)
    })
    print(f"[B] Encolado: {out_json.name}")

def make_envelope_C(solo_path: Path, wid: int, sid: int,
                    landing_root: Path, out_queue: Path, log_csv: Path,
                    policy: str, p85: float, p80: float):
    ingest_id = mk_ingest_id()
    dest_dir = landing_root / datetime.now().strftime("%Y/%m/%d") / ingest_id
    ensure_dir(dest_dir)
    dst = dest_dir / solo_path.name
    shutil.copy2(solo_path, dst)

    attachments = [str(dst)]
    payload = {
        "schema_version": "1.0",
        "ingest_id": ingest_id,
        "source": "watchfolder",
        "received_at": datetime.utcnow().isoformat(),
        "priority": "High",
        "attachments": attachments,
        "hash": sha256_of_paths(attachments),
        "mode": "C",
        "body": {
            # usamos writer_id del nombre del archivo
            "writer_id": wid,
            # el API tomará la genuina de referencia desde REF_INDEX_CSV
            "probe_path": str(dst),
            # opcional: puedes incluir ref_sample_id si quieres forzarlo
            # "ref_sample_id": 1,
            "policy": policy,
            "thresholds": { "p85": p85, "p80": p80 }
        }
    }
    out_json = out_queue / f"{ingest_id}.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    log_line(log_csv, {
        "ts": datetime.utcnow().isoformat(),
        "ingest_id": ingest_id,
        "mode": "C",
        "writer_id": wid,
        "sample_id": sid,
        "attachments": ";".join(attachments)
    })
    print(f"[C] Encolado: {out_json.name}")

def run_ingest(
    inbox: str = r"C:\Users\israe\OneDrive\Documentos\PythonProjects\RLE-signature-verified-for-AA\inbox",
    landing: str = r"C:\Users\israe\OneDrive\Documentos\PythonProjects\RLE-signature-verified-for-AA\landing",
    out_queue: str = r"C:\Users\israe\OneDrive\Documentos\PythonProjects\RLE-signature-verified-for-AA\queue",
    log_csv: str = r"C:\Users\israe\OneDrive\Documentos\PythonProjects\RLE-signature-verified-for-AA\logs\ingest_logs.csv",
    policy: str = "dual",
    p85: float = 0.85,
    p80: float = 0.80,
    force_mode: str | None = None,
    debug: bool = True,
):
    """
    Ejecuta la ingesta una sola pasada (no watcher).
    Retorna contadores útiles para AA o pruebas locales.
    """
    inbox_p = Path(inbox)
    landing_root = Path(landing)
    out_q = Path(out_queue)
    log_csv_p = Path(log_csv)
    ensure_dir(landing_root); ensure_dir(out_q); ensure_dir(log_csv_p.parent)

    if not inbox_p.exists():
        raise FileNotFoundError(f"Inbox no existe: {inbox_p}")

    entries = [p for p in inbox_p.iterdir() if p.is_file()]
    if debug:
        print(f"[INGEST] Inbox: {inbox_p} ({len(entries)} archivos)")
        for e in entries[:20]:
            print("  -", e.name)

    if not entries:
        return {"produced_A":0,"produced_B":0,"produced_C":0,"total_json":0,"reason":"inbox vacío"}

    images = [p for p in entries if is_pair_candidate(p.name)]
    csvs   = [p for p in entries if is_csv_pairs(p)]

    if debug:
        print(f"[INGEST] candidatos imágenes: {len(images)} | CSVs pares: {len(csvs)}")

    groups = {}
    for img in images:
        role, wid, sid = parse_name(img.name)
        if role is None:
            continue
        groups.setdefault((wid, sid), {"original": None, "forgeries": None})
        groups[(wid, sid)][role] = img

    produced_A = produced_B = produced_C = 0

    # B forzado primero
    if force_mode == "B" and csvs:
        for csvp in csvs:
            make_envelope_B(csvp, landing_root, out_q, log_csv_p, policy, p85, p80)
            try: csvp.unlink()
            except: pass
            produced_B += 1
        return {
            "produced_A": produced_A, "produced_B": produced_B, "produced_C": produced_C,
            "total_json": produced_A+produced_B+produced_C
        }

    # A (pares completos)
    if force_mode in (None, "A"):
        for (wid, sid), pair in groups.items():
            if pair["original"] is not None and pair["forgeries"] is not None:
                make_envelope_A(pair["original"], pair["forgeries"], wid, sid,
                                landing_root, out_q, log_csv_p, policy, p85, p80)
                for p in (pair["original"], pair["forgeries"]):
                    try: p.unlink()
                    except: pass
                produced_A += 1

    # B (CSVs sueltos)
    if force_mode in (None, "B"):
        for csvp in csvs:
            make_envelope_B(csvp, landing_root, out_q, log_csv_p, policy, p85, p80)
            try: csvp.unlink()
            except: pass
            produced_B += 1

    # C (restos sueltos)
    if force_mode in (None, "C"):
        for (wid, sid), pair in groups.items():
            solo = pair["original"] or pair["forgeries"]
            if solo and not (pair["original"] and pair["forgeries"]):
                make_envelope_C(solo, wid, sid, landing_root, out_q, log_csv_p, policy, p85, p80)
                try: solo.unlink()
                except: pass
                produced_C += 1

    total_json = produced_A + produced_B + produced_C
    if debug:
        print(f"[INGEST] A={produced_A} B={produced_B} C={produced_C} | total={total_json}")

    return {
        "produced_A": produced_A, "produced_B": produced_B, "produced_C": produced_C,
        "total_json": total_json
    }


def main():
    """ap = argparse.ArgumentParser()
    ap.add_argument("--inbox", required=True)
    ap.add_argument("--landing", required=True)
    ap.add_argument("--out_queue", required=True)
    ap.add_argument("--log_csv", required=True)
    ap.add_argument("--policy", default="dual")
    ap.add_argument("--p85", type=float, default=0.85)
    ap.add_argument("--p80", type=float, default=0.80)
    ap.add_argument("--force_mode", choices=["A","B","C"], default=None)
    args = ap.parse_args()
    run_ingest(args.inbox, args.landing, args.out_queue, args.log_csv,
               policy=args.policy, p85=args.p85, p80=args.p80, force_mode=args.force_mode)"""
    run_ingest(
    inbox=r"C:\Users\israe\OneDrive\Documentos\PythonProjects\RLE-signature-verified-for-AA\inbox",
    landing=r"C:\Users\israe\OneDrive\Documentos\PythonProjects\RLE-signature-verified-for-AA\landing",
    out_queue=r"C:\Users\israe\OneDrive\Documentos\PythonProjects\RLE-signature-verified-for-AA\queue",
    log_csv=r"C:\Users\israe\OneDrive\Documentos\PythonProjects\RLE-signature-verified-for-AA\logs\ingest_logs.csv",
)

if __name__ == "__main__":
    main()
