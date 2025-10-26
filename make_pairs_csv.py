import argparse, pandas as pd, numpy as np
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_csv", type=str, default="out/signatures_index.csv")
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--ref_sample_id", type=int, default=1, help="ID de muestra de la firma de referencia (genuina)")
    ap.add_argument("--mode", choices=["probe_like","positives","negatives","mix"], default="probe_like")
    args = ap.parse_args()

    df = pd.read_csv(args.index_csv)
    df["writer_id"] = df["writer_id"].astype(int)
    rows = []

    if args.mode == "probe_like":
        # Para cada writer: elige una genuina como referencia y compárala con todas sus otras genuinas y forgeries
        for w, g in df.groupby("writer_id"):
            ref = g[(g.label=="genuine") & (g.sample_id==args.ref_sample_id)]
            if ref.empty:
                # si no existe esa sample, coge la primera genuina disponible
                ref = g[g.label=="genuine"].head(1)
                if ref.empty: continue
            ref_path = ref.iloc[0]["path"]

            # candidatos = resto de genuinas + todas las forgeries del mismo writer
            cand = pd.concat([
                g[(g.label=="genuine") & (g.path != ref_path)],
                g[g.label=="forgery"]
            ], ignore_index=True)
            for _, r in cand.iterrows():
                rows.append({"img1": ref_path, "img2": r["path"]})

    elif args.mode == "positives":
        # Solo genuina–genuina (same-writer)
        for w, g in df.groupby("writer_id"):
            orgs = g[g.label=="genuine"]["path"].tolist()
            for i in range(len(orgs)):
                for j in range(i+1, len(orgs)):
                    rows.append({"img1": orgs[i], "img2": orgs[j]})

    elif args.mode == "negatives":
        # Solo genuina–forgery (same-writer)
        for w, g in df.groupby("writer_id"):
            orgs = g[g.label=="genuine"]["path"].tolist()
            forgs = g[g.label=="forgery"]["path"].tolist()
            for a in orgs:
                for b in forgs:
                    rows.append({"img1": a, "img2": b})

    else:  # mix
        for w, g in df.groupby("writer_id"):
            orgs = g[g.label=="genuine"]["path"].tolist()
            forgs = g[g.label=="forgery"]["path"].tolist()
            # unos cuantos positivos
            for i in range(min(5, len(orgs)-1)):
                rows.append({"img1": orgs[0], "img2": orgs[i+1]})
            # y unos cuantos negativos
            for i in range(min(5, len(forgs))):
                rows.append({"img1": orgs[0], "img2": forgs[i]})

    out = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"[OK] Guardado {len(out)} pares en {args.out_csv}")

if __name__ == "__main__":
    main()
