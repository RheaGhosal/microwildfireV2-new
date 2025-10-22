#!/usr/bin/env python3
import argparse, numpy as np, pandas as pd
from pathlib import Path
from tqdm import tqdm
from pystac_client import Client
import stackstac, xarray as xr
from PIL import Image
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

E84_STAC = "https://earth-search.aws.element84.com/v0"

def fetch_chip(lat, lon, date, out_png, size=128, cloud_pct=20):
    api = Client.open(E84_STAC)
    t0 = np.datetime64(date) - np.timedelta64(3,'D')
    t1 = np.datetime64(date) + np.timedelta64(3,'D')
    search = api.search(
        collections=["sentinel-2-l2a"],
        bbox=[lon-0.02, lat-0.02, lon+0.02, lat+0.02],
        datetime=f"{str(t0)}/{str(t1)}",
        query={"eo:cloud_cover": {"lte": cloud_pct}},
        limit=6,
    )
    items = list(search.get_items())
    if not items:
        return False
    bands = ["B04","B03","B02"]
    try:
        arr = stackstac.stack(
            items, assets=bands, resolution=10, dtype="uint16",
            bounds_latlon=(lon-0.005, lat-0.005, lon+0.005, lat+0.005)
        )
        arr = arr.max("time").compute().transpose("band","y","x")
    except Exception:
        return False

    data = arr.values.astype(np.float32)
    p2, p98 = np.percentile(data, 2), np.percentile(data, 98)
    data = np.clip((data - p2)/(p98-p2+1e-6), 0, 1)
    data = (data*255).astype(np.uint8)
    rgb = np.moveaxis(data, 0, 2)
    img = Image.fromarray(rgb)
    if img.size != (size, size):
        img = img.resize((size, size), Image.BILINEAR)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png, format="PNG", optimize=True)
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos_csv", default="data/firms_napa_pos.csv")
    ap.add_argument("--neg_csv", default="data/napa_neg.csv")
    ap.add_argument("--out_dir", default="data/real_chips")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--max_per_class", type=int, default=600)
    ap.add_argument("--cloud_pct", type=int, default=20)
    args = ap.parse_args()

    pos = pd.read_csv(args.pos_csv).head(args.max_per_class).copy(); pos["label"]=1
    neg = pd.read_csv(args.neg_csv).head(args.max_per_class).copy(); neg["label"]=0
    allp = pd.concat([pos,neg], ignore_index=True)
    allp["group_id"] = "Napa"
    allp["timestamp"] = pd.to_datetime(allp["acq_date"]).astype(int)//10**9

    recs = []
    for i, r in tqdm(allp.iterrows(), total=len(allp)):
        out_png = Path(args.out_dir)/f"chip_{i:05d}_{r['label']}.png"
        ok = fetch_chip(r.latitude, r.longitude, r.acq_date, out_png,
                        size=args.size, cloud_pct=args.cloud_pct)
        if ok:
            recs.append({"path": str(out_png), "label": int(r.label),
                         "group_id": r.group_id, "timestamp": int(r.timestamp)})

    df = pd.DataFrame(recs)
    # 60/20/20 split
    n = len(df); idx = np.arange(n); rng = np.random.default_rng(42); rng.shuffle(idx)
    n_tr, n_va = int(0.6*n), int(0.2*n)
    split = np.array(["test"]*n)
    split[idx[:n_tr]]="train"; split[idx[n_tr:n_tr+n_va]]="val"
    df["split"] = split

    out_manifest = Path("data/splits/seed42/train.csv")
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    df[["path","label","group_id","split","timestamp"]].to_csv(out_manifest, index=False)
    print(f"Saved manifest: {out_manifest}  n={len(df)}")

if __name__ == "__main__":
    main()
