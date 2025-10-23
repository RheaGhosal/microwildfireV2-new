#!/usr/bin/env python3
import argparse, numpy as np, pandas as pd
from pathlib import Path
from tqdm import tqdm

import planetary_computer as pc
from pystac_client import Client
import stackstac
import xarray as xr
from PIL import Image

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"

def fetch_chip(lat, lon, date, out_png, size=128, cloud_pct=60, pad_deg=0.007):
    """Fetch an RGB Sentinel-2 chip around (lat,lon) near date, save as PNG."""
    api = Client.open(PC_STAC)
    t0 = np.datetime64(date) - np.timedelta64(3,'D')
    t1 = np.datetime64(date) + np.timedelta64(3,'D')

    search = api.search(
        collections=["sentinel-2-l2a"],
        bbox=[float(lon-0.02), float(lat-0.02), float(lon+0.02), float(lat+0.02)],
        datetime=f"{str(t0)}/{str(t1)}",
        query={"eo:cloud_cover": {"lte": int(cloud_pct)}},
        limit=12,
    )
    items = list(search.items())
    if not items:
        return False
    items = [pc.sign(i) for i in items]

    try:
        arr = stackstac.stack(
            items,
            assets=["B04","B03","B02"],   # RGB
            epsg=3857,                    # pick a common CRS
            resolution=10,                # meters
            dtype="float32",              # allow NaN fill
            fill_value=np.nan,
            bounds_latlon=(float(lon-pad_deg), float(lat-pad_deg),
                           float(lon+pad_deg), float(lat+pad_deg))
        )
        # Max over time for robustness, then to (band,y,x)
        arr = arr.max("time").compute().transpose("band","y","x")
    except Exception:
        return False

    data = arr.values.astype("float32")
    # NaN-robust contrast stretch
    lo, hi = np.nanpercentile(data, 2), np.nanpercentile(data, 98)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return False
    data = (data - lo) / (hi - lo + 1e-6)
    data = np.clip(data, 0, 1)
    data = np.nan_to_num(data, nan=0.0)
    rgb = (np.moveaxis(data, 0, 2) * 255).astype("uint8")  # (H,W,3)

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
    ap.add_argument("--cloud_pct", type=int, default=60)
    ap.add_argument("--pad_deg", type=float, default=0.007)
    args = ap.parse_args()

    pos = pd.read_csv(args.pos_csv).head(args.max_per_class).copy(); pos["label"] = 1
    neg = pd.read_csv(args.neg_csv).head(args.max_per_class).copy(); neg["label"] = 0
    allp = pd.concat([pos, neg], ignore_index=True)

    # Basic manifest columns
    allp["group_id"]  = "Napa"
    allp["timestamp"] = pd.to_datetime(allp["acq_date"]).astype("int64") // 10**9

    recs = []
    for i, r in tqdm(allp.iterrows(), total=len(allp)):
        out_png = Path(args.out_dir) / f"chip_{i:05d}_{int(r['label'])}.png"
        if out_png.exists():   # skip re-download if already present
            ok = True
        else:
            ok = fetch_chip(float(r.latitude), float(r.longitude), str(r.acq_date),
                            str(out_png), size=args.size,
                            cloud_pct=args.cloud_pct, pad_deg=args.pad_deg)
        if ok:
            recs.append({
                "path": str(out_png),
                "label": int(r.label),
                "group_id": r.group_id,
                "timestamp": int(r.timestamp),
            })

    df = pd.DataFrame(recs)
    if df.empty:
        raise SystemExit("No PNG chips were saved. Try increasing --cloud_pct or --pad_deg.")

    # 60/20/20 split – ensure string dtype
    n = len(df)
    idx = np.arange(n, dtype=int)
    rng = np.random.default_rng(42); rng.shuffle(idx)
    n_tr, n_va = int(0.6*n), int(0.2*n)

    split = np.full(n, 'test', dtype=object)
    split[idx[:n_tr]] = 'train'
    split[idx[n_tr:n_tr+n_va]] = 'val'
    df["split"] = split

    out_manifest = Path("data/splits/seed42/train.csv")
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    df[["path","label","group_id","split","timestamp"]].to_csv(out_manifest, index=False)
    print(f"Saved manifest: {out_manifest}  n={len(df)}  "
          f"train={sum(df.split=='train')}  val={sum(df.split=='val')}  test={sum(df.split=='test')}")
    print(f"PNG chips dir: {args.out_dir}")

if __name__ == "__main__":
    main()
