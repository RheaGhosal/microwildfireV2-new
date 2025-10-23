
#!/usr/bin/env python3

Turn a NumPy risk/probability array into a georeferenced WGS84 GeoTIFF.

Usage:
  python make_risk_geotiff_from_npy.py --npy out/risk_probs.npy       --lon-min -118.60 --lon-max -118.30 --lat-min 34.05 --lat-max 34.25       --out out/predicted_risk.tif

Notes:
- The NumPy file must contain a 2D array (H x W) with values in [0,1] (or any numeric range).
- Coordinates are in degrees (EPSG:4326). Make sure the bbox matches your array's region.
- If you only know the pixel size (resolution), compute bounds from a known origin or supply min/max.

import argparse
import numpy as np
import rasterio
from rasterio.transform import from_bounds

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npy", required=True, help="Path to 2D NumPy array (.npy) with risk/probability")
    p.add_argument("--lon-min", type=float, required=True)
    p.add_argument("--lon-max", type=float, required=True)
    p.add_argument("--lat-min", type=float, required=True)
    p.add_argument("--lat-max", type=float, required=True)
    p.add_argument("--out", default="out/predicted_risk.tif")
    args = p.parse_args()

    risk = np.load(args.npy)
    if risk.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {risk.shape}")

    H, W = risk.shape
    transform = from_bounds(args.lon_min, args.lat_min, args.lon_max, args.lat_max, W, H)

    # Write GeoTIFF
    with rasterio.open(
        args.out, "w",
        driver="GTiff",
        height=H, width=W,
        count=1, dtype=risk.dtype,
        crs="EPSG:4326",
        transform=transform,
        compress="lzw"
    ) as dst:
        dst.write(risk, 1)

    print(f"Wrote {args.out} with shape {risk.shape} and bounds "
          f"[{args.lon_min},{args.lat_min}]..[{args.lon_max},{args.lat_max}]")

if __name__ == "__main__":
    main()

