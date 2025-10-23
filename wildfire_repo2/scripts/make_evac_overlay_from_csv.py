#!/usr/bin/env python3
"""
Make an evacuation overlay figure directly from a CSV of (lat, lon, prob).
No GeoTIFF required.

Pipeline:
  1) Interpolate points -> risk grid (default: nearest, robust for few/collinear samples).
  2) Build OSM drivable road graph for the grid's bounding box.
  3) Assign per-edge costs:
        baseline_cost  = travel time (minutes)
        riskaware_cost = time + beta * risk_at_edge_midpoint
  4) Compute two routes (baseline vs risk-aware).
  5) Plot risk heatmap, road network, and both routes.
  6) Save PNG + SVG.

Example:
  python scripts/make_evac_overlay_from_csv.py \
    --csv data/pred_points.csv --lat-col lat --lon-col lon --prob-col prob \
    --origin-lat 34.1234 --origin-lon -118.4567 \
    --dest-lat 34.1567  --dest-lon  -118.5012 \
    --grid-res 0.002 --beta 5.0 \
    --out-png out/fig_evac_routing_overlay.png \
    --out-svg out/fig_evac_routing_overlay.svg
"""

import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import osmnx as ox
import networkx as nx
from shapely.geometry import box
from scipy.interpolate import griddata


# ---------- Interpolation / grid utilities ----------

def grid_from_points(csv_path, lat_col, lon_col, prob_col, res,
                     method="nearest", pad=1):
    """
    Build a regular lon/lat grid and interpolate probabilities onto it.

    - method: "nearest" by default (robust for tiny or nearly-collinear samples).
              "linear"/"cubic" require enough 2D spread; otherwise QHull may fail.
    - pad: add this many grid cells around the min/max bounds to keep edge roads.
    """
    df = pd.read_csv(csv_path)
    required = {lat_col, lon_col, prob_col}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise SystemExit(f"[evac] ERROR: CSV missing required column(s): {missing}")

    lats = df[lat_col].to_numpy(dtype=float)
    lons = df[lon_col].to_numpy(dtype=float)
    probs = df[prob_col].to_numpy(dtype=float)

    lon_min, lon_max = float(lons.min()), float(lons.max())
    lat_min, lat_max = float(lats.min()), float(lats.max())

    # Pad bounds by 'pad' cells
    lon_min -= res * pad
    lon_max += res * pad
    lat_min -= res * pad
    lat_max += res * pad

    xs = np.arange(lon_min, lon_max + 1e-12, res)
    ys = np.arange(lat_min, lat_max + 1e-12, res)
    XX, YY = np.meshgrid(xs, ys)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid = griddata(np.column_stack([lons, lats]), probs, (XX, YY), method=method)

    # Fill any NaNs (linear/cubic) to keep routing stable
    grid = np.nan_to_num(grid, nan=0.0)

    bounds = (lon_min, lon_max, lat_min, lat_max)
    return grid, bounds


def sample_grid_at_lonlat(grid, bounds, lon, lat):
    """Nearest-cell lookup on the risk grid."""
    lon_min, lon_max, lat_min, lat_max = bounds
    H, W = grid.shape
    if not (lon_min <= lon <= lon_max and lat_min <= lat <= lat_max):
        return 0.0
    x = (lon - lon_min) / max(lon_max - lon_min, 1e-9) * (W - 1)
    y = (lat - lat_min) / max(lat_max - lat_min, 1e-9) * (H - 1)
    col = int(round(x))
    row = int(round(y))
    col = max(0, min(W - 1, col))
    row = max(0, min(H - 1, row))
    return float(grid[row, col])


# ---------- Network utilities ----------

def assign_edge_costs_from_grid(G, grid, bounds, alpha=1.0, beta=5.0, default_speed_kph=50.0):
    """
    baseline_cost  = time_min
    riskaware_cost = alpha*time_min + beta*risk_val
    where risk_val is sampled at the edge midpoint.
    """
    # Ensure speeds/travel time exist
    try:
        G = ox.add_edge_speeds(G, fallback=default_speed_kph)   # km/h
        G = ox.add_edge_travel_times(G)                         # seconds
    except Exception:
        pass

    for u, v, k, data in G.edges(keys=True, data=True):
        t_sec = data.get("travel_time")
        if t_sec is None:
            length_m = data.get("length", 1000.0)
            t_sec = length_m / (default_speed_kph * 1000 / 3600.0)
        t_min = float(t_sec) / 60.0

        try:
            x1, y1 = G.nodes[u]["x"], G.nodes[u]["y"]
            x2, y2 = G.nodes[v]["x"], G.nodes[v]["y"]
            mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            risk_val = sample_grid_at_lonlat(grid, bounds, mx, my)
        except Exception:
            risk_val = 0.0

        data["time_min"] = t_min
        data["risk_val"] = risk_val
        data["baseline_cost"] = t_min
        data["riskaware_cost"] = alpha * t_min + beta * risk_val


def path_nodes_to_gdf(G, path_nodes):
    """Convert a node path into a GeoDataFrame of LineStrings for plotting."""
    import geopandas as gpd
    from shapely.geometry import LineString

    lines = []
    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        data = G.get_edge_data(u, v)
        if isinstance(data, dict):
            first_key = next(iter(data))
            attrs = data[first_key]
        else:
            attrs = {}
        if "geometry" in attrs and attrs["geometry"] is not None:
            geom = attrs["geometry"]
        else:
            x1, y1 = G.nodes[u]["x"], G.nodes[u]["y"]
            x2, y2 = G.nodes[v]["x"], G.nodes[v]["y"]
            geom = LineString([(x1, y1), (x2, y2)])
        lines.append(geom)

    return gpd.GeoDataFrame(geometry=lines, crs="EPSG:4326")


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV with lat/lon/prob")
    ap.add_argument("--lat-col", default="lat")
    ap.add_argument("--lon-col", default="lon")
    ap.add_argument("--prob-col", default="prob")
    ap.add_argument("--grid-res", type=float, default=0.001, help="Grid resolution in degrees")
    ap.add_argument("--interp", choices=["nearest", "linear", "cubic"], default="nearest",
                    help="Interpolation method for gridding input points")
    ap.add_argument("--origin-lat", type=float, required=True)
    ap.add_argument("--origin-lon", type=float, required=True)
    ap.add_argument("--dest-lat", type=float, required=True)
    ap.add_argument("--dest-lon", type=float, required=True)
    ap.add_argument("--beta", type=float, default=5.0, help="Risk aversion weight")
    ap.add_argument("--out-png", default="fig_evac_routing_overlay.png")
    ap.add_argument("--out-svg", default="fig_evac_routing_overlay.svg")
    args = ap.parse_args()

    # OSMnx settings: cache + log for clarity/speed
    ox.settings.use_cache = True
    ox.settings.log_console = True
    ox.settings.requests_timeout = 180

    # 1) Build risk grid from points
    grid, bounds = grid_from_points(
        args.csv, args.lat_col, args.lon_col, args.prob_col,
        res=args.grid_res, method=args.interp, pad=1
    )
    lon_min, lon_max, lat_min, lat_max = bounds
    print(f"[evac] bounds lon=[{lon_min:.6f},{lon_max:.6f}] lat=[{lat_min:.6f},{lat_max:.6f}]")

    # 2) Build OSM drivable graph via polygon (works across OSMnx versions)
    print("[evac] downloading OSM drivable graph…")
    bbox_poly = box(lon_min, lat_min, lon_max, lat_max)
    G = ox.graph_from_polygon(bbox_poly, network_type="drive")

    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        raise SystemExit("[evac] ERROR: OSM graph is empty for these bounds. "
                         "Try expanding the area (increase pad or decrease --grid-res), or adjust points.")

    # Keep largest connected component (pure NetworkX: version-agnostic)
    if G.is_directed():
        comps = list(nx.weakly_connected_components(G))
    else:
        comps = list(nx.connected_components(G))
    if not comps:
        raise SystemExit("[evac] ERROR: graph has no connected components for these bounds.")
    largest = max(comps, key=len)
    G = G.subgraph(largest).copy()
    print(f"[evac] graph ready: nodes={G.number_of_nodes()} edges={G.number_of_edges()}")

    # 3) Assign costs from grid
    assign_edge_costs_from_grid(G, grid, bounds, alpha=1.0, beta=args.beta)

    # 4) Snap origin/destination and compute paths
    orig_node = ox.distance.nearest_nodes(G, args.origin_lon, args.origin_lat)
    dest_node = ox.distance.nearest_nodes(G, args.dest_lon, args.dest_lat)

    try:
        baseline_path = nx.shortest_path(G, orig_node, dest_node, weight="baseline_cost")
        riskaware_path = nx.shortest_path(G, orig_node, dest_node, weight="riskaware_cost")
    except nx.NetworkXNoPath:
        raise SystemExit("[evac] ERROR: No path between origin and destination. "
                         "Pick a closer destination or expand the bbox (increase pad).")

    # 5) Plot overlay
    import geopandas as gpd  # used for plotting line geoms
    baseline_gdf = path_nodes_to_gdf(G, baseline_path)
    riskaware_gdf = path_nodes_to_gdf(G, riskaware_path)

    fig, ax = plt.subplots(figsize=(8, 6))
    extent = [lon_min, lon_max, lat_min, lat_max]

    # Risk heatmap
    ax.imshow(grid, extent=extent, origin="lower", alpha=0.5)

    # Road network (light gray)
    try:
        ox.plot_graph(G, ax=ax, node_size=0, edge_color="lightgray", edge_linewidth=0.6, show=False, close=False)
    except Exception:
        pass

    # Routes
    baseline_gdf.plot(ax=ax, linewidth=2.2, linestyle="--", label="Time-only baseline")
    riskaware_gdf.plot(ax=ax, linewidth=2.2, label="Risk-aware route")

    # Origin/Destination
    ax.scatter([args.origin_lon, args.dest_lon], [args.origin_lat, args.dest_lat], s=30, marker="o")

    ax.legend(loc="upper right")
    ax.set_title("Risk-aware Evacuation Routing Overlay")
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=300, bbox_inches="tight")
    plt.savefig(args.out_svg, dpi=300, bbox_inches="tight")
    print(f"[evac] Saved {args.out_png} and {args.out_svg}")


if __name__ == "__main__":
    main()

