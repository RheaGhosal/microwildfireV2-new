#!/usr/bin/env python3

"""
Risk-aware evacuation routing overlay figure generator.

What this script does
---------------------
- Loads your predicted wildfire risk raster (GeoTIFF).
- Downloads an OpenStreetMap (OSM) drivable road network for the raster's bounding box.
- Assigns per-edge costs using time-only (baseline) and time + risk (risk-aware).
- Computes two routes (baseline vs. risk-aware) between an origin and a safe zone.
- Plots:
    - Risk heatmap (background)
    - Road network
    - Baseline route (dashed)
    - Risk-aware route (solid)
- Saves publication-ready PNG and SVG: 
    - fig_evac_routing_overlay.png
    - fig_evac_routing_overlay.svg

Quick start
-----------
1) pip install geopandas osmnx matplotlib rasterio numpy shapely networkx
2) Set the CONFIG values below (paths and coordinates).
3) Run:  python make_evac_overlay.py

Notes
-----
- Internet is required ONCE to download OSM data for the given region.
- For edge risk we sample the raster at the edge midpoint for simplicity and speed.
  (You can switch to max-in-buffer if desired; stub provided.)
- Coordinates must be in latitude, longitude (WGS84).

Author: Rhea Ghosal
"""
from __future__ import annotations

import os
import math
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.transform import rowcol
import osmnx as ox
import geopandas as gpd
from shapely.geometry import LineString, Point
import networkx as nx

# ============== CONFIG: EDIT THESE ==============

# Path to your predicted risk GeoTIFF (single band, higher = higher risk)
RISK_TIFF_PATH = "out/predicted_risk.tif"

# Choose an origin and destination (lat, lon). For example:
#   - origin near a fire-prone area
#   - destination a safe zone (e.g., town, shelter, fire station)
ORIGIN_LATLON = (34.1234, -118.4567)      # (lat, lon)  <-- EDIT
DEST_LATLON   = (34.1567, -118.5012)      # (lat, lon)  <-- EDIT

# Routing weights: cost = alpha*time + beta*risk (+ gamma*slope if you add slopes later)
ALPHA = 1.0        # minutes of travel time coefficient
BETA  = 5.0        # risk coefficient; increase to make routing more risk-averse
# GAMA = 0.25      # slope coefficient (not used in this minimal script)

# Speed assumption for free-flow travel time (km/h) if OSM speed not available
DEFAULT_SPEED_KPH = 50.0

# Optional: Wind annotation (set to None to disable)
# Provide as arrows: [(lon, lat, u, v), ...] in degrees for vector components in map CRS.
WIND_VECTORS = None
# Example:
# WIND_VECTORS = [(-118.48, 34.14, 1.0, 0.2), (-118.46, 34.13, 1.0, 0.2)]

# Output filenames
OUT_PNG = "out/fig_evac_routing_overlay.png"
OUT_SVG = "out/fig_evac_routing_overlay.svg"

# ============== END CONFIG ======================


def load_risk_raster(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Risk raster not found: {path}")
    ds = rasterio.open(path)
    band = ds.read(1)
    # Replace nodata with 0 risk for plotting (optional)
    if ds.nodata is not None:
        band = np.where(band == ds.nodata, 0.0, band)
    # Normalize to [0,1] if values exceed 1 (optional heuristic)
    bmin, bmax = np.nanmin(band), np.nanmax(band)
    if bmax > 1.0:
        band = (band - bmin) / (bmax - bmin + 1e-9)
    extent = [ds.bounds.left, ds.bounds.right, ds.bounds.bottom, ds.bounds.top]
    return ds, band, extent


def graph_for_raster_bbox(ds: rasterio.io.DatasetReader, expand_frac: float = 0.02):
    """Download OSM drivable graph for an expanded raster bbox."""
    left, bottom, right, top = ds.bounds.left, ds.bounds.bottom, ds.bounds.right, ds.bounds.top
    width = right - left
    height = top - bottom
    # Expand a bit so roads at the edge are included
    west  = left   - expand_frac * width
    east  = right  + expand_frac * width
    south = bottom - expand_frac * height
    north = top    + expand_frac * height
    G = ox.graph_from_bbox(north, south, east, west, network_type="drive")
    # Ensure geometry and length fields
    G = ox.add_edge_speeds(G, fallback=DEFAULT_SPEED_KPH)  # km/h
    G = ox.add_edge_travel_times(G)  # seconds
    return G


def sample_risk_at_lonlat(ds: rasterio.io.DatasetReader, lon: float, lat: float) -> float:
    """Sample risk raster at lon/lat. Returns 0.0 if outside or NaN encountered."""
    try:
        row, col = rowcol(ds.transform, lon, lat)
        if 0 <= row < ds.height and 0 <= col < ds.width:
            val = ds.read(1)[row, col]
            if ds.nodata is not None and val == ds.nodata:
                return 0.0
            if math.isnan(val):
                return 0.0
            return float(val)
        return 0.0
    except Exception:
        return 0.0


def assign_edge_costs(G: nx.MultiDiGraph, ds: rasterio.io.DatasetReader, alpha: float, beta: float):
    """Assign per-edge baseline_cost and riskaware_cost."""
    for u, v, k, data in G.edges(keys=True, data=True):
        # Travel time in minutes
        # (ox.add_edge_travel_times stores "travel_time" in seconds)
        t_minutes = data.get("travel_time", data.get("length", 1000.0) / (DEFAULT_SPEED_KPH * 1000/3600.0)) / 60.0

        # Edge midpoint in lon/lat
        try:
            x1, y1 = G.nodes[u]["x"], G.nodes[u]["y"]
            x2, y2 = G.nodes[v]["x"], G.nodes[v]["y"]
        except KeyError:
            # Occasionally nodes may miss coords; skip
            x1 = x2 = data.get("geometry", LineString()).centroid.x if "geometry" in data else 0.0
            y1 = y2 = data.get("geometry", LineString()).centroid.y if "geometry" in data else 0.0
        mx = (x1 + x2) / 2.0
        my = (y1 + y2) / 2.0

        risk_val = sample_risk_at_lonlat(ds, mx, my)

        data["time_min"] = float(t_minutes)
        data["risk_val"] = float(risk_val)
        data["baseline_cost"] = data["time_min"]
        data["riskaware_cost"] = alpha * data["time_min"] + beta * data["risk_val"]


def path_nodes_to_lines(G: nx.MultiDiGraph, path_nodes: list[int]) -> gpd.GeoDataFrame:
    """Convert a node path into a GeoDataFrame of LineStrings for plotting."""
    lines = []
    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        # There may be multiple parallel edges; pick the first
        if G.has_edge(u, v):
            data = G.get_edge_data(u, v)
            # pick any key
            if isinstance(data, dict):
                first_key = next(iter(data))
                attrs = data[first_key]
            else:
                attrs = {}
        else:
            attrs = {}

        if "geometry" in attrs and attrs["geometry"] is not None:
            geom = attrs["geometry"]
        else:
            x1, y1 = G.nodes[u]["x"], G.nodes[u]["y"]
            x2, y2 = G.nodes[v]["x"], G.nodes[v]["y"]
            geom = LineString([(x1, y1), (x2, y2)])
        lines.append(geom)
    gdf = gpd.GeoDataFrame(geometry=lines, crs="EPSG:4326")
    return gdf


def main():
    # 1) Load risk raster
    ds, risk_band, extent = load_risk_raster(RISK_TIFF_PATH)

    # 2) Build OSM graph for raster bbox
    print("Downloading OSM graph...")
    G = graph_for_raster_bbox(ds)

    # 3) Snap ORIGIN/DEST to nearest nodes
    print("Snapping origin/destination to graph...")
    o_lat, o_lon = ORIGIN_LATLON[0], ORIGIN_LATLON[1]
    d_lat, d_lon = DEST_LATLON[0],   DEST_LATLON[1]
    orig_node = ox.distance.nearest_nodes(G, o_lon, o_lat)
    dest_node = ox.distance.nearest_nodes(G, d_lon, d_lat)

    # 4) Assign costs
    print("Assigning edge costs...")
    assign_edge_costs(G, ds, ALPHA, BETA)

    # 5) Compute paths
    print("Computing baseline (time-only) path...")
    baseline_path = nx.shortest_path(G, orig_node, dest_node, weight="baseline_cost")
    print("Computing risk-aware path...")
    riskaware_path = nx.shortest_path(G, orig_node, dest_node, weight="riskaware_cost")

    # 6) Convert to GeoDataFrames
    baseline_gdf = path_nodes_to_lines(G, baseline_path)
    riskaware_gdf = path_nodes_to_lines(G, riskaware_path)

    # 7) Plot
    print("Plotting figure...")
    fig, ax = plt.subplots(figsize=(8, 6))

    # Risk heatmap
    ax.imshow(risk_band, extent=extent, origin="upper", alpha=0.5)  # default colormap

    # Road network (light gray)
    ox.plot_graph(G, ax=ax, node_size=0, edge_color="lightgray", edge_linewidth=0.6, show=False, close=False)

    # Routes
    baseline_gdf.plot(ax=ax, linewidth=2.2, linestyle="--", label="Time-only baseline")
    riskaware_gdf.plot(ax=ax, linewidth=2.2, label="Risk-aware route")

    # Origin/Destination
    ax.scatter([ORIGIN_LATLON[1], DEST_LATLON[1]], [ORIGIN_LATLON[0], DEST_LATLON[0]],
               s=30, marker="o")

    # Optional wind vectors
    if WIND_VECTORS:
        xs = [w[0] for w in WIND_VECTORS]
        ys = [w[1] for w in WIND_VECTORS]
        us = [w[2] for w in WIND_VECTORS]
        vs = [w[3] for w in WIND_VECTORS]
        ax.quiver(xs, ys, us, vs)

    ax.set_title("Risk-aware Evacuation Routing Overlay")
    ax.legend(loc="upper right")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.savefig(OUT_SVG, dpi=300, bbox_inches="tight")
    print(f"Saved {OUT_PNG} and {OUT_SVG}")

if __name__ == "__main__":
    main()

