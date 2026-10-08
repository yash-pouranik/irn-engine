"""Module 3: Road Usable Width & Heterogeneous Class Mix Fusion.

Computes:
- Directional usable road width (width_dir_m):
    * Oneway: full roadway width.
    * Bidirectional: undivided half-width (total_width / 2.0).
- Effective lane capacity (lanes_effective = max(1, round(w_dir / 3.0))).
- Dynamic heterogeneous traffic mix aggregation (two_wheeler, auto, car, bus, truck).
"""

from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd


def fuse_edge_widths(
    edges_df: pd.DataFrame,
    satellite_widths: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """Computes directional usable width and effective lanes for all road edges."""
    df = edges_df.copy()

    directional_widths = []
    effective_lanes = []

    for _, row in df.iterrows():
        edge_id = str(row.get("edge_id", ""))
        osm_width = float(row.get("width_osm_m", 7.0))
        oneway = bool(row.get("oneway", False))

        # Check if satellite transect measurement exists
        if satellite_widths and edge_id in satellite_widths:
            sat_w = satellite_widths[edge_id]
            # Inverse variance weighting: OSM (sigma=1.0m) vs Satellite (sigma=0.5m)
            w_fused = (osm_width / (1.0**2) + sat_w / (0.5**2)) / (1.0 / (1.0**2) + 1.0 / (0.5**2))
        else:
            w_fused = osm_width

        # Directional usable width: undivided two-way roads share total width
        if oneway:
            w_dir = round(w_fused, 2)
        else:
            w_dir = round(w_fused / 2.0, 2)

        # Indian effective lanes (each 3.0m nominal slice)
        lanes_eff = max(1, int(round(w_dir / 3.0)))

        directional_widths.append(w_dir)
        effective_lanes.append(lanes_eff)

    df["width_dir_m"] = directional_widths
    df["lanes_effective"] = effective_lanes
    return df


def aggregate_dynamic_class_mix(
    detections_df: pd.DataFrame,
    matched_edges_map: Dict[int, str]
) -> Dict[str, Dict[str, float]]:
    """Computes heterogeneous traffic percentage split per edge for dynamic agents."""
    edge_counts: Dict[str, Dict[str, int]] = {}

    for _, row in detections_df.iterrows():
        f_idx = int(row.get("frame_idx", 0))
        cls = str(row.get("class_name", ""))
        edge_id = matched_edges_map.get(f_idx)

        if not edge_id:
            continue

        edge_counts.setdefault(edge_id, {})
        edge_counts[edge_id][cls] = edge_counts[edge_id].get(cls, 0) + 1

    mix_result: Dict[str, Dict[str, float]] = {}
    for edge_id, counts in edge_counts.items():
        total = sum(counts.values())
        if total > 0:
            mix_result[edge_id] = {cls: round(c / total, 3) for cls, c in counts.items()}
        else:
            mix_result[edge_id] = {}

    return mix_result
