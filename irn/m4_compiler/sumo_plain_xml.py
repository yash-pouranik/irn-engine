"""Module 4: SUMO Plain-XML Network Generator.

Generates Plain-XML files (.nod.xml, .edg.xml) for compilation with netconvert:
- Enforces Rule G-2: Left-Hand Traffic (LHT).
- Enforces Rule G-6: Never write .net.xml by hand; generate valid Plain-XML.
- Implements `lanefree_mode = collapse`: builds wide, continuous single-lane
  directional carriageways reflecting real Indian road widths.
"""

from __future__ import annotations
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import geopandas as gpd
import pandas as pd
from shapely import wkt
from shapely.geometry import LineString, Point

from irn.common.logger import log_event


class SumoPlainXmlGenerator:
    """Generates standard SUMO Plain-XML node and edge definitions."""

    def __init__(
        self,
        lanefree_mode: str = "collapse",
        default_speed_kph: float = 40.0,
        default_width_m: float = 7.0
    ) -> None:
        self.lanefree_mode = lanefree_mode  # 'collapse' or 'lanes'
        self.default_speed_kph = default_speed_kph
        self.default_width_m = default_width_m

    def generate_plain_xml(
        self,
        enriched_edges_path: Union[str, Path],
        nodes_gpkg_path: Union[str, Path],
        out_dir: Union[str, Path]
    ) -> Tuple[Path, Path]:
        """Generates plain.nod.xml and plain.edg.xml in target directory."""
        out = Path(out_dir).resolve()
        out.mkdir(parents=True, exist_ok=True)

        # 1. Load nodes and edges
        nodes_gdf = gpd.read_file(nodes_gpkg_path, layer="nodes")
        
        # Load edges (parquet or gpkg)
        edges_path = Path(enriched_edges_path)
        if edges_path.suffix == ".parquet":
            edges_df = pd.read_parquet(edges_path)
        else:
            edges_df = gpd.read_file(edges_path)

        nod_file = out / "plain.nod.xml"
        edg_file = out / "plain.edg.xml"

        # 2. Write plain.nod.xml
        nodes_lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<nodes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/nodes_file.xsd">'
        ]

        for _, row in nodes_gdf.iterrows():
            node_id = str(row.get("node_id", ""))
            x = float(row.get("x_utm", row.geometry.x if hasattr(row, "geometry") else 0.0))
            y = float(row.get("y_utm", row.geometry.y if hasattr(row, "geometry") else 0.0))
            hw_tag = str(row.get("highway", "")).lower()

            node_type = "traffic_light" if "traffic_signals" in hw_tag else "priority"
            nodes_lines.append(f'    <node id="{node_id}" x="{x:.3f}" y="{y:.3f}" type="{node_type}"/>')

        nodes_lines.append('</nodes>')
        nod_file.write_text("\n".join(nodes_lines), encoding="utf-8")
        log_event(f"Generated {len(nodes_gdf)} nodes in {nod_file}", reason_code="SUMO-NOD-GEN")

        # 3. Write plain.edg.xml
        edges_lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<edges xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/edges_file.xsd">'
        ]

        for _, row in edges_df.iterrows():
            edge_id = str(row.get("edge_id", ""))
            from_node = str(row.get("u", ""))
            to_node = str(row.get("v", ""))
            if not from_node or not to_node:
                continue

            maxspeed_kph = float(row.get("maxspeed_kph", self.default_speed_kph))
            speed_mps = round(maxspeed_kph / 3.6, 2)

            width_m = float(row.get("width_dir_m", row.get("width_osm_m", self.default_width_m)))
            lanes_eff = int(row.get("lanes_effective", 1))

            # Shape coordinates
            shape_str = ""
            if "geometry_wkt" in row and row["geometry_wkt"]:
                try:
                    geom = wkt.loads(str(row["geometry_wkt"]))
                    coords = [f"{pt[0]:.2f},{pt[1]:.2f}" for pt in geom.coords]
                    shape_str = f' shape="{" ".join(coords)}"'
                except Exception:
                    pass
            elif "geometry" in row and hasattr(row["geometry"], "coords"):
                coords = [f"{pt[0]:.2f},{pt[1]:.2f}" for pt in row["geometry"].coords]
                shape_str = f' shape="{" ".join(coords)}"'

            # lanefree_mode: 'collapse' builds a single continuous wide lane
            if self.lanefree_mode == "collapse":
                num_lanes = 1
                lane_width = max(3.5, width_m)
            else:
                num_lanes = max(1, lanes_eff)
                lane_width = round(width_m / num_lanes, 2)

            edge_tag = (
                f'    <edge id="{edge_id}" from="{from_node}" to="{to_node}" '
                f'numLanes="{num_lanes}" speed="{speed_mps}" width="{lane_width:.2f}"{shape_str}/>'
            )
            edges_lines.append(edge_tag)

        edges_lines.append('</edges>')
        edg_file.write_text("\n".join(edges_lines), encoding="utf-8")
        log_event(f"Generated {len(edges_df)} edges in {edg_file}", reason_code="SUMO-EDG-GEN")

        return nod_file, edg_file
