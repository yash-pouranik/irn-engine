"""Module 4: ASAM OpenDRIVE 1.6.1 Exporter (network.xodr).

Constructs international standard ASAM OpenDRIVE 1.6.1 XML:
- Header with proj4 geoReference.
- Roads with planView geometry, left-hand traffic rule (rule="LHT").
- Embedded <objects> elements placing confirmed anomalies at precise (s, t) offsets.
"""

from __future__ import annotations
import datetime
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from shapely import wkt
from shapely.geometry import LineString, Point

from irn.common.logger import log_event


class OpenDriveBuilder:
    """Builds ASAM OpenDRIVE 1.6.1 compliant XML documents."""

    def __init__(
        self,
        utm_zone: int = 44,
        is_northern: bool = True,
        traffic_rule: str = "LHT"
    ) -> None:
        self.utm_zone = utm_zone
        self.is_northern = is_northern
        self.traffic_rule = traffic_rule

    def build_xodr(
        self,
        enriched_edges_df: pd.DataFrame,
        anomalies_df: pd.DataFrame,
        output_file: Union[str, Path]
    ) -> Path:
        """Constructs and writes network.xodr."""
        out = Path(output_file).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)

        now_str = datetime.datetime.now(datetime.timezone.utc).isoformat()
        hemisphere = "north" if self.is_northern else "south"
        proj_str = f"+proj=utm +zone={self.utm_zone} +{hemisphere} +datum=WGS84 +units=m +no_defs"

        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<OpenDRIVE>',
            f'    <header revMajor="1" revMinor="6" name="IRN_Network" version="1.0" date="{now_str}">',
            f'        <geoReference><![CDATA[{proj_str}]]></geoReference>',
            f'    </header>'
        ]

        # Group anomalies by edge_id
        anomalies_by_edge: Dict[str, List[Dict[str, Any]]] = {}
        if not anomalies_df.empty:
            for _, anom in anomalies_df.iterrows():
                e_id = str(anom.get("edge_id", ""))
                anomalies_by_edge.setdefault(e_id, []).append(anom.to_dict())

        road_counter = 1
        for _, row in enriched_edges_df.iterrows():
            edge_id = str(row.get("edge_id", f"edge_{road_counter}"))
            length_m = float(row.get("length_m", 50.0))
            width_m = float(row.get("width_dir_m", 7.0))

            # Extract start point coordinates
            x_start = 500000.0
            y_start = 2000000.0
            hdg = 0.0

            if "geometry_wkt" in row and row["geometry_wkt"]:
                try:
                    geom = wkt.loads(str(row["geometry_wkt"]))
                    coords = list(geom.coords)
                    if len(coords) >= 2:
                        x_start, y_start = coords[0]
                        dx = coords[1][0] - coords[0][0]
                        dy = coords[1][1] - coords[0][1]
                        hdg = (math.atan2(dx, dy) + 2 * math.pi) % (2 * math.pi)
                except Exception:
                    pass

            lines.append(
                f'    <road name="{edge_id}" length="{length_m:.3f}" id="{road_counter}" '
                f'junction="-1" rule="{self.traffic_rule}">'
            )
            lines.append('        <link/>')
            lines.append('        <planView>')
            lines.append(
                f'            <geometry s="0.000" x="{x_start:.3f}" y="{y_start:.3f}" '
                f'hdg="{hdg:.4f}" length="{length_m:.3f}">'
            )
            lines.append('                <line/>')
            lines.append('            </geometry>')
            lines.append('        </planView>')
            lines.append('        <lanes>')
            lines.append('            <laneSection s="0.000">')
            lines.append('                <center>')
            lines.append('                    <lane id="0" type="none" level="false"/>')
            lines.append('                </center>')
            lines.append('                <left>')
            lines.append('                    <lane id="1" type="driving" level="false">')
            lines.append(f'                        <width sOffset="0.000" a="{width_m:.3f}" b="0.0" c="0.0" d="0.0"/>')
            lines.append('                    </lane>')
            lines.append('                </left>')
            lines.append('            </laneSection>')
            lines.append('        </lanes>')

            # Add objects tag for embedded physical anomalies
            edge_anoms = anomalies_by_edge.get(edge_id, [])
            if edge_anoms:
                lines.append('        <objects>')
                for obj_idx, obj in enumerate(edge_anoms, start=1):
                    s_val = float(obj.get("s", 0.0))
                    t_val = float(obj.get("t", 0.0))
                    cls_name = str(obj.get("class_name", "obstacle"))
                    lines.append(
                        f'            <object id="{obj_idx}" type="obstacle" name="{cls_name}" '
                        f's="{s_val:.3f}" t="{t_val:.3f}" zOffset="0.000" validLength="0.800" '
                        f'width="1.000" length="0.800" hdg="0.000" pitch="0.000" roll="0.000"/>'
                    )
                lines.append('        </objects>')
            else:
                lines.append('        <objects/>')

            lines.append('    </road>')
            road_counter += 1

        lines.append('</OpenDRIVE>')
        out.write_text("\n".join(lines), encoding="utf-8")
        log_event(
            f"Generated ASAM OpenDRIVE 1.6.1 network ({road_counter - 1} roads) in {out}",
            reason_code="XODR-GEN-DONE"
        )
        return out
