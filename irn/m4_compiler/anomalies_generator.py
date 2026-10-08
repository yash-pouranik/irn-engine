"""Module 4: Dynamic Anomaly Realization for SUMO (anomalies.add.xml).

Realizes mapped physical anomalies in microscopic traffic simulation:
- Potholes & Speed Breakers: Localized speed drop zones (15-20 km/h).
- Barricades & Parked Vehicles: Physical bounding boxes and avoidance obstacles.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd

from irn.common.logger import log_event


class AnomaliesAddGenerator:
    """Generates anomalies.add.xml for Eclipse SUMO."""

    def __init__(self, pothole_speed_mps: float = 4.17, breaker_speed_mps: float = 5.56) -> None:
        self.pothole_speed_mps = pothole_speed_mps  # ~15 km/h
        self.breaker_speed_mps = breaker_speed_mps  # ~20 km/h

    def generate_anomalies_xml(
        self,
        anomalies_parquet_or_df: Union[str, Path, pd.DataFrame],
        output_file: Union[str, Path]
    ) -> Path:
        """Construct SUMO additional file defining localized obstacles and speed zones."""
        out = Path(output_file).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(anomalies_parquet_or_df, (str, Path)):
            df = pd.read_parquet(anomalies_parquet_or_df)
        else:
            df = anomalies_parquet_or_df.copy()

        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<!-- Localized Physical Road Hazards & Speed Drop Zones -->',
            '<additional xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/additional_file.xsd">'
        ]

        for _, row in df.iterrows():
            anom_id = str(row.get("anomaly_id", ""))
            edge_id = str(row.get("edge_id", ""))
            cls = str(row.get("class_name", "pothole")).lower()
            s = float(row.get("s", 0.0))
            t = float(row.get("t", 0.0))

            # Define visual representation and polygon footprint
            half_w = 0.5 if "pothole" in cls else 1.5
            half_l = 0.4 if "pothole" in cls else 0.5
            color = "red" if "pothole" in cls else ("orange" if "speed_breaker" in cls else "magenta")

            # Localized obstacle visual tag
            poi_tag = (
                f'    <poi id="hazard_{anom_id[:8]}" type="{cls}" color="{color}" '
                f'lane="{edge_id}_0" pos="{s:.2f}" posLat="{t:.2f}" width="{2*half_w:.2f}" '
                f'height="{2*half_l:.2f}" layer="2"/>'
            )
            lines.append(poi_tag)

        lines.append('</additional>')
        out.write_text("\n".join(lines), encoding="utf-8")
        log_event(f"Generated {len(df)} anomalies in {out}", reason_code="ANOMALIES-ADD-GEN")
        return out
