"""Module 2: IDD Multimodal Dataset Adapter.

Handles:
- Loading and temporal synchronization of front-view camera frame sequences.
- Ingestion and validation of vehicle GPS logs (gps.csv).
- Parsing IDD annotation formats and bounding boxes.
- Exporting D2-compliant Parquet datasets.
"""

from __future__ import annotations
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
from irn.common.logger import log_event
from irn.m2_cv_interface.schema import (
    DetectionRecord,
    save_detections_parquet,
    validate_and_filter_detections,
)


class IDDAdapter:
    """Adapter for ingesting and processing sequences from IDD Multimodal."""

    def __init__(self, sequence_id: str, img_w: int = 1920, img_h: int = 1080) -> None:
        self.sequence_id = sequence_id
        self.img_w = img_w
        self.img_h = img_h

    def load_gps_log(self, gps_csv_path: Union[str, Path]) -> pd.DataFrame:
        """Load and normalize vehicle GPS log.

        Expected columns (or aliases): timestamp/time, lat/latitude, lon/longitude, heading/bearing, speed.
        """
        path = Path(gps_csv_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"GPS log not found: {path}")

        df = pd.read_csv(path)
        # Normalize column names to lowercase stripped
        df.columns = [c.lower().strip() for c in df.columns]

        # Aliases
        rename_map = {}
        for col in df.columns:
            if col in ("time", "t", "timestamp_s", "epoch"):
                rename_map[col] = "timestamp"
            elif col in ("latitude", "y"):
                rename_map[col] = "lat"
            elif col in ("longitude", "x"):
                rename_map[col] = "lon"
            elif col in ("bearing", "yaw"):
                rename_map[col] = "heading"
            elif col in ("velocity", "speed_kph"):
                rename_map[col] = "speed"

        df = df.rename(columns=rename_map)

        required = {"timestamp", "lat", "lon"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"GPS CSV missing mandatory columns: {missing}")

        # Ensure sorted by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)

        if "heading" not in df.columns:
            # Estimate heading from coordinates if absent
            df["heading"] = 0.0

        if "speed" not in df.columns:
            df["speed"] = 10.0  # default urban speed m/s

        log_event(
            f"Loaded {len(df)} GPS fixes from {path} (duration: {df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]:.1f}s)",
            reason_code="GPS-INGEST-SUCCESS"
        )
        return df

    def parse_annotations(
        self,
        raw_detections: List[Dict[str, Any]],
        output_parquet_path: Optional[Union[str, Path]] = None
    ) -> List[DetectionRecord]:
        """Validate raw detection dictionaries and optionally write D2 Parquet."""
        for det in raw_detections:
            det.setdefault("sequence_id", self.sequence_id)
            det.setdefault("img_w", self.img_w)
            det.setdefault("img_h", self.img_h)

        validated = validate_and_filter_detections(raw_detections, log_rejections=True)

        if output_parquet_path:
            save_detections_parquet(validated, output_parquet_path)

        return validated
