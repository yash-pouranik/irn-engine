"""Data Contract D2: Detection Schema and PyArrow Serializer.

Enforces:
- Schema fields: sequence_id, frame_idx, frame_ts, class_name, conf, x1, y1, x2, y2, img_w, img_h, track_id, model_version
- Confidence >= 0.25
- Ground contact point: bottom-center (u = (x1 + x2)/2, v = y2)
- Rejection logging for malformed records (Constraint C-4, NFR-REL-07)
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, Field, field_validator, model_validator
from irn.common.logger import log_event

# Canonical recognized object classes in Indian traffic (Static & Dynamic)
STATIC_CLASSES = {
    "pothole",
    "speed_breaker",
    "barrier",
    "barricade",
    "parked_vehicle_obstruction",
}

DYNAMIC_CLASSES = {
    "two_wheeler",
    "auto_rickshaw",
    "car",
    "bus",
    "truck",
    "bicycle",
    "cycle_rickshaw",
    "pedestrian",
    "stray_animal",
}

RECOGNIZED_CLASSES = STATIC_CLASSES | DYNAMIC_CLASSES

# PyArrow Schema for D2 detections.parquet
DETECTION_PYARROW_SCHEMA = pa.schema([
    ("sequence_id", pa.string()),
    ("frame_idx", pa.int64()),
    ("frame_ts", pa.float64()),
    ("class_name", pa.string()),
    ("conf", pa.float64()),
    ("x1", pa.float64()),
    ("y1", pa.float64()),
    ("x2", pa.float64()),
    ("y2", pa.float64()),
    ("u", pa.float64()),  # Ground contact u = (x1 + x2) / 2
    ("v", pa.float64()),  # Ground contact v = y2
    ("img_w", pa.int64()),
    ("img_h", pa.int64()),
    ("track_id", pa.int64()),  # -1 if untracked
    ("model_version", pa.string()),
])


class DetectionRecord(BaseModel):
    """Pydantic model validating individual detection bounding boxes."""

    sequence_id: str
    frame_idx: int = Field(ge=0)
    frame_ts: float = Field(ge=0.0)
    class_name: str
    conf: float = Field(ge=0.25, le=1.0)
    x1: float = Field(ge=0.0)
    y1: float = Field(ge=0.0)
    x2: float = Field(gt=0.0)
    y2: float = Field(gt=0.0)
    img_w: int = Field(gt=0)
    img_h: int = Field(gt=0)
    track_id: Optional[int] = -1
    model_version: str = "yolov8-idd-v1"

    @model_validator(mode="after")
    def validate_geometry(self) -> "DetectionRecord":
        if self.x2 <= self.x1:
            raise ValueError(f"x2 ({self.x2}) must be strictly greater than x1 ({self.x1})")
        if self.y2 <= self.y1:
            raise ValueError(f"y2 ({self.y2}) must be strictly greater than y1 ({self.y1})")
        if self.x2 > self.img_w:
            raise ValueError(f"x2 ({self.x2}) exceeds img_w ({self.img_w})")
        if self.y2 > self.img_h:
            raise ValueError(f"y2 ({self.y2}) exceeds img_h ({self.img_h})")
        return self

    @property
    def ground_contact_uv(self) -> Tuple[float, float]:
        """Calculates bottom-center contact point on ground plane: (u = (x1+x2)/2, v = y2)."""
        return ((self.x1 + self.x2) / 2.0, self.y2)

    def to_arrow_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary matching PyArrow schema."""
        u, v = self.ground_contact_uv
        return {
            "sequence_id": self.sequence_id,
            "frame_idx": self.frame_idx,
            "frame_ts": self.frame_ts,
            "class_name": self.class_name,
            "conf": self.conf,
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "u": u,
            "v": v,
            "img_w": self.img_w,
            "img_h": self.img_h,
            "track_id": self.track_id if self.track_id is not None else -1,
            "model_version": self.model_version,
        }


def validate_and_filter_detections(
    raw_records: List[Dict[str, Any]],
    log_rejections: bool = True
) -> List[DetectionRecord]:
    """Validate a batch of raw records, filtering out and logging malformed entries."""
    valid_records: List[DetectionRecord] = []
    rejected_count = 0

    for raw in raw_records:
        try:
            record = DetectionRecord(**raw)
            valid_records.append(record)
        except Exception as e:
            rejected_count += 1
            if log_rejections:
                log_event(
                    f"Rejected malformed detection record: {e} | Raw data: {raw}",
                    reason_code="REJECT-SCHEMA",
                    level=30  # WARNING
                )

    if rejected_count > 0:
        log_event(
            f"Filtered {len(valid_records)} valid records, rejected {rejected_count} malformed records",
            reason_code="SCHEMA-FILTER"
        )
    return valid_records


def save_detections_parquet(
    records: List[DetectionRecord],
    output_path: Union[str, Path]
) -> Path:
    """Write validated detections to Parquet using the strict D2 PyArrow schema."""
    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    if not records:
        table = pa.Table.from_arrays(
            [pa.array([], type=f.type) for f in DETECTION_PYARROW_SCHEMA],
            schema=DETECTION_PYARROW_SCHEMA
        )
    else:
        rows = [r.to_arrow_dict() for r in records]
        table = pa.Table.from_pylist(rows, schema=DETECTION_PYARROW_SCHEMA)

    pq.write_table(table, out, compression="snappy")
    log_event(
        f"Wrote {len(records)} detections to {out} (Contract D2)",
        reason_code="IO-WRITE-D2"
    )
    return out


def load_detections_parquet(parquet_path: Union[str, Path]) -> pa.Table:
    """Read detections Parquet file and ensure compliance with D2 schema."""
    p = Path(parquet_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Detections file not found: {p}")
    table = pq.read_table(p)
    return table
