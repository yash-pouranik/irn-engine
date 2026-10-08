"""Module 2: Computer Vision & IDD Ingestion Interface."""

from irn.m2_cv_interface.idd_adapter import IDDAdapter
from irn.m2_cv_interface.detector import MockDetector, YOLOv8Detector
from irn.m2_cv_interface.schema import (
    DetectionRecord,
    load_detections_parquet,
    save_detections_parquet,
    validate_and_filter_detections,
)

__all__ = [
    "IDDAdapter",
    "MockDetector",
    "YOLOv8Detector",
    "DetectionRecord",
    "load_detections_parquet",
    "save_detections_parquet",
    "validate_and_filter_detections",
]
