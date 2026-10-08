"""Module 2: Object Detection Engine & Mock Detector.

Supports:
- YOLOv8 inference with ByteTrack on front-view drive video.
- Mock/calibrated detection generator for offline and synthetic validation.
- Export to Parquet conforming strictly to Contract D2.
"""

from __future__ import annotations
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from irn.common.logger import log_event
from irn.m2_cv_interface.schema import (
    DetectionRecord,
    save_detections_parquet,
    validate_and_filter_detections,
)


class MockDetector:
    """Deterministic mock detector simulating Indian road anomalies and agents."""

    def __init__(
        self,
        sequence_id: str = "seq_idd_demo_01",
        img_w: int = 1920,
        img_h: int = 1080,
        min_conf: float = 0.50,
        seed: int = 42
    ) -> None:
        self.sequence_id = sequence_id
        self.img_w = img_w
        self.img_h = img_h
        self.min_conf = min_conf
        self.rng = np.random.default_rng(seed)

    def generate_synthetic_drive_detections(
        self,
        num_frames: int = 60,
        fps: float = 10.0,
        anomalies_to_plant: Optional[List[Dict[str, Any]]] = None
    ) -> List[DetectionRecord]:
        """Generate a series of frame-by-frame detections simulating a drive sequence."""
        records: List[DetectionRecord] = []
        dt = 1.0 / fps

        # Default planted anomalies along the route if none provided
        if anomalies_to_plant is None:
            anomalies_to_plant = [
                {"class_name": "pothole", "start_frame": 10, "end_frame": 25, "u_center": 900.0, "track_id": 101},
                {"class_name": "speed_breaker", "start_frame": 30, "end_frame": 45, "u_center": 960.0, "track_id": 102},
                {"class_name": "auto_rickshaw", "start_frame": 5, "end_frame": 50, "u_center": 1150.0, "track_id": 201},
                {"class_name": "two_wheeler", "start_frame": 20, "end_frame": 55, "u_center": 750.0, "track_id": 202},
            ]

        for frame_idx in range(num_frames):
            frame_ts = round(frame_idx * dt, 3)

            for item in anomalies_to_plant:
                s_frame = item.get("start_frame", 0)
                e_frame = item.get("end_frame", num_frames)

                if s_frame <= frame_idx <= e_frame:
                    progress = (frame_idx - s_frame) / max(1, (e_frame - s_frame))
                    # Obstacle moves down towards camera in perspective view (v increases from 650 to 950)
                    v_contact = 620.0 + progress * 320.0
                    box_w = 120.0 + progress * 100.0
                    box_h = 60.0 + progress * 50.0

                    u_center = item.get("u_center", 960.0) + self.rng.normal(0, 1.5)
                    x1 = max(0.0, u_center - box_w / 2.0)
                    x2 = min(float(self.img_w), u_center + box_w / 2.0)
                    y2 = min(float(self.img_h), v_contact)
                    y1 = max(0.0, y2 - box_h)

                    conf = float(np.clip(0.70 + self.rng.normal(0, 0.05), self.min_conf, 0.99))

                    record = DetectionRecord(
                        sequence_id=self.sequence_id,
                        frame_idx=frame_idx,
                        frame_ts=frame_ts,
                        class_name=item["class_name"],
                        conf=round(conf, 4),
                        x1=round(x1, 2),
                        y1=round(y1, 2),
                        x2=round(x2, 2),
                        y2=round(y2, 2),
                        img_w=self.img_w,
                        img_h=self.img_h,
                        track_id=item.get("track_id", -1),
                        model_version="mock-idd-v1"
                    )
                    records.append(record)

        log_event(
            f"Generated {len(records)} mock detections across {num_frames} frames",
            reason_code="MOCK-DETECT-GEN"
        )
        return records


class YOLOv8Detector:
    """YOLOv8 inference engine with ByteTrack integration."""

    def __init__(
        self,
        weights_path: Union[str, Path],
        conf_threshold: float = 0.25,
        device: str = "cpu"
    ) -> None:
        self.weights_path = Path(weights_path)
        self.conf_threshold = conf_threshold
        self.device = device
        self._model = None

    def load_model(self) -> None:
        """Dynamically load Ultralytics YOLOv8 model."""
        try:
            from ultralytics import YOLO
            self._model = YOLO(str(self.weights_path))
            log_event(f"Loaded YOLOv8 model from {self.weights_path}", reason_code="YOLO-LOAD-SUCCESS")
        except ImportError:
            raise ImportError(
                "Ultralytics is required for live YOLOv8 inference. Install via 'pip install ultralytics'."
            )

    def run_video_stream(
        self,
        video_path: Union[str, Path],
        sequence_id: str,
        output_parquet: Union[str, Path]
    ) -> Path:
        """Run tracking on video stream and persist Contract D2 Parquet."""
        if self._model is None:
            self.load_model()

        import cv2

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_idx = 0
        raw_detections: List[Dict[str, Any]] = []

        log_event(f"Starting YOLOv8 inference on {video_path} (FPS={fps})", reason_code="YOLO-INFER-START")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            ts = frame_idx / fps

            results = self._model.track(
                source=frame,
                conf=self.conf_threshold,
                device=self.device,
                persist=True,
                verbose=False
            )

            for r in results:
                boxes = r.boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        coords = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        cls_id = int(box.cls[0].cpu().numpy())
                        cls_name = r.names.get(cls_id, f"class_{cls_id}")
                        track_id = int(box.id[0].cpu().numpy()) if box.id is not None else -1

                        raw_detections.append({
                            "sequence_id": sequence_id,
                            "frame_idx": frame_idx,
                            "frame_ts": round(ts, 4),
                            "class_name": cls_name,
                            "conf": round(conf, 4),
                            "x1": float(coords[0]),
                            "y1": float(coords[1]),
                            "x2": float(coords[2]),
                            "y2": float(coords[3]),
                            "img_w": w,
                            "img_h": h,
                            "track_id": track_id,
                            "model_version": "yolov8-live",
                        })

            frame_idx += 1

        cap.release()
        validated = validate_and_filter_detections(raw_detections)
        out = save_detections_parquet(validated, output_parquet)
        log_event(f"Completed inference. Wrote {len(validated)} detections to {out}", reason_code="YOLO-INFER-DONE")
        return out
