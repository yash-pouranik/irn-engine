"""Synthetic World Ground Truth Generator (NFR-QA-01).

Generates a mathematically known ground-truth benchmark:
- Road segment of length 100m, width 7.5m (LHT).
- 3 planted anomalies at exact known (s, t) coordinates:
    (25.0m, -1.2m), (50.0m, 0.8m), (75.0m, -0.5m).
- Vehicle GPS trace with realistic Gaussian noise.
- Exact ground-contact pixel ray-tracing back through camera intrinsics K.
- Synthesizes Contracts D1 (GeoPackages), D2 (Parquet), and GPS log.
"""

from __future__ import annotations
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple
import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point

from irn.common.crs import optimal_utm_epsg, utm_to_wgs84, wgs84_to_utm
from irn.common.io import atomic_write_json
from irn.m2_cv_interface.schema import DetectionRecord, save_detections_parquet


class SyntheticWorld:
    """Generates synthetic, deterministic ground truth datasets."""

    def __init__(
        self,
        road_length_m: float = 100.0,
        road_width_m: float = 7.5,
        start_utm: Tuple[float, float] = (500000.0, 2000000.0),  # UTM Zone 43N
        zone_number: int = 43,
        is_northern: bool = True,
        seed: int = 42
    ) -> None:
        self.road_length_m = road_length_m
        self.road_width_m = road_width_m
        self.start_utm = start_utm
        self.zone_number = zone_number
        self.is_northern = is_northern
        self.utm_epsg = 32600 + zone_number if is_northern else 32700 + zone_number
        self.rng = np.random.default_rng(seed)

        # Planted anomalies at known (s, t)
        self.ground_truth_anomalies = [
            {"id": "anomaly_1", "s": 25.0, "t": -1.2, "class_name": "pothole", "width": 0.8, "length": 0.6},
            {"id": "anomaly_2", "s": 50.0, "t": 0.8, "class_name": "speed_breaker", "width": 6.0, "length": 0.5},
            {"id": "anomaly_3", "s": 75.0, "t": -0.5, "class_name": "pothole", "width": 0.7, "length": 0.7},
        ]

    def build_synthetic_graph(self, out_dir: Path) -> Dict[str, Path]:
        """Constructs and exports D1 edges.gpkg and nodes.gpkg."""
        out_dir.mkdir(parents=True, exist_ok=True)

        x0, y0 = self.start_utm
        # Road runs North (+Y direction)
        x1, y1 = x0, y0 + self.road_length_m

        # WGS84 coordinates
        lon0, lat0 = utm_to_wgs84(x0, y0, self.zone_number, self.is_northern)
        lon1, lat1 = utm_to_wgs84(x1, y1, self.zone_number, self.is_northern)

        # 1. Nodes GeoDataFrame
        nodes_data = [
            {
                "node_id": 1001,
                "x_utm": x0,
                "y_utm": y0,
                "lon": float(lon0),
                "lat": float(lat0),
                "highway": "priority",
                "geometry": Point(x0, y0),
            },
            {
                "node_id": 1002,
                "x_utm": x1,
                "y_utm": y1,
                "lon": float(lon1),
                "lat": float(lat1),
                "highway": "priority",
                "geometry": Point(x1, y1),
            },
        ]
        gdf_nodes = gpd.GeoDataFrame(nodes_data, crs=f"EPSG:{self.utm_epsg}")
        nodes_path = out_dir / "nodes.gpkg"
        gdf_nodes.to_file(nodes_path, layer="nodes", driver="GPKG")

        # 2. Edges GeoDataFrame
        edge_line = LineString([(x0, y0), (x1, y1)])
        edges_data = [
            {
                "edge_id": "1001_1002_0",
                "u": 1001,
                "v": 1002,
                "key": 0,
                "highway": "primary",
                "oneway": True,
                "lanes": 2,
                "width_osm_m": self.road_width_m,
                "maxspeed_kph": 50,
                "surface": "asphalt",
                "length_m": self.road_length_m,
                "geometry": edge_line,
            }
        ]
        gdf_edges = gpd.GeoDataFrame(edges_data, crs=f"EPSG:{self.utm_epsg}")
        edges_path = out_dir / "edges.gpkg"
        gdf_edges.to_file(edges_path, layer="edges", driver="GPKG")

        # 3. Snapshot metadata
        snapshot_path = out_dir / "osm_snapshot.json"
        snapshot_data = {
            "query": {"synthetic": True, "road_length_m": self.road_length_m},
            "utm_epsg": self.utm_epsg,
            "utm_zone": self.zone_number,
            "is_northern": self.is_northern,
            "centroid_lon": float(lon0),
            "centroid_lat": float(lat0),
            "num_nodes": len(gdf_nodes),
            "num_edges": len(gdf_edges),
            "license": "Synthetic Test Fixture",
            "attribution": "IRN Synthetic World Generator",
        }
        atomic_write_json(snapshot_path, snapshot_data)

        return {
            "edges_gpkg": edges_path,
            "nodes_gpkg": nodes_path,
            "osm_snapshot_json": snapshot_path,
        }

    def generate_trajectory_and_detections(
        self,
        out_dir: Path,
        speed_mps: float = 10.0,
        fps: float = 10.0,
        camera_height: float = 1.5,
        fx: float = 1380.0,
        fy: float = 1380.0,
        cx: float = 960.0,
        cy: float = 540.0,
        pitch_rad: float = -0.0436,  # -2.5 degrees tilt
        noise_std_m: float = 1.5
    ) -> Tuple[Path, Path]:
        """Simulates GPS trace and camera detections matching the planted anomalies."""
        out_dir.mkdir(parents=True, exist_ok=True)

        total_time = self.road_length_m / speed_mps
        num_frames = int(total_time * fps)
        dt = 1.0 / fps

        gps_rows = []
        detections: List[DetectionRecord] = []

        x0, y0 = self.start_utm
        R_cam = np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(pitch_rad), -np.sin(pitch_rad)],
            [0.0, np.sin(pitch_rad), np.cos(pitch_rad)],
        ])

        for frame_idx in range(num_frames):
            t = frame_idx * dt
            # Ego vehicle position along centerline (t_ego = 0)
            s_ego = speed_mps * t
            ego_x_utm = x0
            ego_y_utm = y0 + s_ego

            # Noisy GPS
            noise_x = self.rng.normal(0, noise_std_m)
            noise_y = self.rng.normal(0, noise_std_m)
            gps_lon, gps_lat = utm_to_wgs84(
                ego_x_utm + noise_x,
                ego_y_utm + noise_y,
                self.zone_number,
                self.is_northern
            )

            gps_rows.append({
                "timestamp": round(t, 3),
                "lat": float(gps_lat),
                "lon": float(gps_lon),
                "altitude_m": 500.0,
                "speed_mps": speed_mps,
                "heading_deg": 0.0,
            })

            # Check for anomalies visible to camera
            for obs in self.ground_truth_anomalies:
                s_obs = obs["s"]
                t_obs = obs["t"]

                # Relative position in vehicle frame (X = forward along road, Y = left lateral)
                X_fwd = s_obs - s_ego
                Y_left = t_obs  # road heading = 0 => X_fwd = delta_y, Y_left = -delta_x = t_obs

                # Only detect if within forward range [4m, 30m]
                if 4.0 <= X_fwd <= 30.0:
                    # Ground point in vehicle frame: P_veh = [X_fwd, Y_left, -camera_height]
                    P_veh = np.array([X_fwd, Y_left, -camera_height])

                    # Camera frame point: P_cam = R_cam^T * P_veh
                    # In standard camera coordinates: X_c = right = -Y_left, Y_c = down = -(Z_veh), Z_c = forward = X_fwd
                    # Let's align camera coordinate convention:
                    # Vehicle: X_fwd, Y_left, Z_up
                    # Cam (nominal): X_right = -Y_left, Y_down = -Z_up = +h, Z_fwd = X_fwd
                    P_cam_nominal = np.array([-Y_left, camera_height, X_fwd])
                    # Apply pitch rotation (around X_cam axis)
                    P_cam = R_cam @ P_cam_nominal

                    if P_cam[2] > 0.1:  # in front of camera
                        u_proj = fx * (P_cam[0] / P_cam[2]) + cx
                        v_proj = fy * (P_cam[1] / P_cam[2]) + cy

                        if 0 <= u_proj <= 1920 and 0 <= v_proj <= 1080:
                            box_w = max(20.0, fx * (obs["width"] / P_cam[2]))
                            box_h = max(15.0, fy * (obs["length"] / P_cam[2]))

                            det = DetectionRecord(
                                sequence_id="synthetic_seq_01",
                                frame_idx=frame_idx,
                                frame_ts=round(t, 3),
                                class_name=obs["class_name"],
                                conf=0.92,
                                x1=round(max(0.0, u_proj - box_w / 2.0), 2),
                                y1=round(max(0.0, v_proj - box_h), 2),
                                x2=round(min(1920.0, u_proj + box_w / 2.0), 2),
                                y2=round(min(1080.0, v_proj), 2),
                                img_w=1920,
                                img_h=1080,
                                track_id=int(obs["id"].split("_")[-1]),
                                model_version="synthetic-gt-v1",
                            )
                            detections.append(det)

        # Write GPS CSV
        gps_df = pd.DataFrame(gps_rows)
        gps_path = out_dir / "gps.csv"
        gps_df.to_csv(gps_path, index=False)

        # Write Parquet D2
        parquet_path = out_dir / "detections.parquet"
        save_detections_parquet(detections, parquet_path)

        return gps_path, parquet_path
