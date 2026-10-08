"""Module 3 Master Geospatial Fusion Pipeline.

Coordinates the complete end-to-end M3 processing flow:
1. Projects 2D pixels (u, v) -> vehicle ground plane (X, Y) via GroundPlaneProjector.
2. Synchronizes video frame timestamps with vehicle GPS trace via GPSSynchronizer.
3. Decodes vehicle map-matched edges via HMMMapMatcher.
4. Computes road linear referencing (s, t) via LinearReferencer.
5. De-duplicates multi-pass observations via AnomalyDeduplicator.
6. Fuses road usable widths via width_fusion.
7. Persists SQLite ledger (D3) and exports Enriched Network Parquets (D4).
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq

from irn.common.logger import log_event
from irn.m3_fusion.camera_model import CameraCalibration, GroundPlaneProjector
from irn.m3_fusion.deduplication import AnomalyDeduplicator, CanonicalAnomaly
from irn.m3_fusion.linear_ref import LinearReferencer
from irn.m3_fusion.map_matcher import HMMMapMatcher, RoadNetworkSpatialIndex
from irn.m3_fusion.storage import FusionStorage
from irn.m3_fusion.sync import GPSSynchronizer
from irn.m3_fusion.width_fusion import fuse_edge_widths


def run_geospatial_fusion(
    edges_gpkg_path: Union[str, Path],
    detections_parquet_path: Union[str, Path],
    gps_csv_path: Union[str, Path],
    rig_config_path: Union[str, Path],
    utm_zone: int,
    is_northern: bool = True,
    output_d3_db: Optional[Union[str, Path]] = None,
    output_d4_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Execute complete M3 Geospatial Data Fusion pipeline."""
    log_event("Starting M3 Geospatial Fusion Pipeline", reason_code="M3-PIPELINE-START")

    # 1. Load inputs
    edges_gdf = gpd.read_file(edges_gpkg_path, layer="edges")
    detections_table = pq.read_table(detections_parquet_path)
    detections_df = detections_table.to_pandas()
    gps_df = pd.read_csv(gps_csv_path)

    calib = CameraCalibration.from_yaml(rig_config_path)
    projector = GroundPlaneProjector(calib)

    # 2. Camera Ground-Plane Ray Projection (Vectorized)
    u_arr = detections_df["u"].to_numpy(dtype=float)
    v_arr = detections_df["v"].to_numpy(dtype=float)
    proj_res = projector.project_ground_points(u_arr, v_arr)

    detections_df["x_fwd"] = proj_res.x_forward
    detections_df["y_left"] = proj_res.y_left
    detections_df["cov_xx"] = proj_res.cov_xx
    detections_df["cov_yy"] = proj_res.cov_yy
    detections_df["cov_xy"] = proj_res.cov_xy
    detections_df["proj_valid"] = proj_res.valid_mask

    # Filter to valid projections
    valid_dets = detections_df[detections_df["proj_valid"]].copy().reset_index(drop=True)
    log_event(
        f"Projected {len(detections_df)} detections -> {len(valid_dets)} valid on ground plane within range gate",
        reason_code="PROJECTION-FILTER"
    )

    # 3. Temporal GPS Synchronization & Pose Interpolation
    gps_sync = GPSSynchronizer(gps_df, utm_zone=utm_zone, is_northern=is_northern)

    frame_timestamps = valid_dets["frame_ts"].to_numpy(dtype=float)
    poses = gps_sync.interpolate_poses(frame_timestamps)

    valid_dets["x_utm"] = poses.x_utm
    valid_dets["y_utm"] = poses.y_utm
    valid_dets["veh_heading"] = poses.heading_deg
    valid_dets["veh_speed"] = poses.speed_mps
    valid_dets["gps_valid"] = poses.valid_mask

    valid_dets = valid_dets[valid_dets["gps_valid"]].copy().reset_index(drop=True)

    # 4. HMM Viterbi Map-Matching
    spatial_index = RoadNetworkSpatialIndex(edges_gdf)
    matcher = HMMMapMatcher(spatial_index)

    matched_poses = matcher.match_trajectory(
        x_utm_arr=valid_dets["x_utm"].to_numpy(dtype=float),
        y_utm_arr=valid_dets["y_utm"].to_numpy(dtype=float),
        timestamps_arr=valid_dets["frame_ts"].to_numpy(dtype=float),
        headings_deg_arr=valid_dets["veh_heading"].to_numpy(dtype=float)
    )

    # Create map from frame index to matched edge
    matched_edge_map = {m.frame_idx: m for m in matched_poses}

    # 5. Relative Linear Referencing (s, t)
    raw_observations: List[Dict[str, Any]] = []
    edge_len_map = {str(row.get("edge_id", "")): float(row.geometry.length) for _, row in edges_gdf.iterrows()}

    for i, row in valid_dets.iterrows():
        match_info = matched_edge_map.get(i)
        if not match_info:
            continue

        edge_id = match_info.matched_edge_id
        edge_len = edge_len_map.get(edge_id, 100.0)

        frenet = LinearReferencer.project_obstacle_to_frenet(
            s_ego=match_info.s_ego,
            t_ego=match_info.t_ego,
            veh_heading_deg=match_info.vehicle_heading_deg,
            edge_bearing_deg=match_info.edge_bearing_deg,
            x_fwd=float(row["x_fwd"]),
            y_left=float(row["y_left"]),
            cov_xx=float(row["cov_xx"]),
            cov_yy=float(row["cov_yy"]),
            cov_xy=float(row["cov_xy"]),
            edge_id=edge_id,
            edge_length_m=edge_len
        )

        raw_observations.append({
            "sequence_id": row["sequence_id"],
            "frame_idx": row["frame_idx"],
            "frame_ts": row["frame_ts"],
            "edge_id": edge_id,
            "class_name": row["class_name"],
            "conf": float(row["conf"]),
            "track_id": int(row.get("track_id", -1)),
            "s_obs": frenet.s_obs,
            "t_obs": frenet.t_obs,
            "sigma_s": frenet.sigma_s,
            "sigma_t": frenet.sigma_t,
        })

    # 6. Multi-Pass De-duplication (Level 1 & Level 2)
    deduplicator = AnomalyDeduplicator()
    pass_anomalies = deduplicator.deduplicate_pass(raw_observations)
    canonical_anomalies = deduplicator.merge_multi_pass([], pass_anomalies)

    # 7. Road Width Fusion
    enriched_edges = fuse_edge_widths(edges_gdf)

    # 8. Storage & D3/D4 Export
    storage_db = output_d3_db or "data/d3_store/anomalies.db"
    storage = FusionStorage(storage_db)
    storage.persist_observations(raw_observations)
    storage.persist_canonical_anomalies(canonical_anomalies)

    d4_dir = output_d4_dir or "data/d4_enriched"
    d4_edges, d4_anom = storage.export_d4_parquet(enriched_edges, canonical_anomalies, d4_dir)

    log_event("M3 Geospatial Fusion Pipeline Completed Successfully", reason_code="M3-PIPELINE-DONE")

    return {
        "d3_db": Path(storage_db),
        "d4_edges_parquet": d4_edges,
        "d4_anomalies_parquet": d4_anom,
        "num_raw_observations": len(raw_observations),
        "num_canonical_anomalies": len(canonical_anomalies),
        "num_confirmed": sum(1 for a in canonical_anomalies if a.state == "CONFIRMED")
    }
