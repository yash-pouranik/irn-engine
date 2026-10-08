"""Module 3: Temporal GPS-Video Synchronizer and Trajectory Interpolator.

Handles:
- Matching video frame timestamps to GPS log timelines.
- Vectorized linear interpolation of vehicle coordinates (lat, lon, UTM).
- Interpolation of vehicle heading with circular angle unwrapping.
- Gating against GPS dropouts (> 1.0s) and speed spikes (> 45 m/s).
"""

from __future__ import annotations
import math
from typing import Dict, NamedTuple, Optional, Tuple, Union
import numpy as np
import pandas as pd
from irn.common.crs import wgs84_to_utm
from irn.common.logger import log_event


class VehiclePoseState(NamedTuple):
    """Interpolated vehicle kinematic state for a batch of frames."""
    frame_ts: np.ndarray       # Input frame timestamps (seconds)
    lat: np.ndarray            # Latitude (degrees)
    lon: np.ndarray            # Longitude (degrees)
    x_utm: np.ndarray          # UTM Easting (meters)
    y_utm: np.ndarray          # UTM Northing (meters)
    heading_deg: np.ndarray    # Heading angle clockwise from North (degrees [0, 360))
    speed_mps: np.ndarray      # Speed in m/s
    valid_mask: np.ndarray     # Boolean flag indicating valid pose within time/speed gates


class GPSSynchronizer:
    """Synchronizes video frame streams with vehicle GPS logs."""

    def __init__(
        self,
        gps_df: pd.DataFrame,
        utm_zone: int,
        is_northern: bool = True,
        max_dt_sec: float = 0.2,
        max_dropout_sec: float = 1.0,
        max_speed_mps: float = 45.0
    ) -> None:
        self.utm_zone = utm_zone
        self.is_northern = is_northern
        self.max_dt_sec = max_dt_sec
        self.max_dropout_sec = max_dropout_sec
        self.max_speed_mps = max_speed_mps

        # Prepare normalized arrays sorted by timestamp
        df = gps_df.sort_values("timestamp").reset_index(drop=True)
        self.t_gps = df["timestamp"].to_numpy(dtype=float)
        self.lats = df["lat"].to_numpy(dtype=float)
        self.lons = df["lon"].to_numpy(dtype=float)

        # Convert to UTM
        easting, northing = wgs84_to_utm(self.lons, self.lats, self.utm_zone, self.is_northern)
        self.x_utm = np.asarray(easting, dtype=float)
        self.y_utm = np.asarray(northing, dtype=float)

        # Speeds
        if "speed" in df.columns:
            self.speeds = df["speed"].to_numpy(dtype=float)
        elif "speed_mps" in df.columns:
            self.speeds = df["speed_mps"].to_numpy(dtype=float)
        else:
            # Estimate speed by finite difference
            dt = np.diff(self.t_gps, prepend=self.t_gps[0] + 0.1)
            dt = np.where(dt > 1e-4, dt, 0.1)
            dist = np.sqrt(np.diff(self.x_utm, prepend=self.x_utm[0])**2 + np.diff(self.y_utm, prepend=self.y_utm[0])**2)
            self.speeds = dist / dt

        # Headings
        if "heading" in df.columns:
            self.headings = df["heading"].to_numpy(dtype=float)
        elif "heading_deg" in df.columns:
            self.headings = df["heading_deg"].to_numpy(dtype=float)
        else:
            # Compute trajectory bearing from UTM coordinates
            dx = np.diff(self.x_utm, append=self.x_utm[-1])
            dy = np.diff(self.y_utm, append=self.y_utm[-1])
            self.headings = np.degrees(np.arctan2(dx, dy)) % 360.0

        # Unwrap headings for continuous angular interpolation
        self.headings_unwrapped = np.unwrap(np.radians(self.headings))

        log_event(
            f"Initialized GPSSynchronizer with {len(self.t_gps)} fixes (t in [{self.t_gps[0]:.2f}, {self.t_gps[-1]:.2f}])",
            reason_code="GPS-SYNC-INIT"
        )

    def interpolate_poses(
        self,
        frame_timestamps: Union[np.ndarray, list],
        time_offset: float = 0.0
    ) -> VehiclePoseState:
        """Vectorized interpolation of vehicle pose at arbitrary frame timestamps.

        Calculates exact UTM position, coordinates, speed, and heading.
        """
        t_query = np.asarray(frame_timestamps, dtype=float) + time_offset

        # Check boundary bounds and dropouts
        t_min = self.t_gps[0]
        t_max = self.t_gps[-1]

        # Find nearest GPS fix time difference
        idx_nearest = np.searchsorted(self.t_gps, t_query, side="left")
        idx_nearest = np.clip(idx_nearest, 0, len(self.t_gps) - 1)
        nearest_dt = np.abs(self.t_gps[idx_nearest] - t_query)

        # Check if previous neighbor is closer
        idx_prev = np.maximum(idx_nearest - 1, 0)
        prev_dt = np.abs(self.t_gps[idx_prev] - t_query)
        min_dt = np.minimum(nearest_dt, prev_dt)

        # Interpolate Easting and Northing linearly
        interp_x = np.interp(t_query, self.t_gps, self.x_utm)
        interp_y = np.interp(t_query, self.t_gps, self.y_utm)
        interp_lat = np.interp(t_query, self.t_gps, self.lats)
        interp_lon = np.interp(t_query, self.t_gps, self.lons)
        interp_speed = np.interp(t_query, self.t_gps, self.speeds)

        # Angular interpolation via unwrapped radians
        interp_head_rad = np.interp(t_query, self.t_gps, self.headings_unwrapped)
        interp_head_deg = np.degrees(interp_head_rad) % 360.0

        # Validity gating
        valid_mask = (
            (t_query >= t_min)
            & (t_query <= t_max)
            & (min_dt <= self.max_dropout_sec)
            & (interp_speed <= self.max_speed_mps)
        )

        return VehiclePoseState(
            frame_ts=t_query - time_offset,
            lat=interp_lat,
            lon=interp_lon,
            x_utm=interp_x,
            y_utm=interp_y,
            heading_deg=interp_head_deg,
            speed_mps=interp_speed,
            valid_mask=valid_mask
        )
