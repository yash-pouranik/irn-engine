"""Module 3: Relative Linear Referencing Engine (s, t).

Transforms camera-relative ground obstacle coordinates (X_forward, Y_left)
into road centerline curvilinear Frenet coordinates (s_obs, t_obs):
- Ego vehicle position along road: s_ego, t_ego
- Relative offset:
    ds = X * cos(d_psi) - Y * sin(d_psi)
    dt = X * sin(d_psi) + Y * cos(d_psi)
    s_obs = s_ego + ds
    t_obs = t_ego + dt  (t > 0 to the LEFT of road direction)
- Covariance rotation into road frame: Sigma_st = R * Sigma_XY * R^T
- Canonical representation for bidirectional passes: s' = L - s, t' = -t
"""

from __future__ import annotations
import math
from typing import Dict, List, NamedTuple, Optional, Tuple, Union
import numpy as np


class FrenetObstaclePose(NamedTuple):
    edge_id: str
    s_obs: float              # Longitudinal distance along edge in meters [0, L]
    t_obs: float              # Lateral distance from edge centerline in meters (+ to left)
    sigma_s: float            # Standard deviation in longitudinal direction (meters)
    sigma_t: float            # Standard deviation in lateral direction (meters)
    cov_st: float             # Cross-covariance (m^2)
    heading_delta_deg: float  # Delta heading between vehicle and edge tangent


class LinearReferencer:
    """Computes metric curvilinear (s, t) Frenet coordinates for road anomalies."""

    @staticmethod
    def project_obstacle_to_frenet(
        s_ego: float,
        t_ego: float,
        veh_heading_deg: float,
        edge_bearing_deg: float,
        x_fwd: float,
        y_left: float,
        cov_xx: float = 1.0,
        cov_yy: float = 0.2,
        cov_xy: float = 0.0,
        edge_id: str = "edge_0",
        edge_length_m: Optional[float] = None
    ) -> FrenetObstaclePose:
        """Transforms obstacle from vehicle frame (X, Y) to road Frenet frame (s, t)."""
        # Heading delta between vehicle and road direction
        d_psi_deg = (veh_heading_deg - edge_bearing_deg + 180.0) % 360.0 - 180.0
        d_psi_rad = np.radians(d_psi_deg)

        c = np.cos(d_psi_rad)
        s = np.sin(d_psi_rad)

        # Longitudinal and lateral displacement along road
        ds = x_fwd * c - y_left * s
        dt = x_fwd * s + y_left * c

        s_obs = s_ego + ds
        t_obs = t_ego + dt

        # Clamp s_obs to edge length if provided
        if edge_length_m is not None:
            s_obs = max(0.0, min(float(edge_length_m), s_obs))

        # Rotate 2x2 covariance matrix: Sigma_st = R * Sigma_XY * R^T
        R = np.array([[c, -s], [s, c]])
        Sigma_in = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]])
        Sigma_st = R @ Sigma_in @ R.T

        var_s = max(1e-4, float(Sigma_st[0, 0]))
        var_t = max(1e-4, float(Sigma_st[1, 1]))
        cov_st = float(Sigma_st[0, 1])

        return FrenetObstaclePose(
            edge_id=edge_id,
            s_obs=round(s_obs, 3),
            t_obs=round(t_obs, 3),
            sigma_s=round(math.sqrt(var_s), 3),
            sigma_t=round(math.sqrt(var_t), 3),
            cov_st=round(cov_st, 4),
            heading_delta_deg=round(d_psi_deg, 2)
        )

    @staticmethod
    def canonicalize_bidirectional(
        s_obs: float,
        t_obs: float,
        edge_length_m: float,
        is_reverse_direction: bool = False
    ) -> Tuple[float, float]:
        """Maps opposite driving pass on the same road into canonical reference frame."""
        if not is_reverse_direction:
            return s_obs, t_obs
        # In reverse pass: distance from other end is L - s, lateral left becomes right (-t)
        s_canon = max(0.0, min(edge_length_m, edge_length_m - s_obs))
        t_canon = -t_obs
        return s_canon, t_canon
