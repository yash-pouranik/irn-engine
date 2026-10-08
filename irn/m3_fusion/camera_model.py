"""Module 3: Camera Ground-Plane Ray Projection and Uncertainty Propagation.

Implements mathematically exact ground-plane ray-casting:
- Ray d_c = K^-1 * [u, v, 1]^T
- Rotation into vehicle frame d_v = R(pitch, roll, yaw) * d_c_nominal
- Intersect with ground plane z = -h: lambda = -h / d_v[z]
- Ground point: P_veh = lambda * d_v = (X_forward, Y_left)
- Reject rays above horizon or outside range gate [min_range, max_range]
- Vectorized covariance propagation Sigma_XY via analytical Jacobian
"""

from __future__ import annotations
import math
from typing import Dict, NamedTuple, Optional, Tuple, Union
import numpy as np
import yaml
from pathlib import Path


class CameraCalibration:
    """Holds calibrated camera intrinsics, mounting extrinsics, and gating thresholds."""

    def __init__(
        self,
        fx: float = 1380.0,
        fy: float = 1380.0,
        cx: float = 960.0,
        cy: float = 540.0,
        height_m: float = 1.5,
        pitch_deg: float = -2.5,
        roll_deg: float = 0.0,
        yaw_deg: float = 0.0,
        min_range_m: float = 4.0,
        max_range_m: float = 30.0,
        min_confidence: float = 0.35,
        pixel_sigma_v: float = 2.0,
        pitch_sigma_deg: float = 0.5,
        gps_to_cam_translation_m: Tuple[float, float, float] = (1.2, 0.0, 0.3)
    ) -> None:
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.height_m = float(height_m)
        self.pitch_deg = float(pitch_deg)
        self.roll_deg = float(roll_deg)
        self.yaw_deg = float(yaw_deg)
        self.min_range_m = float(min_range_m)
        self.max_range_m = float(max_range_m)
        self.min_confidence = float(min_confidence)
        self.pixel_sigma_v = float(pixel_sigma_v)
        self.pitch_sigma_deg = float(pitch_sigma_deg)
        self.gps_to_cam_translation_m = np.array(gps_to_cam_translation_m, dtype=float)

        # Precompute rotation matrices
        # Camera to nominal vehicle frame:
        # Camera frame: X_c = right, Y_c = down, Z_c = forward
        # Vehicle frame: X_v = forward, Y_v = left, Z_v = up
        # Thus: X_v = Z_c, Y_v = -X_c, Z_v = -Y_c
        self.R_cam_to_veh_nominal = np.array([
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0]
        ], dtype=float)

        pitch_rad = np.radians(self.pitch_deg)
        roll_rad = np.radians(self.roll_deg)
        yaw_rad = np.radians(self.yaw_deg)

        # Rotation around vehicle Y-axis (pitch), X-axis (roll), Z-axis (yaw)
        # Pitch down: tilt nose down -> rotation around Y_v
        cy = np.cos(yaw_rad)
        sy = np.sin(yaw_rad)
        cp = np.cos(pitch_rad)
        sp = np.sin(pitch_rad)
        cr = np.cos(roll_rad)
        sr = np.sin(roll_rad)

        R_yaw = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
        R_pitch = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
        R_roll = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])

        self.R_veh = R_yaw @ R_pitch @ R_roll
        # Combined transformation matrix from camera ray coordinates to vehicle coordinates
        self.R_cam_to_veh = self.R_veh @ self.R_cam_to_veh_nominal

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "CameraCalibration":
        """Load calibration from rig.yaml configuration."""
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        cam = data.get("camera", {})
        mount = data.get("mounting", {})
        gates = data.get("processing_gates", {})

        return cls(
            fx=cam.get("fx", 1380.0),
            fy=cam.get("fy", 1380.0),
            cx=cam.get("cx", 960.0),
            cy=cam.get("cy", 540.0),
            height_m=mount.get("height_m", 1.5),
            pitch_deg=mount.get("pitch_deg", -2.5),
            roll_deg=mount.get("roll_deg", 0.0),
            yaw_deg=mount.get("yaw_deg", 0.0),
            min_range_m=gates.get("min_range_m", 4.0),
            max_range_m=gates.get("max_range_m", 30.0),
            min_confidence=gates.get("min_confidence", 0.35),
            pixel_sigma_v=gates.get("pixel_sigma_v", 2.0),
            pitch_sigma_deg=gates.get("pitch_sigma_deg", 0.5),
            gps_to_cam_translation_m=tuple(mount.get("gps_to_cam_translation_m", [1.2, 0.0, 0.3]))
        )


class ProjectionResult(NamedTuple):
    """Vectorized projection results."""
    x_forward: np.ndarray       # Forward distance X_v (meters)
    y_left: np.ndarray          # Lateral offset Y_v (meters, positive to LEFT)
    cov_xx: np.ndarray          # Variance in forward direction (m^2)
    cov_yy: np.ndarray          # Variance in lateral direction (m^2)
    cov_xy: np.ndarray          # Cross-covariance (m^2)
    valid_mask: np.ndarray      # Boolean mask of valid rays within range gate


class GroundPlaneProjector:
    """Vectorized ground-plane ray-casting engine for vehicle front cameras."""

    def __init__(self, calib: CameraCalibration) -> None:
        self.calib = calib

    def project_ground_points(
        self,
        u_pixels: Union[float, np.ndarray, list],
        v_pixels: Union[float, np.ndarray, list]
    ) -> ProjectionResult:
        """Projects bottom-center ground contact pixels (u, v) onto road ground plane z = -h.

        Fully vectorized across arrays of detections (Rule G-3).
        """
        u = np.atleast_1d(np.asarray(u_pixels, dtype=float))
        v = np.atleast_1d(np.asarray(v_pixels, dtype=float))

        if u.shape != v.shape:
            raise ValueError(f"u and v must have identical shapes, got {u.shape} vs {v.shape}")

        n = len(u)

        # 1. Normalized camera ray d_c = K^-1 * [u, v, 1]^T
        d_cx = (u - self.calib.cx) / self.calib.fx
        d_cy = (v - self.calib.cy) / self.calib.fy
        d_cz = np.ones(n, dtype=float)

        # Camera rays array shape (3, n)
        rays_cam = np.vstack([d_cx, d_cy, d_cz])

        # 2. Transform rays to vehicle coordinate frame: d_v = R_cam_to_veh @ rays_cam
        rays_veh = self.calib.R_cam_to_veh @ rays_cam  # shape (3, n)

        dv_x = rays_veh[0, :]  # forward component
        dv_y = rays_veh[1, :]  # left component
        dv_z = rays_veh[2, :]  # vertical component (downward is negative)

        # 3. Intersect with ground plane z = -h:
        # A valid ray must point downwards (dv_z < -1e-4) to hit the road surface
        points_downward = dv_z < -1e-4

        # Avoid division by zero for rays at or above horizon
        safe_dv_z = np.where(points_downward, dv_z, -1.0)
        lam = -self.calib.height_m / safe_dv_z  # lambda > 0

        # Ground intersection in vehicle frame
        X_fwd = lam * dv_x
        Y_left = lam * dv_y

        # Distance from camera along road ground plane
        range_m = np.sqrt(X_fwd**2 + Y_left**2)

        # 4. Operational range gating [min_range, max_range]
        valid_mask = (
            points_downward
            & (range_m >= self.calib.min_range_m)
            & (range_m <= self.calib.max_range_m)
            & (X_fwd > 0.0)  # obstacle must be in front of the vehicle
        )

        # 5. Covariance propagation Sigma_XY via analytical Jacobian
        # Forward distance X is approximately h / (pitch_angle + (v - cy)/fy)
        # dX/dv ~ -h * fy / (v_eff)^2 ~ -X^2 / (h * fy)
        # dY/du ~ X / fx
        pitch_rad_eff = np.maximum(np.abs(np.radians(self.calib.pitch_deg)), 0.01)
        sigma_v_pix = self.calib.pixel_sigma_v
        sigma_pitch_rad = np.radians(self.calib.pitch_sigma_deg)

        # Analytical Jacobian components
        # Sensitivity of X forward to vertical pixel coordinate v
        dX_dv = (X_fwd**2) / (self.calib.height_m * self.calib.fy)
        dX_dpitch = (X_fwd**2) / self.calib.height_m
        dY_du = X_fwd / self.calib.fx

        # Variance propagation
        cov_xx = (dX_dv * sigma_v_pix)**2 + (dX_dpitch * sigma_pitch_rad)**2
        cov_yy = (dY_du * sigma_v_pix)**2
        cov_xy = np.zeros_like(cov_xx)  # zero cross-correlation under independent noise

        # Set invalid points to NaN or safe values
        X_fwd = np.where(valid_mask, X_fwd, np.nan)
        Y_left = np.where(valid_mask, Y_left, np.nan)
        cov_xx = np.where(valid_mask, cov_xx, np.nan)
        cov_yy = np.where(valid_mask, cov_yy, np.nan)

        return ProjectionResult(
            x_forward=X_fwd,
            y_left=Y_left,
            cov_xx=cov_xx,
            cov_yy=cov_yy,
            cov_xy=cov_xy,
            valid_mask=valid_mask
        )

    def ground_to_pixel(
        self,
        x_forward: Union[float, np.ndarray],
        y_left: Union[float, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Inverse projection: maps ground coordinate (X, Y) back to camera image pixel (u, v).

        Used for synthetic ground-truth verification and projection round-trip tests (FR-M3-10).
        """
        X = np.atleast_1d(np.asarray(x_forward, dtype=float))
        Y = np.atleast_1d(np.asarray(y_left, dtype=float))

        # Ground point in vehicle coordinates: P_veh = [X, Y, -h]
        P_veh = np.vstack([X, Y, -np.full_like(X, self.calib.height_m)])  # shape (3, n)

        # Inverse rotation from vehicle to camera frame: P_cam = R_cam_to_veh^T @ P_veh
        P_cam = self.calib.R_cam_to_veh.T @ P_veh  # shape (3, n)

        P_cx = P_cam[0, :]
        P_cy = P_cam[1, :]
        P_cz = P_cam[2, :]

        # Normalized coordinates and intrinsic matrix projection
        u_proj = self.calib.fx * (P_cx / P_cz) + self.calib.cx
        v_proj = self.calib.fy * (P_cy / P_cz) + self.calib.cy

        return u_proj, v_proj
