"""Verification Tests for Phase 2: Ground-Plane Ray Projection & GPS Sync (FR-M3-10, Gate G-1)."""

import math
import numpy as np
import pandas as pd
import pytest

from irn.m3_fusion.camera_model import CameraCalibration, GroundPlaneProjector
from irn.m3_fusion.sync import GPSSynchronizer


class TestCameraProjection:
    """Mathematical verification of ground-plane ray intersection and uncertainty propagation."""

    @pytest.fixture
    def projector(self):
        calib = CameraCalibration(
            fx=1380.0,
            fy=1380.0,
            cx=960.0,
            cy=540.0,
            height_m=1.5,
            pitch_deg=-2.5,
            roll_deg=0.0,
            yaw_deg=0.0,
            min_range_m=4.0,
            max_range_m=30.0,
            min_confidence=0.35,
            pixel_sigma_v=2.0
        )
        return GroundPlaneProjector(calib)

    def test_round_trip_accuracy_under_1cm(self, projector):
        """Verification Gate G-1 & FR-M3-10: Round-trip error must be < 1 cm (0.01 m)."""
        # Test grid of known ground points in vehicle frame (forward distances from 5m to 25m)
        forward_distances = np.array([5.0, 8.0, 12.0, 15.0, 20.0, 25.0])
        lateral_offsets = np.array([-2.5, -1.0, 0.0, 1.2, 2.0, -1.8])

        # 1. Project ground points to pixel coordinates (u, v)
        u_proj, v_proj = projector.ground_to_pixel(forward_distances, lateral_offsets)

        # 2. Invert pixel rays back to ground points via ray-intersection
        res = projector.project_ground_points(u_proj, v_proj)

        assert np.all(res.valid_mask), "All test points within [5m, 25m] should be valid"

        # 3. Calculate Euclidean recovery error in meters
        errors = np.sqrt((forward_distances - res.x_forward)**2 + (lateral_offsets - res.y_left)**2)

        # Assert < 1 cm (0.01m) accuracy across all points
        max_error = float(np.max(errors))
        assert max_error < 0.01, f"Maximum round-trip projection error {max_error * 100:.3f} cm exceeds 1 cm threshold"

    def test_range_gating_and_horizon_rejection(self, projector):
        """Verify rays outside [4m, 30m] or above horizon are rejected."""
        # 1. Very distant obstacle (> 40m)
        u_far, v_far = projector.ground_to_pixel(45.0, 0.0)
        res_far = projector.project_ground_points(u_far, v_far)
        assert not res_far.valid_mask[0], "Obstacle at 45m should exceed range gate max of 30m"

        # 2. Too close (< 3m)
        u_near, v_near = projector.ground_to_pixel(2.5, 0.0)
        res_near = projector.project_ground_points(u_near, v_near)
        assert not res_near.valid_mask[0], "Obstacle at 2.5m should be within range gate min of 4m"

        # 3. Pixel above horizon (e.g. sky at v = 100)
        res_sky = projector.project_ground_points(960.0, 100.0)
        assert not res_sky.valid_mask[0], "Sky rays must be rejected"

    def test_covariance_properties(self, projector):
        """Verify forward uncertainty Sigma_XX grows quadratically with distance."""
        dists = np.array([6.0, 12.0, 24.0])
        u_pts, v_pts = projector.ground_to_pixel(dists, np.zeros_like(dists))
        res = projector.project_ground_points(u_pts, v_pts)

        # Variances must be strictly positive
        assert np.all(res.cov_xx > 0.0)
        assert np.all(res.cov_yy > 0.0)

        # As distance doubles, forward uncertainty grows substantially due to perspective foreshortening
        std_6m = np.sqrt(res.cov_xx[0])
        std_12m = np.sqrt(res.cov_xx[1])
        std_24m = np.sqrt(res.cov_xx[2])

        assert std_12m > std_6m * 2.0
        assert std_24m > std_12m * 2.0


class TestGPSSynchronizer:
    """Verify temporal GPS interpolation, heading unwrapping, and gating."""

    @pytest.fixture
    def gps_sync(self):
        # 10 seconds of simulated straight trajectory traveling North at 10 m/s
        timestamps = np.arange(0.0, 10.1, 0.5)  # fixes every 0.5s
        lats = 17.440 + timestamps * (10.0 / 111139.0)  # ~10 m/s in latitude
        lons = np.full_like(timestamps, 78.350)
        speeds = np.full_like(timestamps, 10.0)
        headings = np.full_like(timestamps, 0.0)

        df = pd.DataFrame({
            "timestamp": timestamps,
            "lat": lats,
            "lon": lons,
            "speed": speeds,
            "heading": headings
        })
        return GPSSynchronizer(df, utm_zone=44, is_northern=True)

    def test_pose_interpolation(self, gps_sync):
        # Query at frame timestamps between fixes (e.g. 1.25s, 3.75s)
        query_ts = [1.25, 3.75, 7.10]
        poses = gps_sync.interpolate_poses(query_ts)

        assert np.all(poses.valid_mask)
        assert pytest.approx(poses.speed_mps[0], abs=0.1) == 10.0
        assert pytest.approx(poses.heading_deg[0], abs=0.1) == 0.0

        # Position at t=3.75 should be midway between t=3.5 and t=4.0
        y_dist = poses.y_utm[1] - poses.y_utm[0]
        expected_dist = (3.75 - 1.25) * 10.0  # 25 meters
        assert pytest.approx(y_dist, abs=0.2) == expected_dist

    def test_dropout_rejection(self, gps_sync):
        # Timestamp far beyond the log (> 10s + 1s dropout)
        poses = gps_sync.interpolate_poses([15.0])
        assert not poses.valid_mask[0], "Frames beyond GPS log duration must be flagged invalid"
