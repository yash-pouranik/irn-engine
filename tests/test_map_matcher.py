"""Verification Tests for HMM Map Matching and Linear Referencing (Gate G-2)."""

import numpy as np
import pytest
from shapely.geometry import LineString, Point
import geopandas as gpd

from irn.m3_fusion.linear_ref import LinearReferencer
from irn.m3_fusion.map_matcher import HMMMapMatcher, RoadNetworkSpatialIndex


class TestHMMMapMatcher:
    """Tests HMM Viterbi edge assignment and parallel carriageway disambiguation."""

    @pytest.fixture
    def parallel_roads_network(self):
        """Creates two parallel roads: Road A (y = 0 to 100 at x = 0) and Road B (flyover at x = 15)."""
        line_a = LineString([(0.0, 0.0), (0.0, 100.0)])
        line_b = LineString([(15.0, 0.0), (15.0, 100.0)])

        gdf = gpd.GeoDataFrame([
            {"edge_id": "road_A_surface", "geometry": line_a, "highway": "primary", "oneway": True},
            {"edge_id": "road_B_flyover", "geometry": line_b, "highway": "motorway", "oneway": True},
        ], crs="EPSG:32644")
        return gdf

    def test_viterbi_continuous_route_selection(self, parallel_roads_network):
        """Gate G-2: Asserts >= 95% precision on trajectory along Road A despite noisy GPS."""
        index = RoadNetworkSpatialIndex(parallel_roads_network, sample_step_m=5.0, search_radius_m=35.0)
        matcher = HMMMapMatcher(index)

        # Vehicle is actually driving along Road A (x = 0), but with 4m noisy GPS points towards Road B
        y_pts = np.linspace(5.0, 95.0, 20)
        rng = np.random.default_rng(42)
        noise_x = rng.normal(1.0, 2.5, size=len(y_pts))  # slight noise bias towards Road B
        x_pts = noise_x

        timestamps = np.linspace(0.0, 10.0, len(y_pts))
        headings = np.full_like(y_pts, 0.0)  # driving North (heading 0)

        matched = matcher.match_trajectory(x_pts, y_pts, timestamps, headings)

        assert len(matched) == len(y_pts)
        correct_matches = sum(1 for m in matched if m.matched_edge_id == "road_A_surface")
        accuracy = correct_matches / len(y_pts)

        # Must maintain >= 95% accuracy
        assert accuracy >= 0.95, f"HMM matching accuracy {accuracy*100:.1f}% below 95% threshold"

    def test_linear_referencing_frenet_projection(self):
        """Verify (s, t) Frenet coordinates and covariance rotation."""
        # Road going North (bearing = 0 deg)
        # Ego vehicle is at s=20.0m, t=0.0m, heading North (heading = 0 deg)
        # Camera observes obstacle at X_fwd = 10.0m, Y_left = -1.5m (1.5m to the right)
        frenet = LinearReferencer.project_obstacle_to_frenet(
            s_ego=20.0,
            t_ego=0.0,
            veh_heading_deg=0.0,
            edge_bearing_deg=0.0,
            x_fwd=10.0,
            y_left=-1.5,
            cov_xx=1.0,
            cov_yy=0.2,
            edge_id="test_edge",
            edge_length_m=100.0
        )

        assert pytest.approx(frenet.s_obs, abs=0.01) == 30.0  # 20m + 10m
        assert pytest.approx(frenet.t_obs, abs=0.01) == -1.5  # -1.5m to the right

    def test_bidirectional_canonical_mapping(self):
        """Verify reverse driving pass maps to canonical frame."""
        # On a 100m road, obstacle at s=30m, t=-1.5m in forward direction
        # In reverse pass, vehicle drives from s=100m to 0m
        s_canon, t_canon = LinearReferencer.canonicalize_bidirectional(
            s_obs=70.0,
            t_obs=1.5,
            edge_length_m=100.0,
            is_reverse_direction=True
        )
        assert pytest.approx(s_canon, abs=0.01) == 30.0  # 100 - 70 = 30
        assert pytest.approx(t_canon, abs=0.01) == -1.5  # -(+1.5) = -1.5
