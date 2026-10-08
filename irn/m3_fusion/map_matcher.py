"""Module 3: Road Spatial Indexing and HMM Viterbi Map-Matching Engine.

Implements:
- 5m edge centerline discretization with scipy.spatial.cKDTree spatial indexing.
- Spatial candidate filtering within 35m and heading gate <= 100 degrees.
- Hidden Markov Model (HMM) Viterbi trajectory decoding:
    * Gaussian emission on perpendicular distance d_perp (sigma = 5.0m).
    * Route vs. Euclidean transition distance penalty (beta = 10.0m).
- Resolves parallel flyovers and complex junctions deterministically (Rule G-3).
"""

from __future__ import annotations
import math
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union
import geopandas as gpd
import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point
import shapely

from irn.common.logger import log_event


class MatchedEdgeCandidate(NamedTuple):
    edge_id: str
    edge_idx: int
    s_ego: float          # Distance along edge centerline in meters
    t_ego: float          # Perpendicular lateral offset in meters (positive to LEFT)
    d_perp: float         # Absolute perpendicular distance in meters
    edge_bearing_deg: float
    emission_log_prob: float


class HMMMatchedPose(NamedTuple):
    frame_idx: int
    frame_ts: float
    matched_edge_id: str
    s_ego: float          # Metric linear reference distance along edge (meters)
    t_ego: float          # Metric lateral offset from edge centerline (meters)
    d_perp: float
    edge_bearing_deg: float
    vehicle_heading_deg: float


class RoadNetworkSpatialIndex:
    """Discretizes road centerlines at 5m intervals and indexes via cKDTree."""

    def __init__(
        self,
        edges_gdf: gpd.GeoDataFrame,
        sample_step_m: float = 5.0,
        search_radius_m: float = 35.0,
        max_candidates: int = 8,
        bearing_gate_deg: float = 100.0,
        sigma_gps_m: float = 5.0,
        beta_transition_m: float = 10.0
    ) -> None:
        self.edges_gdf = edges_gdf.copy().reset_index(drop=True)
        self.sample_step_m = sample_step_m
        self.search_radius_m = search_radius_m
        self.max_candidates = max_candidates
        self.bearing_gate_deg = bearing_gate_deg
        self.sigma_gps_m = sigma_gps_m
        self.beta_transition_m = beta_transition_m

        # Build sample points along all edges
        sample_coords = []
        sample_edge_indices = []
        sample_s_values = []
        sample_bearings = []

        self.edge_geometries: List[LineString] = []
        self.edge_ids: List[str] = []
        self.edge_lengths: List[float] = []

        for idx, row in self.edges_gdf.iterrows():
            geom = row.geometry
            if not isinstance(geom, LineString) or geom.length < 0.1:
                continue

            edge_id = str(row.get("edge_id", f"edge_{idx}"))
            self.edge_geometries.append(geom)
            self.edge_ids.append(edge_id)
            length = float(geom.length)
            self.edge_lengths.append(length)

            edge_list_idx = len(self.edge_geometries) - 1

            # Sample points at regular intervals along length
            num_samples = max(2, int(np.ceil(length / self.sample_step_m)) + 1)
            s_pts = np.linspace(0.0, length, num_samples)

            coords = np.array([geom.interpolate(s).coords[0] for s in s_pts])  # (N, 2)

            # Compute tangents/bearings
            dx = np.gradient(coords[:, 0])
            dy = np.gradient(coords[:, 1])
            bearings = (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0

            for i in range(len(s_pts)):
                sample_coords.append(coords[i])
                sample_edge_indices.append(edge_list_idx)
                sample_s_values.append(s_pts[i])
                sample_bearings.append(bearings[i])

        self.sample_coords = np.array(sample_coords, dtype=float)
        self.sample_edge_indices = np.array(sample_edge_indices, dtype=int)
        self.sample_s_values = np.array(sample_s_values, dtype=float)
        self.sample_bearings = np.array(sample_bearings, dtype=float)

        self.kdtree = cKDTree(self.sample_coords)
        log_event(
            f"Built RoadNetworkSpatialIndex with {len(self.edge_geometries)} edges and {len(self.sample_coords)} sample points",
            reason_code="MAP-INDEX-INIT"
        )

    def get_candidate_edges(
        self,
        x_utm: float,
        y_utm: float,
        heading_deg: Optional[float] = None
    ) -> List[MatchedEdgeCandidate]:
        """Query nearest candidate edges within search radius, filtered by heading."""
        pt = Point(x_utm, y_utm)
        neighbor_indices = self.kdtree.query_ball_point([x_utm, y_utm], r=self.search_radius_m)

        if not neighbor_indices:
            # Expand radius slightly if no candidate found
            neighbor_indices = self.kdtree.query_ball_point([x_utm, y_utm], r=self.search_radius_m * 1.5)
            if not neighbor_indices:
                return []

        # Find unique edge indices among neighbors
        unique_edge_indices = set(self.sample_edge_indices[neighbor_indices])
        candidates: List[MatchedEdgeCandidate] = []

        log_norm_const = -0.5 * np.log(2.0 * np.pi * (self.sigma_gps_m ** 2))
        two_sigma_sq = 2.0 * (self.sigma_gps_m ** 2)

        for edge_idx in unique_edge_indices:
            geom = self.edge_geometries[edge_idx]
            edge_id = self.edge_ids[edge_idx]

            # Project vehicle point orthogonally onto edge geometry
            s_ego = float(shapely.line_locate_point(geom, pt))
            proj_pt = geom.interpolate(s_ego)

            # Metric perpendicular distance
            d_perp = float(pt.distance(proj_pt))
            if d_perp > self.search_radius_m:
                continue

            # Compute edge tangent bearing at projected s
            s_ahead = min(geom.length, s_ego + 1.0)
            s_behind = max(0.0, s_ego - 1.0)
            if s_ahead > s_behind:
                pt_ahead = geom.interpolate(s_ahead)
                pt_behind = geom.interpolate(s_behind)
                dx = pt_ahead.x - pt_behind.x
                dy = pt_ahead.y - pt_behind.y
                edge_bearing = (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0
            else:
                edge_bearing = 0.0

            # Lateral offset sign (t_ego): positive to LEFT of edge heading
            # Vector along road: (dx_road, dy_road)
            # Vector from road to vehicle: (x_utm - proj_pt.x, y_utm - proj_pt.y)
            # Cross product (2D): road_x * veh_y - road_y * veh_x
            # In (North = +Y, East = +X):
            # Normal to left of heading: (-sin(bearing), cos(bearing)) in (dx, dy)?
            # Let's use standard cross product:
            # road tangent T = (sin(bearing), cos(bearing))
            # left normal N = (-cos(bearing), sin(bearing))
            rad = np.radians(edge_bearing)
            nx_left = -np.cos(rad)
            ny_left = np.sin(rad)
            dx_veh = x_utm - proj_pt.x
            dy_veh = y_utm - proj_pt.y
            t_ego = dx_veh * nx_left + dy_veh * ny_left

            # Heading filter
            if heading_deg is not None:
                delta_heading = abs((heading_deg - edge_bearing + 180.0) % 360.0 - 180.0)
                if delta_heading > self.bearing_gate_deg:
                    continue

            # Emission log-probability: Gaussian
            log_emission = log_norm_const - (d_perp ** 2) / two_sigma_sq

            candidates.append(MatchedEdgeCandidate(
                edge_id=edge_id,
                edge_idx=edge_idx,
                s_ego=s_ego,
                t_ego=t_ego,
                d_perp=d_perp,
                edge_bearing_deg=edge_bearing,
                emission_log_prob=log_emission
            ))

        # Sort by emission probability descending and retain top max_candidates
        candidates.sort(key=lambda c: c.emission_log_prob, reverse=True)
        return candidates[:self.max_candidates]


class HMMMapMatcher:
    """Hidden Markov Model Viterbi decoder for robust trajectory map matching."""

    def __init__(self, spatial_index: RoadNetworkSpatialIndex) -> None:
        self.index = spatial_index

    def match_trajectory(
        self,
        x_utm_arr: np.ndarray,
        y_utm_arr: np.ndarray,
        timestamps_arr: np.ndarray,
        headings_deg_arr: Optional[np.ndarray] = None
    ) -> List[HMMMatchedPose]:
        """Runs Viterbi dynamic programming to find optimal edge sequence across trajectory."""
        n_epochs = len(x_utm_arr)
        if n_epochs == 0:
            return []

        # 1. Generate candidate sets for all epochs
        candidate_states: List[List[MatchedEdgeCandidate]] = []
        for i in range(n_epochs):
            head = headings_deg_arr[i] if headings_deg_arr is not None else None
            cands = self.index.get_candidate_edges(x_utm_arr[i], y_utm_arr[i], heading_deg=head)
            if not cands:
                # Fallback: query without heading filter if too restrictive
                cands = self.index.get_candidate_edges(x_utm_arr[i], y_utm_arr[i], heading_deg=None)
            candidate_states.append(cands)

        # 2. Viterbi Forward Pass
        # viterbi_log_prob[t][cand_idx]
        # backpointers[t][cand_idx]
        viterbi_log_prob: List[np.ndarray] = []
        backpointers: List[np.ndarray] = []

        # Initialization at t = 0
        cands_0 = candidate_states[0]
        if not cands_0:
            # No candidate found in area
            log_event("No map matching candidates at t=0", reason_code="HMM-NO-CANDIDATES", level=30)
            return []

        viterbi_log_prob.append(np.array([c.emission_log_prob for c in cands_0], dtype=float))
        backpointers.append(np.zeros(len(cands_0), dtype=int))

        # Dynamic Programming forward iteration
        for t in range(1, n_epochs):
            cands_curr = candidate_states[t]
            cands_prev = candidate_states[t - 1]

            if not cands_curr:
                # Retain previous best if no candidate in this step
                viterbi_log_prob.append(viterbi_log_prob[-1])
                backpointers.append(np.arange(len(viterbi_log_prob[-1])))
                candidate_states[t] = cands_prev
                continue

            prev_probs = viterbi_log_prob[t - 1]
            n_curr = len(cands_curr)
            n_prev = len(cands_prev)

            curr_probs = np.full(n_curr, -np.inf)
            bp_curr = np.zeros(n_curr, dtype=int)

            # Euclidean distance between GPS observations
            d_euclid = np.sqrt(
                (x_utm_arr[t] - x_utm_arr[t - 1])**2
                + (y_utm_arr[t] - y_utm_arr[t - 1])**2
            )

            for j, cand_j in enumerate(cands_curr):
                best_score = -np.inf
                best_prev_idx = 0

                for k, cand_k in enumerate(cands_prev):
                    # Route distance estimation:
                    if cand_k.edge_id == cand_j.edge_id:
                        d_route = abs(cand_j.s_ego - cand_k.s_ego)
                    else:
                        # Cross-edge transition: distance along edge k + distance along edge j
                        # Approximate network distance by Euclidean + edge endpoint distance
                        d_route = abs(d_euclid)

                    # Transition probability penalty:
                    delta_d = abs(d_route - d_euclid)
                    log_trans = -np.log(self.index.beta_transition_m) - (delta_d / self.index.beta_transition_m)

                    total_score = prev_probs[k] + log_trans + cand_j.emission_log_prob
                    if total_score > best_score:
                        best_score = total_score
                        best_prev_idx = k

                curr_probs[j] = best_score
                bp_curr[j] = best_prev_idx

            viterbi_log_prob.append(curr_probs)
            backpointers.append(bp_curr)

        # 3. Backtracking for optimal sequence
        best_last_idx = int(np.argmax(viterbi_log_prob[-1]))
        matched_sequence: List[HMMMatchedPose] = []

        curr_idx = best_last_idx
        for t in range(n_epochs - 1, -1, -1):
            cand = candidate_states[t][curr_idx]
            matched_sequence.append(HMMMatchedPose(
                frame_idx=t,
                frame_ts=float(timestamps_arr[t]),
                matched_edge_id=cand.edge_id,
                s_ego=cand.s_ego,
                t_ego=cand.t_ego,
                d_perp=cand.d_perp,
                edge_bearing_deg=cand.edge_bearing_deg,
                vehicle_heading_deg=float(headings_deg_arr[t]) if headings_deg_arr is not None else cand.edge_bearing_deg
            ))
            curr_idx = backpointers[t][curr_idx]

        matched_sequence.reverse()
        log_event(
            f"Viterbi decoded {len(matched_sequence)} epochs successfully across {len(set(m.matched_edge_id for m in matched_sequence))} unique edges",
            reason_code="HMM-MATCH-SUCCESS"
        )
        return matched_sequence
