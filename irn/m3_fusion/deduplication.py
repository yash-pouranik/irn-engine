"""Module 3: Multi-Pass Anomaly De-duplication Engine (Level 1 & Level 2).

Implements:
- Level 1 (Within-pass): Group observations by track_id and perform inverse-variance Bayesian fusion.
- Level 2 (Across-pass): Bipartite matching via Hungarian algorithm (linear_sum_assignment)
  with Mahalanobis distance cost metric.
- Chi-square gating threshold: chi2(df=2, p=0.01) = 9.21. Incompatible classes never merge.
- Lifecycle state machine: CANDIDATE -> CONFIRMED -> STALE -> RESOLVED.
"""

from __future__ import annotations
import math
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.optimize import linear_sum_assignment

from irn.common.logger import log_event
from irn.m3_fusion.linear_ref import FrenetObstaclePose

# Chi-square gating threshold for 2 degrees of freedom at p = 0.01
CHI2_GATE_THRESHOLD = 9.21


class CanonicalAnomaly:
    """Represents a unique, de-duplicated physical road anomaly."""

    def __init__(
        self,
        anomaly_id: Optional[str] = None,
        class_name: str = "pothole",
        edge_id: str = "edge_0",
        s: float = 0.0,
        t: float = 0.0,
        var_s: float = 1.0,
        var_t: float = 0.2,
        confidence: float = 0.8,
        observation_count: int = 1,
        state: str = "CANDIDATE"
    ) -> None:
        self.anomaly_id = anomaly_id or str(uuid.uuid4())
        self.class_name = class_name
        self.edge_id = edge_id
        self.s = float(s)
        self.t = float(t)
        self.var_s = float(var_s)
        self.var_t = float(var_t)
        self.confidence = float(confidence)
        self.observation_count = observation_count
        self.state = state

    @property
    def sigma_s(self) -> float:
        return math.sqrt(max(1e-4, self.var_s))

    @property
    def sigma_t(self) -> float:
        return math.sqrt(max(1e-4, self.var_t))

    def update_with_observation(
        self,
        obs_s: float,
        obs_t: float,
        obs_var_s: float,
        obs_var_t: float,
        obs_conf: float
    ) -> None:
        """Inverse-variance Bayesian Kalman fusion update."""
        # 1D independent Kalman updates along s and t
        w_s_curr = 1.0 / max(1e-4, self.var_s)
        w_s_new = 1.0 / max(1e-4, obs_var_s)
        new_var_s = 1.0 / (w_s_curr + w_s_new)
        self.s = new_var_s * (w_s_curr * self.s + w_s_new * obs_s)
        self.var_s = new_var_s

        w_t_curr = 1.0 / max(1e-4, self.var_t)
        w_t_new = 1.0 / max(1e-4, obs_var_t)
        new_var_t = 1.0 / (w_t_curr + w_t_new)
        self.t = new_var_t * (w_t_curr * self.t + w_t_new * obs_t)
        self.var_t = new_var_t

        self.observation_count += 1
        self.confidence = min(0.99, max(self.confidence, obs_conf) + 0.05)

        # State transition: CANDIDATE -> CONFIRMED
        if self.state == "CANDIDATE":
            if self.observation_count >= 2 and self.confidence >= 0.5:
                self.state = "CONFIRMED"
            elif self.observation_count >= 5:
                self.state = "CONFIRMED"


class AnomalyDeduplicator:
    """Two-level de-duplication engine for multi-pass obstacle observations."""

    def __init__(self, chi2_gate: float = CHI2_GATE_THRESHOLD) -> None:
        self.chi2_gate = chi2_gate

    @staticmethod
    def mahalanobis_dist_sq(
        s1: float,
        t1: float,
        var_s1: float,
        var_t1: float,
        s2: float,
        t2: float,
        var_s2: float,
        var_t2: float
    ) -> float:
        """Computes squared Mahalanobis distance D_M^2 between two 2D Gaussians."""
        ds = s1 - s2
        dt = t1 - t2
        total_var_s = max(1e-4, var_s1 + var_s2)
        total_var_t = max(1e-4, var_t1 + var_t2)
        return (ds**2) / total_var_s + (dt**2) / total_var_t

    def deduplicate_pass(
        self,
        raw_observations: List[Dict[str, Any]]
    ) -> List[CanonicalAnomaly]:
        """Level 1: Within-pass grouping by track_id or spatial proximity."""
        if not raw_observations:
            return []

        # Group by (edge_id, class_name, track_id)
        grouped_by_track: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = {}
        untracked: List[Dict[str, Any]] = []

        for obs in raw_observations:
            track_id = obs.get("track_id", -1)
            edge_id = obs["edge_id"]
            cls_name = obs["class_name"]

            if track_id is not None and track_id >= 0:
                key = (edge_id, cls_name, track_id)
                grouped_by_track.setdefault(key, []).append(obs)
            else:
                untracked.append(obs)

        pass_anomalies: List[CanonicalAnomaly] = []

        # Merge tracked clusters
        for (edge_id, cls_name, tid), cluster in grouped_by_track.items():
            # Fuse all observations in cluster
            first = cluster[0]
            canon = CanonicalAnomaly(
                class_name=cls_name,
                edge_id=edge_id,
                s=first["s_obs"],
                t=first["t_obs"],
                var_s=first["sigma_s"]**2,
                var_t=first["sigma_t"]**2,
                confidence=first.get("conf", 0.7),
                observation_count=1,
                state="CONFIRMED" if len(cluster) >= 5 else "CANDIDATE"
            )
            for item in cluster[1:]:
                canon.update_with_observation(
                    obs_s=item["s_obs"],
                    obs_t=item["t_obs"],
                    obs_var_s=item["sigma_s"]**2,
                    obs_var_t=item["sigma_t"]**2,
                    obs_conf=item.get("conf", 0.7)
                )
            pass_anomalies.append(canon)

        # Untracked items added individually
        for item in untracked:
            pass_anomalies.append(CanonicalAnomaly(
                class_name=item["class_name"],
                edge_id=item["edge_id"],
                s=item["s_obs"],
                t=item["t_obs"],
                var_s=item["sigma_s"]**2,
                var_t=item["sigma_t"]**2,
                confidence=item.get("conf", 0.6),
                observation_count=1,
                state="CANDIDATE"
            ))

        log_event(
            f"Level 1 within-pass de-duplication: reduced {len(raw_observations)} detections to {len(pass_anomalies)} anomalies",
            reason_code="DEDUP-LEVEL1-DONE"
        )
        return pass_anomalies

    def merge_multi_pass(
        self,
        existing_canonical: List[CanonicalAnomaly],
        new_pass_anomalies: List[CanonicalAnomaly]
    ) -> List[CanonicalAnomaly]:
        """Level 2: Across-pass matching using Hungarian algorithm with Mahalanobis gating."""
        if not existing_canonical:
            return list(new_pass_anomalies)
        if not new_pass_anomalies:
            return list(existing_canonical)

        # Build bipartite cost matrix between existing and new
        n_exist = len(existing_canonical)
        n_new = len(new_pass_anomalies)

        INF_COST = 1e9
        cost_matrix = np.full((n_exist, n_new), INF_COST, dtype=float)

        for i, can in enumerate(existing_canonical):
            for j, new_anom in enumerate(new_pass_anomalies):
                # Must match edge_id and class_name
                if can.edge_id != new_anom.edge_id or can.class_name != new_anom.class_name:
                    continue

                d_m_sq = self.mahalanobis_dist_sq(
                    can.s, can.t, can.var_s, can.var_t,
                    new_anom.s, new_anom.t, new_anom.var_s, new_anom.var_t
                )

                if d_m_sq <= self.chi2_gate:
                    cost_matrix[i, j] = d_m_sq

        # Solve optimal bipartite matching
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_new_indices = set()
        for r, c in zip(row_ind, col_ind):
            cost = cost_matrix[r, c]
            if cost <= self.chi2_gate:
                # Merge into existing canonical anomaly
                new_a = new_pass_anomalies[c]
                existing_canonical[r].update_with_observation(
                    obs_s=new_a.s,
                    obs_t=new_a.t,
                    obs_var_s=new_a.var_s,
                    obs_var_t=new_a.var_t,
                    obs_conf=new_a.confidence
                )
                matched_new_indices.add(c)

        # Unmatched new anomalies become new canonical entries
        for j, new_a in enumerate(new_pass_anomalies):
            if j not in matched_new_indices:
                existing_canonical.append(new_a)

        confirmed_count = sum(1 for a in existing_canonical if a.state == "CONFIRMED")
        log_event(
            f"Level 2 across-pass de-duplication: total canonical ledger {len(existing_canonical)} ({confirmed_count} confirmed)",
            reason_code="DEDUP-LEVEL2-DONE"
        )
        return existing_canonical
