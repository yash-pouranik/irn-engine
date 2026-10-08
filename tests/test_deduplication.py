"""Verification Tests for Multi-Pass De-duplication and Mahalanobis Gating (Gate G-3)."""

import pytest
import numpy as np

from irn.m3_fusion.deduplication import (
    AnomalyDeduplicator,
    CanonicalAnomaly,
    CHI2_GATE_THRESHOLD,
)


class TestAnomalyDeduplication:
    """Tests Hungarian algorithm, Mahalanobis distance gating, and covariance shrinkage."""

    def test_multi_pass_merge_and_covariance_shrinkage(self):
        """Gate G-3: Repeated passes merge into single UUID with shrinking covariance."""
        dedup = AnomalyDeduplicator()

        # Pass 1: Observed pothole at s = 25.0m, t = -1.2m
        pass1_anomalies = [
            CanonicalAnomaly(
                class_name="pothole",
                edge_id="edge_101",
                s=25.0,
                t=-1.2,
                var_s=0.5,
                var_t=0.2,
                confidence=0.8,
                observation_count=1,
                state="CANDIDATE"
            )
        ]

        # Pass 2: Second independent pass observes same pothole at s = 25.2m, t = -1.15m
        pass2_anomalies = [
            CanonicalAnomaly(
                class_name="pothole",
                edge_id="edge_101",
                s=25.2,
                t=-1.15,
                var_s=0.5,
                var_t=0.2,
                confidence=0.85,
                observation_count=1,
                state="CANDIDATE"
            )
        ]

        merged = dedup.merge_multi_pass(pass1_anomalies, pass2_anomalies)

        # Must merge into single canonical anomaly
        assert len(merged) == 1
        anom = merged[0]

        # State must promote to CONFIRMED
        assert anom.state == "CONFIRMED"
        assert anom.observation_count == 2

        # Position should be optimal weighted mean (~25.1, ~-1.175)
        assert pytest.approx(anom.s, abs=0.1) == 25.1
        assert pytest.approx(anom.t, abs=0.05) == -1.175

        # Variance MUST shrink (var_merged = 1 / (1/0.5 + 1/0.5) = 0.25 < 0.5)
        assert anom.var_s < 0.5
        assert pytest.approx(anom.var_s, abs=0.01) == 0.25
        assert pytest.approx(anom.var_t, abs=0.01) == 0.10

    def test_distant_and_incompatible_anomalies_not_merged(self):
        """Verify anomalies beyond chi2 gate or of different classes do not merge."""
        dedup = AnomalyDeduplicator()

        existing = [
            CanonicalAnomaly(
                class_name="pothole",
                edge_id="edge_101",
                s=25.0,
                t=-1.2,
                var_s=0.2,
                var_t=0.1
            )
        ]

        # New anomaly 1: Same class but far away (s = 50.0m)
        new_distant = [
            CanonicalAnomaly(
                class_name="pothole",
                edge_id="edge_101",
                s=50.0,
                t=-1.2,
                var_s=0.2,
                var_t=0.1
            )
        ]
        res1 = dedup.merge_multi_pass(list(existing), new_distant)
        assert len(res1) == 2, "Distant anomalies must remain separate entries"

        # New anomaly 2: Same location but incompatible class (speed_breaker vs pothole)
        new_diff_class = [
            CanonicalAnomaly(
                class_name="speed_breaker",
                edge_id="edge_101",
                s=25.0,
                t=-1.2,
                var_s=0.2,
                var_t=0.1
            )
        ]
        res2 = dedup.merge_multi_pass(list(existing), new_diff_class)
        assert len(res2) == 2, "Incompatible classes must never merge"
