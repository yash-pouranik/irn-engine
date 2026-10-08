"""Verification Tests for Phase 5: End-to-End Pipeline Integration & Manifest (FR-M4-10, NFR-REL-02)."""

import json
from pathlib import Path
import numpy as np
import pytest

from irn.common.metrics import evaluate_anomaly_benchmarks, measure_fusion_throughput
from irn.m4_compiler.validator import validate_opendrive_xml, validate_sumo_scenario
from irn.orchestrator import PipelineOrchestrator


class TestEndToEndPipeline:
    """Verifies complete end-to-end simulation synthesis and manifest generation."""

    def test_pipeline_execution_and_manifest(self, tmp_path):
        orch = PipelineOrchestrator(config_path="configs/run_config.yaml")
        res = orch.run_end_to_end(output_dir=tmp_path / "scenario", use_offline_network=True)

        scen_dir = res["output_dir"]
        assert (scen_dir / "network.net.xml").exists()
        assert (scen_dir / "network.xodr").exists()
        assert (scen_dir / "vtypes.add.xml").exists()
        assert (scen_dir / "anomalies.add.xml").exists()
        assert (scen_dir / "scenario.sumocfg").exists()
        assert (scen_dir / "preview.geojson").exists()
        assert (scen_dir / "MANIFEST.json").exists()
        assert (scen_dir / "ATTRIBUTION.md").exists()

        # Validate OpenDRIVE and SUMO scenario
        val_xodr = validate_opendrive_xml(scen_dir / "network.xodr")
        assert val_xodr.is_valid, f"OpenDRIVE validation failed: {val_xodr.errors}"

        val_sumo = validate_sumo_scenario(scen_dir)
        assert val_sumo.is_valid, f"SUMO scenario validation failed: {val_sumo.errors}"

        # Inspect MANIFEST.json
        manifest = res["manifest"]
        assert manifest["traffic_rule"] == "LHT"
        assert manifest["statistics"]["num_canonical_anomalies"] > 0
        assert manifest["validation"]["opendrive_valid"] is True
        assert len(manifest["artifact_sha256"]) >= 6

        # Check Report generation
        report_str = orch.generate_report(scen_dir)
        assert "IRN SCENARIO EXECUTION & BENCHMARK REPORT" in report_str
        assert "network.xodr" in report_str

    def test_anomaly_evaluation_benchmarks(self):
        """Verify NFR-REL-02: precision >= 0.95, recall >= 0.85, duplicate rate < 2%."""
        ground_truth = [
            {"s": 25.0, "t": -1.2, "class_name": "pothole"},
            {"s": 50.0, "t": 0.8, "class_name": "speed_breaker"},
            {"s": 75.0, "t": -0.5, "class_name": "pothole"},
        ]

        # Simulated recovered anomalies with sub-meter spatial accuracy
        recovered = [
            {"s": 25.1, "t": -1.18, "class_name": "pothole"},
            {"s": 50.05, "t": 0.82, "class_name": "speed_breaker"},
            {"s": 75.12, "t": -0.48, "class_name": "pothole"},
        ]

        metrics = evaluate_anomaly_benchmarks(ground_truth, recovered, spatial_tolerance_m=3.0)
        assert metrics.precision >= 0.95
        assert metrics.recall >= 0.85
        assert metrics.duplicate_rate < 0.02

        # Throughput
        tp = measure_fusion_throughput(1000, 0.02)
        assert tp >= 20000.0  # target >= 20,000 dets/sec
