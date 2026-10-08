"""Lead Orchestrator Engine for IRN Pipeline.

Coordinates:
- Modular and end-to-end execution of M1, M2, M3, and M4.
- Atomic file writes and strict data contract handoffs.
- Provenance manifest generation (MANIFEST.json) with SHA-256 hashes.
- Scenario summary report generation.
"""

from __future__ import annotations
import datetime
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import yaml

from irn.common.io import atomic_write_json, atomic_write_text
from irn.common.logger import log_event
from irn.common.metrics import evaluate_anomaly_benchmarks, measure_fusion_throughput
from irn.m1_fetcher.osm_fetcher import OSMFetcher
from irn.m2_cv_interface.detector import MockDetector
from irn.m3_fusion.pipeline import run_geospatial_fusion
from irn.m4_compiler.compiler import compile_simulation_scenario
from irn.m4_compiler.validator import validate_opendrive_xml, validate_sumo_scenario


def compute_sha256(filepath: Union[str, Path]) -> str:
    """Calculates SHA-256 hash of a file."""
    p = Path(filepath)
    if not p.exists():
        return ""
    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class PipelineOrchestrator:
    """Coordinates end-to-end execution, manifests, and verification."""

    def __init__(self, config_path: Union[str, Path] = "configs/run_config.yaml") -> None:
        self.config_path = Path(config_path)
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

    def run_end_to_end(
        self,
        output_dir: Union[str, Path] = "output_scenario",
        use_offline_network: bool = True
    ) -> Dict[str, Any]:
        """Runs complete IRN pipeline from raw acquisition to compiled scenario bundle."""
        start_time = time.time()
        out = Path(output_dir).resolve()
        out.mkdir(parents=True, exist_ok=True)

        data_dir = out / "pipeline_data"
        d1_dir = data_dir / "d1_graph"
        d2_dir = data_dir / "d2_detections"
        d3_db = data_dir / "d3_store" / "anomalies.db"
        d4_dir = data_dir / "d4_enriched"

        log_event("=== Launching End-to-End IRN Pipeline ===", reason_code="ORCHESTRATOR-START")

        # ----------------------------------------------------
        # Step 1: Module 1 (OSM Fetcher)
        # ----------------------------------------------------
        m1_cfg = self.config.get("m1_fetcher", {})
        bbox = tuple(m1_cfg.get("bbox", [78.345, 17.440, 78.360, 17.452]))
        fetcher = OSMFetcher(timeout=m1_cfg.get("timeout_sec", 15))

        if use_offline_network:
            G = fetcher.generate_offline_grid(bbox)
        else:
            G = fetcher.fetch_by_bbox(bbox, allow_offline_fallback=True)

        d1_res = fetcher.process_and_export(G, d1_dir, query_metadata={"bbox": list(bbox)})

        # ----------------------------------------------------
        # Step 2: Module 2 (CV Ingestion / Detections)
        # ----------------------------------------------------
        d2_dir.mkdir(parents=True, exist_ok=True)
        d2_file = d2_dir / "detections.parquet"
        detector = MockDetector(sequence_id="seq_e2e_run", min_conf=0.35)
        records = detector.generate_synthetic_drive_detections(num_frames=60)
        from irn.m2_cv_interface.schema import save_detections_parquet
        save_detections_parquet(records, d2_file)

        # Generate corresponding vehicle GPS log
        gps_csv = d2_dir / "gps.csv"
        # 60 frames from South to North across the bbox
        t_arr = np.linspace(0.0, 10.0, 60)
        lat_arr = np.linspace(bbox[1], bbox[3], 60)
        lon_arr = np.full(60, (bbox[0] + bbox[2]) / 2.0)
        gps_df = pd.DataFrame({
            "timestamp": t_arr,
            "lat": lat_arr,
            "lon": lon_arr,
            "speed": 10.0,
            "heading": 0.0
        })
        gps_df.to_csv(gps_csv, index=False)

        # ----------------------------------------------------
        # Step 3: Module 3 (Geospatial Data Fusion)
        # ----------------------------------------------------
        utm_zone = d1_res.get("utm_zone", 44)
        fusion_start = time.time()
        fusion_res = run_geospatial_fusion(
            edges_gpkg_path=d1_res["edges_gpkg"],
            detections_parquet_path=d2_file,
            gps_csv_path=gps_csv,
            rig_config_path="configs/rig.yaml",
            utm_zone=utm_zone,
            output_d3_db=d3_db,
            output_d4_dir=d4_dir
        )
        fusion_elapsed = time.time() - fusion_start
        throughput = measure_fusion_throughput(len(records), fusion_elapsed)

        # ----------------------------------------------------
        # Step 4: Module 4 (Simulation Compiler)
        # ----------------------------------------------------
        compile_res = compile_simulation_scenario(
            enriched_edges_path=fusion_res["d4_edges_parquet"],
            anomalies_parquet_path=fusion_res["d4_anomalies_parquet"],
            nodes_gpkg_path=d1_res["nodes_gpkg"],
            out_dir=out,
            lanefree_mode=self.config.get("m4_compiler", {}).get("lanefree_mode", "collapse"),
            utm_zone=utm_zone
        )

        # ----------------------------------------------------
        # Step 5: Validation & Manifest Generation
        # ----------------------------------------------------
        val_xodr = validate_opendrive_xml(compile_res["network_xodr"])
        val_sumo = validate_sumo_scenario(out)

        elapsed_total = round(time.time() - start_time, 2)

        # Build MANIFEST.json (Constraint C-7, Provenance)
        artifact_hashes = {
            "edges_gpkg": compute_sha256(d1_res["edges_gpkg"]),
            "nodes_gpkg": compute_sha256(d1_res["nodes_gpkg"]),
            "detections_parquet": compute_sha256(d2_file),
            "anomalies_db": compute_sha256(d3_db),
            "enriched_edges_parquet": compute_sha256(fusion_res["d4_edges_parquet"]),
            "anomalies_parquet": compute_sha256(fusion_res["d4_anomalies_parquet"]),
            "network_net_xml": compute_sha256(compile_res["network_net_xml"]),
            "network_xodr": compute_sha256(compile_res["network_xodr"]),
            "scenario_sumocfg": compute_sha256(compile_res["scenario_sumocfg"]),
        }

        manifest = {
            "project": "IRN Engine",
            "version": "0.1.0",
            "execution_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "total_elapsed_sec": elapsed_total,
            "fusion_throughput_dets_per_sec": throughput,
            "traffic_rule": "LHT",
            "utm_zone": utm_zone,
            "statistics": {
                "num_detections_ingested": len(records),
                "num_raw_observations_fused": fusion_res["num_raw_observations"],
                "num_canonical_anomalies": fusion_res["num_canonical_anomalies"],
                "num_confirmed_anomalies": fusion_res["num_confirmed"],
            },
            "validation": {
                "opendrive_valid": val_xodr.is_valid,
                "sumo_scenario_valid": val_sumo.is_valid,
            },
            "artifact_sha256": artifact_hashes,
        }
        manifest_path = out / "MANIFEST.json"
        atomic_write_json(manifest_path, manifest)

        # Write ATTRIBUTION.md
        attribution_content = """# IRN Simulation Scenario Attribution & Licensing

This simulation network was synthesized by the **IRN Engine (Automated Road Network Modeling Engine for Lane-Free Indian Traffic Simulations)**.

- **Institution:** Shri Vaishnav Institute of Information Technology (SVIIT), Indore
- **Project Team:** Yash Pouranik, Yashika Goswami, Vipul Yadav, Vivek Tekwani
- **Map Data Attribution:** © OpenStreetMap contributors
- **Map Data License:** Open Database License (ODbL) 1.0 (https://opendatacommons.org/licenses/odbl/)
- **Simulation Models:** Eclipse SUMO Sublane Model (EPL-2.0), ASAM OpenDRIVE 1.6.1 Standard
"""
        attribution_path = out / "ATTRIBUTION.md"
        atomic_write_text(attribution_path, attribution_content)

        log_event(f"=== Pipeline Finished Successfully in {elapsed_total}s ===", reason_code="ORCHESTRATOR-DONE")

        return {
            "output_dir": out,
            "manifest_path": manifest_path,
            "attribution_path": attribution_path,
            "manifest": manifest,
            "compile_results": compile_res,
        }

    def generate_report(self, scenario_dir: Union[str, Path]) -> str:
        """Inspects scenario directory and generates formatted summary string."""
        sdir = Path(scenario_dir)
        manifest_path = sdir / "MANIFEST.json"

        lines = [
            "=" * 65,
            "IRN SCENARIO EXECUTION & BENCHMARK REPORT",
            "=" * 65,
        ]

        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            stats = data.get("statistics", {})
            val = data.get("validation", {})
            lines.append(f"Execution Time:    {data.get('total_elapsed_sec', 0.0)} seconds")
            lines.append(f"Throughput:        {data.get('fusion_throughput_dets_per_sec', 0.0)} detections/sec")
            lines.append(f"Traffic Rule:      {data.get('traffic_rule', 'LHT')} (Strict Indian Convention)")
            lines.append(f"Detections Ingest: {stats.get('num_detections_ingested', 0)}")
            lines.append(f"Anomalies Mapped:  {stats.get('num_canonical_anomalies', 0)} ({stats.get('num_confirmed_anomalies', 0)} Confirmed)")
            lines.append(f"OpenDRIVE 1.6.1:   {'PASS' if val.get('opendrive_valid') else 'CHECK'}")
            lines.append(f"SUMO Scenario:     {'PASS' if val.get('sumo_scenario_valid') else 'CHECK'}")
        else:
            lines.append(f"Scenario Directory: {sdir}")
            lines.append("Note: MANIFEST.json not present.")

        lines.append("-" * 65)
        lines.append("Generated Artifacts:")
        for fname in ["network.net.xml", "network.xodr", "vtypes.add.xml", "anomalies.add.xml", "scenario.sumocfg", "preview.geojson"]:
            exists = (sdir / fname).exists()
            lines.append(f"  [{'x' if exists else ' '}] {fname}")
        lines.append("=" * 65)

        return "\n".join(lines)
