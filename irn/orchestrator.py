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

    def run_idd_sequence(
        self,
        sequence_dir: Union[str, Path],
        output_scenario_dir: Union[str, Path],
        data_dir: Optional[Union[str, Path]] = None,
        shared_graph_dir: Optional[Union[str, Path]] = None,
        shared_db_path: Optional[Union[str, Path]] = None,
        pothole_weights: Optional[Union[str, Path]] = None,
        vehicle_weights: Optional[Union[str, Path]] = None,
        sample_step: int = 10,
        max_frames: Optional[int] = 100,
        utm_zone: int = 44,
    ) -> Dict[str, Any]:
        """Runs the complete IRN pipeline on a real IDD Multimodal sequence."""
        start_time = time.time()
        s_dir = Path(sequence_dir).resolve()
        seq_id = s_dir.name
        out_scen = Path(output_scenario_dir).resolve()
        out_scen.mkdir(parents=True, exist_ok=True)

        d_dir = Path(data_dir or f"data/idd_{seq_id}").resolve()
        d_dir.mkdir(parents=True, exist_ok=True)

        log_event(f"=== Processing IDD Sequence: {seq_id} ===", reason_code="ORCHESTRATOR-IDD-START")

        # Step 1: IDD Adapter (YOLO Inference + GPS Normalization)
        from irn.m2_cv_interface.idd_adapter import IDDAdapter
        adapter = IDDAdapter(sequence_id=seq_id)
        ingest_res = adapter.process_sequence(
            sequence_dir=s_dir,
            out_dir=d_dir,
            pothole_weights=pothole_weights,
            vehicle_weights=vehicle_weights,
            sample_step=sample_step,
            max_frames=max_frames,
        )

        # Step 2: Road Network (Fetch or reuse shared graph)
        if shared_graph_dir and (Path(shared_graph_dir) / "edges.gpkg").exists():
            edges_gpkg = Path(shared_graph_dir) / "edges.gpkg"
            nodes_gpkg = Path(shared_graph_dir) / "nodes.gpkg"
            log_event(f"Reusing shared OSM graph from {shared_graph_dir}", reason_code="GRAPH-REUSED")
        else:
            fetcher = OSMFetcher()
            bbox = ingest_res["bbox"]
            g_dir = d_dir / "d1_graph"
            G = fetcher.fetch_by_bbox(bbox, allow_offline_fallback=True)
            d1_res = fetcher.process_and_export(G, g_dir, query_metadata={"bbox": list(bbox)})
            edges_gpkg = d1_res["edges_gpkg"]
            nodes_gpkg = d1_res["nodes_gpkg"]

        # Step 3: M3 Geospatial Fusion
        d3_db = Path(shared_db_path) if shared_db_path else (d_dir / "d3_store" / "anomalies.db")
        d4_dir = d_dir / "d4_enriched"
        fusion_start = time.time()
        fusion_res = run_geospatial_fusion(
            edges_gpkg_path=edges_gpkg,
            detections_parquet_path=ingest_res["detections_parquet"],
            gps_csv_path=ingest_res["gps_csv"],
            rig_config_path="configs/rig.yaml",
            utm_zone=utm_zone,
            output_d3_db=d3_db,
            output_d4_dir=d4_dir,
        )
        fusion_elapsed = time.time() - fusion_start
        throughput = measure_fusion_throughput(ingest_res["num_detections"], fusion_elapsed)

        # Step 4: M4 Simulation Compilation
        compile_res = compile_simulation_scenario(
            enriched_edges_path=fusion_res["d4_edges_parquet"],
            anomalies_parquet_path=fusion_res["d4_anomalies_parquet"],
            nodes_gpkg_path=nodes_gpkg,
            out_dir=out_scen,
            lanefree_mode="collapse",
            utm_zone=utm_zone,
        )

        val_xodr = validate_opendrive_xml(compile_res["network_xodr"])
        val_sumo = validate_sumo_scenario(out_scen)
        elapsed_total = round(time.time() - start_time, 2)

        # Manifest
        artifact_hashes = {
            "edges_gpkg": compute_sha256(edges_gpkg),
            "nodes_gpkg": compute_sha256(nodes_gpkg),
            "detections_parquet": compute_sha256(ingest_res["detections_parquet"]),
            "anomalies_db": compute_sha256(d3_db),
            "enriched_edges_parquet": compute_sha256(fusion_res["d4_edges_parquet"]),
            "anomalies_parquet": compute_sha256(fusion_res["d4_anomalies_parquet"]),
            "network_net_xml": compute_sha256(compile_res["network_net_xml"]),
            "network_xodr": compute_sha256(compile_res["network_xodr"]),
            "scenario_sumocfg": compute_sha256(compile_res["scenario_sumocfg"]),
        }

        manifest = {
            "project": "IRN Engine",
            "sequence_id": seq_id,
            "version": "0.1.0",
            "execution_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "total_elapsed_sec": elapsed_total,
            "fusion_throughput_dets_per_sec": throughput,
            "traffic_rule": "LHT",
            "utm_zone": utm_zone,
            "statistics": {
                "num_gps_fixes": ingest_res["num_gps_fixes"],
                "num_detections_ingested": ingest_res["num_detections"],
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
        manifest_path = out_scen / "MANIFEST.json"
        atomic_write_json(manifest_path, manifest)

        log_event(f"=== IDD Sequence {seq_id} Finished in {elapsed_total}s ===", reason_code="ORCHESTRATOR-IDD-DONE")

        return {
            "sequence_id": seq_id,
            "output_dir": out_scen,
            "manifest_path": manifest_path,
            "manifest": manifest,
            "compile_results": compile_res,
            "shared_graph_dir": Path(edges_gpkg).parent,
        }

    def run_idd_batch(
        self,
        dataset_dir: Union[str, Path],
        output_base_dir: Union[str, Path] = "scenarios",
        sequences: Optional[List[str]] = None,
        pothole_weights: Optional[Union[str, Path]] = None,
        vehicle_weights: Optional[Union[str, Path]] = None,
        sample_step: int = 10,
        max_frames: Optional[int] = 100,
        unified_multi_pass: bool = True,
        utm_zone: int = 44,
    ) -> Dict[str, Any]:
        """Runs batch processing over all sequences with cross-pass anomaly accumulation."""
        import geopandas as gpd
        from irn.m3_fusion.storage import FusionStorage
        from irn.m3_fusion.width_fusion import fuse_edge_widths

        ds_path = Path(dataset_dir).resolve()
        out_base = Path(output_base_dir).resolve()
        out_base.mkdir(parents=True, exist_ok=True)

        if not sequences:
            seq_candidates = sorted([d.name for d in ds_path.iterdir() if d.is_dir() and (d / "train.csv").exists()])
        else:
            seq_candidates = sequences

        log_event(f"Batch processing {len(seq_candidates)} sequences: {seq_candidates}", reason_code="BATCH-START")

        shared_db = (Path("data") / "multi_pass" / "anomalies.db").resolve() if unified_multi_pass else None
        if shared_db:
            shared_db.parent.mkdir(parents=True, exist_ok=True)

        shared_graph_dir: Optional[Path] = None
        results: Dict[str, Any] = {}

        for seq_name in seq_candidates:
            seq_dir = ds_path / seq_name
            scen_dir = out_base / f"scenario_{seq_name}"

            res = self.run_idd_sequence(
                sequence_dir=seq_dir,
                output_scenario_dir=scen_dir,
                shared_graph_dir=shared_graph_dir,
                shared_db_path=shared_db,
                pothole_weights=pothole_weights,
                vehicle_weights=vehicle_weights,
                sample_step=sample_step,
                max_frames=max_frames,
                utm_zone=utm_zone,
            )
            if not shared_graph_dir:
                shared_graph_dir = res["shared_graph_dir"]
            results[seq_name] = res

        # If unified multi-pass, compile the master corridor scenario from the merged ledger
        master_scen = None
        if unified_multi_pass and shared_db and shared_graph_dir:
            log_event("Compiling Unified Multi-Pass Corridor Scenario", reason_code="CORRIDOR-COMPILE")
            corridor_dir = out_base / "corridor_multi_pass"
            corridor_d4 = Path("data") / "multi_pass" / "d4_enriched"

            storage = FusionStorage(shared_db)
            all_canonical = storage.load_canonical_anomalies()
            edges_gdf = gpd.read_file(shared_graph_dir / "edges.gpkg", layer="edges")
            enriched_edges = fuse_edge_widths(edges_gdf)
            storage.export_d4_parquet(enriched_edges, all_canonical, corridor_d4)

            master_scen = compile_simulation_scenario(
                enriched_edges_path=corridor_d4 / "enriched_edges.parquet",
                anomalies_parquet_path=corridor_d4 / "anomalies.parquet",
                nodes_gpkg_path=shared_graph_dir / "nodes.gpkg",
                out_dir=corridor_dir,
                lanefree_mode="collapse",
                utm_zone=utm_zone,
            )

            val_xodr = validate_opendrive_xml(master_scen["network_xodr"])
            val_sumo = validate_sumo_scenario(corridor_dir)
            total_dets = sum(r["manifest"]["statistics"]["num_detections_ingested"] for r in results.values())
            total_fixes = sum(r["manifest"]["statistics"]["num_gps_fixes"] for r in results.values())
            confirmed_anoms = sum(1 for a in all_canonical if a.state == "CONFIRMED")

            corridor_manifest = {
                "project": "IRN Engine",
                "scenario": "corridor_multi_pass",
                "version": "0.1.0",
                "execution_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "traffic_rule": "LHT",
                "utm_zone": utm_zone,
                "multi_pass_sequences": list(results.keys()),
                "statistics": {
                    "num_gps_fixes": total_fixes,
                    "num_detections_ingested": total_dets,
                    "num_canonical_anomalies": len(all_canonical),
                    "num_confirmed_anomalies": confirmed_anoms,
                },
                "validation": {
                    "opendrive_valid": val_xodr.is_valid,
                    "sumo_scenario_valid": val_sumo.is_valid,
                },
            }
            atomic_write_json(corridor_dir / "MANIFEST.json", corridor_manifest)

        return {
            "sequences_processed": list(results.keys()),
            "sequence_results": results,
            "corridor_scenario": master_scen,
        }

