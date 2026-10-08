"""IRN Command-Line Interface (CLI) orchestrated via Typer."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Optional
import typer

from irn.common.logger import log_event, logger

app = typer.Typer(
    name="irn",
    help="IRN: Automated Road Network Modeling Engine for Lane-Free Indian Traffic Simulations",
    add_completion=False
)


@app.command()
def info() -> None:
    """Display engine environment, dependencies, and CRS projection info."""
    import osmnx as ox
    import pyarrow as pa
    import shapely

    typer.echo("=================================================================")
    typer.echo("IRN: Road Network Modeling Engine (SVIIT Indore Major Project)")
    typer.echo("=================================================================")
    typer.echo(f"OSMnx Version:    {ox.__version__}")
    typer.echo(f"Shapely Version:  {shapely.__version__}")
    typer.echo(f"PyArrow Version:  {pa.__version__}")
    typer.echo("Traffic Rule:     Strict Left-Hand Traffic (LHT)")
    typer.echo("Coordinate Grid:  WGS84 Storage (EPSG:4326) / Local Projected UTM")
    typer.echo("=================================================================")


@app.command()
def run(
    config: Path = typer.Option(
        Path("configs/run_config.yaml"),
        "--config",
        help="Path to pipeline run_config.yaml"
    ),
    out_dir: Path = typer.Option(
        Path("output_scenario"),
        "--out",
        help="Destination directory for scenario bundle"
    ),
    offline: bool = typer.Option(
        True,
        "--offline/--live",
        help="Whether to use calibrated offline network fallback or query live Overpass API"
    ),
) -> None:
    """Execute end-to-end IRN pipeline (M1 -> M2 -> M3 -> M4 -> Manifest)."""
    from irn.orchestrator import PipelineOrchestrator

    typer.secho("Starting End-to-End IRN Simulation Synthesis...", fg=typer.colors.CYAN)
    orch = PipelineOrchestrator(config_path=config)
    res = orch.run_end_to_end(output_dir=out_dir, use_offline_network=offline)

    typer.secho(f"Scenario successfully synthesized at {out_dir}!", fg=typer.colors.GREEN)
    report_text = orch.generate_report(out_dir)
    typer.echo(report_text)


@app.command()
def report(
    scenario: Path = typer.Option(
        Path("output_scenario"),
        "--scenario",
        help="Path to compiled scenario directory"
    ),
) -> None:
    """Display execution statistics and validation report for a scenario bundle."""
    from irn.orchestrator import PipelineOrchestrator

    orch = PipelineOrchestrator()
    report_text = orch.generate_report(scenario)
    typer.echo(report_text)


@app.command()
def fetch(
    bbox: Optional[str] = typer.Option(
        None,
        "--bbox",
        help="Bounding box as 'west,south,east,north' in WGS84 degrees (EPSG:4326)"
    ),
    place: Optional[str] = typer.Option(
        None,
        "--place",
        help="OSM place query string (e.g. 'Gachibowli, Hyderabad, India')"
    ),
    out_dir: Path = typer.Option(
        Path("data/d1_graph"),
        "--out",
        help="Destination directory for D1 artifacts (edges.gpkg, nodes.gpkg, osm_snapshot.json)"
    ),
    timeout: int = typer.Option(10, "--timeout", help="Overpass request timeout in seconds"),
    offline: bool = typer.Option(False, "--offline", help="Generate calibrated offline network without external Overpass API request"),
) -> None:
    """Fetch drivable road network from OSM, project to local UTM, clean, and export Contract D1."""
    from irn.m1_fetcher.osm_fetcher import OSMFetcher

    fetcher = OSMFetcher(timeout=timeout)

    if bbox:
        try:
            parts = [float(x.strip()) for x in bbox.split(",")]
            if len(parts) != 4:
                raise ValueError("Must provide 4 float values: west,south,east,north")
            bbox_tuple = (parts[0], parts[1], parts[2], parts[3])
        except Exception as e:
            typer.secho(f"Invalid bbox parameter: {e}", fg=typer.colors.RED)
            raise typer.Exit(code=1)

        typer.secho(f"Acquiring network for bbox: {bbox_tuple}...", fg=typer.colors.CYAN)
        if offline:
            G = fetcher.generate_offline_grid(bbox_tuple)
        else:
            G = fetcher.fetch_by_bbox(bbox_tuple)
        query_meta = {"bbox": list(bbox_tuple), "offline": offline}
    elif place:
        typer.secho(f"Fetching network for place: '{place}'...", fg=typer.colors.CYAN)
        G = fetcher.fetch_by_place(place)
        query_meta = {"place": place}
    else:
        # Default bounding box: Gachibowli / IIIT-H corridor
        bbox_tuple = (78.345, 17.440, 78.360, 17.452)
        typer.secho(f"No area specified. Using default IIIT-H bbox: {bbox_tuple}...", fg=typer.colors.YELLOW)
        if offline:
            G = fetcher.generate_offline_grid(bbox_tuple)
        else:
            G = fetcher.fetch_by_bbox(bbox_tuple)
        query_meta = {"bbox": list(bbox_tuple), "default": True, "offline": offline}

    res = fetcher.process_and_export(G, out_dir, query_metadata=query_meta)
    typer.secho(f"D1 Road Graph Exported Successfully to {out_dir}", fg=typer.colors.GREEN)
    typer.echo(f"  - Edges:    {res['edges_gpkg']}")
    typer.echo(f"  - Nodes:    {res['nodes_gpkg']}")
    typer.echo(f"  - Snapshot: {res['osm_snapshot_json']}")
    typer.echo(f"  - Projected UTM: EPSG:{res['utm_epsg']}")


@app.command()
def fuse(
    graph: Path = typer.Option(
        Path("data/d1_graph/edges.gpkg"),
        "--graph",
        help="Path to D1 edges.gpkg"
    ),
    detections: Path = typer.Option(
        Path("data/d2_detections/detections.parquet"),
        "--detections",
        help="Path to D2 detections.parquet"
    ),
    gps: Path = typer.Option(
        Path("data/synthetic/d2_data/gps.csv"),
        "--gps",
        help="Path to GPS log CSV (gps.csv)"
    ),
    rig: Path = typer.Option(
        Path("configs/rig.yaml"),
        "--rig",
        help="Path to rig calibration YAML"
    ),
    out_d3: Path = typer.Option(
        Path("data/d3_store/anomalies.db"),
        "--out-d3",
        help="Path to SQLite ledger output (Contract D3)"
    ),
    out_d4: Path = typer.Option(
        Path("data/d4_enriched"),
        "--out-d4",
        help="Output directory for enriched GeoParquet (Contract D4)"
    ),
    utm_zone: int = typer.Option(44, "--utm-zone", help="UTM zone number (default 44 for Hyderabad)"),
) -> None:
    """Execute M3 Geospatial Fusion pipeline: ray projection, HMM map matching, (s,t) referencing, and de-duplication."""
    from irn.m3_fusion.pipeline import run_geospatial_fusion

    typer.secho("Launching M3 Geospatial Fusion Engine...", fg=typer.colors.CYAN)
    res = run_geospatial_fusion(
        edges_gpkg_path=graph,
        detections_parquet_path=detections,
        gps_csv_path=gps,
        rig_config_path=rig,
        utm_zone=utm_zone,
        output_d3_db=out_d3,
        output_d4_dir=out_d4,
    )
    typer.secho("M3 Fusion Complete!", fg=typer.colors.GREEN)
    typer.echo(f"  - D3 Ledger (SQLite):   {res['d3_db']}")
    typer.echo(f"  - D4 Enriched Edges:    {res['d4_edges_parquet']}")
    typer.echo(f"  - D4 Anomalies Parquet: {res['d4_anomalies_parquet']}")
    typer.echo(f"  - Total Observations:   {res['num_raw_observations']}")
    typer.echo(f"  - Canonical Anomalies:  {res['num_canonical_anomalies']} ({res['num_confirmed']} confirmed)")


@app.command()
def compile(
    enriched: Path = typer.Option(
        Path("data/d4_enriched/enriched_edges.parquet"),
        "--enriched",
        help="Path to D4 enriched_edges.parquet"
    ),
    anomalies: Path = typer.Option(
        Path("data/d4_enriched/anomalies.parquet"),
        "--anomalies",
        help="Path to D4 anomalies.parquet"
    ),
    nodes: Path = typer.Option(
        Path("data/synthetic/d1_graph/nodes.gpkg"),
        "--nodes",
        help="Path to D1 nodes.gpkg"
    ),
    out_dir: Path = typer.Option(
        Path("output_scenario"),
        "--out",
        help="Destination directory for scenario bundle"
    ),
    lanefree_mode: str = typer.Option(
        "collapse",
        "--lanefree-mode",
        help="Sublane mode: 'collapse' for continuous width or 'lanes'"
    ),
    utm_zone: int = typer.Option(44, "--utm-zone", help="UTM zone number"),
) -> None:
    """Compile enriched network into Eclipse SUMO and ASAM OpenDRIVE 1.6.1 scenario."""
    from irn.m4_compiler.compiler import compile_simulation_scenario

    typer.secho("Compiling simulation scenario bundle...", fg=typer.colors.CYAN)
    res = compile_simulation_scenario(
        enriched_edges_path=enriched,
        anomalies_parquet_path=anomalies,
        nodes_gpkg_path=nodes,
        out_dir=out_dir,
        lanefree_mode=lanefree_mode,
        utm_zone=utm_zone
    )
    typer.secho(f"Scenario compiled successfully in {out_dir}", fg=typer.colors.GREEN)
    typer.echo(f"  - SUMO Net:       {res['network_net_xml']}")
    typer.echo(f"  - OpenDRIVE:      {res['network_xodr']}")
    typer.echo(f"  - Fleet Profiles: {res['vtypes_add_xml']}")
    typer.echo(f"  - Anomalies:      {res['anomalies_add_xml']}")
    typer.echo(f"  - SUMO Config:    {res['scenario_sumocfg']}")
    typer.echo(f"  - Preview:        {res['preview_geojson']}")


@app.command()
def detect(
    sequence_id: str = typer.Option("seq_demo_01", "--seq", help="Identifier for IDD drive sequence"),
    out_file: Path = typer.Option(
        Path("data/d2_detections/detections.parquet"),
        "--out",
        help="Destination path for Contract D2 detections.parquet"
    ),
    frames: int = typer.Option(60, "--frames", help="Number of frames to generate in mock mode"),
) -> None:
    """Run detection engine (or mock generator) and export Contract D2 detections.parquet."""
    from irn.m2_cv_interface.detector import MockDetector

    typer.secho(f"Running detection ingestion for sequence '{sequence_id}'...", fg=typer.colors.CYAN)
    detector = MockDetector(sequence_id=sequence_id)
    records = detector.generate_synthetic_drive_detections(num_frames=frames)

    from irn.m2_cv_interface.schema import save_detections_parquet
    out_path = save_detections_parquet(records, out_file)
    typer.secho(f"D2 Detections Exported Successfully ({len(records)} items) to {out_path}", fg=typer.colors.GREEN)


@app.command("ingest-idd")
def ingest_idd(
    seq_dir: Path = typer.Option(
        Path(r"D:\final\DATASET\Primary\idd_multimodal\primary\d0"),
        "--seq-dir",
        help="Path to IDD Multimodal sequence directory (e.g. primary/d0)"
    ),
    out_dir: Path = typer.Option(
        Path("data/idd_d0"),
        "--out",
        help="Destination directory for normalized GPS and D2 Parquet"
    ),
    pothole_model: Optional[Path] = typer.Option(
        Path(r"D:\Major_2\roadscope\backend\models\potholes.pt") if Path(r"D:\Major_2\roadscope\backend\models\potholes.pt").exists() else None,
        "--pothole-model",
        help="Path to trained YOLO pothole detector weights (.pt)"
    ),
    vehicle_model: Optional[Path] = typer.Option(
        Path(r"D:\Major_2\roadscope\backend\yolo11n.pt") if Path(r"D:\Major_2\roadscope\backend\yolo11n.pt").exists() else None,
        "--vehicle-model",
        help="Path to YOLO vehicle detector weights (.pt)"
    ),
    sample_step: int = typer.Option(10, "--sample-step", help="Subsample video frames by this factor"),
    max_frames: Optional[int] = typer.Option(150, "--max-frames", help="Maximum frames to run inference on"),
    seq_id: str = typer.Option("idd_d0", "--seq-id", help="Sequence ID identifier"),
) -> None:
    """Ingest real IDD Multimodal driving sequence, run YOLO inference, and export D2 parquet + normalized GPS."""
    from irn.m2_cv_interface.idd_adapter import IDDAdapter

    typer.secho(f"Ingesting IDD Multimodal sequence from: {seq_dir}", fg=typer.colors.CYAN)
    adapter = IDDAdapter(sequence_id=seq_id)
    res = adapter.process_sequence(
        sequence_dir=seq_dir,
        out_dir=out_dir,
        pothole_weights=pothole_model,
        vehicle_weights=vehicle_model,
        sample_step=sample_step,
        max_frames=max_frames,
    )
    typer.secho("IDD Multimodal Ingestion Successful!", fg=typer.colors.GREEN)
    typer.echo(f"  - Sequence ID:    {res['sequence_id']}")
    typer.echo(f"  - Trajectory Bbox: {res['bbox']}")
    typer.echo(f"  - Normalized GPS: {res['gps_csv']} ({res['num_gps_fixes']} fixes)")
    typer.echo(f"  - D2 Detections:  {res['detections_parquet']} ({res['num_detections']} records)")
    typer.secho(f"\nNext Steps to run full pipeline on this sequence:", fg=typer.colors.YELLOW)
    bbox_str = f"{res['bbox'][0]},{res['bbox'][1]},{res['bbox'][2]},{res['bbox'][3]}"
    typer.echo(f"  1. Fetch OSM Graph:  irn fetch --bbox \"{bbox_str}\" --out {out_dir}/d1_graph")
    typer.echo(f"  2. Geospatial Fusion: irn fuse --graph {out_dir}/d1_graph/edges.gpkg --detections {res['detections_parquet']} --gps {res['gps_csv']} --out-d3 {out_dir}/d3_store/anomalies.db --out-d4 {out_dir}/d4_enriched --utm-zone 44")
    typer.echo(f"  3. Compile Scenario:  irn compile --enriched {out_dir}/d4_enriched/enriched_edges.parquet --anomalies {out_dir}/d4_enriched/anomalies.parquet --nodes {out_dir}/d1_graph/nodes.gpkg --out output_{seq_id} --utm-zone 44")


@app.command("batch-idd")
def batch_idd(
    dataset_dir: Path = typer.Option(
        Path(r"D:\final\DATASET\Primary\idd_multimodal\primary"),
        "--dataset-dir",
        help="Root path containing IDD sequence directories (d0, d1, d2...)"
    ),
    out_dir: Path = typer.Option(
        Path("scenarios"),
        "--out",
        help="Base directory for output scenarios and multi-pass corridor bundle"
    ),
    sequences: Optional[str] = typer.Option(
        None,
        "--sequences",
        help="Comma-separated sequence names to process (e.g. 'd0,d1,d2'). Default: all found"
    ),
    pothole_model: Optional[Path] = typer.Option(
        Path(r"D:\Major_2\roadscope\backend\models\potholes.pt") if Path(r"D:\Major_2\roadscope\backend\models\potholes.pt").exists() else None,
        "--pothole-model",
        help="Path to trained YOLO pothole detector weights (.pt)"
    ),
    vehicle_model: Optional[Path] = typer.Option(
        Path(r"D:\Major_2\roadscope\backend\yolo11n.pt") if Path(r"D:\Major_2\roadscope\backend\yolo11n.pt").exists() else None,
        "--vehicle-model",
        help="Path to YOLO vehicle detector weights (.pt)"
    ),
    sample_step: int = typer.Option(10, "--sample-step", help="Subsample video frames by this factor"),
    max_frames: Optional[int] = typer.Option(100, "--max-frames", help="Maximum frames per sequence to run inference on"),
    unified: bool = typer.Option(True, "--unified/--isolated", help="Enable multi-pass Hungarian cross-pass fusion ledger"),
) -> None:
    """Run end-to-end IRN batch processing across all IDD multimodal sequences with multi-pass fusion."""
    from irn.orchestrator import PipelineOrchestrator

    seq_list = [s.strip() for s in sequences.split(",")] if sequences else None
    orch = PipelineOrchestrator()
    typer.secho(f"Starting IDD Batch Pipeline for dataset: {dataset_dir}", fg=typer.colors.CYAN)
    
    batch_res = orch.run_idd_batch(
        dataset_dir=dataset_dir,
        output_base_dir=out_dir,
        sequences=seq_list,
        pothole_weights=pothole_model,
        vehicle_weights=vehicle_model,
        sample_step=sample_step,
        max_frames=max_frames,
        unified_multi_pass=unified,
    )

    typer.secho("\n==================================================", fg=typer.colors.GREEN)
    typer.secho("IDD MULTI-SEQUENCE BATCH PROCESSING COMPLETE!", fg=typer.colors.GREEN)
    typer.secho("==================================================", fg=typer.colors.GREEN)
    for seq, res in batch_res["sequence_results"].items():
        st = res["manifest"]["statistics"]
        typer.echo(f"  * Sequence {seq}: {st['num_detections_ingested']} dets -> {st['num_canonical_anomalies']} anomalies mapped -> {res['output_dir']}")

    if batch_res.get("corridor_scenario"):
        typer.secho(f"\nUnified Multi-Pass Corridor Scenario compiled in: {out_dir}/corridor_multi_pass", fg=typer.colors.YELLOW)
        typer.echo(f"  - OpenDRIVE: {out_dir}/corridor_multi_pass/network.xodr")
        typer.echo(f"  - SUMO Config: {out_dir}/corridor_multi_pass/scenario.sumocfg")
        typer.echo(f"  - Preview: {out_dir}/corridor_multi_pass/preview.geojson")


@app.command()
def synth(
    out_dir: Path = typer.Option(Path("data/synthetic"), "--out", help="Output directory for synthetic world"),
    length: float = typer.Option(100.0, "--length", help="Length of road segment in meters"),
) -> None:
    """Generate mathematically verified synthetic ground truth world (NFR-QA-01)."""
    from tests.fixtures.synthetic_world import SyntheticWorld

    typer.secho(f"Building synthetic world of length {length}m...", fg=typer.colors.CYAN)
    world = SyntheticWorld(road_length_m=length)
    d1_res = world.build_synthetic_graph(out_dir / "d1_graph")
    gps_path, d2_path = world.generate_trajectory_and_detections(out_dir / "d2_data")

    typer.secho(f"Synthetic World generated at {out_dir}", fg=typer.colors.GREEN)
    typer.echo(f"  - Graph Edges:  {d1_res['edges_gpkg']}")
    typer.echo(f"  - Detections:   {d2_path}")
    typer.echo(f"  - Vehicle GPS:  {gps_path}")


if __name__ == "__main__":
    app()
