"""Module 4 Master Simulation Compiler Coordinator.

Assembles the complete ready-to-run simulation scenario bundle:
- network.net.xml (via Plain-XML + netconvert --lefthand=true)
- network.xodr (ASAM OpenDRIVE 1.6.1)
- vtypes.add.xml (Indian vehicle profiles)
- anomalies.add.xml (Dynamic obstacles & speed zones)
- scenario.sumocfg (Sublane Model configuration)
- preview.geojson (for instant QGIS validation)
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import geopandas as gpd
import pandas as pd
from shapely import wkt
from shapely.geometry import mapping

from irn.common.logger import log_event
from irn.m4_compiler.anomalies_generator import AnomaliesAddGenerator
from irn.m4_compiler.netconvert_runner import NetconvertRunner
from irn.m4_compiler.sumo_config_generator import SumoConfigGenerator
from irn.m4_compiler.sumo_plain_xml import SumoPlainXmlGenerator
from irn.m4_compiler.vtypes_generator import VTypesGenerator
from irn.m4_compiler.xodr_builder import OpenDriveBuilder


def compile_simulation_scenario(
    enriched_edges_path: Union[str, Path],
    anomalies_parquet_path: Union[str, Path],
    nodes_gpkg_path: Union[str, Path],
    out_dir: Union[str, Path],
    lanefree_mode: str = "collapse",
    utm_zone: int = 44,
    is_northern: bool = True
) -> Dict[str, Path]:
    """Compiles all simulation artifacts into output scenario directory."""
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    plain_dir = out / "plain_xml"
    plain_dir.mkdir(parents=True, exist_ok=True)

    log_event(f"Compiling simulation scenario into {out}", reason_code="M4-COMPILE-START")

    # 1. Plain-XML network generation
    plain_gen = SumoPlainXmlGenerator(lanefree_mode=lanefree_mode)
    nod_file, edg_file = plain_gen.generate_plain_xml(
        enriched_edges_path=enriched_edges_path,
        nodes_gpkg_path=nodes_gpkg_path,
        out_dir=plain_dir
    )

    # 2. Automated netconvert compilation
    net_file = out / "network.net.xml"
    runner = NetconvertRunner()
    runner.compile_network(
        node_file=nod_file,
        edge_file=edg_file,
        output_net_file=net_file,
        lefthand=True
    )

    # 3. Indian vehicle definitions
    vtypes_gen = VTypesGenerator()
    vtypes_file = vtypes_gen.generate_vtypes_xml(out / "vtypes.add.xml")

    # 4. Anomaly realization
    anom_gen = AnomaliesAddGenerator()
    anom_file = anom_gen.generate_anomalies_xml(anomalies_parquet_path, out / "anomalies.add.xml")

    # 5. Master Sublane scenario.sumocfg
    cfg_gen = SumoConfigGenerator(lateral_resolution_m=0.6, step_length_s=0.1)
    sumocfg_file = cfg_gen.generate_sumocfg(
        net_filename="network.net.xml",
        additional_files="vtypes.add.xml,anomalies.add.xml",
        output_file=out / "scenario.sumocfg"
    )

    # 6. ASAM OpenDRIVE 1.6.1 Exporter
    edges_df = pd.read_parquet(enriched_edges_path)
    anom_df = pd.read_parquet(anomalies_parquet_path)
    xodr_builder = OpenDriveBuilder(utm_zone=utm_zone, is_northern=is_northern, traffic_rule="LHT")
    xodr_file = xodr_builder.build_xodr(edges_df, anom_df, out / "network.xodr")

    # 7. Generate preview.geojson (reprojected to standard WGS84 EPSG:4326 for GIS/web viewers)
    from pyproj import Transformer
    from shapely.ops import transform
    utm_epsg = 32600 + utm_zone if is_northern else 32700 + utm_zone
    to_wgs84 = Transformer.from_crs(f"EPSG:{utm_epsg}", "EPSG:4326", always_xy=True).transform

    preview_file = out / "preview.geojson"
    features = []
    edge_geom_map = {}
    for _, r in edges_df.iterrows():
        geom_wkt = r.get("geometry_wkt", None)
        if geom_wkt:
            try:
                g_utm = wkt.loads(geom_wkt)
                edge_geom_map[str(r.get("edge_id", ""))] = g_utm
                g_wgs84 = transform(to_wgs84, g_utm)
                features.append({
                    "type": "Feature",
                    "geometry": mapping(g_wgs84),
                    "properties": {
                        "edge_id": str(r.get("edge_id", "")),
                        "width_dir_m": float(r.get("width_dir_m", 7.0)),
                        "lanes_effective": int(r.get("lanes_effective", 1))
                    }
                })
            except Exception:
                pass

    # Include confirmed road anomalies as Point markers
    for _, a in anom_df.iterrows():
        eid = str(a.get("edge_id", ""))
        s = float(a.get("s", 0.0))
        if eid in edge_geom_map:
            edge_geom = edge_geom_map[eid]
            frac = max(0.0, min(1.0, s / max(1.0, edge_geom.length)))
            pt_utm = edge_geom.interpolate(frac, normalized=True)
            pt_wgs84 = transform(to_wgs84, pt_utm)
            features.append({
                "type": "Feature",
                "geometry": mapping(pt_wgs84),
                "properties": {
                    "anomaly_id": str(a.get("anomaly_id", "")),
                    "class_name": str(a.get("class_name", "pothole")),
                    "confidence": round(float(a.get("confidence", 0.8)), 3),
                    "marker-color": "#e11d48",
                    "marker-symbol": "danger"
                }
            })

    geojson_data = {"type": "FeatureCollection", "features": features}
    preview_file.write_text(json.dumps(geojson_data, indent=2), encoding="utf-8")

    log_event("Simulation Scenario Compilation Complete!", reason_code="M4-COMPILE-DONE")

    return {
        "network_net_xml": net_file,
        "network_xodr": xodr_file,
        "vtypes_add_xml": vtypes_file,
        "anomalies_add_xml": anom_file,
        "scenario_sumocfg": sumocfg_file,
        "preview_geojson": preview_file,
    }
