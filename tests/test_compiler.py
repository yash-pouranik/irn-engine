"""Verification Tests for Module 4 Simulation Compiler (SUMO & OpenDRIVE)."""

from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd
import pytest

from irn.m4_compiler.anomalies_generator import AnomaliesAddGenerator
from irn.m4_compiler.compiler import compile_simulation_scenario
from irn.m4_compiler.sumo_config_generator import SumoConfigGenerator
from irn.m4_compiler.sumo_plain_xml import SumoPlainXmlGenerator
from irn.m4_compiler.vtypes_generator import VTypesGenerator
from irn.m4_compiler.xodr_builder import OpenDriveBuilder


class TestSumoPlainXmlGenerator:
    """Verifies Plain-XML node and edge file generation and collapse mode."""

    def test_plain_xml_structure(self, tmp_path):
        gen = SumoPlainXmlGenerator(lanefree_mode="collapse")

        edges_df = pd.DataFrame([{
            "edge_id": "1001_1002_0",
            "u": "1001",
            "v": "1002",
            "maxspeed_kph": 50.0,
            "width_dir_m": 7.5,
            "lanes_effective": 2,
            "geometry_wkt": "LINESTRING (500000.0 2000000.0, 500000.0 2000100.0)"
        }])
        edges_parquet = tmp_path / "test_edges.parquet"
        edges_df.to_parquet(edges_parquet)

        # Nodes GeoDataFrame
        import geopandas as gpd
        from shapely.geometry import Point
        nodes_gdf = gpd.GeoDataFrame([
            {"node_id": "1001", "x_utm": 500000.0, "y_utm": 2000000.0, "highway": "priority", "geometry": Point(500000, 2000000)},
            {"node_id": "1002", "x_utm": 500000.0, "y_utm": 2000100.0, "highway": "traffic_signals", "geometry": Point(500000, 2000100)},
        ], crs="EPSG:32644")
        nodes_gpkg = tmp_path / "test_nodes.gpkg"
        nodes_gdf.to_file(nodes_gpkg, layer="nodes", driver="GPKG")

        nod_file, edg_file = gen.generate_plain_xml(edges_parquet, nodes_gpkg, tmp_path / "plain")

        # Verify XML well-formedness
        tree_nod = ET.parse(nod_file)
        root_nod = tree_nod.getroot()
        assert root_nod.tag == "nodes"
        assert len(root_nod.findall("node")) == 2
        # Verify traffic light type
        tl_node = root_nod.find(".//node[@id='1002']")
        assert tl_node is not None
        assert tl_node.attrib["type"] == "traffic_light"

        tree_edg = ET.parse(edg_file)
        root_edg = tree_edg.getroot()
        assert root_edg.tag == "edges"
        edge_elem = root_edg.find(".//edge[@id='1001_1002_0']")
        assert edge_elem is not None
        # In collapse mode, numLanes is 1 with wide carriageway (7.5m)
        assert edge_elem.attrib["numLanes"] == "1"
        assert float(edge_elem.attrib["width"]) == 7.5
        assert pytest.approx(float(edge_elem.attrib["speed"]), abs=0.1) == 13.89  # 50 km/h in m/s


class TestVTypesGenerator:
    """Verifies heterogeneous Indian vehicle definitions and Sublane parameters."""

    def test_vtypes_generation(self, tmp_path):
        gen = VTypesGenerator()
        vtypes_file = gen.generate_vtypes_xml(tmp_path / "vtypes.add.xml")

        tree = ET.parse(vtypes_file)
        root = tree.getroot()
        assert root.tag == "additional"

        # Check two_wheeler
        tw = root.find(".//vType[@id='two_wheeler']")
        assert tw is not None
        assert tw.attrib["latAlignment"] == "arbitrary"
        assert tw.attrib["minGapLat"] == "0.2"

        # Check auto_rickshaw
        auto = root.find(".//vType[@id='auto_rickshaw']")
        assert auto is not None
        assert auto.attrib["latAlignment"] == "arbitrary"
        assert auto.attrib["width"] == "1.4"


class TestAnomaliesAndSublaneConfig:
    """Verifies anomaly realization and Sublane lateral resolution."""

    def test_anomalies_add_xml(self, tmp_path):
        gen = AnomaliesAddGenerator()
        df = pd.DataFrame([{
            "anomaly_id": "anom_uuid_12345",
            "edge_id": "1001_1002_0",
            "class_name": "pothole",
            "s": 25.0,
            "t": -1.2
        }])
        out = gen.generate_anomalies_xml(df, tmp_path / "anomalies.add.xml")

        tree = ET.parse(out)
        root = tree.getroot()
        assert root.tag == "additional"
        poi = root.find(".//poi")
        assert poi is not None
        assert poi.attrib["type"] == "pothole"
        assert pytest.approx(float(poi.attrib["pos"]), abs=0.1) == 25.0
        assert pytest.approx(float(poi.attrib["posLat"]), abs=0.1) == -1.2

    def test_scenario_sumocfg(self, tmp_path):
        cfg_gen = SumoConfigGenerator(lateral_resolution_m=0.6, step_length_s=0.1)
        sumocfg = cfg_gen.generate_sumocfg(output_file=tmp_path / "scenario.sumocfg")

        tree = ET.parse(sumocfg)
        root = tree.getroot()
        assert root.tag == "configuration"
        lat_res = root.find(".//lateral-resolution")
        assert lat_res is not None
        assert float(lat_res.attrib["value"]) == 0.6


class TestOpenDriveBuilder:
    """Verifies ASAM OpenDRIVE 1.6.1 XML export with Left-Hand Traffic (Rule G-2)."""

    def test_xodr_export_lht_and_objects(self, tmp_path):
        builder = OpenDriveBuilder(utm_zone=44, is_northern=True, traffic_rule="LHT")

        edges_df = pd.DataFrame([{
            "edge_id": "road_101",
            "length_m": 100.0,
            "width_dir_m": 7.5,
            "geometry_wkt": "LINESTRING (500000.0 2000000.0, 500000.0 2000100.0)"
        }])
        anom_df = pd.DataFrame([{
            "edge_id": "road_101",
            "class_name": "pothole",
            "s": 50.0,
            "t": 0.8
        }])

        xodr_file = builder.build_xodr(edges_df, anom_df, tmp_path / "network.xodr")

        tree = ET.parse(xodr_file)
        root = tree.getroot()
        assert root.tag == "OpenDRIVE"

        # Check geoReference in header
        geo_ref = root.find(".//header/geoReference")
        assert geo_ref is not None
        assert "+proj=utm +zone=44" in geo_ref.text

        # Check road element declares rule="LHT" (Rule G-2)
        road = root.find(".//road")
        assert road is not None
        assert road.attrib["rule"] == "LHT"

        # Check embedded obstacle object
        obj = root.find(".//objects/object")
        assert obj is not None
        assert obj.attrib["name"] == "pothole"
        assert float(obj.attrib["s"]) == 50.0
        assert float(obj.attrib["t"]) == 0.8
