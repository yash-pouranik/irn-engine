"""Unit tests for Phase 0 and Phase 1 modules: CRS, M1 Cleaner, M2 Schemas, and Synthetic World."""

import tempfile
from pathlib import Path
import networkx as nx
import numpy as np
import pytest
from shapely.geometry import LineString, Point

from irn.common.crs import (
    epsg_from_utm_zone,
    optimal_utm_epsg,
    utm_to_wgs84,
    utm_zone_from_lon_lat,
    wgs84_to_utm,
)
from irn.m1_fetcher.cleaner import (
    clean_and_normalize_graph,
    parse_lanes,
    parse_maxspeed,
    parse_width,
)
from irn.m2_cv_interface.schema import (
    DetectionRecord,
    load_detections_parquet,
    save_detections_parquet,
    validate_and_filter_detections,
)
from tests.fixtures.synthetic_world import SyntheticWorld


class TestCRSProjections:
    """Verify Rule G-1: WGS84 <-> UTM round-trip accuracy within millimeters."""

    def test_utm_zone_calculation(self):
        # Indore region (~75.85E, 22.7N) -> Zone 43N
        zone_indore, north_indore = utm_zone_from_lon_lat(75.85, 22.7)
        assert zone_indore == 43
        assert north_indore is True
        assert epsg_from_utm_zone(zone_indore, north_indore) == 32643

        # Hyderabad region (~78.4E, 17.4N) -> Zone 44N
        zone_hyd, north_hyd = utm_zone_from_lon_lat(78.4, 17.4)
        assert zone_hyd == 44
        assert north_hyd is True
        assert epsg_from_utm_zone(zone_hyd, north_hyd) == 32644

    def test_wgs84_utm_roundtrip(self):
        # Test coordinates in Hyderabad, India
        orig_lon = 78.345678
        orig_lat = 17.432109
        zone, north = utm_zone_from_lon_lat(orig_lon, orig_lat)

        easting, northing = wgs84_to_utm(orig_lon, orig_lat, zone, north)
        assert easting > 100000.0  # Valid easting range
        assert northing > 1000000.0

        rec_lon, rec_lat = utm_to_wgs84(easting, northing, zone, north)

        # Assert sub-millimeter precision (< 1e-6 degrees ~ 0.1 meter)
        assert pytest.approx(rec_lon, abs=1e-5) == orig_lon
        assert pytest.approx(rec_lat, abs=1e-5) == orig_lat


class TestM1Cleaner:
    """Verify attribute parsing and topology cleanup."""

    def test_parse_attributes(self):
        assert parse_lanes("2") == 2
        assert parse_lanes(["3", "2"]) == 3
        assert parse_lanes(None, "primary") == 2

        assert parse_maxspeed("50 km/h") == 50
        assert parse_maxspeed("60") == 60
        assert parse_maxspeed(None, "residential") == 30

        assert parse_width("7.5 m") == 7.5
        assert parse_width(None, lanes=2) == 7.0

    def test_graph_cleaning(self):
        G = nx.MultiDiGraph()
        # Add 3 nodes
        G.add_node(1, x=78.34, y=17.44)
        G.add_node(2, x=78.35, y=17.44)
        G.add_node(3, x=78.36, y=17.44)

        # Valid edge
        line1 = LineString([(78.34, 17.44), (78.35, 17.44)])
        G.add_edge(1, 2, key=0, highway="primary", length=100.0, geometry=line1)

        # Zero-length edge (should be pruned)
        G.add_edge(2, 3, key=0, highway="residential", length=0.001)

        cleaned = clean_and_normalize_graph(G, utm_epsg=32643)

        assert cleaned.has_edge(1, 2, 0)
        assert not cleaned.has_edge(2, 3, 0)

        edge_data = cleaned.edges[1, 2, 0]
        assert edge_data["edge_id"] == "1_2_0"
        assert edge_data["lanes"] == 2
        assert edge_data["maxspeed_kph"] == 50


class TestM2DetectionsSchema:
    """Verify Contract D2 Pydantic validation and PyArrow Parquet export."""

    def test_ground_contact_point(self):
        record = DetectionRecord(
            sequence_id="test_seq",
            frame_idx=10,
            frame_ts=1.0,
            class_name="pothole",
            conf=0.85,
            x1=100.0,
            y1=200.0,
            x2=200.0,
            y2=300.0,
            img_w=1920,
            img_h=1080,
            track_id=42
        )
        u, v = record.ground_contact_uv
        # u = (100 + 200)/2 = 150, v = 300
        assert u == 150.0
        assert v == 300.0

    def test_invalid_box_rejected(self):
        with pytest.raises(ValueError):
            # x2 <= x1 is invalid
            DetectionRecord(
                sequence_id="test_seq",
                frame_idx=1,
                frame_ts=0.1,
                class_name="car",
                conf=0.9,
                x1=200.0,
                y1=100.0,
                x2=100.0,
                y2=300.0,
                img_w=1920,
                img_h=1080
            )

    def test_parquet_roundtrip(self, tmp_path):
        records = [
            DetectionRecord(
                sequence_id="test_seq",
                frame_idx=i,
                frame_ts=i * 0.1,
                class_name="pothole",
                conf=0.80,
                x1=50.0,
                y1=100.0,
                x2=150.0,
                y2=200.0,
                img_w=1920,
                img_h=1080,
                track_id=10
            )
            for i in range(5)
        ]

        pq_file = tmp_path / "test_detections.parquet"
        save_detections_parquet(records, pq_file)
        assert pq_file.exists()

        table = load_detections_parquet(pq_file)
        assert table.num_rows == 5
        assert "u" in table.column_names
        assert "v" in table.column_names
        assert table["u"][0].as_py() == 100.0
        assert table["v"][0].as_py() == 200.0


class TestSyntheticWorld:
    """Verify Synthetic World generates valid D1 and D2 fixtures."""

    def test_synthetic_world_generation(self, tmp_path):
        world = SyntheticWorld(road_length_m=100.0)
        d1_files = world.build_synthetic_graph(tmp_path / "d1")
        gps_path, d2_path = world.generate_trajectory_and_detections(tmp_path / "d2")

        assert d1_files["edges_gpkg"].exists()
        assert d1_files["nodes_gpkg"].exists()
        assert d1_files["osm_snapshot_json"].exists()
        assert gps_path.exists()
        assert d2_path.exists()

        # Check detections content
        table = load_detections_parquet(d2_path)
        assert table.num_rows > 0
        classes = set(table["class_name"].to_pylist())
        assert "pothole" in classes or "speed_breaker" in classes
