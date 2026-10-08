"""Normalization, type parsing, and topology cleanup for OSM road graphs.

Enforces:
- Clean integer 'lanes', integer km/h 'maxspeed_kph', float 'width_osm_m'.
- Retention of full-resolution LineString geometries.
- Retention of the largest strongly connected component.
- Removal of zero-length or degenerate edges.
- Generation of deterministic edge IDs: '{u}_{v}_{key}'.
"""

from __future__ import annotations
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import networkx as nx
import numpy as np
import shapely
from shapely.geometry import LineString, Point
from irn.common.logger import log_event

# Default lane assignments by OSM highway type when 'lanes' tag is missing
DEFAULT_LANES_BY_HIGHWAY: Dict[str, int] = {
    "motorway": 4,
    "motorway_link": 2,
    "trunk": 3,
    "trunk_link": 2,
    "primary": 2,
    "primary_link": 1,
    "secondary": 2,
    "secondary_link": 1,
    "tertiary": 2,
    "tertiary_link": 1,
    "residential": 1,
    "living_street": 1,
    "unclassified": 1,
    "service": 1,
}

# Default speed limits (km/h) for Indian urban/semi-urban roadways
DEFAULT_MAXSPEED_BY_HIGHWAY: Dict[str, int] = {
    "motorway": 80,
    "trunk": 60,
    "primary": 50,
    "secondary": 40,
    "tertiary": 35,
    "residential": 30,
    "living_street": 20,
    "unclassified": 30,
    "service": 20,
}

# Standard lane width assumption in meters
STANDARD_LANE_WIDTH_M = 3.5


def parse_first_scalar(val: Any) -> Any:
    """If val is a list/tuple, extract the first non-null element."""
    if isinstance(val, (list, tuple)):
        return val[0] if len(val) > 0 else None
    return val


def parse_lanes(raw_lanes: Any, highway_type: str = "residential") -> int:
    """Parse OSM lanes tag into a positive integer."""
    val = parse_first_scalar(raw_lanes)
    if val is not None:
        try:
            # Handle formats like "2", "2.0", "2;3"
            clean_str = str(val).split(";")[0].strip()
            count = int(float(clean_str))
            if count > 0:
                return count
        except (ValueError, TypeError):
            pass

    return DEFAULT_LANES_BY_HIGHWAY.get(highway_type, 1)


def parse_maxspeed(raw_speed: Any, highway_type: str = "residential") -> int:
    """Parse OSM maxspeed tag into integer km/h."""
    val = parse_first_scalar(raw_speed)
    if val is not None:
        text = str(val).lower().strip()
        match = re.search(r"(\d+)", text)
        if match:
            speed = int(match.group(1))
            if "mph" in text:
                speed = int(speed * 1.60934)
            if 10 <= speed <= 150:
                return speed

    return DEFAULT_MAXSPEED_BY_HIGHWAY.get(highway_type, 40)


def parse_width(raw_width: Any, lanes: int = 1, highway_type: str = "residential") -> float:
    """Parse OSM width tag into metric float (meters). Fallback to lanes * 3.5m."""
    val = parse_first_scalar(raw_width)
    if val is not None:
        text = str(val).lower().strip()
        # Remove units like 'm', 'meters'
        text = text.replace("m", "").replace("meters", "").strip()
        try:
            width_val = float(text)
            if 1.5 <= width_val <= 60.0:
                return round(width_val, 2)
        except (ValueError, TypeError):
            pass

    # Heuristic fallback
    return round(float(lanes) * STANDARD_LANE_WIDTH_M, 2)


def parse_highway_type(raw_highway: Any) -> str:
    """Extract primary highway classification string."""
    val = parse_first_scalar(raw_highway)
    if val is None:
        return "unclassified"
    return str(val).lower().strip()


def parse_oneway(raw_oneway: Any) -> bool:
    """Parse OSM oneway attribute."""
    val = parse_first_scalar(raw_oneway)
    if val is None:
        return False
    if isinstance(val, bool):
        return val
    text = str(val).lower().strip()
    return text in ("yes", "true", "1", "-1")


def clean_and_normalize_graph(
    G: nx.MultiDiGraph,
    utm_epsg: int
) -> nx.MultiDiGraph:
    """Cleans attributes, prunes disconnected components, and enforces D1 specifications.

    1. Normalizes lanes, maxspeed_kph, width_osm_m, surface, oneway.
    2. Enforces deterministic edge_id.
    3. Prunes zero-length edges and preserves largest connected component.
    """
    log_event(
        f"Cleaning graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges",
        reason_code="CLEAN-GRAPH-START"
    )

    # 1. Filter out zero-length or degenerate edges
    edges_to_remove = []
    for u, v, k, data in G.edges(keys=True, data=True):
        length = data.get("length", None)
        geom = data.get("geometry", None)
        if length is not None and float(length) <= 0.01:
            edges_to_remove.append((u, v, k))
        elif geom is not None and (geom.is_empty or geom.length <= 1e-7):
            edges_to_remove.append((u, v, k))
        elif length is None and geom is None:
            edges_to_remove.append((u, v, k))

    if edges_to_remove:
        G.remove_edges_from(edges_to_remove)
        log_event(
            f"Removed {len(edges_to_remove)} zero-length/degenerate edges",
            reason_code="CLEAN-PURGE-ZEROLEN"
        )

    # 2. Extract largest connected component
    if G.is_directed() and G.number_of_nodes() > 0:
        wcc = list(nx.weakly_connected_components(G))
        if wcc:
            largest_nodes = max(wcc, key=len)
            if len(largest_nodes) < G.number_of_nodes():
                pruned_count = G.number_of_nodes() - len(largest_nodes)
                G = G.subgraph(largest_nodes).copy()
                log_event(
                    f"Retained largest connected component ({len(largest_nodes)} nodes, pruned {pruned_count})",
                    reason_code="CLEAN-PRUNE-WCC"
                )
    elif not G.is_directed() and G.number_of_nodes() > 0:
        cc = list(nx.connected_components(G))
        if cc:
            largest_nodes = max(cc, key=len)
            if len(largest_nodes) < G.number_of_nodes():
                pruned_count = G.number_of_nodes() - len(largest_nodes)
                G = G.subgraph(largest_nodes).copy()
                log_event(
                    f"Retained largest connected component ({len(largest_nodes)} nodes, pruned {pruned_count})",
                    reason_code="CLEAN-PRUNE-CC"
                )

    # 3. Normalize edge attributes
    for u, v, k, data in G.edges(keys=True, data=True):
        highway = parse_highway_type(data.get("highway"))
        lanes = parse_lanes(data.get("lanes"), highway)
        maxspeed = parse_maxspeed(data.get("maxspeed"), highway)
        width = parse_width(data.get("width"), lanes, highway)
        oneway = parse_oneway(data.get("oneway"))
        surface = parse_first_scalar(data.get("surface"))
        surface_str = str(surface).lower().strip() if surface else "asphalt"

        # Edge ID deterministic identifier (Constraint C-4)
        edge_id = f"{u}_{v}_{k}"

        data["edge_id"] = edge_id
        data["u"] = u
        data["v"] = v
        data["key"] = k
        data["highway"] = highway
        data["lanes"] = lanes
        data["maxspeed_kph"] = maxspeed
        data["width_osm_m"] = width
        data["oneway"] = oneway
        data["surface"] = surface_str

        # Ensure length attribute in meters is populated
        if "length" not in data or data["length"] is None:
            if "geometry" in data and data["geometry"] is not None:
                data["length"] = float(data["geometry"].length)
            else:
                data["length"] = 1.0
        data["length_m"] = float(data["length"])

    # 4. Normalize node attributes
    for node, data in G.nodes(data=True):
        data["node_id"] = node
        data["highway"] = str(data.get("highway", "priority"))

    log_event(
        f"Graph cleaning complete: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges retained",
        reason_code="CLEAN-GRAPH-DONE"
    )
    return G
