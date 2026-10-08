"""Module 3: Geospatial Data Fusion Engine & Mathematical Models."""

from irn.m3_fusion.camera_model import (
    CameraCalibration,
    GroundPlaneProjector,
    ProjectionResult,
)
from irn.m3_fusion.sync import GPSSynchronizer, VehiclePoseState
from irn.m3_fusion.map_matcher import (
    RoadNetworkSpatialIndex,
    HMMMapMatcher,
    MatchedEdgeCandidate,
    HMMMatchedPose,
)
from irn.m3_fusion.linear_ref import LinearReferencer, FrenetObstaclePose
from irn.m3_fusion.deduplication import (
    AnomalyDeduplicator,
    CanonicalAnomaly,
    CHI2_GATE_THRESHOLD,
)
from irn.m3_fusion.width_fusion import fuse_edge_widths, aggregate_dynamic_class_mix
from irn.m3_fusion.storage import FusionStorage
from irn.m3_fusion.pipeline import run_geospatial_fusion

__all__ = [
    "CameraCalibration",
    "GroundPlaneProjector",
    "ProjectionResult",
    "GPSSynchronizer",
    "VehiclePoseState",
    "RoadNetworkSpatialIndex",
    "HMMMapMatcher",
    "MatchedEdgeCandidate",
    "HMMMatchedPose",
    "LinearReferencer",
    "FrenetObstaclePose",
    "AnomalyDeduplicator",
    "CanonicalAnomaly",
    "CHI2_GATE_THRESHOLD",
    "fuse_edge_widths",
    "aggregate_dynamic_class_mix",
    "FusionStorage",
    "run_geospatial_fusion",
]
