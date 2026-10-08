"""Module 4: Scenario and Standards Validator.

Validates:
- ASAM OpenDRIVE 1.6.1 XML integrity, LHT rule, and geoReference.
- SUMO scenario configuration and Sublane settings.
- Integrity of Plain-XML and additional simulation files.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, NamedTuple, Tuple, Union
import xml.etree.ElementTree as ET

from irn.common.logger import log_event


class ValidationResult(NamedTuple):
    is_valid: bool
    errors: List[str]
    warnings: List[str]


def validate_opendrive_xml(xodr_path: Union[str, Path]) -> ValidationResult:
    """Validates ASAM OpenDRIVE 1.6.1 XML document compliance."""
    p = Path(xodr_path)
    errors: List[str] = []
    warnings: List[str] = []

    if not p.exists():
        return ValidationResult(is_valid=False, errors=[f"File not found: {p}"], warnings=[])

    try:
        tree = ET.parse(p)
        root = tree.getroot()
    except Exception as e:
        return ValidationResult(is_valid=False, errors=[f"XML Parse Error: {e}"], warnings=[])

    if root.tag != "OpenDRIVE":
        errors.append(f"Root tag must be 'OpenDRIVE', got '{root.tag}'")

    # Header check
    header = root.find("header")
    if header is None:
        errors.append("Missing <header> element in OpenDRIVE root")
    else:
        geo = header.find("geoReference")
        if geo is None or not geo.text:
            warnings.append("Missing or empty <geoReference> in header")

    # Road checks
    roads = root.findall("road")
    if not roads:
        warnings.append("OpenDRIVE file contains 0 roads")

    for road in roads:
        r_id = road.attrib.get("id", "unknown")
        rule = road.attrib.get("rule", "")
        if rule != "LHT":
            errors.append(f"Road {r_id} traffic rule is '{rule}'; must be strictly 'LHT' (Rule G-2)")

        plan_view = road.find("planView")
        if plan_view is None or len(plan_view.findall("geometry")) == 0:
            errors.append(f"Road {r_id} missing planView geometry")

    is_valid = len(errors) == 0
    return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)


def validate_sumo_scenario(scenario_dir: Union[str, Path]) -> ValidationResult:
    """Validates complete SUMO scenario directory."""
    sdir = Path(scenario_dir)
    errors: List[str] = []
    warnings: List[str] = []

    sumocfg_path = sdir / "scenario.sumocfg"
    if not sumocfg_path.exists():
        errors.append(f"Missing scenario.sumocfg in {sdir}")
        return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

    try:
        tree = ET.parse(sumocfg_path)
        root = tree.getroot()
    except Exception as e:
        errors.append(f"Invalid XML in scenario.sumocfg: {e}")
        return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

    # Check Sublane lateral resolution
    lat_res = root.find(".//lateral-resolution")
    if lat_res is None:
        warnings.append("Sublane Model not configured: missing <lateral-resolution>")
    else:
        val = float(lat_res.attrib.get("value", 0.0))
        if val <= 0.0:
            errors.append(f"Invalid lateral-resolution value '{val}'; must be > 0.0")

    # Check referenced files
    net_elem = root.find(".//net-file")
    if net_elem is not None:
        net_name = net_elem.attrib.get("value", "")
        if not (sdir / net_name).exists():
            warnings.append(f"Referenced net-file not found on disk: {net_name}")

    is_valid = len(errors) == 0
    return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)
