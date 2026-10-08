"""Module 4: Heterogeneous Indian Vehicle Definition Generator (vtypes.add.xml).

Defines realistic Indian vehicle profiles with Sublane Model attributes:
- Two-wheelers: lateral-alignment="arbitrary", minGapLat="0.2m".
- Auto-rickshaws: compact 3-wheeler footprint, arbitrary lateral placement.
- Passenger cars, buses, and commercial trucks.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Union

from irn.common.logger import log_event


DEFAULT_INDIAN_VEHICLE_TYPES = [
    {
        "id": "two_wheeler",
        "vClass": "motorcycle",
        "length": "1.9",
        "width": "0.7",
        "height": "1.2",
        "minGap": "1.0",
        "minGapLat": "0.2",
        "lateral_alignment": "arbitrary",
        "maxSpeed": "22.2",   # 80 km/h
        "accel": "3.5",
        "decel": "4.5",
        "color": "red"
    },
    {
        "id": "auto_rickshaw",
        "vClass": "passenger",
        "length": "2.7",
        "width": "1.4",
        "height": "1.8",
        "minGap": "1.2",
        "minGapLat": "0.3",
        "lateral_alignment": "arbitrary",
        "maxSpeed": "13.9",   # 50 km/h
        "accel": "2.2",
        "decel": "3.5",
        "color": "yellow"
    },
    {
        "id": "car",
        "vClass": "passenger",
        "length": "4.2",
        "width": "1.7",
        "height": "1.5",
        "minGap": "2.0",
        "minGapLat": "0.4",
        "lateral_alignment": "center",
        "maxSpeed": "27.8",   # 100 km/h
        "accel": "2.6",
        "decel": "4.5",
        "color": "white"
    },
    {
        "id": "bus",
        "vClass": "bus",
        "length": "10.5",
        "width": "2.5",
        "height": "3.2",
        "minGap": "2.5",
        "minGapLat": "0.6",
        "lateral_alignment": "center",
        "maxSpeed": "16.7",   # 60 km/h
        "accel": "1.2",
        "decel": "3.0",
        "color": "blue"
    },
    {
        "id": "truck",
        "vClass": "truck",
        "length": "9.0",
        "width": "2.5",
        "height": "3.5",
        "minGap": "2.5",
        "minGapLat": "0.6",
        "lateral_alignment": "center",
        "maxSpeed": "16.7",   # 60 km/h
        "accel": "1.0",
        "decel": "2.8",
        "color": "cyan"
    },
    {
        "id": "bicycle",
        "vClass": "bicycle",
        "length": "1.8",
        "width": "0.6",
        "height": "1.1",
        "minGap": "0.8",
        "minGapLat": "0.2",
        "lateral_alignment": "arbitrary",
        "maxSpeed": "5.5",    # 20 km/h
        "accel": "1.5",
        "decel": "3.0",
        "color": "green"
    }
]


class VTypesGenerator:
    """Generates vtypes.add.xml for SUMO Sublane simulations."""

    def __init__(self, vehicle_types: Optional[List[Dict[str, str]]] = None) -> None:
        self.vehicle_types = vehicle_types or DEFAULT_INDIAN_VEHICLE_TYPES

    def generate_vtypes_xml(self, output_file: Union[str, Path]) -> Path:
        """Write vehicle types definition file."""
        out = Path(output_file).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<!-- Heterogeneous Indian Vehicle Definitions with Sublane Model -->',
            '<additional xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/additional_file.xsd">'
        ]

        for vt in self.vehicle_types:
            tag = (
                f'    <vType id="{vt["id"]}" vClass="{vt["vClass"]}" length="{vt["length"]}" '
                f'width="{vt["width"]}" height="{vt["height"]}" minGap="{vt["minGap"]}" '
                f'minGapLat="{vt["minGapLat"]}" latAlignment="{vt["lateral_alignment"]}" '
                f'maxSpeed="{vt["maxSpeed"]}" accel="{vt["accel"]}" decel="{vt["decel"]}" '
                f'color="{vt["color"]}"/>'
            )
            lines.append(tag)

        lines.append('</additional>')
        out.write_text("\n".join(lines), encoding="utf-8")
        log_event(f"Generated Indian fleet vtypes definition in {out}", reason_code="VTYPES-GEN-DONE")
        return out
