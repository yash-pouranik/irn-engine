"""Module 4: SUMO Master Sublane Configuration Generator (scenario.sumocfg).

Configures:
- Lateral resolution = 0.6m (Sublane Model activation).
- Simulation step length = 0.1s.
- Links compiled network.net.xml, vtypes.add.xml, and anomalies.add.xml.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Union

from irn.common.logger import log_event


class SumoConfigGenerator:
    """Generates scenario.sumocfg and accompanying gui settings."""

    def __init__(
        self,
        lateral_resolution_m: float = 0.6,
        step_length_s: float = 0.1
    ) -> None:
        self.lateral_resolution_m = lateral_resolution_m
        self.step_length_s = step_length_s

    def generate_sumocfg(
        self,
        net_filename: str = "network.net.xml",
        route_files: Optional[str] = None,
        additional_files: str = "vtypes.add.xml,anomalies.add.xml",
        output_file: Union[str, Path] = "scenario.sumocfg",
        begin_time: int = 0,
        end_time: Optional[int] = None
    ) -> Path:
        """Write scenario.sumocfg file."""
        out = Path(output_file).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)

        if not route_files and (out.parent / "routes.rou.xml").exists():
            route_files = "routes.rou.xml"

        route_tag = f'        <route-files value="{route_files}"/>\n' if route_files else ""
        end_tag = f'<end value="{end_time}"/>' if end_time else ""

        content = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    <input>
        <net-file value="{net_filename}"/>
{route_tag}        <additional-files value="{additional_files}"/>
    </input>
    <processing>
        <!-- Eclipse SUMO Sublane Model Activation -->
        <lateral-resolution value="{self.lateral_resolution_m:.2f}"/>
        <step-length value="{self.step_length_s:.2f}"/>
        <collision.action value="warn"/>
        <collision.stoptime value="2"/>
    </processing>
    <time>
        <begin value="{begin_time}"/>
        {end_tag}
    </time>
</configuration>
"""
        out.write_text(content.strip(), encoding="utf-8")
        log_event(f"Generated Sublane simulator configuration in {out}", reason_code="SUMOCFG-GEN-DONE")
        return out
