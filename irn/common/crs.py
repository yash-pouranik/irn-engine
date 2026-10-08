"""Coordinate Reference System (CRS) transformations and UTM utilities.

Strictly enforces Rule G-1:
- Stored geometry is always WGS84 (EPSG:4326).
- All metric spatial calculations use local projected UTM based on the area centroid.
"""

from __future__ import annotations
import math
from typing import Tuple, Union
import numpy as np

# WGS84 Ellipsoid constants
WGS84_A = 6378137.0  # semi-major axis (meters)
WGS84_F = 1.0 / 298.257223563  # flattening
WGS84_B = WGS84_A * (1.0 - WGS84_F)  # semi-minor axis
WGS84_E2 = 2.0 * WGS84_F - WGS84_F**2  # first eccentricity squared
WGS84_E_PRIME2 = WGS84_E2 / (1.0 - WGS84_E2)  # second eccentricity squared
UTM_K0 = 0.9996  # central scale factor


def utm_zone_from_lon_lat(lon: float, lat: float) -> Tuple[int, bool]:
    """Calculate UTM zone number and hemisphere (True for North, False for South) from lon/lat."""
    zone_number = int((lon + 180.0) / 6.0) + 1
    if zone_number < 1:
        zone_number = 1
    elif zone_number > 60:
        zone_number = 60
    is_northern = lat >= 0.0
    return zone_number, is_northern


def epsg_from_utm_zone(zone_number: int, is_northern: bool = True) -> int:
    """Return EPSG code for a given UTM zone (e.g. 32643 for UTM 43N, 32644 for UTM 44N)."""
    return (32600 if is_northern else 32700) + zone_number


def optimal_utm_epsg(lon: float, lat: float) -> int:
    """Find the optimal UTM EPSG code for a given centroid (Rule G-1)."""
    zone, north = utm_zone_from_lon_lat(lon, lat)
    return epsg_from_utm_zone(zone, north)


def wgs84_to_utm(
    lon: Union[float, np.ndarray],
    lat: Union[float, np.ndarray],
    zone_number: int,
    is_northern: bool = True
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """Vectorized conversion of (longitude, latitude) in degrees to UTM (easting, northing) in meters.

    Uses high-precision Karney/Snyder Transverse Mercator formulation.
    """
    lon_rad = np.radians(lon)
    lat_rad = np.radians(lat)

    lon_origin = (zone_number - 1) * 6.0 - 180.0 + 3.0
    lon_origin_rad = np.radians(lon_origin)

    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    tan_lat = np.tan(lat_rad)

    N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat**2)
    T = tan_lat**2
    C = WGS84_E_PRIME2 * cos_lat**2
    A = cos_lat * (lon_rad - lon_origin_rad)

    # Meridional distance M
    M = WGS84_A * (
        (1.0 - WGS84_E2 / 4.0 - 3.0 * WGS84_E2**2 / 64.0 - 5.0 * WGS84_E2**3 / 256.0) * lat_rad
        - (3.0 * WGS84_E2 / 8.0 + 3.0 * WGS84_E2**2 / 32.0 + 45.0 * WGS84_E2**3 / 1024.0) * np.sin(2.0 * lat_rad)
        + (15.0 * WGS84_E2**2 / 256.0 + 45.0 * WGS84_E2**3 / 1024.0) * np.sin(4.0 * lat_rad)
        - (35.0 * WGS84_E2**3 / 3072.0) * np.sin(6.0 * lat_rad)
    )

    easting = UTM_K0 * N * (
        A
        + (1.0 - T + C) * A**3 / 6.0
        + (5.0 - 18.0 * T + T**2 + 72.0 * C - 58.0 * WGS84_E_PRIME2) * A**5 / 120.0
    ) + 500000.0

    northing = UTM_K0 * (
        M
        + N * tan_lat * (
            A**2 / 2.0
            + (5.0 - T + 9.0 * C + 4.0 * C**2) * A**4 / 24.0
            + (61.0 - 58.0 * T + T**2 + 600.0 * C - 330.0 * WGS84_E_PRIME2) * A**6 / 720.0
        )
    )

    if not is_northern:
        northing += 10000000.0

    return easting, northing


def utm_to_wgs84(
    easting: Union[float, np.ndarray],
    northing: Union[float, np.ndarray],
    zone_number: int,
    is_northern: bool = True
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """Vectorized conversion of UTM (easting, northing) in meters to (longitude, latitude) in degrees."""
    x = np.asarray(easting, dtype=float) - 500000.0
    y = np.asarray(northing, dtype=float)
    if not is_northern:
        y -= 10000000.0

    lon_origin = (zone_number - 1) * 6.0 - 180.0 + 3.0

    M = y / UTM_K0
    mu = M / (WGS84_A * (1.0 - WGS84_E2 / 4.0 - 3.0 * WGS84_E2**2 / 64.0 - 5.0 * WGS84_E2**3 / 256.0))

    e1 = (1.0 - np.sqrt(1.0 - WGS84_E2)) / (1.0 + np.sqrt(1.0 - WGS84_E2))

    phi1 = (
        mu
        + (3.0 * e1 / 2.0 - 27.0 * e1**3 / 32.0) * np.sin(2.0 * mu)
        + (21.0 * e1**2 / 16.0 - 55.0 * e1**4 / 32.0) * np.sin(4.0 * mu)
        + (151.0 * e1**3 / 96.0) * np.sin(6.0 * mu)
        + (1097.0 * e1**4 / 512.0) * np.sin(8.0 * mu)
    )

    sin_phi1 = np.sin(phi1)
    cos_phi1 = np.cos(phi1)
    tan_phi1 = np.tan(phi1)

    N1 = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_phi1**2)
    R1 = WGS84_A * (1.0 - WGS84_E2) / ((1.0 - WGS84_E2 * sin_phi1**2) ** 1.5)
    D = x / (N1 * UTM_K0)

    T1 = tan_phi1**2
    C1 = WGS84_E_PRIME2 * cos_phi1**2

    lat_rad = phi1 - (N1 * tan_phi1 / R1) * (
        D**2 / 2.0
        - (5.0 + 3.0 * T1 + 10.0 * C1 - 4.0 * C1**2 - 9.0 * WGS84_E_PRIME2) * D**4 / 24.0
        + (61.0 + 90.0 * T1 + 298.0 * C1 + 45.0 * T1**2 - 252.0 * WGS84_E_PRIME2 - 3.0 * C1**2) * D**6 / 720.0
    )

    lon_rad = np.radians(lon_origin) + (
        D
        - (1.0 + 2.0 * T1 + C1) * D**3 / 6.0
        + (5.0 - 2.0 * C1 + 28.0 * T1 - 3.0 * C1**2 + 8.0 * WGS84_E_PRIME2 + 24.0 * T1**2) * D**5 / 120.0
    ) / cos_phi1

    return np.degrees(lon_rad), np.degrees(lat_rad)
