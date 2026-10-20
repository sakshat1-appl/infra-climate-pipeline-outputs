"""Tests for the geodesic primitives in build_highway_features.py.

All computation must be geodesic on WGS84 (pyproj.Geod) - never planar
arithmetic on degrees. Reference values and their sources:
  * One degree of latitude at the equator = 110574.389 m (meridional arc
    of the WGS84 ellipsoid, a = 6378137 m, f = 1/298.257223563; the same
    value any geodesy reference such as GeographicLib reports).
  * One degree of longitude at the equator = 111319.491 m (equatorial
    radius times pi/180).
  * Real-distance check: the in-test spherical haversine (mean Earth
    radius 6371.0088 km - the constant already used by the rainfall join
    gate in this repository) agrees with the ellipsoidal Geod length to
    well within 1 %; the test pins that agreement on the
    Gateway-of-India to India-Gate pair.

Offline: pure computation, no network, no cache files.
"""

import math
import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

import build_highway_features as B

# Hand-checked WGS84 references (see module docstring).
M_PER_DEG_LAT_EQUATOR = 110574.389
M_PER_DEG_LON_EQUATOR = 111319.491

# Gateway of India, Mumbai -> India Gate, Delhi (city-centre landmarks).
MUMBAI = (18.9220, 72.8330)
DELHI = (28.6129, 77.2295)


def haversine_m(lat1, lon1, lat2, lon2):
    """Independent spherical reference (repo constant R = 6371.0088 km)."""
    R = 6371.0088 * 1000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def test_polyline_length_matches_wgs84_degree_arcs():
    north = B.polyline_length_m([(0.0, 0.0), (1.0, 0.0)])
    assert abs(north - M_PER_DEG_LAT_EQUATOR) < 2.0, north
    east = B.polyline_length_m([(0.0, 0.0), (0.0, 1.0)])
    assert abs(east - M_PER_DEG_LON_EQUATOR) < 2.0, east
    # Multi-segment sums.
    two_deg = B.polyline_length_m([(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)])
    assert abs(two_deg - 2 * M_PER_DEG_LAT_EQUATOR) < 4.0, two_deg


def test_point_to_segment_hand_cases():
    # Point 1 degree north of the equator segment -> one meridional arc.
    d = B.point_to_segment_m(1.0, 0.5, 0.0, 0.0, 0.0, 1.0)
    assert abs(d - M_PER_DEG_LAT_EQUATOR) < 60.0, d
    # Point exactly on the segment -> zero.
    d = B.point_to_segment_m(0.0, 0.4, 0.0, 0.0, 0.0, 1.0)
    assert d < 1e-6, d
    # Degenerate segment (A == B) -> plain geodesic point distance:
    # diagonal of a 1x1 equatorial degree square.
    d = B.point_to_segment_m(0.0, 0.0, 1.0, 1.0, 1.0, 1.0)
    expected = math.hypot(M_PER_DEG_LAT_EQUATOR, M_PER_DEG_LON_EQUATOR)
    assert abs(d - expected) < 5.0, (d, expected)
    # Beyond the endpoint: perpendicular foot falls outside, distance goes
    # to the nearer endpoint (1 degree of longitude at the equator).
    d = B.point_to_segment_m(0.0, 2.0, 0.0, 0.0, 0.0, 1.0)
    assert abs(d - M_PER_DEG_LON_EQUATOR) < 5.0, d


def test_real_distance_within_one_percent_of_haversine():
    geod_len = B.polyline_length_m([MUMBAI, DELHI])
    sphere_len = haversine_m(*MUMBAI, *DELHI)
    print("Mumbai-Delhi: geod %.1f m, haversine %.1f m" % (geod_len, sphere_len))
    assert abs(geod_len - sphere_len) / sphere_len < 0.01
    # Ground the magnitude: the pair is ~1150 km apart, not ~115 or ~11500.
    assert 1_000_000 < geod_len < 1_300_000, geod_len


def test_clipped_length_crossing_and_outside_segments():
    # Meridian segment crossing the 10 km circle: inside part is a
    # 20 km diameter to within metres.
    d = B.clipped_length_within_radius_m(0.0, 0.0, [(-1.0, 0.0), (1.0, 0.0)], 10000.0)
    assert abs(d - 20000.0) < 5.0, d
    # Fully outside segment contributes nothing.
    d = B.clipped_length_within_radius_m(0.0, 0.0, [(5.0, 5.0), (6.0, 6.0)], 10000.0)
    assert d == 0.0, d
    # Fully inside segment equals its geodesic length.
    seg = [(0.01, 0.0), (0.02, 0.0)]
    d = B.clipped_length_within_radius_m(0.0, 0.0, seg, 10000.0)
    assert abs(d - B.polyline_length_m(seg)) < 1e-6, d
    # Partial overlap: from inside (equator) to 1 degree north, only the
    # first 10 km count.
    d = B.clipped_length_within_radius_m(0.0, 0.0, [(0.0, 0.0), (1.0, 0.0)], 10000.0)
    assert abs(d - 10000.0) < 5.0, d


def test_point_to_polyline_picks_nearest_leg():
    # L-shaped polyline; the point sits 0.5 deg south of the E-W leg and
    # 0.5 deg east of the N-S leg, so the E-W leg (half a meridional
    # degree) must win.
    d = B.point_to_polyline_m(0.5, 0.5, [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)])
    expected = 0.5 * M_PER_DEG_LAT_EQUATOR
    assert abs(d - expected) < 250.0, (d, expected)
    assert d < 0.5 * M_PER_DEG_LON_EQUATOR, d
    assert B.point_to_polyline_m(0.0, 0.0, [(0.0, 0.0)]) is None
    assert B.point_to_polyline_m(0.0, 0.0, []) is None
