"""Maritime / port exposure features for every asset with coordinates.

Reads:
  * ``integration/data_reference/assets_verified.csv`` (asset sidecar)
  * ``integration/data_raw_cache/ports/portwatch_ports.csv`` (PortWatch
    global ports via HDX; 2,065 rows, 37 for India)
  * ``integration/data_raw_cache/ports/hotosm_ind_sea_ports_geojson.zip``
    (HOT Sea Ports of India; 772 features, points + polygons + 1 line)
  * ``integration/data_raw_cache/ports/ne_10m_coastline.zip`` (Natural
    Earth 1:10m coastline SHP bundle)
  * ``integration/data_raw_cache/ports/MANIFEST.json`` (exact URLs,
    retrieval dates, licences)

Writes:
  * ``integration/data_features/maritime_features_asset.csv`` (one row per
    asset with non-null coordinates, sorted by asset_id)
  * extends ``integration/data_output/weather_coverage.json`` in place by
    appending one source entry per maritime source (``portwatch_ports``,
    ``hdx_sea_ports``, ``natural_earth_coastline``) to each asset and
    adding a top-level ``mt`` family block. Pre-existing ``wx``/``rd``
    content is preserved verbatim.

Feature definitions (exactly these - no composite indices, no weights,
no modelling). Per asset with non-null coordinates:
  * ``mt_dist_major_port_m``: minimum geodesic distance (m) from the asset
    point to the nearest feature of the reference port set (see REFERENCE
    PORT SET below). Zero when the asset lies inside a reference port
    polygon.
  * ``mt_nearest_major_port_id`` / ``mt_nearest_major_port_name``: the
    nearest reference feature (PortWatch ``portid``/``portname`` for
    PortWatch rows, OSM ``id`` for HOT features; the name falls back to
    the id where OSM carries no name).
  * ``mt_ports_within_100km``: count of reference port points within
    100 km plus reference port polygons whose boundary (or interior) lies
    within 100 km.
  * ``mt_in_port_polygon``: true iff the asset point falls inside a
    reference port Polygon/MultiPolygon (ray casting with holes;
    topological on degrees, no metric involved). Non-port assets are
    expected to be false; only assets physically inside a mapped port
    estate are true.
  * ``mt_coast_dist_m`` / ``mt_coastal_flag``: geodesic distance (m) to
    the nearest Natural Earth 1:10m coastline segment in the India
    window; coastal iff <= 50 km.
  * ``mt_throughput_teu_yr`` / ``mt_throughput_source`` /
    ``mt_throughput_year``: ALWAYS NULL. No licenced throughput source is
    programmatically obtainable (IPA/Sagarmala publish HTML/PDF only and
    are unreachable from this network), so throughput is gated, never
    estimated, modelled or imputed. Reason ``throughput_gated``.
  * ``mt_pw_nearest_port_id`` / ``mt_pw_nearest_port_name`` /
    ``mt_pw_dist_m``: the nearest PortWatch India row specifically (a
    subset of the reference set, but identified separately because only
    PortWatch rows carry activity figures).
  * ``mt_pw_vessel_count_total_yr`` / ``mt_pw_vessel_count_container_yr``:
    that row's vessel-call counts as published. These are activity
    figures, explicitly NOT throughput, and are never presented as such.
  * ``mt_pw_share_import_pct`` / ``mt_pw_share_export_pct``: that row's
    country-maritime-trade shares as published (percent scale).

REFERENCE PORT SET. 42 reference points (37 PortWatch India rows - every
PortWatch-tracked commercial port in India, no activity cutoff - plus 5
HOT point features tagged ``harbour=yes``) and 60 HOT Polygon /
MultiPolygon features with a port-identifying tag (``industrial=port``,
``harbour=yes``, or a non-null ``port`` tag). Deliberately EXCLUDED: 625
HOT ``amenity=ferry_terminal`` points (jetties and water-metro stops, not
ports), 56 ferry-terminal polygons, 24 named-but-untagged or untagged
polygons (no port-identifying tag; commercial ports among them, e.g.
Karwar and Pipavav, remain covered as PortWatch points), and 1
``harbour=yes`` LineString (a line is not a port location). The builder
prints the exact composition every run. Known consequence of the
tag-only rule: one predicate polygon is the inland container depot
``ICD CONCOR Kanpur`` (tagged ``industrial=port``); it enters the set
as tagged, and is the nearest reference feature for a few deep-inland
assets.

LINEAR ASSETS. Pipelines and transmission corridors have no line geometry
in the sidecar - only an ``approximate_midpoint`` coordinate. All
distances are computed from that point. A midpoint far from the coast
does not mean every kilometre of the line is; this is a documented
limitation, not a line-based analysis.

GEODESY. All distances via ``pyproj.Geod`` (WGS84) on EPSG:4326 - never
planar arithmetic on degrees. Point-to-segment distances run in a local
azimuthal-equidistant plane centred on the asset (same construction as
phase 5). Coastline SHP polylines are read with a minimal stdlib
shapefile reader (no new dependency) and pre-filtered to an India window
(lon 60-100, lat 0-40) before matching. Polygon rings are measured with
the same point-to-segment machinery.

MISSING DATA. ``is_imputed`` is 0 on every row. Where a source file is
absent its features are null with a coverage reason
(``portwatch_source_unavailable`` / ``hdx_ports_unavailable`` /
``coastline_source_unavailable``); ``asset_coordinate_unverified`` covers
the 19 assets with no row. ``mt_coastal_flag`` is null exactly when no
coastline source was available.

Deterministic: stdlib + pyproj, fixed constants, sorted iteration, LF
line endings. Byte-identical across runs. No network access.
"""

import csv
import json
import math
import os
import struct
import zipfile

from pyproj import Geod

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
ASSETS_CSV = os.path.join(
    REPO_ROOT, "integration", "data_reference", "assets_verified.csv"
)
PORTS_CACHE_DIR = os.path.join(
    REPO_ROOT, "integration", "data_raw_cache", "ports"
)
PORTWATCH_CSV = os.path.join(PORTS_CACHE_DIR, "portwatch_ports.csv")
HOTOSM_ZIP = os.path.join(
    PORTS_CACHE_DIR, "hotosm_ind_sea_ports_geojson.zip"
)
NE_COAST_ZIP = os.path.join(PORTS_CACHE_DIR, "ne_10m_coastline.zip")
PORTS_MANIFEST_JSON = os.path.join(PORTS_CACHE_DIR, "MANIFEST.json")
FEATURES_CSV = os.path.join(
    REPO_ROOT, "integration", "data_features", "maritime_features_asset.csv"
)
COVERAGE_JSON = os.path.join(
    REPO_ROOT, "integration", "data_output", "weather_coverage.json"
)

# Reference-set construction (no numeric thresholds anywhere: membership
# is by source inclusion and OSM tags only - see module docstring).
PORTWATCH_INDIA_ISO3 = "IND"
RADIUS_100KM_M = 100000.0
COASTAL_THRESHOLD_M = 50000.0
# India-window pre-filter for coastline segments (lon/lat degrees).
COAST_LON_MIN, COAST_LON_MAX = 60.0, 100.0
COAST_LAT_MIN, COAST_LAT_MAX = 0.0, 40.0

PORTWATCH_SOURCE = "IMF PortWatch (via HDX)"
POLY_SOURCE = "HOT Sea Ports of India (via HDX)"
COAST_SOURCE = "Natural Earth 1:10m coastline"
SOURCE_NAME = "IMF PortWatch (via HDX); HOT Sea Ports; Natural Earth"
LICENCE = (
    "IMF copyright and terms (PortWatch via HDX); "
    "ODbL 1.0, (c) OpenStreetMap contributors (HOT Sea Ports); "
    "public domain (Natural Earth coastline)"
)
THROUGHPUT_REASON = "throughput_gated"

FEATURE_COLUMNS = [
    "asset_id",
    "sector",
    "name",
    "mt_dist_major_port_m",
    "mt_nearest_major_port_id",
    "mt_nearest_major_port_name",
    "mt_ports_within_100km",
    "mt_in_port_polygon",
    "mt_coast_dist_m",
    "mt_coastal_flag",
    "mt_throughput_teu_yr",
    "mt_throughput_source",
    "mt_throughput_year",
    "mt_pw_nearest_port_id",
    "mt_pw_nearest_port_name",
    "mt_pw_dist_m",
    "mt_pw_vessel_count_total_yr",
    "mt_pw_vessel_count_container_yr",
    "mt_pw_share_import_pct",
    "mt_pw_share_export_pct",
    "portwatch_source",
    "portwatch_url",
    "portwatch_licence",
    "portwatch_retrieval_date",
    "poly_source",
    "poly_url",
    "poly_licence",
    "poly_retrieval_date",
    "coast_source",
    "coast_url",
    "coast_licence",
    "coast_retrieval_date",
    "source_name",
    "source_url",
    "licence",
    "retrieval_date",
    "is_observed",
    "is_imputed",
]

GEOD = Geod(ellps="WGS84")


# --------------------------------------------------------------------------
# Geodesy helpers (same local-AEQD construction as phase 5)
# --------------------------------------------------------------------------


def _aeqd_xy(plat, plon, lat, lon):
    az, _, dist = GEOD.inv(plon, plat, lon, lat)
    theta = math.radians(az)
    return (dist * math.sin(theta), dist * math.cos(theta))


def point_to_segment_m(plat, plon, alat, alon, blat, blon):
    """Minimum geodesic distance (m) from point P to segment A-B."""
    ax, ay = _aeqd_xy(plat, plon, alat, alon)
    bx, by = _aeqd_xy(plat, plon, blat, blon)
    dx, dy = bx - ax, by - ay
    seg_len2 = dx * dx + dy * dy
    if seg_len2 == 0.0:
        return math.hypot(ax, ay)
    t = max(0.0, min(1.0, -(ax * dx + ay * dy) / seg_len2))
    return math.hypot(ax + t * dx, ay + t * dy)


def point_to_polyline_m(plat, plon, coords):
    """Minimum geodesic distance (m) from P to a [(lat, lon), ...] polyline.

    Returns None for a polyline with no usable segment.
    """
    best = None
    for (alat, alon), (blat, blon) in zip(coords, coords[1:]):
        d = point_to_segment_m(plat, plon, alat, alon, blat, blon)
        if best is None or d < best:
            best = d
    return best


def point_in_ring(lon, lat, ring):
    """Ray casting on degrees (topological; no metric involved)."""
    inside = False
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % n]
        if (y1 > lat) != (y2 > lat):
            xinters = x1 + (lat - y1) * (x2 - x1) / (y2 - y1)
            if xinters > lon:
                inside = not inside
    return inside


def point_in_polygon_geom(lon, lat, geometry):
    """True iff (lon, lat) is inside a Polygon/MultiPolygon (holes out)."""
    gtype = geometry.get("type")
    coords = geometry.get("coordinates")
    if gtype == "Polygon":
        polys = [coords]
    elif gtype == "MultiPolygon":
        polys = coords
    else:
        return False
    for poly in polys:
        if not poly or not point_in_ring(lon, lat, poly[0]):
            continue
        if any(point_in_ring(lon, lat, hole) for hole in poly[1:]):
            continue
        return True
    return False


def polygon_boundary_distance_m(lat, lon, geometry):
    """Minimum geodesic distance (m) from P to a polygon's outer rings."""
    gtype = geometry.get("type")
    coords = geometry.get("coordinates")
    if gtype == "Polygon":
        polys = [coords]
    elif gtype == "MultiPolygon":
        polys = coords
    else:
        return None
    best = None
    for poly in polys:
        if not poly:
            continue
        ring = [(pt[1], pt[0]) for pt in poly[0]]
        d = point_to_polyline_m(lat, lon, ring)
        if d is not None and (best is None or d < best):
            best = d
    return best


# --------------------------------------------------------------------------
# Source loading
# --------------------------------------------------------------------------


def is_hot_reference_point(props):
    """HOT point features admitted to the reference set: harbour=yes."""
    return (props or {}).get("harbour") == "yes"


def is_hot_reference_polygon(props):
    """HOT Polygon/MultiPolygon features admitted to the reference set."""
    props = props or {}
    return (
        props.get("industrial") == "port"
        or props.get("harbour") == "yes"
        or props.get("port") not in (None, "")
    )


def load_portwatch(path=PORTWATCH_CSV):
    """PortWatch rows filtered to India; None when the file is absent."""
    if not os.path.exists(path):
        return None
    with open(path, newline="", encoding="utf-8") as f:
        rows = [
            r
            for r in csv.DictReader(f)
            if (r.get("ISO3") or "").strip() == PORTWATCH_INDIA_ISO3
        ]
    ports = []
    for r in rows:
        try:
            lat = float(r["lat"])
            lon = float(r["lon"])
        except (ValueError, TypeError):
            continue
        ports.append(
            {
                "ref_id": (r.get("portid") or "").strip(),
                "ref_name": (r.get("portname") or "").strip(),
                "lat": lat,
                "lon": lon,
                "kind": "portwatch",
                "vessel_count_total": _to_int(r.get("vessel_count_total")),
                "vessel_count_container": _to_int(
                    r.get("vessel_count_container")
                ),
                "share_import_pct": _to_float(
                    r.get("share_country_maritime_import")
                ),
                "share_export_pct": _to_float(
                    r.get("share_country_maritime_export")
                ),
            }
        )
    ports.sort(key=lambda p: p["ref_id"])
    return ports


def _to_int(text):
    try:
        return int(float((text or "").strip()))
    except (ValueError, TypeError, AttributeError):
        return None


def _to_float(text):
    try:
        return float((text or "").strip())
    except (ValueError, TypeError, AttributeError):
        return None


def load_hotosm_reference(path=HOTOSM_ZIP):
    """(points, polygons) admitted to the reference set.

    Points: [(ref_id, ref_name, lat, lon)]. Polygons: [(ref_id, ref_name,
    geometry)]. Returns (None, None) when the file is absent.
    """
    if not os.path.exists(path):
        return None, None
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        member = next(
            (n for n in names if n.lower().endswith(".geojson")), None
        )
        if member is None:
            raise ValueError("hotosm zip has no .geojson member: %s" % names)
        payload = json.loads(archive.read(member).decode("utf-8"))
    points, polygons = [], []
    for feat in payload.get("features") or []:
        feat = feat or {}
        geom = feat.get("geometry") or {}
        props = feat.get("properties") or {}
        gtype = geom.get("type")
        if gtype == "Point" and is_hot_reference_point(props):
            coords = geom.get("coordinates") or []
            if len(coords) != 2:
                continue
            ref_id = (props.get("id") or "").strip()
            points.append(
                {
                    "ref_id": ref_id,
                    "ref_name": (props.get("name") or "").strip()
                    or ref_id,
                    "lat": float(coords[1]),
                    "lon": float(coords[0]),
                    "kind": "hotosm",
                }
            )
        elif gtype in ("Polygon", "MultiPolygon") and (
            is_hot_reference_polygon(props)
        ):
            ref_id = (props.get("id") or "").strip()
            polygons.append(
                {
                    "ref_id": ref_id,
                    "ref_name": (props.get("name") or "").strip()
                    or ref_id,
                    "geometry": geom,
                }
            )
    points.sort(key=lambda p: p["ref_id"])
    polygons.sort(key=lambda p: p["ref_id"])
    return points, polygons


def read_shp_polylines(data):
    """Minimal .shp PolyLine reader -> list of [[(lon, lat), ...], ...]."""
    if len(data) < 104 or struct.unpack(">i", data[:4])[0] != 9994:
        raise ValueError("bad shapefile header")
    offset = 100
    records = []
    while offset + 8 <= len(data):
        content_words = struct.unpack(">i", data[offset + 4 : offset + 8])[0]
        if content_words < 2:
            raise ValueError("short shapefile record")
        rec = data[offset + 8 : offset + 8 + content_words * 2]
        shape_type = struct.unpack("<i", rec[:4])[0]
        # 3 = PolyLine, 13 = PolyLineZ (same XY prefix layout; Z/M
        # ranges after the points are skipped via content length).
        if shape_type not in (3, 13):
            raise ValueError(
                "non-polyline record in coastline shapefile: %d" % shape_type
            )
        num_parts, num_points = struct.unpack("<ii", rec[36:44])
        parts = struct.unpack("<%di" % num_parts, rec[44 : 44 + 4 * num_parts])
        pts = struct.unpack(
            "<%dd" % (2 * num_points),
            rec[44 + 4 * num_parts : 44 + 4 * num_parts + 16 * num_points],
        )
        coords = [(pts[2 * i], pts[2 * i + 1]) for i in range(num_points)]
        for i in range(num_parts):
            end = parts[i + 1] if i + 1 < num_parts else num_points
            records.append(coords[parts[i] : end])
        offset += 8 + content_words * 2
    if offset != len(data):
        raise ValueError("shapefile has trailing bytes")
    return records


def load_coastline_segments(path=NE_COAST_ZIP):
    """Coastline segments in the India window, with bboxes for pruning."""
    if not os.path.exists(path):
        return None
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        member = next((n for n in names if n.lower().endswith(".shp")), None)
        if member is None:
            raise ValueError("coastline zip has no .shp member: %s" % names)
        polylines = read_shp_polylines(archive.read(member))
    segments = []
    for line in polylines:
        for (lon1, lat1), (lon2, lat2) in zip(line, line[1:]):
            if (
                max(lon1, lon2) < COAST_LON_MIN
                or min(lon1, lon2) > COAST_LON_MAX
                or max(lat1, lat2) < COAST_LAT_MIN
                or min(lat1, lat2) > COAST_LAT_MAX
            ):
                continue
            segments.append(
                {
                    "a": (lat1, lon1),
                    "b": (lat2, lon2),
                    "bbox": (
                        min(lat1, lat2),
                        min(lon1, lon2),
                        max(lat1, lat2),
                        max(lon1, lon2),
                    ),
                }
            )
    return segments


def coast_distance_m(lat, lon, segments):
    """Min geodesic distance to coastline segments (bbox-pruned)."""
    best = None
    for seg in segments:
        slat0, slon0, slat1, slon1 = seg["bbox"]
        if best is not None:
            # Cheap prune: bbox corner proximity already exceeds best.
            clon = min(max(lon, slon0), slon1)
            clat = min(max(lat, slat0), slat1)
            _, _, rough = GEOD.inv(lon, lat, clon, clat)
            if rough >= best:
                continue
        d = point_to_segment_m(
            lat, lon, seg["a"][0], seg["a"][1], seg["b"][0], seg["b"][1]
        )
        if best is None or d < best:
            best = d
    return best


# --------------------------------------------------------------------------
# Feature computation
# --------------------------------------------------------------------------


def compute_asset_features(lat, lon, points, polygons, segments):
    """All maritime features for one asset point (raw values, None=null).

    ``points``/``polygons`` are the reference port set (either may be an
    empty list when its source file is absent); ``segments`` is the
    coastline list (None when absent). Iteration is in sorted ref_id
    order with strict improvement, so ties resolve deterministically.
    """
    dist, nearest_id, nearest_name = None, None, None
    within = None
    if points is not None and polygons is not None:
        within = 0
        for pt in points:
            _, _, d = GEOD.inv(lon, lat, pt["lon"], pt["lat"])
            if dist is None or d < dist:
                dist, nearest_id, nearest_name = (
                    d,
                    pt["ref_id"],
                    pt["ref_name"],
                )
            if d <= RADIUS_100KM_M:
                within += 1
        inside = False
        for poly in polygons:
            if point_in_polygon_geom(lon, lat, poly["geometry"]):
                inside = True
                within += 1
                if dist is None or 0.0 < dist:
                    dist, nearest_id, nearest_name = (
                        0.0,
                        poly["ref_id"],
                        poly["ref_name"],
                    )
                continue
            bdist = polygon_boundary_distance_m(lat, lon, poly["geometry"])
            if bdist is None:
                continue
            if bdist <= RADIUS_100KM_M:
                within += 1
            if dist is None or bdist < dist:
                dist, nearest_id, nearest_name = (
                    bdist,
                    poly["ref_id"],
                    poly["ref_name"],
                )
    else:
        inside = None

    pw_id, pw_name, pw_d = None, None, None
    pw_total, pw_cont, pw_imp, pw_exp = None, None, None, None
    if points is not None:
        for pt in points:
            if pt["kind"] != "portwatch":
                continue
            _, _, d = GEOD.inv(lon, lat, pt["lon"], pt["lat"])
            if pw_d is None or d < pw_d:
                pw_d = d
                pw_id, pw_name = pt["ref_id"], pt["ref_name"]
                pw_total = pt["vessel_count_total"]
                pw_cont = pt["vessel_count_container"]
                pw_imp = pt["share_import_pct"]
                pw_exp = pt["share_export_pct"]

    coast_d, coastal = None, None
    if segments is not None:
        coast_d = coast_distance_m(lat, lon, segments)
        coastal = coast_d is not None and coast_d <= COASTAL_THRESHOLD_M

    return {
        "mt_dist_major_port_m": dist,
        "mt_nearest_major_port_id": nearest_id,
        "mt_nearest_major_port_name": nearest_name,
        "mt_ports_within_100km": within,
        "mt_in_port_polygon": inside,
        "mt_coast_dist_m": coast_d,
        "mt_coastal_flag": coastal,
        "mt_throughput_teu_yr": None,
        "mt_throughput_source": None,
        "mt_throughput_year": None,
        "mt_pw_nearest_port_id": pw_id,
        "mt_pw_nearest_port_name": pw_name,
        "mt_pw_dist_m": pw_d,
        "mt_pw_vessel_count_total_yr": pw_total,
        "mt_pw_vessel_count_container_yr": pw_cont,
        "mt_pw_share_import_pct": pw_imp,
        "mt_pw_share_export_pct": pw_exp,
    }


def fmt_m(value):
    return "" if value is None else "%.3f" % value


def fmt_bool(value):
    if value is None:
        return ""
    return "true" if value else "false"


def fmt_num(value):
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return "%d" % int(value)
    return str(value)


def fmt_text(value):
    return "" if value is None else str(value)


def load_manifest_info(path=PORTS_MANIFEST_JSON):
    """Manifest URLs, licences and retrieval dates; missing file raises."""
    try:
        with open(path, encoding="utf-8") as f:
            manifest = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            "ports manifest not found: %s (run ingest_ports.py first)" % path
        )
    files = (manifest or {}).get("files") or {}
    info = {}
    for key in ("portwatch", "hotosm_ports", "ne_coastline"):
        entry = files.get(key) or {}
        info[key] = {
            "url": entry.get("url") or "",
            "licence": entry.get("licence") or "",
            "retrieval_date": entry.get("retrieval_date") or "",
        }
    return info


def format_row(asset, row, info, observed):
    dates = {
        "portwatch": info["portwatch"]["retrieval_date"],
        "hdx_sea_ports": info["hotosm_ports"]["retrieval_date"],
        "natural_earth_coastline": info["ne_coastline"]["retrieval_date"],
    }
    return {
        "asset_id": asset["asset_id"],
        "sector": asset["sector"],
        "name": asset["name"],
        "mt_dist_major_port_m": fmt_m(row["mt_dist_major_port_m"]),
        "mt_nearest_major_port_id": fmt_text(
            row["mt_nearest_major_port_id"]
        ),
        "mt_nearest_major_port_name": fmt_text(
            row["mt_nearest_major_port_name"]
        ),
        "mt_ports_within_100km": fmt_text(row["mt_ports_within_100km"]),
        "mt_in_port_polygon": fmt_bool(row["mt_in_port_polygon"]),
        "mt_coast_dist_m": fmt_m(row["mt_coast_dist_m"]),
        "mt_coastal_flag": fmt_bool(row["mt_coastal_flag"]),
        "mt_throughput_teu_yr": "",
        "mt_throughput_source": "",
        "mt_throughput_year": "",
        "mt_pw_nearest_port_id": fmt_text(row["mt_pw_nearest_port_id"]),
        "mt_pw_nearest_port_name": fmt_text(row["mt_pw_nearest_port_name"]),
        "mt_pw_dist_m": fmt_m(row["mt_pw_dist_m"]),
        "mt_pw_vessel_count_total_yr": fmt_text(
            row["mt_pw_vessel_count_total_yr"]
        ),
        "mt_pw_vessel_count_container_yr": fmt_text(
            row["mt_pw_vessel_count_container_yr"]
        ),
        "mt_pw_share_import_pct": fmt_num(row["mt_pw_share_import_pct"]),
        "mt_pw_share_export_pct": fmt_num(row["mt_pw_share_export_pct"]),
        "portwatch_source": PORTWATCH_SOURCE,
        "portwatch_url": info["portwatch"]["url"],
        "portwatch_licence": info["portwatch"]["licence"],
        "portwatch_retrieval_date": dates["portwatch"],
        "poly_source": POLY_SOURCE,
        "poly_url": info["hotosm_ports"]["url"],
        "poly_licence": info["hotosm_ports"]["licence"],
        "poly_retrieval_date": dates["hdx_sea_ports"],
        "coast_source": COAST_SOURCE,
        "coast_url": info["ne_coastline"]["url"],
        "coast_licence": info["ne_coastline"]["licence"],
        "coast_retrieval_date": dates["natural_earth_coastline"],
        "source_name": SOURCE_NAME,
        "source_url": info["portwatch"]["url"],
        "licence": LICENCE,
        "retrieval_date": max(
            d for d in dates.values() if d
        )
        if any(dates.values())
        else "",
        "is_observed": "1" if observed else "0",
        "is_imputed": "0",
    }


# --------------------------------------------------------------------------
# Coverage update
# --------------------------------------------------------------------------

MT_SOURCES = ("portwatch_ports", "hdx_sea_ports", "natural_earth_coastline")
MT_SOURCE_NAMES = {
    "portwatch_ports": PORTWATCH_SOURCE,
    "hdx_sea_ports": POLY_SOURCE,
    "natural_earth_coastline": COAST_SOURCE,
}
MT_MISSING_REASONS = {
    "portwatch_ports": "portwatch_source_unavailable",
    "hdx_sea_ports": "hdx_ports_unavailable",
    "natural_earth_coastline": "coastline_source_unavailable",
}
MT_INFO_KEYS = {
    "portwatch_ports": "portwatch",
    "hdx_sea_ports": "hotosm_ports",
    "natural_earth_coastline": "ne_coastline",
}


def build_mt_source_entry(
    name, has_coords, present, row, info, dist_m
):
    """One maritime coverage entry for an asset."""
    if not has_coords:
        return {
            "name": name,
            "status": "no_coverage",
            "reason_code": "asset_coordinate_unverified",
            "source_name": MT_SOURCE_NAMES[name],
            "retrieval_date": None,
            "distance_km": None,
        }
    if not present:
        return {
            "name": name,
            "status": "no_coverage",
            "reason_code": MT_MISSING_REASONS[name],
            "source_name": MT_SOURCE_NAMES[name],
            "retrieval_date": None,
            "distance_km": None,
        }
    entry = {
        "name": name,
        "status": "covered",
        "reason_code": None,
        "source_name": MT_SOURCE_NAMES[name],
        "retrieval_date": info[MT_INFO_KEYS[name]]["retrieval_date"] or None,
        "distance_km": (
            round(dist_m / 1000.0, 3) if dist_m is not None else None
        ),
    }
    if name == "portwatch_ports":
        # Throughput is gated on every row: vessel counts are activity,
        # never throughput (see module docstring).
        entry["throughput_status"] = "no_coverage"
        entry["throughput_reason_code"] = THROUGHPUT_REASON
    return entry


def update_coverage(payload, ordered_assets, rows_by_id, info,
                    present):
    """Append maritime entries and the mt family block (wx/rd untouched).

    ``present`` maps each MT_SOURCES name to whether its source file was
    available to the builder.
    """
    # Strip any previous maritime entries first: this makes rebuilds
    # byte-identical (append-only would duplicate them on re-run).
    for existing in payload["assets"]:
        existing["sources"] = [
            s for s in existing["sources"] if s["name"] not in MT_SOURCES
        ]
    payload.pop("mt", None)

    counts = {
        name: {"covered": 0, "no_coverage": 0, "by_reason_code": {}}
        for name in MT_SOURCES
    }
    by_id = {e["asset_id"]: e for e in payload["assets"]}
    for asset in ordered_assets:
        asset_id = asset["asset_id"]
        lat_text = (asset.get("verified_lat") or "").strip()
        lon_text = (asset.get("verified_lon") or "").strip()
        has_coords = bool(lat_text and lon_text)
        row = rows_by_id[asset_id]
        entries = [
            build_mt_source_entry(
                "portwatch_ports",
                has_coords,
                present["portwatch_ports"],
                row,
                info,
                row["mt_dist_major_port_m"],
            ),
            build_mt_source_entry(
                "hdx_sea_ports",
                has_coords,
                present["hdx_sea_ports"],
                row,
                info,
                None,
            ),
            build_mt_source_entry(
                "natural_earth_coastline",
                has_coords,
                present["natural_earth_coastline"],
                row,
                info,
                row["mt_coast_dist_m"],
            ),
        ]
        for entry in entries:
            by_id[asset_id]["sources"].append(entry)
            bucket = counts[entry["name"]]
            bucket[entry["status"]] += 1
            if entry["reason_code"] is not None:
                bucket["by_reason_code"][entry["reason_code"]] = (
                    bucket["by_reason_code"].get(entry["reason_code"], 0) + 1
                )

    payload["mt"] = {
        "family": "mt",
        "sources": [
            {
                "name": name,
                "source_name": MT_SOURCE_NAMES[name],
                "description": {
                    "portwatch_ports": (
                        "IMF PortWatch global ports via HDX (2,065 rows, "
                        "37 for India); per asset the nearest India row "
                        "with its vessel-call counts and trade shares "
                        "(activity, not throughput)."
                    ),
                    "hdx_sea_ports": (
                        "HOT Sea Ports of India (ODbL, 772 features); "
                        "reference port polygons for the "
                        "asset-inside-polygon test and port-boundary "
                        "distances."
                    ),
                    "natural_earth_coastline": (
                        "Natural Earth 1:10m coastline (public domain); "
                        "geodesic distance to the nearest segment in the "
                        "India window; coastal iff within 50 km."
                    ),
                }[name],
                "status_counts": {
                    "covered": counts[name]["covered"],
                    "no_coverage": counts[name]["no_coverage"],
                },
                "by_reason_code": dict(
                    sorted(counts[name]["by_reason_code"].items())
                ),
            }
            for name in MT_SOURCES
        ],
        "join_rule": {
            "description": (
                "Per asset with non-null coordinates (linear assets use "
                "their sidecar midpoint coordinate): geodesic distance to "
                "the nearest reference port feature (42 reference points "
                "plus 60 HOT port polygons; zero when inside a polygon); "
                "count of reference features within 100 km; "
                "inside-polygon flag; geodesic distance to the Natural "
                "Earth coastline. Unverified-coordinate assets receive no "
                "maritime features. Throughput is gated "
                "(throughput_gated): no licenced programmatic source "
                "exists."
            ),
        },
        "throughput": {
            "status": "unavailable",
            "reason_code": THROUGHPUT_REASON,
            "explanation": (
                "No licenced programmatic throughput (TEU) source exists: "
                "official IPA/Sagarmala tables are HTML/PDF-only and "
                "unreachable from this network. PortWatch vessel counts "
                "are activity, not throughput, and are never presented "
                "as such. mt_throughput_teu_yr is NULL on every row."
            ),
        },
        "totals": {
            "assets": len(ordered_assets),
            "per_source": {
                name: {
                    "covered": counts[name]["covered"],
                    "no_coverage": counts[name]["no_coverage"],
                }
                for name in MT_SOURCES
            },
        },
    }
    return payload


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------


def load_assets(path=ASSETS_CSV):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def null_row():
    row = compute_asset_features(0.0, 0.0, None, None, None)
    row["_present"] = {
        "portwatch_ports": False,
        "hdx_sea_ports": False,
        "natural_earth_coastline": False,
    }
    return row


def build():
    assets = load_assets()
    ordered = sorted(assets, key=lambda r: r["asset_id"])
    info = load_manifest_info(PORTS_MANIFEST_JSON)
    pw_points = load_portwatch()
    hot_points, hot_polygons = load_hotosm_reference()
    segments = load_coastline_segments()

    points = (pw_points or []) + (hot_points or [])
    points.sort(key=lambda p: p["ref_id"])
    polygons = hot_polygons or []
    present = {
        "portwatch_ports": pw_points is not None,
        "hdx_sea_ports": hot_polygons is not None,
        "natural_earth_coastline": segments is not None,
    }
    points_arg = points if all(
        present[k] for k in ("portwatch_ports", "hdx_sea_ports")
    ) else None
    polygons_arg = polygons if present["hdx_sea_ports"] else None

    print(
        "portwatch India points: %d, hotosm reference points: %d, "
        "hotosm reference polygons: %d"
        % (
            len(pw_points) if pw_points is not None else 0,
            len(hot_points) if hot_points is not None else 0,
            len(hot_polygons) if hot_polygons is not None else 0,
        )
    )
    print(
        "coastline segments (India window): %s"
        % (len(segments) if segments is not None else 0)
    )

    formatted_rows = []
    rows_by_id = {}
    n_inside = 0
    for asset in ordered:
        asset_id = asset["asset_id"]
        lat_text = (asset.get("verified_lat") or "").strip()
        lon_text = (asset.get("verified_lon") or "").strip()
        if not lat_text or not lon_text:
            rows_by_id[asset_id] = null_row()
            continue
        lat, lon = float(lat_text), float(lon_text)
        row = compute_asset_features(
            lat, lon, points_arg, polygons_arg, segments
        )
        if row["mt_in_port_polygon"]:
            n_inside += 1
        observed = all(present.values())
        rows_by_id[asset_id] = row
        formatted_rows.append(format_row(asset, row, info, observed))

    formatted_rows.sort(key=lambda r: r["asset_id"])
    os.makedirs(os.path.dirname(FEATURES_CSV), exist_ok=True)
    with open(FEATURES_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=FEATURE_COLUMNS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(formatted_rows)

    with open(COVERAGE_JSON, encoding="utf-8") as f:
        payload = json.load(f)
    payload = update_coverage(payload, ordered, rows_by_id, info, present)
    with open(COVERAGE_JSON, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    print("reference points: %d" % len(points))
    print("feature rows: %d" % len(formatted_rows))
    print("assets inside a port polygon: %d" % n_inside)
    print("wrote %s" % FEATURES_CSV)
    print("updated %s" % COVERAGE_JSON)
    return {"n_rows": len(formatted_rows), "n_inside": n_inside}


if __name__ == "__main__":
    build()
