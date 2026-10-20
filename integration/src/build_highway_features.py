"""Highway-network exposure features for every asset with coordinates.

Reads:
  * ``integration/data_reference/assets_verified.csv`` (asset sidecar)
  * ``integration/data_raw_cache/osm_highways/<asset_id>.json`` (raw
    Overpass tiles written by ``ingest_osm_highways.py``)
  * ``integration/data_raw_cache/osm_highways/MANIFEST.json`` (per-tile
    retrieval dates and OSM extract timestamps)
  * ``integration/config/corridors.yaml`` (named corridor polylines)

Writes:
  * ``integration/data_features/highway_features_asset.csv`` (one row per
    asset with non-null coordinates, sorted by asset_id)
  * extends ``integration/data_output/weather_coverage.json`` in place by
    appending one ``osm_highways`` source entry per asset and adding a
    top-level ``rd`` family block. The pre-existing ``wx`` content is
    preserved verbatim (same keys, same values, same formatting).

Feature definitions (exactly these - no composite indices, no weights).
Per asset with non-null coordinates, from the asset's Overpass tile (asset
coordinate plus a 25 km buffer fetch box):
  * ``rd_dist_nh_m``: minimum geodesic distance in metres from the asset
    point to the nearest way tagged as a national highway (see NH_REF_RE).
  * ``rd_hwy_len_10km_m``: total length in metres of all ways tagged
    highway in {motorway, trunk, primary, secondary} within 10 km of the
    asset, each way's geometry clipped to that radius.
  * ``rd_nh_len_10km_m``: the same, restricted to national-highway ways.
  * ``rd_node_degree_mean_10km`` / ``rd_node_count_10km``: mean node degree
    of the road graph within 10 km (see NODE DEGREE below) and the number
    of graph nodes it was computed over.
  * ``rd_on_corridor_flag`` / ``rd_corridor_name``: whether the asset lies
    within 5 km of a corridor from corridors.yaml, and the matched name.

NH IDENTIFICATION. A way counts as a national highway iff at least one
``;``-separated token of its ``ref`` tag, stripped and upper-cased, fully
matches::

    NH_REF_RE = ^NH[\\s\\-]*\\d{1,3}[A-Z]?$

so "NH 44", "NH44" and "NH-44B" match while "SH 44", "NHDP", "NE 4" and
"NH 44 bypass" do not. Expressway refs (NE/NS/E series) are deliberately
EXCLUDED: they are not national highways. The builder prints the total
number of matched NH ways; the count is also stored in the coverage file.

NODE DEGREE. The road graph is built from the fetched ways' ``nodes``
arrays: each consecutive node pair in a way is one undirected segment, and
a node's degree is the number of such segments incident to it (counted
with multiplicity, i.e. a node shared by two ways accrues both). Only
nodes whose geodesic distance from the asset is <= 10 km enter the mean;
``rd_node_count_10km`` reports how many did. Small samples are reported
as-is WITH their node count (documented limitation, not a silent bare
number): a mean over 3 nodes is far less stable than one over 3000. With
zero nodes in radius the mean is null with reason ``no_nodes_in_radius``.

GEODESY. All distances and lengths use proper geodesic computation via
``pyproj.Geod`` (WGS84) - never planar arithmetic on degrees. Point to
polyline-segment distances and circle clipping are evaluated in a local
azimuthal-equidistant plane centred on the asset point (each endpoint
placed by its true geodesic azimuth and distance from the asset), which is
accurate to centimetres over these radii; sub-segment lengths are measured
with Geod on EPSG:4326 coordinates. Corridor polylines are densified to
<= 5 km sub-segments before matching so long straight legs cannot skip
over a nearby asset.

MISSING DATA. ``is_imputed`` is 0 on every row: nothing is ever filled.
Where a feature cannot be computed the value is null (empty CSV cell) and
the asset's ``osm_highways`` coverage entry is ``no_coverage`` with a
reason code: ``asset_coordinate_unverified`` (19 assets, no row emitted),
``osm_tile_not_cached`` / ``osm_fetch_failed`` (tile absent or fetch
recorded as failed in the manifest), ``no_nh_in_25km`` (tile fetched but
no NH way matched), ``no_nodes_in_radius`` (nearest NH found but no graph
node within 10 km). Lengths are 0.0 - never null - when the tile fetched
fine but no ways fall in radius (an empty neighbourhood is an observed
fact). The corridor flag is always computed (it needs no OSM data).

Deterministic: stdlib + pyproj + PyYAML, fixed constants, sorted
iteration, LF line endings. Byte-identical across runs. No network access.
"""

import csv
import datetime
import json
import math
import os
import re
import sys

from pyproj import Geod

import yaml

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
ASSETS_CSV = os.path.join(
    REPO_ROOT, "integration", "data_reference", "assets_verified.csv"
)
OSM_CACHE_DIR = os.path.join(
    REPO_ROOT, "integration", "data_raw_cache", "osm_highways"
)
OSM_MANIFEST_JSON = os.path.join(OSM_CACHE_DIR, "MANIFEST.json")
CORRIDORS_YAML = os.path.join(
    REPO_ROOT, "integration", "config", "corridors.yaml"
)
FEATURES_CSV = os.path.join(
    REPO_ROOT, "integration", "data_features", "highway_features_asset.csv"
)
COVERAGE_JSON = os.path.join(
    REPO_ROOT, "integration", "data_output", "weather_coverage.json"
)

RADIUS_10KM_M = 10000.0
CORRIDOR_THRESHOLD_M = 5000.0
CORRIDOR_MAX_SEG_M = 5000.0

# Indian national-highway ref pattern, applied per ';'-separated token of
# the way's ref tag after strip()/upper(). See module docstring.
NH_REF_RE = re.compile(r"^NH[\s\-]*\d{1,3}[A-Z]?$")

HIGHWAY_CLASSES = ("motorway", "trunk", "primary", "secondary")

OSM_SOURCE_NAME = "OpenStreetMap"
OSM_SOURCE_URL = "https://overpass-api.de/api/interpreter"
OSM_LICENCE = "ODbL 1.0, (c) OpenStreetMap contributors"
OSM_FAMILY_SOURCE = "osm_highways"

FEATURE_COLUMNS = [
    "asset_id",
    "sector",
    "name",
    "rd_dist_nh_m",
    "rd_hwy_len_10km_m",
    "rd_nh_len_10km_m",
    "rd_node_degree_mean_10km",
    "rd_node_count_10km",
    "rd_on_corridor_flag",
    "rd_corridor_name",
    "osm_extract_date",
    "source_name",
    "source_url",
    "licence",
    "retrieval_date",
    "is_observed",
    "is_imputed",
]

GEOD = Geod(ellps="WGS84")


# --------------------------------------------------------------------------
# Geodesy helpers (importable; validated by tests/test_geodesy.py)
# --------------------------------------------------------------------------


def _aeqd_xy(plat, plon, lat, lon):
    """Place (lat, lon) in a local metre plane centred on (plat, plon).

    x points east, y points north; each point is placed by its true
    geodesic azimuth and distance from the centre, i.e. a local
    azimuthal-equidistant projection. Exact at the centre, accurate to
    centimetres within tens of kilometres.
    """
    az, _, dist = GEOD.inv(plon, plat, lon, lat)
    theta = math.radians(az)
    return (dist * math.sin(theta), dist * math.cos(theta))


def point_to_segment_m(plat, plon, alat, alon, blat, blon):
    """Minimum geodesic distance (m) from point P to segment A-B.

    Evaluated in the local AEQD plane around P; degenerate segments
    (A == B) reduce to the geodesic point distance.
    """
    ax, ay = _aeqd_xy(plat, plon, alat, alon)
    bx, by = _aeqd_xy(plat, plon, blat, blon)
    dx, dy = bx - ax, by - ay
    seg_len2 = dx * dx + dy * dy
    if seg_len2 == 0.0:
        return math.hypot(ax, ay)
    t = max(0.0, min(1.0, -(ax * dx + ay * dy) / seg_len2))
    return math.hypot(ax + t * dx, ay + t * dy)


def point_to_polyline_m(plat, plon, coords):
    """Minimum geodesic distance (m) from P to a polyline.

    ``coords`` is a [(lat, lon), ...] sequence. Returns None for a polyline
    with no usable segment.
    """
    best = None
    for (alat, alon), (blat, blon) in zip(coords, coords[1:]):
        d = point_to_segment_m(plat, plon, alat, alon, blat, blon)
        if best is None or d < best:
            best = d
    return best


def clipped_length_within_radius_m(plat, plon, coords, radius_m):
    """Geodesic length (m) of the parts of a polyline within radius_m of P.

    Each segment is clipped to the circle in the local AEQD plane around P
    (solving the segment-circle intersection), and the inside portion is
    measured with Geod on EPSG:4326 coordinates.
    """
    total = 0.0
    for (alat, alon), (blat, blon) in zip(coords, coords[1:]):
        ax, ay = _aeqd_xy(plat, plon, alat, alon)
        bx, by = _aeqd_xy(plat, plon, blat, blon)
        dx, dy = bx - ax, by - ay
        a = dx * dx + dy * dy
        if a == 0.0:
            if ax * ax + ay * ay <= radius_m * radius_m:
                continue  # zero-length point contributes no length
            continue
        # Solve |A + t*D|^2 = R^2 for t in [0, 1].
        b = 2.0 * (ax * dx + ay * dy)
        c = ax * ax + ay * ay - radius_m * radius_m
        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            inside_a = c <= 0.0
            t0, t1 = (0.0, 1.0) if inside_a else (1.0, 0.0)
        else:
            sq = math.sqrt(disc)
            r0 = (-b - sq) / (2.0 * a)
            r1 = (-b + sq) / (2.0 * a)
            t0 = max(0.0, r0)
            t1 = min(1.0, r1)
        if t1 <= t0:
            continue
        # Map the inside parameter range back to lon/lat and measure.
        lon0 = alon + t0 * (blon - alon)
        lat0 = alat + t0 * (blat - alat)
        lon1 = alon + t1 * (blon - alon)
        lat1 = alat + t1 * (blat - alat)
        _, _, dist = GEOD.inv(lon0, lat0, lon1, lat1)
        total += dist
    return total


def polyline_length_m(coords):
    """Total geodesic length (m) of a [(lat, lon), ...] polyline."""
    total = 0.0
    for (alat, alon), (blat, blon) in zip(coords, coords[1:]):
        _, _, dist = GEOD.inv(alon, alat, blon, blat)
        total += dist
    return total


# --------------------------------------------------------------------------
# OSM tile handling
# --------------------------------------------------------------------------


def is_national_highway(tags):
    """True iff a way's ref tag matches the Indian NH pattern (NH_REF_RE)."""
    ref = (tags or {}).get("ref") or ""
    for token in ref.split(";"):
        if NH_REF_RE.match(token.strip().upper()):
            return True
    return False


def parse_tile(payload):
    """Overpass JSON payload -> list of way dicts.

    Each way: {"id", "highway", "tags", "node_ids", "coords"} with coords
    as [(lat, lon), ...] aligned to node_ids. Only ways whose highway tag
    is in HIGHWAY_CLASSES are returned (the server query already filters;
    this is a client-side re-check).
    """
    ways = []
    for el in payload.get("elements") or []:
        if not isinstance(el, dict) or el.get("type") != "way":
            continue
        tags = el.get("tags") or {}
        if tags.get("highway") not in HIGHWAY_CLASSES:
            continue
        node_ids = el.get("nodes") or []
        geometry = el.get("geometry") or []
        coords = [
            (g["lat"], g["lon"])
            for g in geometry
            if isinstance(g, dict) and "lat" in g and "lon" in g
        ]
        ways.append(
            {
                "id": el.get("id"),
                "highway": tags.get("highway"),
                "tags": tags,
                "node_ids": list(node_ids),
                "coords": coords,
            }
        )
    return ways


def load_tile(cache_dir, asset_id):
    """Parse one cached tile; returns None when the file is absent."""
    path = os.path.join(cache_dir, "%s.json" % asset_id)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return parse_tile(json.load(f))


def build_node_index(ways):
    """(node_id -> (lat, lon), {node_id: degree}) for the tile's ways."""
    node_coords = {}
    degree = {}
    for way in ways:
        ids = way["node_ids"]
        coords = way["coords"]
        for i, nid in enumerate(ids):
            if nid not in node_coords and i < len(coords):
                node_coords[nid] = coords[i]
            degree.setdefault(nid, 0)
        for a, b in zip(ids, ids[1:]):
            degree[a] = degree.get(a, 0) + 1
            degree[b] = degree.get(b, 0) + 1
    return node_coords, degree


# --------------------------------------------------------------------------
# Corridors
# --------------------------------------------------------------------------


def densify_legs(waypoints, max_seg_m=CORRIDOR_MAX_SEG_M):
    """Split a [[lat, lon], ...] polyline into <= max_seg_m sub-segments."""
    segments = []
    for (alat, alon), (blat, blon) in zip(waypoints, waypoints[1:]):
        _, _, leg_len = GEOD.inv(alon, alat, blon, blat)
        n = max(1, int(math.ceil(leg_len / max_seg_m)))
        for i in range(n):
            t0, t1 = i / n, (i + 1) / n
            segments.append(
                (
                    (alat + t0 * (blat - alat), alon + t0 * (blon - alon)),
                    (alat + t1 * (blat - alat), alon + t1 * (blon - alon)),
                )
            )
    return segments


def load_corridors(path=CORRIDORS_YAML):
    """corridors.yaml -> {name: {"label", "segments", "source_url"}}."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    corridors = {}
    for entry in (data or {}).get("corridors") or []:
        name = entry["name"]
        waypoints = [(float(p[0]), float(p[1])) for p in entry["waypoints"]]
        if len(waypoints) < 2:
            raise ValueError("corridor %r has fewer than 2 waypoints" % name)
        corridors[name] = {
            "label": entry.get("label", name),
            "segments": densify_legs(waypoints),
            "source_url": entry.get("source_url", ""),
        }
    if not corridors:
        raise ValueError("no corridors defined in %s" % path)
    return corridors


def match_corridor(lat, lon, corridors):
    """(corridor_name or None, distance_m) to the nearest corridor leg."""
    best_name, best_d = None, None
    for name in sorted(corridors):
        for (a, b) in corridors[name]["segments"]:
            d = point_to_segment_m(
                lat, lon, a[0], a[1], b[0], b[1]
            )
            if best_d is None or d < best_d:
                best_d = d
                best_name = name
    if best_name is not None and best_d < CORRIDOR_THRESHOLD_M:
        return best_name, best_d
    return None, best_d


# --------------------------------------------------------------------------
# Feature computation
# --------------------------------------------------------------------------


def compute_asset_features(lat, lon, ways, corridors):
    """All highway features for one asset point.

    Returns (row_values, info) where row_values holds the raw numeric
    values (None for nulls) keyed by CSV column, and info carries
    diagnostics (n_ways, n_nh_ways, corridor_distance_m).
    """
    nh_ways = [w for w in ways if is_national_highway(w["tags"])]

    dist_nh = None
    for way in nh_ways:
        d = point_to_polyline_m(lat, lon, way["coords"])
        if d is not None and (dist_nh is None or d < dist_nh):
            dist_nh = d

    hwy_len = sum(
        clipped_length_within_radius_m(lat, lon, w["coords"], RADIUS_10KM_M)
        for w in ways
    )
    nh_len = sum(
        clipped_length_within_radius_m(lat, lon, w["coords"], RADIUS_10KM_M)
        for w in nh_ways
    )

    node_coords, degree = build_node_index(ways)
    in_radius = []
    for nid, (nlat, nlon) in node_coords.items():
        _, _, d = GEOD.inv(lon, lat, nlon, nlat)
        if d <= RADIUS_10KM_M:
            in_radius.append(nid)
    node_count = len(in_radius)
    degree_mean = (
        sum(degree[n] for n in in_radius) / node_count if node_count else None
    )

    corridor_name, corridor_d = match_corridor(lat, lon, corridors)

    row = {
        "rd_dist_nh_m": dist_nh,
        "rd_hwy_len_10km_m": hwy_len,
        "rd_nh_len_10km_m": nh_len,
        "rd_node_degree_mean_10km": degree_mean,
        "rd_node_count_10km": node_count,
        "rd_on_corridor_flag": corridor_name is not None,
        "rd_corridor_name": corridor_name,
    }
    info = {
        "n_ways": len(ways),
        "n_nh_ways": len(nh_ways),
        "corridor_distance_m": corridor_d,
    }
    return row, info


def fmt_m(value):
    return "" if value is None else "%.3f" % value


def fmt_degree(value):
    return "" if value is None else "%.6f" % value


def format_row(asset, row, osm_extract_date, retrieval_date, observed):
    corridor_name = row["rd_corridor_name"] or ""
    return {
        "asset_id": asset["asset_id"],
        "sector": asset["sector"],
        "name": asset["name"],
        "rd_dist_nh_m": fmt_m(row["rd_dist_nh_m"]),
        "rd_hwy_len_10km_m": fmt_m(row["rd_hwy_len_10km_m"]),
        "rd_nh_len_10km_m": fmt_m(row["rd_nh_len_10km_m"]),
        "rd_node_degree_mean_10km": fmt_degree(
            row["rd_node_degree_mean_10km"]
        ),
        "rd_node_count_10km": row["rd_node_count_10km"],
        "rd_on_corridor_flag": "true"
        if row["rd_on_corridor_flag"]
        else "false",
        "rd_corridor_name": corridor_name,
        "osm_extract_date": osm_extract_date,
        "source_name": OSM_SOURCE_NAME,
        "source_url": OSM_SOURCE_URL,
        "licence": OSM_LICENCE,
        "retrieval_date": retrieval_date,
        "is_observed": "1" if observed else "0",
        "is_imputed": "0",
    }


# --------------------------------------------------------------------------
# Coverage update
# --------------------------------------------------------------------------


def build_osm_source_entry(asset_id, has_coords, row, manifest_tiles,
                           manifest_failures):
    """One osm_highways coverage entry for an asset."""
    if not has_coords:
        return {
            "name": OSM_FAMILY_SOURCE,
            "status": "no_coverage",
            "reason_code": "asset_coordinate_unverified",
            "source_name": OSM_SOURCE_NAME,
            "retrieval_date": None,
            "distance_km": None,
        }
    tile = manifest_tiles.get(asset_id)
    if tile is None:
        reason = (
            "osm_fetch_failed"
            if asset_id in manifest_failures
            else "osm_tile_not_cached"
        )
        return {
            "name": OSM_FAMILY_SOURCE,
            "status": "no_coverage",
            "reason_code": reason,
            "source_name": OSM_SOURCE_NAME,
            "retrieval_date": None,
            "distance_km": None,
        }
    if row["rd_dist_nh_m"] is None:
        # Tile fetched but no national-highway way matched in the tile.
        # Lengths (possibly 0.0) and corridor columns are still reported.
        return {
            "name": OSM_FAMILY_SOURCE,
            "status": "no_coverage",
            "reason_code": "no_nh_in_25km",
            "source_name": OSM_SOURCE_NAME,
            "retrieval_date": tile.get("retrieval_date"),
            "distance_km": None,
        }
    if row["rd_node_degree_mean_10km"] is None:
        # A nearest NH exists but no graph node falls within 10 km
        # (possible only with very sparse node spacing near the boundary).
        return {
            "name": OSM_FAMILY_SOURCE,
            "status": "no_coverage",
            "reason_code": "no_nodes_in_radius",
            "source_name": OSM_SOURCE_NAME,
            "retrieval_date": tile.get("retrieval_date"),
            "distance_km": round(row["rd_dist_nh_m"] / 1000.0, 3),
        }
    return {
        "name": OSM_FAMILY_SOURCE,
        "status": "covered",
        "reason_code": None,
        "source_name": OSM_SOURCE_NAME,
        "retrieval_date": tile.get("retrieval_date"),
        "distance_km": round(row["rd_dist_nh_m"] / 1000.0, 3),
    }


def update_coverage(payload, ordered_assets, rows_by_id, manifest):
    """Append osm_highways entries and the rd family block.

    ``payload`` is the parsed pre-existing weather_coverage.json (its wx
    content is never modified, only added to); ``rows_by_id`` maps
    asset_id -> raw numeric row dict; returns the extended payload.
    """
    manifest_tiles = (manifest or {}).get("tiles") or {}
    manifest_failures = (manifest or {}).get("failures") or {}

    # Strip any previous osm_highways entries first: this makes rebuilds
    # byte-identical (append-only would duplicate them on re-run).
    for existing in payload["assets"]:
        existing["sources"] = [
            s for s in existing["sources"] if s["name"] != OSM_FAMILY_SOURCE
        ]
    payload.pop("rd", None)

    counts = {"covered": 0, "no_coverage": 0, "by_reason_code": {}}
    by_id = {e["asset_id"]: e for e in payload["assets"]}
    for asset in ordered_assets:
        asset_id = asset["asset_id"]
        lat_text = (asset.get("verified_lat") or "").strip()
        lon_text = (asset.get("verified_lon") or "").strip()
        entry = build_osm_source_entry(
            asset_id,
            bool(lat_text and lon_text),
            rows_by_id[asset_id],
            manifest_tiles,
            manifest_failures,
        )
        by_id[asset_id]["sources"].append(entry)
        counts[entry["status"]] += 1
        if entry["reason_code"] is not None:
            counts["by_reason_code"][entry["reason_code"]] = (
                counts["by_reason_code"].get(entry["reason_code"], 0) + 1
            )

    payload["rd"] = {
        "family": "rd",
        "sources": [
            {
                "name": OSM_FAMILY_SOURCE,
                "source_name": OSM_SOURCE_NAME,
                "description": (
                    "OpenStreetMap highways (motorway/trunk/primary/"
                    "secondary) fetched per asset tile (25 km buffer) "
                    "from the Overpass API; national highways "
                    "identified by ref tag (see NH_REF_RE in "
                    "build_highway_features.py)."
                ),
                "status_counts": {
                    "covered": counts["covered"],
                    "no_coverage": counts["no_coverage"],
                },
                "by_reason_code": dict(sorted(counts["by_reason_code"].items())),
            }
        ],
        "join_rule": {
            "description": (
                "Per asset with non-null coordinates: rd_dist_nh_m is the "
                "minimum geodesic distance to the nearest ref-tagged "
                "national-highway way in the asset tile; lengths are "
                "clipped to 10 km; node degree is the mean incident "
                "way-segment count over graph nodes within 10 km; "
                "corridor flag is true within 5 km of a corridors.yaml "
                "polyline. Assets with unverified coordinates receive no "
                "highway features."
            ),
        },
        "totals": {
            "assets": len(ordered_assets),
            "covered": counts["covered"],
            "no_coverage": counts["no_coverage"],
            "per_source": {
                OSM_FAMILY_SOURCE: {
                    "covered": counts["covered"],
                    "no_coverage": counts["no_coverage"],
                }
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


def build():
    assets = load_assets()
    ordered = sorted(assets, key=lambda r: r["asset_id"])
    corridors = load_corridors()
    try:
        with open(OSM_MANIFEST_JSON, encoding="utf-8") as f:
            manifest = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            "OSM manifest not found: %s (run ingest_osm_highways.py first)"
            % OSM_MANIFEST_JSON
        )
    manifest_tiles = (manifest or {}).get("tiles") or {}

    formatted_rows = []
    rows_by_id = {}
    n_nh_ways_total = 0
    n_corridor_hits = 0
    n_observed = 0
    for asset in ordered:
        asset_id = asset["asset_id"]
        lat_text = (asset.get("verified_lat") or "").strip()
        lon_text = (asset.get("verified_lon") or "").strip()
        if not lat_text or not lon_text:
            rows_by_id[asset_id] = {
                "rd_dist_nh_m": None,
                "rd_hwy_len_10km_m": None,
                "rd_nh_len_10km_m": None,
                "rd_node_degree_mean_10km": None,
                "rd_node_count_10km": None,
                "rd_on_corridor_flag": False,
                "rd_corridor_name": None,
            }
            continue
        lat, lon = float(lat_text), float(lon_text)
        tile = manifest_tiles.get(asset_id)
        ways = load_tile(OSM_CACHE_DIR, asset_id) if tile else None
        if ways is None:
            row = {
                "rd_dist_nh_m": None,
                "rd_hwy_len_10km_m": None,
                "rd_nh_len_10km_m": None,
                "rd_node_degree_mean_10km": None,
                "rd_node_count_10km": None,
                "rd_on_corridor_flag": False,
                "rd_corridor_name": None,
            }
            corridor_name, _ = match_corridor(lat, lon, corridors)
            row["rd_on_corridor_flag"] = corridor_name is not None
            row["rd_corridor_name"] = corridor_name
            rows_by_id[asset_id] = row
            formatted_rows.append(
                format_row(asset, row, "", "", observed=False)
            )
            continue
        row, info = compute_asset_features(lat, lon, ways, corridors)
        n_nh_ways_total += info["n_nh_ways"]
        if row["rd_corridor_name"]:
            n_corridor_hits += 1
        if row["rd_dist_nh_m"] is not None:
            n_observed += 1
        # A tile that fetched fine but holds zero ways still observed an
        # (empty) neighbourhood: lengths are 0.0, only dist/degree are null.
        observed = True
        rows_by_id[asset_id] = row
        formatted_rows.append(
            format_row(
                asset,
                row,
                tile.get("osm_extract_date") or "",
                tile.get("retrieval_date") or "",
                observed=observed,
            )
        )

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
    payload = update_coverage(payload, ordered, rows_by_id, manifest)
    with open(COVERAGE_JSON, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    n_coord = sum(
        1
        for a in ordered
        if (a.get("verified_lat") or "").strip()
        and (a.get("verified_lon") or "").strip()
    )
    print("assets with coordinates: %d" % n_coord)
    print("feature rows: %d" % len(formatted_rows))
    print("NH ways matched (all tiles): %d" % n_nh_ways_total)
    print("corridor hits: %d" % n_corridor_hits)
    print("covered (dist computed): %d" % n_observed)
    print("wrote %s" % FEATURES_CSV)
    print("updated %s" % COVERAGE_JSON)
    return {
        "n_rows": len(formatted_rows),
        "n_nh_ways": n_nh_ways_total,
        "n_corridor_hits": n_corridor_hits,
    }


if __name__ == "__main__":
    build()
