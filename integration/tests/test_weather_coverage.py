"""Tests for the unified two-source weather coverage report and join gates.

Offline: reads the committed sidecar, coverage JSON and both feature-file
families. The UP district join gate is re-derived independently with a
from-scratch haversine implementation; the NASA POWER point source is
verified against the sidecar coordinates and the manifest.
"""

import csv
import json
import math
import os

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
SIDECAR_CSV = os.path.join(
    REPO_ROOT, "integration", "data_reference", "assets_verified.csv"
)
REFERENCE_CSV = os.path.join(
    REPO_ROOT, "integration", "data_clean", "district_reference.csv"
)
COVERAGE_JSON = os.path.join(
    REPO_ROOT, "integration", "data_output", "weather_coverage.json"
)
UP_FEATURES_DAY_CSV = os.path.join(
    REPO_ROOT, "integration", "data_features", "rainfall_features_asset_day.csv"
)
UP_FEATURES_MONTH_CSV = os.path.join(
    REPO_ROOT, "integration", "data_features", "rainfall_features_asset_month.csv"
)
PANINDIA_FEATURES_DAY_GZ = os.path.join(
    REPO_ROOT,
    "integration",
    "data_features",
    "rainfall_features_panindia_asset_day.csv.gz",
)
PANINDIA_FEATURES_MONTH_CSV = os.path.join(
    REPO_ROOT,
    "integration",
    "data_features",
    "rainfall_features_panindia_asset_month.csv",
)
MANIFEST_JSON = os.path.join(
    REPO_ROOT, "integration", "data_raw_cache", "power_points", "MANIFEST.json"
)

EARTH_RADIUS_KM = 6371.0088
UP_BOX_LAT_MIN, UP_BOX_LAT_MAX = 23.5, 31.0
UP_BOX_LON_MIN, UP_BOX_LON_MAX = 77.0, 85.0
MAX_JOIN_DISTANCE_KM = 75.0
UP_SOURCE = "up_district_panel"
POINT_SOURCE = "nasa_power_point"
OSM_SOURCE = "osm_highways"
MT_SOURCES = ("portwatch_ports", "hdx_sea_ports", "natural_earth_coastline")
REASON_CODES = {
    "non_UP_asset",
    "no_district_within_75km",
    "asset_coordinate_unverified",
}
# Reason codes for the highway family (asserted in detail by
# test_highway_features.py; listed here so the honest-accounting test
# below accepts them on osm_highways entries only).
OSM_REASON_CODES = {
    "asset_coordinate_unverified",
    "osm_tile_not_cached",
    "osm_fetch_failed",
    "no_nh_in_25km",
    "no_nodes_in_radius",
}
# Reason codes for the maritime family (asserted in detail by
# test_maritime_features.py; listed here so the honest-accounting test
# below accepts them on maritime entries only).
MT_REASON_CODES = {
    "asset_coordinate_unverified",
    "portwatch_source_unavailable",
    "hdx_ports_unavailable",
    "coastline_source_unavailable",
}


def haversine_km(lat1, lon1, lat2, lon2):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def load_sidecar():
    with open(SIDECAR_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_centroids():
    centroids = {}
    with open(REFERENCE_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            centroids[row["district_canonical"]] = (
                float(row["centroid_lat"]),
                float(row["centroid_lon"]),
            )
    return centroids


def load_coverage():
    with open(COVERAGE_JSON, encoding="utf-8") as f:
        return json.load(f)


def load_feature_asset_ids(path, compressed=False):
    ids = set()
    if compressed:
        import gzip

        opener = gzip.open(path, "rt", encoding="utf-8", newline="")
    else:
        opener = open(path, newline="", encoding="utf-8")
    with opener as f:
        for row in csv.DictReader(f):
            ids.add(row["asset_id"])
    return ids


def inside_up_box(lat, lon):
    return (
        UP_BOX_LAT_MIN <= lat <= UP_BOX_LAT_MAX
        and UP_BOX_LON_MIN <= lon <= UP_BOX_LON_MAX
    )


def source_entry(asset_entry, name):
    matches = [s for s in asset_entry["sources"] if s["name"] == name]
    assert len(matches) == 1, "%s: expected exactly one %s entry" % (
        asset_entry["asset_id"],
        name,
    )
    return matches[0]


# --------------------------------------------------------------------------
# Unified accounting
# --------------------------------------------------------------------------


def test_coverage_accounting():
    coverage = load_coverage()
    entries = coverage["assets"]
    sidecar_ids = {r["asset_id"] for r in load_sidecar()}
    assert len(entries) == 100
    assert {e["asset_id"] for e in entries} == sidecar_ids

    covered = sum(1 for e in entries if any(s["status"] == "covered" for s in e["sources"]))
    no_coverage = sum(1 for e in entries if not any(s["status"] == "covered" for s in e["sources"]))
    assert covered + no_coverage == 100
    totals = coverage["totals"]
    assert totals["assets"] == 100
    assert totals["covered"] == covered == totals["covered_from_at_least_one_source"]
    assert totals["no_coverage"] == no_coverage

    # Per-source blocks exist, are internally consistent and match the data.
    # The wx blocks below are frozen by phases 3-4; the highway family lives
    # in coverage["rd"] and per-asset osm_highways entries (see
    # test_highway_features.py). Both are checked here for honest accounting.
    source_names = [s["name"] for s in coverage["sources"]]
    assert source_names == [UP_SOURCE, POINT_SOURCE]
    assert "rd" in coverage, "highway family block missing from coverage"
    rd_blocks = {s["name"]: s for s in coverage["rd"]["sources"]}
    assert set(rd_blocks) == {OSM_SOURCE}
    assert "mt" in coverage, "maritime family block missing from coverage"
    mt_blocks = {s["name"]: s for s in coverage["mt"]["sources"]}
    assert set(mt_blocks) == set(MT_SOURCES)
    for block in coverage["sources"]:
        counts = block["status_counts"]
        assert counts["covered"] + counts["no_coverage"] == 100
        assert sum(block["by_reason_code"].values()) == counts["no_coverage"]
        assert totals["per_source"][block["name"]] == {
            "covered": counts["covered"],
            "no_coverage": counts["no_coverage"],
        }
    for block in coverage["mt"]["sources"]:
        counts = block["status_counts"]
        assert counts["covered"] + counts["no_coverage"] == 100
        assert sum(block["by_reason_code"].values()) == counts["no_coverage"]

    for entry in entries:
        assert entry["family"] == "wx"
        names = [s["name"] for s in entry["sources"]]
        assert names == [UP_SOURCE, POINT_SOURCE, OSM_SOURCE] + list(
            MT_SOURCES
        )
        for source in entry["sources"]:
            assert source["source_name"]
            if source["name"] == OSM_SOURCE:
                # Highway entries are owned by test_highway_features.py;
                # here only the status/reason contract is pinned.
                assert source["status"] in ("covered", "no_coverage")
                if source["status"] == "covered":
                    assert source["reason_code"] is None
                    assert source["retrieval_date"]
                    assert source["distance_km"] is not None
                else:
                    assert source["reason_code"] in OSM_REASON_CODES
                    assert source["distance_km"] is None
                continue
            if source["name"] in MT_SOURCES:
                # Maritime entries are owned by test_maritime_features.py;
                # here only the status/reason contract is pinned.
                assert source["status"] in ("covered", "no_coverage")
                if source["status"] == "covered":
                    assert source["reason_code"] is None
                    assert source["retrieval_date"]
                else:
                    assert source["reason_code"] in MT_REASON_CODES
                continue
            if source["status"] == "covered":
                assert source["reason_code"] is None
                assert source["retrieval_date"]
            else:
                assert source["reason_code"] in REASON_CODES
                # retrieval_date is null exactly when nothing was retrieved.
                # The point source never fetched unverified assets; the UP
                # source still carries its fixed panel constant for them.
                if (
                    source["reason_code"] == "asset_coordinate_unverified"
                    and source["name"] == POINT_SOURCE
                ):
                    assert source["retrieval_date"] is None, source
                else:
                    assert source["retrieval_date"], source


def test_up_join_gate():
    """The UP district entries match an independent haversine recomputation."""
    coverage = load_coverage()
    entries = {e["asset_id"]: e for e in coverage["assets"]}
    centroids = load_centroids()
    up_day_ids = load_feature_asset_ids(UP_FEATURES_DAY_CSV)
    up_month_ids = load_feature_asset_ids(UP_FEATURES_MONTH_CSV)
    assert up_day_ids == up_month_ids, "UP day and month tables cover the same assets"

    for asset in load_sidecar():
        aid = asset["asset_id"]
        up = source_entry(entries[aid], UP_SOURCE)
        lat_text = (asset.get("verified_lat") or "").strip()
        lon_text = (asset.get("verified_lon") or "").strip()
        lat = float(lat_text) if lat_text else None
        lon = float(lon_text) if lon_text else None
        status = (asset.get("verification_status") or "").strip()

        if status == "unverified" or lat is None or lon is None:
            assert up["status"] == "no_coverage", aid
            assert up["reason_code"] == "asset_coordinate_unverified", aid
            assert up["nearest_district"] is None, aid
            assert up["distance_km"] is None, aid
            assert aid not in up_day_ids
            continue

        # Independent nearest-district computation.
        nearest = min(
            centroids.items(),
            key=lambda kv: haversine_km(lat, lon, kv[1][0], kv[1][1]),
        )
        dist = haversine_km(lat, lon, nearest[1][0], nearest[1][1])

        if not inside_up_box(lat, lon):
            assert up["status"] == "no_coverage", aid
            assert up["reason_code"] == "non_UP_asset", aid
            assert up["nearest_district"] is None, aid
            assert up["distance_km"] is None, aid
            assert aid not in up_day_ids, (
                "non-UP asset %s must have no UP-district features" % aid
            )
            continue

        if dist > MAX_JOIN_DISTANCE_KM:
            assert up["status"] == "no_coverage", aid
            assert up["reason_code"] == "no_district_within_75km", aid
            assert up["nearest_district"] == nearest[0], aid
            assert math.isclose(up["distance_km"], round(dist, 3), abs_tol=1e-9), aid
            assert aid not in up_day_ids
            continue

        # Covered: recorded district and distance match the independent
        # recomputation, and the 75 km gate holds.
        assert up["status"] == "covered", aid
        assert up["nearest_district"] == nearest[0], aid
        assert math.isclose(up["distance_km"], round(dist, 3), abs_tol=1e-9), aid
        assert up["distance_km"] <= MAX_JOIN_DISTANCE_KM, aid
        assert aid in up_day_ids


def test_up_feature_tables_cover_exactly_the_up_source_assets():
    coverage = load_coverage()
    up_covered = {
        e["asset_id"]
        for e in coverage["assets"]
        if source_entry(e, UP_SOURCE)["status"] == "covered"
    }
    day_ids = load_feature_asset_ids(UP_FEATURES_DAY_CSV)
    month_ids = load_feature_asset_ids(UP_FEATURES_MONTH_CSV)
    assert day_ids == month_ids == up_covered
    assert len(up_covered) == 11


def test_nasa_power_point_source_covers_every_coordinate_asset():
    coverage = load_coverage()
    entries = {e["asset_id"]: e for e in coverage["assets"]}
    pan_day_ids = load_feature_asset_ids(PANINDIA_FEATURES_DAY_GZ, compressed=True)
    pan_month_ids = load_feature_asset_ids(PANINDIA_FEATURES_MONTH_CSV)
    assert pan_day_ids == pan_month_ids
    assert len(pan_day_ids) == 81

    with open(MANIFEST_JSON, encoding="utf-8") as f:
        manifest = json.load(f)

    for asset in load_sidecar():
        aid = asset["asset_id"]
        point = source_entry(entries[aid], POINT_SOURCE)
        lat_text = (asset.get("verified_lat") or "").strip()
        lon_text = (asset.get("verified_lon") or "").strip()
        if not lat_text or not lon_text:
            # Unverified coordinates: no fetch possible, documented reason.
            assert point["status"] == "no_coverage", aid
            assert point["reason_code"] == "asset_coordinate_unverified", aid
            assert point["nearest_asset_coordinate"] is None, aid
            assert aid not in pan_day_ids
            continue
        assert point["status"] == "covered", aid
        assert point["reason_code"] is None, aid
        assert point["nearest_asset_coordinate"] == [
            float(lat_text),
            float(lon_text),
        ], aid
        assert point["distance_km"] == 0.0, aid
        assert point["retrieval_date"] == manifest["assets"][aid]["retrieval_date"], aid
        assert aid in pan_day_ids


def test_coverage_reason_codes_are_honest():
    """Every asset is accounted for exactly once per source, with reasons."""
    coverage = load_coverage()
    buckets = {}
    for entry in coverage["assets"]:
        for source in entry["sources"]:
            assert source["status"] in ("covered", "no_coverage")
            if source["status"] == "no_coverage":
                if source["name"] == OSM_SOURCE:
                    allowed = OSM_REASON_CODES
                elif source["name"] in MT_SOURCES:
                    allowed = MT_REASON_CODES
                else:
                    allowed = REASON_CODES
                assert source["reason_code"] in allowed, source
                key = source["name"]
                buckets[key] = buckets.get(key, {})
                buckets[key][source["reason_code"]] = (
                    buckets[key].get(source["reason_code"], 0) + 1
                )
    for block in coverage["sources"]:
        expected = buckets.get(block["name"], {})
        assert block["by_reason_code"] == expected
    for block in coverage["rd"]["sources"]:
        expected = buckets.get(block["name"], {})
        assert block["by_reason_code"] == expected
    for block in coverage["mt"]["sources"]:
        expected = buckets.get(block["name"], {})
        assert block["by_reason_code"] == expected


def test_covered_assets_have_nonnull_features():
    coverage = load_coverage()
    up_rows = {}
    with open(UP_FEATURES_DAY_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            up_rows.setdefault(r["asset_id"], []).append(r)
    pan_rows = {}
    import gzip

    with gzip.open(
        PANINDIA_FEATURES_DAY_GZ, "rt", encoding="utf-8", newline=""
    ) as f:
        for r in csv.DictReader(f):
            pan_rows.setdefault(r["asset_id"], []).append(r)

    for entry in coverage["assets"]:
        aid = entry["asset_id"]
        # Mid-panel rows for covered assets carry real (non-null) features.
        if source_entry(entry, UP_SOURCE)["status"] == "covered":
            sample = [
                r
                for r in up_rows[aid]
                if r["date"] == "2015-07-15" and r["wx_precip_30d_mm"] != ""
            ]
            assert len(sample) == 1, aid
        if source_entry(entry, POINT_SOURCE)["status"] == "covered":
            sample = [
                r
                for r in pan_rows[aid]
                if r["date"] == "2015-07-15" and r["wx_precip_30d_mm"] != ""
            ]
            assert len(sample) == 1, aid
