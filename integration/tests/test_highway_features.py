"""Tests for the OSM highway exposure features.

Offline: reads the committed sidecar, feature CSV, coverage JSON, OSM
MANIFEST and the committed fixture tiles under tests/fixtures/osm_*.json.
Never touches the network.

The committed fixture tiles are full raw Overpass responses for a few
small-footprint assets (see tests/fixtures/README.md). The recomputation
test below re-derives rd_dist_nh_m and rd_hwy_len_10km_m from them with
test-local geodesic code (pyproj.Geod directly - never the production
helpers) and requires agreement with the committed CSV.
"""

import csv
import glob
import json
import math
import os
import re
import sys

import pytest
from pyproj import Geod

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

import build_highway_features as B
import ingest_osm_highways as ING
from firewall import FORBIDDEN_MASTER_COLUMNS, assert_no_forbidden_columns

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
SIDECAR_CSV = os.path.join(
    REPO_ROOT, "integration", "data_reference", "assets_verified.csv"
)
FEATURES_CSV = os.path.join(
    REPO_ROOT, "integration", "data_features", "highway_features_asset.csv"
)
COVERAGE_JSON = os.path.join(
    REPO_ROOT, "integration", "data_output", "weather_coverage.json"
)
OSM_MANIFEST_JSON = os.path.join(
    REPO_ROOT, "integration", "data_raw_cache", "osm_highways", "MANIFEST.json"
)
FIXTURE_GLOB = os.path.join(
    REPO_ROOT, "integration", "tests", "fixtures", "osm_*.json"
)

OSM_SOURCE = "osm_highways"
LICENCE = "ODbL 1.0, (c) OpenStreetMap contributors"
# The documented NH ref pattern, restated here so the test - not just the
# implementation - pins the exact definition.
NH_PATTERN = r"^NH[\s\-]*\d{1,3}[A-Z]?$"
NUMERIC_COLS = [
    "rd_dist_nh_m",
    "rd_hwy_len_10km_m",
    "rd_nh_len_10km_m",
    "rd_node_degree_mean_10km",
]

GEOD = Geod(ellps="WGS84")


def load_sidecar():
    with open(SIDECAR_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_features():
    with open(FEATURES_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_coverage():
    with open(COVERAGE_JSON, encoding="utf-8") as f:
        return json.load(f)


def osm_entry(asset_entry):
    matches = [s for s in asset_entry["sources"] if s["name"] == OSM_SOURCE]
    assert len(matches) == 1, asset_entry["asset_id"]
    return matches[0]


def fnum(text):
    text = (text or "").strip()
    return None if not text else float(text)


# --------------------------------------------------------------------------
# Independent (test-local) geodesic recomputation - no production helpers
# --------------------------------------------------------------------------


def t_aeqd(p_lat, p_lon, lat, lon):
    az, _, dist = GEOD.inv(p_lon, p_lat, lon, lat)
    th = math.radians(az)
    return dist * math.sin(th), dist * math.cos(th)


def t_point_seg(p_lat, p_lon, a_lat, a_lon, b_lat, b_lon):
    ax, ay = t_aeqd(p_lat, p_lon, a_lat, a_lon)
    bx, by = t_aeqd(p_lat, p_lon, b_lat, b_lon)
    dx, dy = bx - ax, by - ay
    den = dx * dx + dy * dy
    if den == 0.0:
        return math.hypot(ax, ay)
    t = max(0.0, min(1.0, -(ax * dx + ay * dy) / den))
    return math.hypot(ax + t * dx, ay + t * dy)


def t_clipped_length(p_lat, p_lon, coords, radius_m):
    total = 0.0
    for (a_lat, a_lon), (b_lat, b_lon) in zip(coords, coords[1:]):
        ax, ay = t_aeqd(p_lat, p_lon, a_lat, a_lon)
        bx, by = t_aeqd(p_lat, p_lon, b_lat, b_lon)
        dx, dy = bx - ax, by - ay
        a = dx * dx + dy * dy
        if a == 0.0:
            continue
        b = 2.0 * (ax * dx + ay * dy)
        c = ax * ax + ay * ay - radius_m * radius_m
        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            if c > 0.0:
                continue
            t0, t1 = 0.0, 1.0
        else:
            sq = math.sqrt(disc)
            t0 = max(0.0, (-b - sq) / (2.0 * a))
            t1 = min(1.0, (-b + sq) / (2.0 * a))
            if t1 <= t0:
                continue
        lon0 = a_lon + t0 * (b_lon - a_lon)
        lat0 = a_lat + t0 * (b_lat - a_lat)
        lon1 = a_lon + t1 * (b_lon - a_lon)
        lat1 = a_lat + t1 * (b_lat - a_lat)
        _, _, dist = GEOD.inv(lon0, lat0, lon1, lat1)
        total += dist
    return total


def t_is_nh(tags):
    ref = (tags or {}).get("ref") or ""
    return any(
        re.match(NH_PATTERN, tok.strip().upper()) for tok in ref.split(";")
    )


# --------------------------------------------------------------------------
# Table shape
# --------------------------------------------------------------------------


def test_exactly_one_row_per_coordinate_asset_sorted():
    sidecar = load_sidecar()
    with_coords = {
        r["asset_id"]
        for r in sidecar
        if (r.get("verified_lat") or "").strip()
        and (r.get("verified_lon") or "").strip()
    }
    without_coords = {r["asset_id"] for r in sidecar} - with_coords
    assert len(with_coords) == 81
    assert len(without_coords) == 19

    rows = load_features()
    assert [r["asset_id"] for r in rows] == sorted(r["asset_id"] for r in rows)
    assert {r["asset_id"] for r in rows} == with_coords
    assert not ({r["asset_id"] for r in rows} & without_coords)
    assert list(rows[0].keys()) == B.FEATURE_COLUMNS


def test_values_in_range_and_internally_consistent():
    for row in load_features():
        dist = fnum(row["rd_dist_nh_m"])
        hwy = fnum(row["rd_hwy_len_10km_m"])
        nh = fnum(row["rd_nh_len_10km_m"])
        deg = fnum(row["rd_node_degree_mean_10km"])
        count = (row["rd_node_count_10km"] or "").strip()
        for value in (dist, hwy, nh, deg):
            assert value is None or value >= 0.0, row["asset_id"]
        if hwy is not None and nh is not None:
            assert nh <= hwy + 1e-6, row["asset_id"]
        if deg is None:
            assert count in ("", "0"), row["asset_id"]
        else:
            assert int(count) > 0, row["asset_id"]
        flag = row["rd_on_corridor_flag"]
        assert flag in ("true", "false"), row["asset_id"]
        assert (row["rd_corridor_name"] != "") == (flag == "true"), row[
            "asset_id"
        ]


def test_licence_source_and_no_imputation():
    for row in load_features():
        assert row["licence"] == LICENCE, row["asset_id"]
        assert (row["source_url"] or "").strip(), row["asset_id"]
        assert row["source_name"].strip(), row["asset_id"]
        assert row["is_imputed"] == "0", row["asset_id"]
        assert row["is_observed"] in ("0", "1"), row["asset_id"]


def test_null_features_have_no_coverage_entries():
    coverage = load_coverage()
    by_id = {e["asset_id"]: e for e in coverage["assets"]}
    for row in load_features():
        entry = osm_entry(by_id[row["asset_id"]])
        nulls = [c for c in NUMERIC_COLS if not (row[c] or "").strip()]
        if nulls:
            assert entry["status"] == "no_coverage", (
                row["asset_id"],
                nulls,
            )
        if entry["status"] == "covered":
            assert not nulls, (row["asset_id"], nulls)


def test_firewall_holds():
    rows = load_features()
    assert rows
    assert_no_forbidden_columns(rows[0].keys(), "highway_features_asset.csv")
    assert not (set(rows[0].keys()) & set(FORBIDDEN_MASTER_COLUMNS))


def test_nh_regex_is_the_documented_definition():
    assert B.NH_REF_RE.pattern == NH_PATTERN
    cases = {
        "NH 44": True,
        "NH44": True,
        "NH-44B": True,
        "nh 19": True,
        "SH 22; NH 48": True,
        "SH 44": False,
        "NE 4": False,
        "NHDP": False,
        "NH 44 bypass": False,
        "": False,
    }
    for ref, expected in cases.items():
        assert t_is_nh({"ref": ref}) is expected, ref
        assert B.is_national_highway({"ref": ref}) is expected, ref


# --------------------------------------------------------------------------
# Fixture recomputation (offline re-derivation from committed raw tiles)
# --------------------------------------------------------------------------


def fixture_assets():
    paths = sorted(glob.glob(FIXTURE_GLOB))
    assert paths, "no committed OSM fixture tiles found"
    return paths


def test_fixture_recomputation_matches_committed_csv():
    sidecar = {r["asset_id"]: r for r in load_sidecar()}
    by_id = {r["asset_id"]: r for r in load_features()}
    assert len(fixture_assets()) >= 2
    for path in fixture_assets():
        asset_id = os.path.basename(path)[len("osm_") : -len(".json")]
        asset = sidecar[asset_id]
        lat, lon = float(asset["verified_lat"]), float(asset["verified_lon"])
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        ways = [
            el
            for el in payload["elements"]
            if el.get("type") == "way"
            and (el.get("tags") or {}).get("highway")
            in ("motorway", "trunk", "primary", "secondary")
        ]
        nh_ways = [w for w in ways if t_is_nh(w.get("tags"))]
        print("%s: %d ways, %d NH ways" % (asset_id, len(ways), len(nh_ways)))

        best = None
        for w in nh_ways:
            coords = [(g["lat"], g["lon"]) for g in w["geometry"]]
            for a, b in zip(coords, coords[1:]):
                d = t_point_seg(lat, lon, a[0], a[1], b[0], b[1])
                if best is None or d < best:
                    best = d
        row = by_id[asset_id]
        csv_dist = fnum(row["rd_dist_nh_m"])
        if best is None:
            assert csv_dist is None, asset_id
        else:
            assert csv_dist is not None, asset_id
            assert abs(best - csv_dist) <= 0.005, (asset_id, best, csv_dist)

        total = sum(
            t_clipped_length(
                lat,
                lon,
                [(g["lat"], g["lon"]) for g in w["geometry"]],
                10000.0,
            )
            for w in ways
        )
        assert abs(total - fnum(row["rd_hwy_len_10km_m"])) <= 0.05, (
            asset_id,
            total,
            row["rd_hwy_len_10km_m"],
        )


# --------------------------------------------------------------------------
# Grounded sanity on real assets (catches a silently empty join)
# --------------------------------------------------------------------------


def test_highway_assets_near_a_highway_and_ports_not_all_zero():
    sidecar = {r["asset_id"]: r for r in load_sidecar()}
    by_id = {r["asset_id"]: r for r in load_features()}
    hw = [
        (aid, sidecar[aid]["name"], fnum(by_id[aid]["rd_dist_nh_m"]))
        for aid in by_id
        if sidecar[aid]["sector"] == "Highways"
    ]
    for aid, name, dist in sorted(hw, key=lambda t: (t[2] is None, t[2])):
        print("HW %s %-22s dist_nh=%s" % (aid, name, dist))
    near = [(aid, n, d) for aid, n, d in hw if d is not None and d < 5000]
    assert near, "no expressway/highway asset within 5 km of an NH way"

    pairs = set()
    for aid in by_id:
        if sidecar[aid]["sector"] in ("Ports", "Renewable Energy"):
            row = by_id[aid]
            pairs.add(
                (
                    row["rd_dist_nh_m"],
                    row["rd_hwy_len_10km_m"],
                    row["rd_nh_len_10km_m"],
                )
            )
    print("distinct port/solar (dist,hwy,nh) tuples: %d" % len(pairs))
    assert len(pairs) > 1, "port/solar assets are identically valued"
    assert any(
        (fnum(by_id[aid]["rd_hwy_len_10km_m"]) or 0) > 0
        for aid in by_id
        if sidecar[aid]["sector"] in ("Ports", "Renewable Energy")
    ), "no port/solar asset has any highway length"


# --------------------------------------------------------------------------
# Coverage consistency for the highway family
# --------------------------------------------------------------------------


def test_osm_coverage_matches_feature_table_and_manifest():
    coverage = load_coverage()
    with open(OSM_MANIFEST_JSON, encoding="utf-8") as f:
        manifest = json.load(f)
    tiles = manifest.get("tiles") or {}
    failures = manifest.get("failures") or {}
    sidecar = {r["asset_id"]: r for r in load_sidecar()}
    by_id = {r["asset_id"]: r for r in load_features()}

    n_covered = 0
    reasons = {}
    for entry in coverage["assets"]:
        aid = entry["asset_id"]
        src = osm_entry(entry)
        has_coords = bool(
            (sidecar[aid].get("verified_lat") or "").strip()
            and (sidecar[aid].get("verified_lon") or "").strip()
        )
        if not has_coords:
            assert aid not in by_id
            assert src["status"] == "no_coverage"
            assert src["reason_code"] == "asset_coordinate_unverified"
            assert src["retrieval_date"] is None
            assert src["distance_km"] is None
            reasons[src["reason_code"]] = reasons.get(src["reason_code"], 0) + 1
            continue
        row = by_id[aid]
        dist = fnum(row["rd_dist_nh_m"])
        if src["status"] == "covered":
            n_covered += 1
            assert dist is not None, aid
            assert src["reason_code"] is None, aid
            assert src["distance_km"] == round(dist / 1000.0, 3), aid
            assert src["retrieval_date"] == tiles[aid]["retrieval_date"], aid
        else:
            assert dist is None, aid
            assert src["reason_code"] in (
                "osm_tile_not_cached",
                "osm_fetch_failed",
                "no_nh_in_25km",
                "no_nodes_in_radius",
            ), aid
            if src["reason_code"] == "osm_fetch_failed":
                assert aid in failures, aid
            reasons[src["reason_code"]] = reasons.get(src["reason_code"], 0) + 1
    print("osm_highways covered=%d reasons=%s" % (n_covered, reasons))
    rd = coverage["rd"]
    assert rd["totals"]["covered"] == n_covered
    assert rd["totals"]["assets"] == 100
    assert rd["sources"][0]["by_reason_code"] == dict(sorted(reasons.items()))


# --------------------------------------------------------------------------
# Offline behaviour: clear failures, silent gaps never
# --------------------------------------------------------------------------


def test_ingest_offline_fails_clearly_on_missing_cache(tmp_path):
    with pytest.raises(ING.OfflineCacheError):
        ING.fetch_all(
            offline=True,
            cache_dir=str(tmp_path / "empty"),
            manifest_path=str(tmp_path / "no_manifest.json"),
            session=None,
        )


def test_builder_missing_manifest_fails_clearly(monkeypatch, tmp_path):
    monkeypatch.setattr(
        B, "OSM_MANIFEST_JSON", str(tmp_path / "no_manifest.json")
    )
    with pytest.raises(FileNotFoundError):
        B.build()


def test_empty_tile_gives_zero_lengths_and_null_distance():
    corridors = B.load_corridors()
    row, info = B.compute_asset_features(28.0, 77.0, [], corridors)
    assert row["rd_dist_nh_m"] is None
    assert row["rd_hwy_len_10km_m"] == 0.0
    assert row["rd_nh_len_10km_m"] == 0.0
    assert row["rd_node_degree_mean_10km"] is None
    assert row["rd_node_count_10km"] == 0
    assert info["n_nh_ways"] == 0
