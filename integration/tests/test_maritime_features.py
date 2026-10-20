"""Tests for the maritime / port exposure features.

Offline: reads the committed sidecar, feature CSV, coverage JSON, ports
MANIFEST and the committed fixtures (tests/fixtures/portwatch_india.csv,
hot_ports_reference.geojson, ne_coast_india_segments.json). Never touches
the network.

The fixtures carry the full reference port set (37 PortWatch India rows
verbatim, 65 verbatim HOT reference features) and every Natural Earth
coastline segment in the India window, so the recomputation test below
re-derives every feature for every coordinate-bearing asset with
test-local geodesic code (pyproj.Geod directly - never the production
helpers) and requires agreement with the committed CSV.
"""

import csv
import json
import math
import os
import sys

import pytest
from pyproj import Geod

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

import build_maritime_features as B
import ingest_ports as ING
from firewall import FORBIDDEN_MASTER_COLUMNS, assert_no_forbidden_columns

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
SIDECAR_CSV = os.path.join(
    REPO_ROOT, "integration", "data_reference", "assets_verified.csv"
)
FEATURES_CSV = os.path.join(
    REPO_ROOT, "integration", "data_features", "maritime_features_asset.csv"
)
COVERAGE_JSON = os.path.join(
    REPO_ROOT, "integration", "data_output", "weather_coverage.json"
)
PORTS_MANIFEST_JSON = os.path.join(
    REPO_ROOT, "integration", "data_raw_cache", "ports", "MANIFEST.json"
)
FIXTURE_PW = os.path.join(
    REPO_ROOT, "integration", "tests", "fixtures", "portwatch_india.csv"
)
FIXTURE_HOT = os.path.join(
    REPO_ROOT,
    "integration",
    "tests",
    "fixtures",
    "hot_ports_reference.geojson",
)
FIXTURE_COAST = os.path.join(
    REPO_ROOT,
    "integration",
    "tests",
    "fixtures",
    "ne_coast_india_segments.json",
)

MT_SOURCES = ("portwatch_ports", "hdx_sea_ports", "natural_earth_coastline")
EARLIER_SOURCES = ("up_district_panel", "nasa_power_point", "osm_highways")
# Restated here so the test - not just the implementation - pins the
# exact reference-set predicate and thresholds.
RADIUS_100KM_M = 100000.0
COASTAL_THRESHOLD_M = 50000.0
COAST_WINDOW = (60.0, 100.0, 0.0, 40.0)
THROUGHPUT_REASON = "throughput_gated"
MT_MISSING_REASONS = {
    "portwatch_ports": "portwatch_source_unavailable",
    "hdx_sea_ports": "hdx_ports_unavailable",
    "natural_earth_coastline": "coastline_source_unavailable",
}

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


def load_reference():
    """(points, polygons) from the committed fixtures.

    Points: [(ref_id, ref_name, lat, lon, kind)]. Polygons: [(ref_id,
    ref_name, geometry)]. Sorted by ref_id, as the builder iterates.
    """
    with open(FIXTURE_PW, newline="", encoding="utf-8") as f:
        pw_rows = [
            r
            for r in csv.DictReader(f)
            if (r.get("ISO3") or "").strip() == "IND"
        ]
    points = [
        (
            (r.get("portid") or "").strip(),
            (r.get("portname") or "").strip(),
            float(r["lat"]),
            float(r["lon"]),
            "portwatch",
        )
        for r in pw_rows
    ]
    with open(FIXTURE_HOT, encoding="utf-8") as f:
        hot = json.load(f)
    for feat in hot["features"]:
        props = feat.get("properties") or {}
        geom = feat.get("geometry") or {}
        ref_id = (props.get("id") or "").strip()
        ref_name = (props.get("name") or "").strip() or ref_id
        if geom.get("type") == "Point":
            lon, lat = geom["coordinates"]
            points.append((ref_id, ref_name, lat, lon, "hotosm"))
    polygons = [
        (
            ((feat.get("properties") or {}).get("id") or "").strip(),
            ((feat.get("properties") or {}).get("name") or "").strip()
            or ((feat.get("properties") or {}).get("id") or "").strip(),
            feat["geometry"],
        )
        for feat in hot["features"]
        if (feat.get("geometry") or {}).get("type")
        in ("Polygon", "MultiPolygon")
    ]
    points.sort(key=lambda p: p[0])
    polygons.sort(key=lambda p: p[0])
    return points, polygons


def load_coast_segments():
    with open(FIXTURE_COAST, encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["provenance"]["window"] == {
        "lon_min": COAST_WINDOW[0],
        "lon_max": COAST_WINDOW[1],
        "lat_min": COAST_WINDOW[2],
        "lat_max": COAST_WINDOW[3],
    }
    return payload["segments"]


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


def t_in_ring(lon, lat, ring):
    inside = False
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % n]
        if (y1 > lat) != (y2 > lat):
            if x1 + (lat - y1) * (x2 - x1) / (y2 - y1) > lon:
                inside = not inside
    return inside


def t_in_geom(lon, lat, geometry):
    polys = (
        [geometry["coordinates"]]
        if geometry["type"] == "Polygon"
        else geometry["coordinates"]
    )
    for poly in polys:
        if not poly or not t_in_ring(lon, lat, poly[0]):
            continue
        if any(t_in_ring(lon, lat, hole) for hole in poly[1:]):
            continue
        return True
    return False


def t_expected(lat, lon, points, polygons, segments):
    """Full independent expectation for one asset point."""
    dist, nid, nname, within = None, None, None, 0
    for ref_id, ref_name, plat, plon, _kind in points:
        _, _, d = GEOD.inv(lon, lat, plon, plat)
        if dist is None or d < dist:
            dist, nid, nname = d, ref_id, ref_name
        if d <= RADIUS_100KM_M:
            within += 1
    inside = False
    for ref_id, ref_name, geom in polygons:
        if t_in_geom(lon, lat, geom):
            inside = True
            within += 1
            if dist is None or 0.0 < dist:
                dist, nid, nname = 0.0, ref_id, ref_name
            continue
        rings = (
            [geom["coordinates"]]
            if geom["type"] == "Polygon"
            else geom["coordinates"]
        )
        bdist = None
        for poly in rings:
            for a, b in zip(poly[0], poly[0][1:]):
                d = t_point_seg(lat, lon, a[1], a[0], b[1], b[0])
                if bdist is None or d < bdist:
                    bdist = d
        if bdist is not None:
            if bdist <= RADIUS_100KM_M:
                within += 1
            if dist is None or bdist < dist:
                dist, nid, nname = bdist, ref_id, ref_name
    pw = None
    for ref_id, ref_name, plat, plon, kind in points:
        if kind != "portwatch":
            continue
        _, _, d = GEOD.inv(lon, lat, plon, plat)
        if pw is None or d < pw[0]:
            pw = (d, ref_id, ref_name)
    coast = None
    for a, b in segments:
        d = t_point_seg(lat, lon, a[1], a[0], b[1], b[0])
        if coast is None or d < coast:
            coast = d
    return {
        "dist": dist,
        "nid": nid,
        "nname": nname,
        "within": within,
        "inside": inside,
        "pw": pw,
        "coast": coast,
    }


# --------------------------------------------------------------------------
# Fixture composition (pins the reference-set definition itself)
# --------------------------------------------------------------------------


def test_fixture_reference_set_composition():
    with open(FIXTURE_PW, newline="", encoding="utf-8") as f:
        pw_rows = list(csv.DictReader(f))
    assert len(pw_rows) == 37
    assert all((r.get("ISO3") or "").strip() == "IND" for r in pw_rows)
    assert len({r["portid"] for r in pw_rows}) == 37

    with open(FIXTURE_HOT, encoding="utf-8") as f:
        hot = json.load(f)
    feats = hot["features"]
    hot_points = [x for x in feats if x["geometry"]["type"] == "Point"]
    hot_polys = [
        x
        for x in feats
        if x["geometry"]["type"] in ("Polygon", "MultiPolygon")
    ]
    assert len(hot_points) == 5
    assert len(hot_polys) == 60
    assert len(feats) == 65
    # Independent restatement of the predicate: every HOT point has
    # harbour=yes; every HOT polygon has a port-identifying tag.
    for feat in hot_points:
        assert (feat["properties"] or {}).get("harbour") == "yes", feat[
            "properties"
        ].get("id")
    for feat in hot_polys:
        props = feat["properties"] or {}
        assert (
            props.get("industrial") == "port"
            or props.get("harbour") == "yes"
            or props.get("port") not in (None, "")
        ), props.get("id")
    # The builder implements exactly this definition.
    assert B.RADIUS_100KM_M == RADIUS_100KM_M
    assert B.COASTAL_THRESHOLD_M == COASTAL_THRESHOLD_M
    assert (
        B.COAST_LON_MIN,
        B.COAST_LON_MAX,
        B.COAST_LAT_MIN,
        B.COAST_LAT_MAX,
    ) == COAST_WINDOW
    assert B.THROUGHPUT_REASON == THROUGHPUT_REASON


# --------------------------------------------------------------------------
# Table shape and value contracts
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


def test_distances_counts_non_negative_and_flags_boolean():
    for row in load_features():
        for col in ("mt_dist_major_port_m", "mt_coast_dist_m", "mt_pw_dist_m"):
            value = fnum(row[col])
            assert value is None or value >= 0.0, (row["asset_id"], col)
        within = (row["mt_ports_within_100km"] or "").strip()
        assert within != "" and int(within) >= 0, row["asset_id"]
        assert row["mt_nearest_major_port_id"].strip(), row["asset_id"]
        assert row["mt_nearest_major_port_name"].strip(), row["asset_id"]
        assert row["mt_in_port_polygon"] in ("true", "false"), row["asset_id"]
        assert row["mt_coastal_flag"] in ("true", "false"), row["asset_id"]
        assert row["mt_pw_nearest_port_id"].strip(), row["asset_id"]
        assert row["is_imputed"] == "0", row["asset_id"]
        assert row["is_observed"] in ("0", "1"), row["asset_id"]


def test_throughput_is_null_everywhere_with_gated_reason():
    coverage = load_coverage()
    by_id = {e["asset_id"]: e for e in coverage["assets"]}
    for row in load_features():
        assert row["mt_throughput_teu_yr"] == "", row["asset_id"]
        assert row["mt_throughput_source"] == "", row["asset_id"]
        assert row["mt_throughput_year"] == "", row["asset_id"]
        pw = [
            s
            for s in by_id[row["asset_id"]]["sources"]
            if s["name"] == "portwatch_ports"
        ][0]
        assert pw["throughput_status"] == "no_coverage", row["asset_id"]
        assert pw["throughput_reason_code"] == THROUGHPUT_REASON, row[
            "asset_id"
        ]
    mt = coverage["mt"]
    assert mt["throughput"]["status"] == "unavailable"
    assert mt["throughput"]["reason_code"] == THROUGHPUT_REASON


def test_licence_source_and_no_imputation():
    with open(PORTS_MANIFEST_JSON, encoding="utf-8") as f:
        manifest = json.load(f)
    files = manifest["files"]
    for row in load_features():
        assert "IMF copyright and terms" in row["licence"], row["asset_id"]
        assert "ODbL 1.0" in row["licence"], row["asset_id"]
        assert "public domain" in row["licence"], row["asset_id"]
        # Per-source licence strings equal the manifest record exactly.
        assert row["portwatch_licence"] == files["portwatch"]["licence"]
        assert row["poly_licence"] == files["hotosm_ports"]["licence"]
        assert row["coast_licence"] == files["ne_coastline"]["licence"]
        for col in (
            "portwatch_url",
            "poly_url",
            "coast_url",
            "source_url",
            "portwatch_retrieval_date",
            "poly_retrieval_date",
            "coast_retrieval_date",
            "retrieval_date",
        ):
            assert (row[col] or "").strip(), (row["asset_id"], col)
        assert row["portwatch_url"] == files["portwatch"]["url"]
        assert row["poly_url"] == files["hotosm_ports"]["url"]
        assert row["coast_url"] == files["ne_coastline"]["url"]
        assert row["is_imputed"] == "0", row["asset_id"]


def test_firewall_holds():
    rows = load_features()
    assert rows
    assert_no_forbidden_columns(rows[0].keys(), "maritime_features_asset.csv")
    assert not (set(rows[0].keys()) & set(FORBIDDEN_MASTER_COLUMNS))


def test_missing_sources_give_nulls():
    row = B.compute_asset_features(28.0, 77.0, None, None, None)
    assert row["mt_dist_major_port_m"] is None
    assert row["mt_nearest_major_port_id"] is None
    assert row["mt_ports_within_100km"] is None
    assert row["mt_in_port_polygon"] is None
    assert row["mt_coast_dist_m"] is None
    assert row["mt_coastal_flag"] is None
    assert row["mt_throughput_teu_yr"] is None
    assert row["mt_pw_nearest_port_id"] is None


# --------------------------------------------------------------------------
# Fixture recomputation (offline re-derivation from committed sources)
# --------------------------------------------------------------------------


def test_fixture_recomputation_matches_committed_csv():
    sidecar = {r["asset_id"]: r for r in load_sidecar()}
    by_id = {r["asset_id"]: r for r in load_features()}
    points, polygons = load_reference()
    assert len(points) == 42
    assert len(polygons) == 60
    segments = load_coast_segments()
    assert len(segments) == 13232
    with open(FIXTURE_PW, newline="", encoding="utf-8") as f:
        pw_rows = {
            (r.get("portid") or "").strip(): r
            for r in csv.DictReader(f)
        }

    for asset_id in sorted(by_id):
        asset = sidecar[asset_id]
        lat, lon = float(asset["verified_lat"]), float(asset["verified_lon"])
        exp = t_expected(lat, lon, points, polygons, segments)
        row = by_id[asset_id]
        assert abs(exp["dist"] - fnum(row["mt_dist_major_port_m"])) <= 0.01, (
            asset_id,
            exp["dist"],
            row["mt_dist_major_port_m"],
        )
        assert row["mt_nearest_major_port_id"] == exp["nid"], asset_id
        assert row["mt_nearest_major_port_name"] == exp["nname"], asset_id
        assert int(row["mt_ports_within_100km"]) == exp["within"], (
            asset_id,
            exp["within"],
            row["mt_ports_within_100km"],
        )
        assert (row["mt_in_port_polygon"] == "true") == exp["inside"], asset_id
        assert abs(exp["coast"] - fnum(row["mt_coast_dist_m"])) <= 0.05, (
            asset_id,
            exp["coast"],
            row["mt_coast_dist_m"],
        )
        assert (
            row["mt_coastal_flag"] == "true"
        ) == (exp["coast"] <= COASTAL_THRESHOLD_M), asset_id
        # PortWatch companion columns against the verbatim fixture row.
        assert row["mt_pw_nearest_port_id"] == exp["pw"][1], asset_id
        assert row["mt_pw_nearest_port_name"] == exp["pw"][2], asset_id
        assert abs(exp["pw"][0] - fnum(row["mt_pw_dist_m"])) <= 0.01, asset_id
        prow = pw_rows[exp["pw"][1]]
        for col, key in (
            ("mt_pw_vessel_count_total_yr", "vessel_count_total"),
            ("mt_pw_vessel_count_container_yr", "vessel_count_container"),
        ):
            assert (row[col] or "").strip() == str(
                int(float(prow[key] or 0))
            ), (asset_id, col)
        for col, key in (
            ("mt_pw_share_import_pct", "share_country_maritime_import"),
            ("mt_pw_share_export_pct", "share_country_maritime_export"),
        ):
            assert abs(float(row[col]) - float(prow[key] or 0)) < 1e-9, (
                asset_id,
                col,
            )
        if asset_id in ("PT_021", "PT_022", "PT_032", "PT_039"):
            print(
                "%s: dist=%.1f nearest=%s within=%d inside=%s coast=%.1f"
                % (
                    asset_id,
                    exp["dist"],
                    exp["nid"],
                    exp["within"],
                    exp["inside"],
                    exp["coast"],
                )
            )


# --------------------------------------------------------------------------
# Grounded sanity on real assets (catches a silently empty join)
# --------------------------------------------------------------------------


def test_port_assets_close_to_ports_and_inland_far_from_coast():
    sidecar = {r["asset_id"]: r for r in load_sidecar()}
    by_id = {r["asset_id"]: r for r in load_features()}
    points, _polygons = load_reference()
    pw_points = [p for p in points if p[4] == "portwatch"]

    for aid in ("PT_021", "PT_022"):
        lat = float(sidecar[aid]["verified_lat"])
        lon = float(sidecar[aid]["verified_lon"])
        d = min(GEOD.inv(lon, lat, p[3], p[2])[2] for p in pw_points)
        csv_dist = fnum(by_id[aid]["mt_dist_major_port_m"])
        print(
            "%s nearest PW=%.1f csv_dist=%.1f nearest=%s inside=%s" % (
                aid,
                d,
                csv_dist,
                by_id[aid]["mt_nearest_major_port_id"],
                by_id[aid]["mt_in_port_polygon"],
            )
        )
        # A port terminal asset is geographically close to its own port
        # feature (JNPT and Mundra both observed under 10 km).
        assert csv_dist is not None and csv_dist < 15000.0, (aid, csv_dist)

    # Assets physically inside mapped port estates report zero distance.
    inside = [
        aid
        for aid in by_id
        if by_id[aid]["mt_in_port_polygon"] == "true"
    ]
    print("inside port polygons: %s" % inside)
    assert set(inside) == {"PT_032", "PT_039"}, inside
    for aid in inside:
        assert fnum(by_id[aid]["mt_dist_major_port_m"]) == 0.0, aid

    # Inland assets are far from the coast and not flagged coastal.
    for aid, minimum in (("RE_067", 300000.0), ("HW_002", 300000.0)):
        coast = fnum(by_id[aid]["mt_coast_dist_m"])
        assert coast is not None and coast > minimum, (aid, coast)
        assert by_id[aid]["mt_coastal_flag"] == "false", aid
    print(
        "RE_067 coast=%.1f HW_002 coast=%.1f"
        % (
            fnum(by_id["RE_067"]["mt_coast_dist_m"]),
            fnum(by_id["HW_002"]["mt_coast_dist_m"]),
        )
    )


# --------------------------------------------------------------------------
# Coverage consistency for the maritime family (earlier families intact)
# --------------------------------------------------------------------------


def test_mt_coverage_matches_feature_table_and_manifest():
    coverage = load_coverage()
    with open(PORTS_MANIFEST_JSON, encoding="utf-8") as f:
        manifest = json.load(f)
    dates = {
        "portwatch_ports": manifest["files"]["portwatch"]["retrieval_date"],
        "hdx_sea_ports": manifest["files"]["hotosm_ports"]["retrieval_date"],
        "natural_earth_coastline": manifest["files"]["ne_coastline"][
            "retrieval_date"
        ],
    }
    sidecar = {r["asset_id"]: r for r in load_sidecar()}
    by_id = {r["asset_id"]: r for r in load_features()}

    for entry in coverage["assets"]:
        aid = entry["asset_id"]
        names = [s["name"] for s in entry["sources"]]
        assert names == list(EARLIER_SOURCES) + list(MT_SOURCES), (aid, names)
        has_coords = bool(
            (sidecar[aid].get("verified_lat") or "").strip()
            and (sidecar[aid].get("verified_lon") or "").strip()
        )
        mt_entries = {s["name"]: s for s in entry["sources"][-3:]}
        if not has_coords:
            assert aid not in by_id
            for name in MT_SOURCES:
                assert mt_entries[name]["status"] == "no_coverage", aid
                assert (
                    mt_entries[name]["reason_code"]
                    == "asset_coordinate_unverified"
                ), aid
                assert mt_entries[name]["retrieval_date"] is None, aid
                assert mt_entries[name]["distance_km"] is None, aid
            continue
        row = by_id[aid]
        pw = mt_entries["portwatch_ports"]
        assert pw["status"] == "covered", aid
        assert pw["retrieval_date"] == dates["portwatch_ports"], aid
        assert pw["distance_km"] == round(
            fnum(row["mt_dist_major_port_m"]) / 1000.0, 3
        ), aid
        assert pw["throughput_status"] == "no_coverage", aid
        assert pw["throughput_reason_code"] == THROUGHPUT_REASON, aid
        hx = mt_entries["hdx_sea_ports"]
        assert hx["status"] == "covered", aid
        assert hx["retrieval_date"] == dates["hdx_sea_ports"], aid
        cs = mt_entries["natural_earth_coastline"]
        assert cs["status"] == "covered", aid
        assert cs["retrieval_date"] == dates["natural_earth_coastline"], aid
        assert cs["distance_km"] == round(
            fnum(row["mt_coast_dist_m"]) / 1000.0, 3
        ), aid

    mt = coverage["mt"]
    assert mt["totals"]["per_source"] == {
        name: {"covered": 81, "no_coverage": 19} for name in MT_SOURCES
    }
    for block in mt["sources"]:
        assert block["by_reason_code"] == {"asset_coordinate_unverified": 19}
    assert mt["throughput"]["reason_code"] == THROUGHPUT_REASON


def test_earlier_families_intact():
    """wx and rd entries are byte-identical in meaning to before."""
    coverage = load_coverage()
    assert [s["name"] for s in coverage["sources"]] == [
        "up_district_panel",
        "nasa_power_point",
    ]
    assert coverage["totals"]["covered"] == 81
    assert coverage["totals"]["per_source"] == {
        "up_district_panel": {"covered": 11, "no_coverage": 89},
        "nasa_power_point": {"covered": 81, "no_coverage": 19},
    }
    rd = coverage["rd"]
    assert rd["totals"]["covered"] == 78
    assert rd["totals"]["per_source"] == {
        "osm_highways": {"covered": 78, "no_coverage": 22}
    }
    for entry in coverage["assets"]:
        for source in entry["sources"]:
            assert source["status"] in ("covered", "no_coverage")


# --------------------------------------------------------------------------
# Offline behaviour: clear failures, silent gaps never
# --------------------------------------------------------------------------


def test_ingest_offline_fails_clearly_on_missing_cache(tmp_path):
    with pytest.raises(ING.OfflineCacheError):
        ING.fetch_all(
            cache_dir=str(tmp_path / "empty"),
            manifest_path=str(tmp_path / "no_manifest.json"),
            offline=True,
            session=None,
        )


def test_builder_missing_manifest_fails_clearly(monkeypatch, tmp_path):
    monkeypatch.setattr(
        B, "PORTS_MANIFEST_JSON", str(tmp_path / "no_manifest.json")
    )
    with pytest.raises(FileNotFoundError):
        B.build()
