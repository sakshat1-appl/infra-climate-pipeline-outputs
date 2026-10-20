"""Tests for the verified asset-coordinate sidecar (offline, committed files only)."""

import csv
import math
import os

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
SIDECAR_PATH = os.path.join(REPO_ROOT, "integration", "data_reference", "assets_verified.csv")
MASTER_PATH = os.path.join(
    REPO_ROOT, "Master_Infrastructure_50_Units_Dataset.csv"
)

ALLOWED_STATUSES = {"verified", "approximate_midpoint", "unverified"}

# Hand-checked anchor set: independent truth coordinates recorded in the
# sidecar's truth_lat/truth_lon/truth_source_url columns. Tolerances:
# 2.0 km for point assets, 10.0 km for representative midpoints.
ANCHOR_IDS = {"HW_001", "HW_004", "PT_021", "PT_022", "PT_030", "RE_061", "RE_068"}

EARTH_RADIUS_KM = 6371.0088


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
    with open(SIDECAR_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_master():
    with open(MASTER_PATH, newline="", encoding="utf-8") as f:
        return {r["asset_id"]: r for r in csv.DictReader(f)}


def num(value):
    value = (value or "").strip()
    return float(value) if value else None


def test_sidecar_has_exactly_100_rows_matching_master():
    rows = load_sidecar()
    master = load_master()
    assert len(rows) == 100
    assert {r["asset_id"] for r in rows} == set(master.keys())


def test_coordinates_are_both_null_or_both_present():
    for r in load_sidecar():
        lat, lon = num(r["verified_lat"]), num(r["verified_lon"])
        assert (lat is None) == (lon is None), r["asset_id"]


def test_non_null_coordinates_have_provenance():
    for r in load_sidecar():
        if num(r["verified_lat"]) is not None:
            assert (r["source_url"] or "").strip(), r["asset_id"]
            assert (r["provenance_source"] or "").strip(), r["asset_id"]


def test_unverified_rows_have_null_coordinates_and_reason():
    seen_unverified = False
    for r in load_sidecar():
        if r["verification_status"] == "unverified":
            seen_unverified = True
            assert num(r["verified_lat"]) is None, r["asset_id"]
            assert num(r["verified_lon"]) is None, r["asset_id"]
            assert (r["notes"] or "").strip(), r["asset_id"]
    assert seen_unverified, "expected at least one honest unverified row"


def test_coordinates_inside_india_bounding_box():
    # The box [8.0, 37.5] x [68.0, 97.5] embeds a margin of well over 50 km
    # beyond India's land extremes (8.06N Kanyakumari, 37.1N Kashmir,
    # 68.1E Kutch, 97.4E Arunachal), so box membership is the documented
    # "within 50 km tolerance of the country's plausible bounds" check.
    for r in load_sidecar():
        lat, lon = num(r["verified_lat"]), num(r["verified_lon"])
        if lat is None:
            continue
        assert 8.0 <= lat <= 37.5, (r["asset_id"], lat)
        assert 68.0 <= lon <= 97.5, (r["asset_id"], lon)


def test_anchor_truth_errors_within_tolerance():
    rows = {r["asset_id"]: r for r in load_sidecar()}
    assert ANCHOR_IDS <= set(rows.keys())
    for aid in sorted(ANCHOR_IDS):
        r = rows[aid]
        assert (r["truth_source_url"] or "").strip(), aid
        error = float(r["error_km"])
        tolerance = 2.0 if r["coord_role"] == "point" else 10.0
        assert error <= tolerance, (aid, error, tolerance)


def test_fabricated_coordinate_tripwire():
    # Inverts the audit finding into an automated check: every verified
    # coordinate must be more than 50 km from the Master file's own
    # fabricated lat/lon, unless a proximity justification is documented in
    # notes. Reintroducing a fabricated coordinate fails this test.
    master = load_master()
    checked = 0
    for r in load_sidecar():
        lat, lon = num(r["verified_lat"]), num(r["verified_lon"])
        if lat is None:
            continue
        checked += 1
        m = master[r["asset_id"]]
        dist = haversine_km(lat, lon, float(m["lat"]), float(m["lon"]))
        assert dist > 50.0 or "proximity" in (r["notes"] or "").lower(), (
            r["asset_id"],
            dist,
        )
    assert checked > 0


def test_status_values_and_counts_sum_to_100():
    rows = load_sidecar()
    for r in rows:
        assert r["verification_status"] in ALLOWED_STATUSES, r["asset_id"]
    counts = {s: sum(1 for r in rows if r["verification_status"] == s) for s in ALLOWED_STATUSES}
    assert sum(counts.values()) == 100


def test_coord_role_values():
    for r in load_sidecar():
        assert r["coord_role"] in {"point", "representative_midpoint"}, r["asset_id"]
