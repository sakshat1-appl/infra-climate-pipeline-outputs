"""Pan-India asset-level rainfall features from the NASA POWER point panel.

Reads:
  * ``integration/data_clean/weather_asset_daily.csv.gz`` (canonical asset
    daily panel written by ``clean_power_points.py``)
  * ``integration/data_raw_cache/power_points/MANIFEST.json`` (per-asset
    retrieval dates for the coverage report)
  * ``integration/data_reference/assets_verified.csv`` (asset sidecar)
  * ``integration/data_clean/district_reference.csv`` (UP district
    centroids, to report the UP-district-panel source in the coverage file)

Writes:
  * ``integration/data_clean/asset_climatology.csv`` (per-asset baseline
    stats, 2005-01-01..2020-12-31 - same window as the district baseline)
  * ``integration/data_features/rainfall_features_panindia_asset_day.csv.gz``
    (gzip because the 81 x 7638-day table exceeds the 20 MB committed
    artifact limit - see README for the read path)
  * ``integration/data_features/rainfall_features_panindia_asset_month.csv``
  * ``integration/data_output/weather_coverage.json`` (unified per-asset
    report covering BOTH sources: the UP district panel and the NASA
    POWER point API)

Feature definitions are exactly the phase-3 definitions and are computed by
the SAME shared core (``rainfall_features_core``) as the UP district path -
there is deliberately one implementation. Baselines are per asset per
calendar month over 2005-01-01..2020-12-31, so the two sources align.
All windows are trailing and backward-looking; a window that touches a
missing day is null, never filled.

Deterministic: stdlib only, fixed constants, sorted iteration, LF line
endings, gzip headers pinned to mtime=0. Byte-identical across runs. No
network access.
"""

import csv
import datetime
import gzip
import io
import json
import os
import sys

from rainfall_features_core import (
    DAILY_FEATURE_COLUMNS,
    MONTH_FEATURE_COLUMNS,
    RETRIEVAL_DATE,
    SOURCE_NAME,
    compute_climatology,
    compute_daily_features,
    compute_month_features,
    fmt_number,
    full_calendar,
    join_assets,
    load_assets,
    load_centroids,
    panel_months,
    write_climatology_csv,
)

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
ASSETS_CSV = os.path.join(
    REPO_ROOT, "integration", "data_reference", "assets_verified.csv"
)
REFERENCE_CSV = os.path.join(
    REPO_ROOT, "integration", "data_clean", "district_reference.csv"
)
MANIFEST_JSON = os.path.join(
    REPO_ROOT, "integration", "data_raw_cache", "power_points", "MANIFEST.json"
)
PANEL_GZ = os.path.join(
    REPO_ROOT, "integration", "data_clean", "weather_asset_daily.csv.gz"
)
ASSET_CLIMATOLOGY_CSV = os.path.join(
    REPO_ROOT, "integration", "data_clean", "asset_climatology.csv"
)
FEATURES_DAY_GZ = os.path.join(
    REPO_ROOT,
    "integration",
    "data_features",
    "rainfall_features_panindia_asset_day.csv.gz",
)
FEATURES_MONTH_CSV = os.path.join(
    REPO_ROOT,
    "integration",
    "data_features",
    "rainfall_features_panindia_asset_month.csv",
)
COVERAGE_JSON = os.path.join(
    REPO_ROOT, "integration", "data_output", "weather_coverage.json"
)

UP_SOURCE_NAME = "up_district_panel"
POINT_SOURCE_NAME = "nasa_power_point"

# Pan-India feature tables: asset_id + date + the shared feature columns.
PANINDIA_DAY_COLUMNS = ["asset_id", "date"] + list(DAILY_FEATURE_COLUMNS)
PANINDIA_MONTH_COLUMNS = ["asset_id", "year_month"] + list(MONTH_FEATURE_COLUMNS)


def load_asset_panel(path=PANEL_GZ):
    """Read the gzipped asset panel -> {asset_id: {date: precip_mm_day}}.

    Rows with an empty precip_mm_day (fill-value days omitted upstream or
    partially observed) are treated as absent, matching the UP panel's
    null-on-gap policy.
    """
    precip = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            asset_id = row["asset_id"]
            day = datetime.date.fromisoformat(row["date"])
            text = (row.get("precip_mm_day") or "").strip()
            if text:
                precip.setdefault(asset_id, {})[day] = float(text)
            else:
                precip.setdefault(asset_id, {})
    return precip


def compute_panindia_features(precip_by_asset, climatology, calendar, months):
    """Run the shared feature core over every asset.

    Returns ``(daily, monthly)`` keyed by asset_id; ``daily[aid]`` maps
    date_iso -> feature dict and ``monthly[aid]`` maps (year, month) ->
    feature dict. Pure functions of the panel: no lookahead, no filling.
    """
    daily = {}
    monthly = {}
    for asset_id in sorted(precip_by_asset):
        daily[asset_id] = compute_daily_features(
            asset_id, precip_by_asset[asset_id], calendar, climatology
        )
        monthly[asset_id] = compute_month_features(
            asset_id, precip_by_asset[asset_id], months, climatology
        )
    return daily, monthly


def write_panindia_day_gz(daily, calendar, path=FEATURES_DAY_GZ):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            with io.TextIOWrapper(gz, encoding="utf-8", newline="") as text:
                writer = csv.DictWriter(
                    text, fieldnames=PANINDIA_DAY_COLUMNS, lineterminator="\n"
                )
                writer.writeheader()
                for asset_id in sorted(daily):
                    feats = daily[asset_id]
                    for day in calendar:
                        row = feats[day.isoformat()]
                        writer.writerow(
                            {
                                "asset_id": asset_id,
                                "date": day.isoformat(),
                                "wx_precip_7d_mm": fmt_number(row["wx_precip_7d_mm"]),
                                "wx_precip_30d_mm": fmt_number(
                                    row["wx_precip_30d_mm"]
                                ),
                                "wx_heavy_days_7d": row["wx_heavy_days_7d"],
                                "wx_very_heavy_days_30d": row[
                                    "wx_very_heavy_days_30d"
                                ],
                                "wx_rain_days_30d": row["wx_rain_days_30d"],
                                "wx_cwd_30d": row["wx_cwd_30d"],
                                "wx_dry_spell_90d": row["wx_dry_spell_90d"],
                                "wx_p95_exceed_days_30d": row[
                                    "wx_p95_exceed_days_30d"
                                ],
                                "wx_monsoon_to_date_mm": fmt_number(
                                    row["wx_monsoon_to_date_mm"]
                                ),
                            }
                        )


def write_panindia_month_csv(monthly, months, path=FEATURES_MONTH_CSV):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=PANINDIA_MONTH_COLUMNS, lineterminator="\n"
        )
        writer.writeheader()
        for asset_id in sorted(monthly):
            feats = monthly[asset_id]
            for year, month in months:
                row = feats[(year, month)]
                writer.writerow(
                    {
                        "asset_id": asset_id,
                        "year_month": "%04d-%02d" % (year, month),
                        "wx_precip_month_mm": fmt_number(row["wx_precip_month_mm"]),
                        "wx_rain_days_month": row["wx_rain_days_month"],
                        "wx_monsoon_JJAS_mm": fmt_number(row["wx_monsoon_JJAS_mm"]),
                        "wx_anomaly_z": fmt_number(row["wx_anomaly_z"]),
                    }
                )


def build_unified_coverage(assets, centroids, manifest):
    """Per-asset ``sources`` entries for both sources, plus totals.

    The UP district source reuses the fixed phase-3 join gate from the
    shared core. The NASA POWER point source covers every asset with
    non-null coordinates that has a manifest entry (all of them once the
    ingest ran); unverified-coordinate assets get no_coverage with reason
    ``asset_coordinate_unverified`` for both sources.
    """
    up_entries, _covered = join_assets(assets, centroids)
    manifest_entries = (manifest or {}).get("assets") or {}

    per_source = {
        UP_SOURCE_NAME: {"covered": 0, "no_coverage": 0, "by_reason_code": {}},
        POINT_SOURCE_NAME: {"covered": 0, "no_coverage": 0, "by_reason_code": {}},
    }

    asset_entries = []
    for asset in sorted(assets, key=lambda r: r["asset_id"]):
        asset_id = asset["asset_id"]
        lat_text = (asset.get("verified_lat") or "").strip()
        lon_text = (asset.get("verified_lon") or "").strip()
        has_coords = bool(lat_text and lon_text)

        up_entry = next(e for e in up_entries if e["asset_id"] == asset_id)
        up_source = {
            "name": UP_SOURCE_NAME,
            "status": up_entry["status"],
            "reason_code": up_entry["reason_code"],
            "nearest_district": up_entry["nearest_district"],
            "distance_km": up_entry["distance_km"],
            "source_name": up_entry["source_name"],
            "retrieval_date": up_entry["retrieval_date"],
        }

        if has_coords and asset_id in manifest_entries:
            entry = manifest_entries[asset_id]
            lat = float(lat_text)
            lon = float(lon_text)
            point_source = {
                "name": POINT_SOURCE_NAME,
                "status": "covered",
                "reason_code": None,
                "nearest_asset_coordinate": [lat, lon],
                "distance_km": 0.0,
                "source_name": SOURCE_NAME,
                "retrieval_date": entry.get("retrieval_date"),
            }
        elif not has_coords:
            point_source = {
                "name": POINT_SOURCE_NAME,
                "status": "no_coverage",
                "reason_code": "asset_coordinate_unverified",
                "nearest_asset_coordinate": None,
                "distance_km": None,
                "source_name": SOURCE_NAME,
                "retrieval_date": None,
            }
        else:
            point_source = {
                "name": POINT_SOURCE_NAME,
                "status": "no_coverage",
                "reason_code": "power_point_not_cached",
                "nearest_asset_coordinate": None,
                "distance_km": None,
                "source_name": SOURCE_NAME,
                "retrieval_date": None,
            }

        for source in (up_source, point_source):
            counts = per_source[source["name"]]
            counts[source["status"]] += 1
            if source["reason_code"] is not None:
                counts["by_reason_code"][source["reason_code"]] = (
                    counts["by_reason_code"].get(source["reason_code"], 0) + 1
                )

        asset_entries.append(
            {
                "asset_id": asset_id,
                "family": "wx",
                "sources": [up_source, point_source],
            }
        )

    covered_overall = sum(
        1
        for e in asset_entries
        if any(s["status"] == "covered" for s in e["sources"])
    )
    # Note: reason codes are reported per source only. An asset can be
    # no_coverage from one source and covered from the other, so a single
    # overall reason code per asset does not exist.
    payload = {
        "family": "wx",
        "sources": [
            {
                "name": UP_SOURCE_NAME,
                "source_name": SOURCE_NAME,
                "description": (
                    "UP district panel join: nearest district centroid "
                    "within 75 km, assets inside the UP bounding box only."
                ),
                "status_counts": {
                    "covered": per_source[UP_SOURCE_NAME]["covered"],
                    "no_coverage": per_source[UP_SOURCE_NAME]["no_coverage"],
                },
                "by_reason_code": dict(
                    sorted(per_source[UP_SOURCE_NAME]["by_reason_code"].items())
                ),
            },
            {
                "name": POINT_SOURCE_NAME,
                "source_name": SOURCE_NAME,
                "description": (
                    "NASA POWER point API fetched at each asset's own "
                    "coordinate; covers every asset with non-null "
                    "coordinates."
                ),
                "status_counts": {
                    "covered": per_source[POINT_SOURCE_NAME]["covered"],
                    "no_coverage": per_source[POINT_SOURCE_NAME]["no_coverage"],
                },
                "by_reason_code": dict(
                    sorted(per_source[POINT_SOURCE_NAME]["by_reason_code"].items())
                ),
            },
        ],
        "join_rule": {
            "description": (
                "UP district source: each asset with non-null coordinates "
                "and verification_status != 'unverified' is joined to the "
                "nearest UP district centroid by haversine distance "
                "(mean Earth radius 6371.0088 km) ONLY IF the asset lies "
                "inside the Uttar Pradesh bounding box (latitude "
                "23.5..31.0, longitude 77.0..85.0) AND the distance is "
                "<= 75 km. The NASA POWER point source is joined at the "
                "asset's own coordinate (distance 0) for every asset with "
                "non-null coordinates. UP rainfall is never joined to a "
                "non-UP asset."
            ),
        },
        "assets": asset_entries,
        "totals": {
            "assets": len(asset_entries),
            "covered": covered_overall,
            "no_coverage": len(asset_entries) - covered_overall,
            "covered_from_at_least_one_source": covered_overall,
            "per_source": {
                name: {
                    "covered": per_source[name]["covered"],
                    "no_coverage": per_source[name]["no_coverage"],
                }
                for name in (UP_SOURCE_NAME, POINT_SOURCE_NAME)
            },
        },
    }
    return payload


def write_unified_coverage(payload, path=COVERAGE_JSON):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def build(panel_gz=PANEL_GZ):
    precip_by_asset = load_asset_panel(panel_gz)
    if not precip_by_asset:
        raise ValueError("asset panel %s is empty" % panel_gz)
    assets = load_assets(ASSETS_CSV)
    centroids = load_centroids(REFERENCE_CSV)
    with open(MANIFEST_JSON, encoding="utf-8") as f:
        manifest = json.load(f)

    calendar = full_calendar(precip_by_asset)
    months = panel_months(calendar)

    climatology = compute_climatology(precip_by_asset)
    write_climatology_csv(climatology, ASSET_CLIMATOLOGY_CSV, key_name="asset_id")

    daily, monthly = compute_panindia_features(
        precip_by_asset, climatology, calendar, months
    )
    write_panindia_day_gz(daily, calendar)
    write_panindia_month_csv(monthly, months)

    coverage = build_unified_coverage(assets, centroids, manifest)
    write_unified_coverage(coverage)

    print("assets: %d" % len(precip_by_asset))
    print("climatology rows: %d" % (len(precip_by_asset) * 12))
    print("daily feature rows: %d" % (len(precip_by_asset) * len(calendar)))
    print("month feature rows: %d" % (len(precip_by_asset) * len(months)))
    print(
        "coverage: %d covered, %d no_coverage (of %d)"
        % (
            coverage["totals"]["covered"],
            coverage["totals"]["no_coverage"],
            coverage["totals"]["assets"],
        )
    )
    print("wrote %s" % ASSET_CLIMATOLOGY_CSV)
    print("wrote %s" % FEATURES_DAY_GZ)
    print("wrote %s" % FEATURES_MONTH_CSV)
    print("wrote %s" % COVERAGE_JSON)
    return {"n_assets": len(precip_by_asset), "coverage": coverage["totals"]}


if __name__ == "__main__":
    sys.exit(build() and 0)
