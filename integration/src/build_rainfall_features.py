"""Climatology and asset-level rainfall features from the cleaned UP panel.

Reads:
  * ``integration/data_clean/weather_up_daily.csv``   (canonical district panel)
  * ``integration/data_clean/district_reference.csv`` (district centroids)
  * ``integration/data_reference/assets_verified.csv`` (asset sidecar)

Writes:
  * ``integration/data_clean/district_climatology.csv``
  * ``integration/data_features/rainfall_features_asset_day.csv``
  * ``integration/data_features/rainfall_features_asset_month.csv``

All climatology, feature and join logic lives in
``rainfall_features_core`` so the UP district path and the pan-India asset
path share ONE implementation (see ``build_panindia_features.py``). This
module only loads the UP panel, writes the district climatology and the
two per-asset feature tables, and prints the join summary. The unified
two-source coverage report is written by ``build_panindia_features.py``.

Join rule (fixed, do not redesign): for each asset with non-null coordinates
whose ``verification_status`` is not ``unverified``, find the nearest UP
district centroid by haversine distance (mean Earth radius 6371.0088 km).
Join that district's daily series ONLY IF the asset lies inside the Uttar
Pradesh bounding box (latitude 23.5..31.0, longitude 77.0..85.0) AND the
distance is <= 75 km. Every other asset receives no weather: its feature
rows are absent and it appears in the unified coverage report with a
reason code. Never join UP rainfall to a non-UP asset.

Deterministic: stdlib only, fixed constants, sorted iteration, LF line
endings. Byte-identical across runs. No network access.
"""

import csv
import os
import sys

from rainfall_features_core import (
    DAILY_FEATURE_COLUMNS,
    MONTH_FEATURE_COLUMNS,
    compute_climatology,
    compute_daily_features,
    compute_month_features,
    fmt_number,
    full_calendar,
    join_assets,
    load_assets,
    load_centroids,
    load_panel,
    panel_months,
    write_climatology_csv,
)

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
PANEL_CSV = os.path.join(REPO_ROOT, "integration", "data_clean", "weather_up_daily.csv")
REFERENCE_CSV = os.path.join(
    REPO_ROOT, "integration", "data_clean", "district_reference.csv"
)
ASSETS_CSV = os.path.join(
    REPO_ROOT, "integration", "data_reference", "assets_verified.csv"
)
CLIMATOLOGY_CSV = os.path.join(
    REPO_ROOT, "integration", "data_clean", "district_climatology.csv"
)
FEATURES_DAY_CSV = os.path.join(
    REPO_ROOT, "integration", "data_features", "rainfall_features_asset_day.csv"
)
FEATURES_MONTH_CSV = os.path.join(
    REPO_ROOT, "integration", "data_features", "rainfall_features_asset_month.csv"
)

# Phase-3 UP feature tables: asset_id + district_canonical + date + features.
UP_DAILY_FEATURE_COLUMNS = ["asset_id", "district_canonical", "date"] + list(
    DAILY_FEATURE_COLUMNS
)
UP_MONTH_FEATURE_COLUMNS = ["asset_id", "district_canonical", "year_month"] + list(
    MONTH_FEATURE_COLUMNS
)


def build():
    precip_by_district = load_panel(PANEL_CSV)
    centroids = load_centroids(REFERENCE_CSV)
    assets = load_assets(ASSETS_CSV)

    entries, covered = join_assets(assets, centroids)

    climatology = compute_climatology(precip_by_district)
    write_climatology_csv(climatology, CLIMATOLOGY_CSV)

    calendar = full_calendar(precip_by_district)
    months = panel_months(calendar)

    # Daily features, computed once per district (assets may share one).
    daily_by_district = {}
    for asset_id in sorted(covered):
        district = covered[asset_id][0]
        if district not in daily_by_district:
            daily_by_district[district] = compute_daily_features(
                district, precip_by_district[district], calendar, climatology
            )

    os.makedirs(os.path.dirname(FEATURES_DAY_CSV), exist_ok=True)
    with open(FEATURES_DAY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=UP_DAILY_FEATURE_COLUMNS, lineterminator="\n"
        )
        writer.writeheader()
        for asset_id in sorted(covered):
            district = covered[asset_id][0]
            feats = daily_by_district[district]
            for day in calendar:
                row = feats[day.isoformat()]
                writer.writerow(
                    {
                        "asset_id": asset_id,
                        "district_canonical": district,
                        "date": day.isoformat(),
                        "wx_precip_7d_mm": fmt_number(row["wx_precip_7d_mm"]),
                        "wx_precip_30d_mm": fmt_number(row["wx_precip_30d_mm"]),
                        "wx_heavy_days_7d": row["wx_heavy_days_7d"],
                        "wx_very_heavy_days_30d": row["wx_very_heavy_days_30d"],
                        "wx_rain_days_30d": row["wx_rain_days_30d"],
                        "wx_cwd_30d": row["wx_cwd_30d"],
                        "wx_dry_spell_90d": row["wx_dry_spell_90d"],
                        "wx_p95_exceed_days_30d": row["wx_p95_exceed_days_30d"],
                        "wx_monsoon_to_date_mm": fmt_number(
                            row["wx_monsoon_to_date_mm"]
                        ),
                    }
                )

    month_by_district = {}
    for asset_id in sorted(covered):
        district = covered[asset_id][0]
        if district not in month_by_district:
            month_by_district[district] = compute_month_features(
                district, precip_by_district[district], months, climatology
            )

    os.makedirs(os.path.dirname(FEATURES_MONTH_CSV), exist_ok=True)
    with open(FEATURES_MONTH_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=UP_MONTH_FEATURE_COLUMNS, lineterminator="\n"
        )
        writer.writeheader()
        for asset_id in sorted(covered):
            district = covered[asset_id][0]
            feats = month_by_district[district]
            for year, month in months:
                row = feats[(year, month)]
                writer.writerow(
                    {
                        "asset_id": asset_id,
                        "district_canonical": district,
                        "year_month": "%04d-%02d" % (year, month),
                        "wx_precip_month_mm": fmt_number(row["wx_precip_month_mm"]),
                        "wx_rain_days_month": row["wx_rain_days_month"],
                        "wx_monsoon_JJAS_mm": fmt_number(
                            row["wx_monsoon_JJAS_mm"]
                        ),
                        "wx_anomaly_z": fmt_number(row["wx_anomaly_z"]),
                    }
                )

    print("assets: %d" % len(entries))
    print("covered: %d" % sum(1 for e in entries if e["status"] == "covered"))
    print(
        "no_coverage: %d"
        % sum(1 for e in entries if e["status"] == "no_coverage")
    )
    for entry in entries:
        if entry["status"] == "covered":
            print(
                "  covered %s -> %s (%.3f km)"
                % (
                    entry["asset_id"],
                    entry["nearest_district"],
                    entry["distance_km"],
                )
            )
    print("climatology rows: %d" % (len(precip_by_district) * 12))
    print("daily feature rows: %d" % (len(covered) * len(calendar)))
    print("month feature rows: %d" % (len(covered) * len(months)))
    print("wrote %s" % CLIMATOLOGY_CSV)
    print("wrote %s" % FEATURES_DAY_CSV)
    print("wrote %s" % FEATURES_MONTH_CSV)
    return {
        "n_assets": len(entries),
        "n_covered": sum(1 for e in entries if e["status"] == "covered"),
        "covered": {
            aid: {"district": d, "distance_km": round(dist, 3)}
            for aid, (d, dist) in covered.items()
        },
    }


if __name__ == "__main__":
    sys.exit(build() and 0)
