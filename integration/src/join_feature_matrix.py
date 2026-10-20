"""Join the per-family feature tables into release matrices.

Reads (never the Master file's quarantined columns - firewall enforced):
  * ``integration/data_reference/assets_verified.csv`` (identity +
    coordinates + coord_role/verification_status for all 100 assets)
  * ``integration/data_features/highway_features_asset.csv`` (81 rows)
  * ``integration/data_features/maritime_features_asset.csv`` (81 rows)
  * ``integration/data_features/rainfall_features_asset_month.csv``
    (UP district join, 11 assets x 251 months, with district_canonical)
  * ``integration/data_features/rainfall_features_panindia_asset_month.csv``
    (per-asset pull, 81 assets x 251 months)
  * ``integration/data_output/weather_coverage.json`` (reasons for NULLs,
    UP district/distance metadata)

Writes:
  * ``integration/data_output/feature_matrix_asset.csv`` - ONE row per
    asset (ALL 100, sorted by asset_id): identity (asset_id, sector,
    name, coord_role, verification_status, verified_lat, verified_lon),
    every ``rd_``/``mt_`` feature column verbatim, wx join metadata
    (which rainfall sources cover the asset; the UP district and its
    distance where joined), per-family provenance (``rd_``/``mt_``
    prefixed copies of each family's licence/source_url/retrieval_date/
    is_observed/is_imputed) and per-source coverage status + reason.
    The 19 unverified-coordinate assets appear with NULL feature cells
    and explicit reasons - never silently dropped. Rainfall MEASUREMENTS
    live in the month matrix; the asset matrix carries only wx join
    metadata (documented split, not missing data).
  * ``integration/data_output/feature_matrix_asset_month.csv`` -
    asset_id x year_month for the 81 coordinate-bearing assets x all 251
    months (20,331 rows, sorted): pan-India monthly columns for every
    row plus the UP-district monthly columns (NULL with empty
    district_canonical where the district join does not cover the
    asset). Grain proof: both inputs are unique per (asset_id,
    year_month), so the join cannot multiply rows (asserted).
  * ``integration/data_output/coverage_report.json`` - consolidated
    machine-readable coverage derived from weather_coverage.json: per
    asset per family {status, reason_code} plus totals overall and per
    family. weather_coverage.json itself is left intact.

Guarantees (asserted in code, tested): no row multiplication (row count
equals distinct key count), unique keys, deterministic sort, firewall on
every input and both outputs, every NULL feature cell explained by a
coverage reason (unverified coordinates, a no_coverage family entry, or
the family-level throughput gate).

Deterministic: stdlib only, sorted iteration, verbatim string copies
(no float round-trips), LF line endings. Byte-identical across runs.
No network access.
"""

import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from firewall import assert_no_forbidden_columns  # noqa: E402

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
SIDECAR_CSV = os.path.join(
    REPO_ROOT, "integration", "data_reference", "assets_verified.csv"
)
HIGHWAY_CSV = os.path.join(
    REPO_ROOT, "integration", "data_features", "highway_features_asset.csv"
)
MARITIME_CSV = os.path.join(
    REPO_ROOT, "integration", "data_features", "maritime_features_asset.csv"
)
UP_MONTH_CSV = os.path.join(
    REPO_ROOT, "integration", "data_features", "rainfall_features_asset_month.csv"
)
PANINDIA_MONTH_CSV = os.path.join(
    REPO_ROOT,
    "integration",
    "data_features",
    "rainfall_features_panindia_asset_month.csv",
)
COVERAGE_JSON = os.path.join(
    REPO_ROOT, "integration", "data_output", "weather_coverage.json"
)
MATRIX_ASSET_CSV = os.path.join(
    REPO_ROOT, "integration", "data_output", "feature_matrix_asset.csv"
)
MATRIX_MONTH_CSV = os.path.join(
    REPO_ROOT, "integration", "data_output", "feature_matrix_asset_month.csv"
)
COVERAGE_REPORT_JSON = os.path.join(
    REPO_ROOT, "integration", "data_output", "coverage_report.json"
)

UP_SOURCE = "up_district_panel"
POINT_SOURCE = "nasa_power_point"
OSM_SOURCE = "osm_highways"
MT_SOURCES = ("portwatch_ports", "hdx_sea_ports", "natural_earth_coastline")
ALL_SOURCES = (UP_SOURCE, POINT_SOURCE, OSM_SOURCE) + MT_SOURCES

# Provenance columns copied verbatim from each family's feature table,
# stored under an rd_/mt_ prefix that cannot collide: every rd_ feature
# column starts rd_dist/rd_hwy/rd_nh/rd_node/rd_on/rd_corridor/osm_, and
# every mt_ feature column starts mt_dist/mt_nearest/mt_ports/mt_in/
# mt_coast/mt_throughput/mt_pw_.
RD_PROV = ["source_name", "source_url", "licence", "retrieval_date",
           "is_observed", "is_imputed"]
MT_PROV = RD_PROV
# mt_ feature columns explained by the family-level throughput gate
# (not by a per-source status).
MT_THROUGHPUT_COLS = ("mt_throughput_teu_yr", "mt_throughput_source",
                      "mt_throughput_year")
THROUGHPUT_REASON = "throughput_gated"


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def check_keys(rows, key_fields, context):
    """No row multiplication: row count equals distinct key count."""
    keys = [tuple(r[k] for k in key_fields) for r in rows]
    assert len(keys) == len(set(keys)), (
        "%s: %d rows but only %d distinct keys %s"
        % (context, len(keys), len(set(keys)), key_fields)
    )
    return keys


def load_inputs():
    sidecar = read_csv(SIDECAR_CSV)
    highway = read_csv(HIGHWAY_CSV)
    maritime = read_csv(MARITIME_CSV)
    up_month = read_csv(UP_MONTH_CSV)
    pan_month = read_csv(PANINDIA_MONTH_CSV)
    for rows, context in (
        (sidecar, "assets_verified.csv"),
        (highway, "highway_features_asset.csv"),
        (maritime, "maritime_features_asset.csv"),
        (up_month, "rainfall_features_asset_month.csv"),
        (pan_month, "rainfall_features_panindia_asset_month.csv"),
    ):
        assert_no_forbidden_columns(rows[0].keys(), context)
    check_keys(sidecar, ["asset_id"], "sidecar")
    check_keys(highway, ["asset_id"], "highway features")
    check_keys(maritime, ["asset_id"], "maritime features")
    check_keys(up_month, ["asset_id", "year_month"], "UP monthly features")
    check_keys(
        pan_month, ["asset_id", "year_month"], "pan-India monthly features"
    )
    with open(COVERAGE_JSON, encoding="utf-8") as f:
        coverage = json.load(f)
    return sidecar, highway, maritime, up_month, pan_month, coverage


def coverage_lookup(coverage):
    """asset_id -> source name -> full source entry dict."""
    return {
        entry["asset_id"]: {s["name"]: s for s in entry["sources"]}
        for entry in coverage["assets"]
    }


def family_feature_columns(rows, prefix, exclude=()):
    cols = sorted(
        c for c in rows[0].keys()
        if c.startswith(prefix) and c not in exclude
    )
    # The rd_/mt_ provenance copies we add ourselves start with the same
    # prefix scheme but are not in the inputs; inputs carry bare names
    # (source_name, ...) so no input column collides. Guard anyway.
    assert not any(c.startswith("rd_prov_") or c.startswith("mt_prov_")
                   for c in cols)
    return cols


def build_asset_matrix(sidecar, highway, maritime, cov_lookup,
                       path=MATRIX_ASSET_CSV):
    """One row per asset (100), verbatim family values, NULLs explained."""
    hw_by_id = {r["asset_id"]: r for r in highway}
    mt_by_id = {r["asset_id"]: r for r in maritime}
    rd_cols = family_feature_columns(highway, "rd_")
    mt_cols = family_feature_columns(maritime, "mt_")

    header = (
        ["asset_id", "sector", "name", "coord_role", "verification_status",
         "verified_lat", "verified_lon"]
        + rd_cols
        + mt_cols
        + ["wx_up_covered", "wx_up_district", "wx_up_distance_km",
           "wx_point_covered"]
        + ["rd_prov_%s" % c for c in RD_PROV]
        + ["mt_prov_%s" % c for c in MT_PROV]
        + ["cov_%s_status" % s for s in ALL_SOURCES]
        + ["cov_%s_reason" % s for s in ALL_SOURCES]
        + ["null_reason"]
    )
    rows = []
    for asset in sorted(sidecar, key=lambda r: r["asset_id"]):
        aid = asset["asset_id"]
        lat = (asset.get("verified_lat") or "").strip()
        lon = (asset.get("verified_lon") or "").strip()
        has_coords = bool(lat and lon)
        hw = hw_by_id.get(aid, {})
        mt = mt_by_id.get(aid, {})
        cov = cov_lookup.get(aid, {})

        def cov_of(source):
            entry = cov.get(source, {})
            return entry.get("status", "no_coverage"), (
                entry.get("reason_code") or ""
            )

        row = {
            "asset_id": aid,
            "sector": asset["sector"],
            "name": asset["name"],
            "coord_role": asset.get("coord_role", ""),
            "verification_status": asset.get("verification_status", ""),
            "verified_lat": lat,
            "verified_lon": lon,
        }
        for c in rd_cols:
            row[c] = hw.get(c, "")
        for c in mt_cols:
            row[c] = mt.get(c, "")
        up = cov.get(UP_SOURCE, {})
        row["wx_up_covered"] = (
            "true" if up.get("status") == "covered" else "false"
        )
        row["wx_up_district"] = up.get("nearest_district") or ""
        row["wx_up_distance_km"] = (
            "" if up.get("distance_km") is None else str(up["distance_km"])
        )
        row["wx_point_covered"] = (
            "true" if cov_of(POINT_SOURCE)[0] == "covered" else "false"
        )
        for c in RD_PROV:
            row["rd_prov_%s" % c] = hw.get(c, "")
        for c in MT_PROV:
            row["mt_prov_%s" % c] = mt.get(c, "")
        for s in ALL_SOURCES:
            status, reason = cov_of(s)
            row["cov_%s_status" % s] = status
            row["cov_%s_reason" % s] = reason
        if not has_coords:
            row["null_reason"] = "asset_coordinate_unverified"
        else:
            missing = sorted(
                s for s in ALL_SOURCES if cov_of(s)[0] != "covered"
            )
            row["null_reason"] = (
                "" if not missing else "no_coverage:" + ",".join(missing)
            )
        rows.append(row)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows, header


def build_month_matrix(up_month, pan_month, path=MATRIX_MONTH_CSV):
    """asset_id x year_month (81 x 251): pan-India + UP-district columns."""
    pan_by_key = {(r["asset_id"], r["year_month"]): r for r in pan_month}
    up_by_key = {(r["asset_id"], r["year_month"]): r for r in up_month}
    pan_cols = [c for c in pan_month[0].keys()
                if c not in ("asset_id", "year_month")]
    up_cols = [c for c in up_month[0].keys()
               if c not in ("asset_id", "district_canonical", "year_month")]
    header = (
        ["asset_id", "year_month", "district_canonical"]
        + ["pan_%s" % c for c in pan_cols]
        + ["up_%s" % c for c in up_cols]
    )
    rows = []
    for asset_id, year_month in sorted(pan_by_key):
        pan = pan_by_key[(asset_id, year_month)]
        up = up_by_key.get((asset_id, year_month), {})
        row = {
            "asset_id": asset_id,
            "year_month": year_month,
            "district_canonical": up.get("district_canonical", ""),
        }
        for c in pan_cols:
            row["pan_%s" % c] = pan.get(c, "")
        for c in up_cols:
            row["up_%s" % c] = up.get(c, "")
        rows.append(row)
    check_keys(rows, ["asset_id", "year_month"], "month matrix")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows, header


def build_coverage_report(coverage, path=COVERAGE_REPORT_JSON):
    """Consolidated per-asset per-family status+reason with totals."""
    families = {
        "wx": {
            "sources": {
                name: {
                    "covered": block["status_counts"]["covered"],
                    "no_coverage": block["status_counts"]["no_coverage"],
                    "by_reason_code": block["by_reason_code"],
                }
                for block in coverage["sources"]
                for name in [block["name"]]
            }
        },
        "rd": {
            "sources": {
                block["name"]: {
                    "covered": block["status_counts"]["covered"],
                    "no_coverage": block["status_counts"]["no_coverage"],
                    "by_reason_code": block["by_reason_code"],
                }
                for block in coverage["rd"]["sources"]
            }
        },
        "mt": {
            "sources": {
                block["name"]: {
                    "covered": block["status_counts"]["covered"],
                    "no_coverage": block["status_counts"]["no_coverage"],
                    "by_reason_code": block["by_reason_code"],
                }
                for block in coverage["mt"]["sources"]
            },
            "throughput": coverage["mt"]["throughput"],
        },
    }
    assets = []
    for entry in sorted(coverage["assets"], key=lambda e: e["asset_id"]):
        by_source = {s["name"]: s for s in entry["sources"]}
        assets.append(
            {
                "asset_id": entry["asset_id"],
                "families": {
                    name: {
                        "status": by_source[name]["status"],
                        "reason_code": by_source[name].get("reason_code"),
                    }
                    for name in ALL_SOURCES
                },
            }
        )
    report = {"families": families, "assets": assets}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    return report


def assert_nulls_explained(asset_rows, header, cov_lookup, coverage):
    """Every NULL feature cell maps to an unverified asset, a no_coverage
    family entry, or the throughput gate. Returns (n_null, n_explained)."""
    mt_throughput_reason = (coverage.get("mt") or {}).get(
        "throughput", {}).get("reason_code")
    feature_cols = [c for c in header
                    if c.startswith("rd_") or c.startswith("mt_")]
    feature_cols = [c for c in feature_cols
                    if not c.startswith(("rd_prov_", "mt_prov_"))]
    # rd_corridor_name is empty BY CONVENTION exactly when the flag is
    # false (a name exists only for a matched corridor) - not a NULL.
    feature_cols = [c for c in feature_cols
                    if c != "rd_corridor_name"]
    for row in asset_rows:
        if (row["rd_corridor_name"] == "") != (
                row["rd_on_corridor_flag"] != "true"):
            raise ValueError(
                "corridor name/flag mismatch on %s" % row["asset_id"])
    n_null = 0
    n_explained = 0
    for row in asset_rows:
        aid = row["asset_id"]
        lat, lon = row["verified_lat"], row["verified_lon"]
        for col in feature_cols:
            if (row[col] or "") != "":
                continue
            n_null += 1
            if not lat or not lon:
                n_explained += 1
                continue
            if col in MT_THROUGHPUT_COLS:
                if mt_throughput_reason == THROUGHPUT_REASON:
                    n_explained += 1
                continue
            if col.startswith("rd_"):
                fams = [OSM_SOURCE]
            elif col.startswith(("mt_dist_", "mt_nearest_", "mt_ports_",
                                 "mt_pw_")):
                fams = ["portwatch_ports"]
            elif col.startswith("mt_in_port_polygon"):
                fams = ["hdx_sea_ports"]
            elif col.startswith(("mt_coast_", "mt_coastal_")):
                fams = ["natural_earth_coastline"]
            else:  # pragma: no cover - header guard below catches drift
                fams = []
            statuses = [
                cov_lookup.get(aid, {}).get(f, {}).get("status")
                for f in fams
            ]
            if fams and all(st != "covered" for st in statuses):
                n_explained += 1
    assert n_null == n_explained, (
        "unexplained NULLs: %d of %d" % (n_null - n_explained, n_null)
    )
    return n_null, n_explained


def build(asset_path=MATRIX_ASSET_CSV, month_path=MATRIX_MONTH_CSV,
          report_path=COVERAGE_REPORT_JSON):
    sidecar, highway, maritime, up_month, pan_month, coverage = load_inputs()
    cov_lookup = coverage_lookup(coverage)
    asset_rows, asset_header = build_asset_matrix(
        sidecar, highway, maritime, cov_lookup, path=asset_path
    )
    for row in asset_rows:
        assert_no_forbidden_columns(row.keys(), "feature_matrix_asset.csv")
    check_keys(asset_rows, ["asset_id"], "asset matrix")
    assert len(asset_rows) == 100
    # Header drift guard: every feature column must map to a known family.
    known = set(
        [c for c in asset_header if c.startswith("rd_")
         and not c.startswith("rd_prov_")]
        + [c for c in asset_header if c.startswith("mt_")
           and not c.startswith("mt_prov_")]
    )
    mappable = (
        lambda c: c.startswith("rd_") or c.startswith("mt_dist_")
        or c.startswith("mt_nearest_") or c.startswith("mt_ports_")
        or c.startswith("mt_pw_")
        or c.startswith("mt_in_port_polygon") or c.startswith("mt_coast_")
        or c.startswith("mt_coastal_") or c in MT_THROUGHPUT_COLS
    )
    assert all(mappable(c) for c in known), "unmapped feature column"
    n_null, n_explained = assert_nulls_explained(
        asset_rows, asset_header, cov_lookup, coverage
    )
    month_rows, month_header = build_month_matrix(
        up_month, pan_month, path=month_path
    )
    for row in month_rows:
        assert_no_forbidden_columns(
            row.keys(), "feature_matrix_asset_month.csv")
    report = build_coverage_report(coverage, path=report_path)
    print("asset matrix rows: %d" % len(asset_rows))
    print("month matrix rows: %d" % len(month_rows))
    print("NULL feature cells: %d (explained %d)" % (n_null, n_explained))
    print("wrote %s" % asset_path)
    print("wrote %s" % month_path)
    print("wrote %s" % report_path)
    return {
        "n_asset_rows": len(asset_rows),
        "n_month_rows": len(month_rows),
        "n_null": n_null,
    }


if __name__ == "__main__":
    build()
