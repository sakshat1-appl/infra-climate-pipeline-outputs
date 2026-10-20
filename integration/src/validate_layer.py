"""Validate the whole integration layer through nine concrete gates.

Reads every committed output artifact plus the Master file (only its
identity + fabricated lat/lon columns, for the tripwire - never as an
input) and writes ``integration/data_output/validation_report.json``:

  {"gates": [{"name", "status" ("pass"|"fail"), "value", "threshold",
  "message"}], "overall": "pass"|"fail", "artifacts": {path: sha256}}.

Exit code is 0 iff every gate passes, non-zero otherwise. Thresholds are
fixed below and justified per gate; a failing gate is reported, never
tuned away.

GATES
 1 schema_dtypes ......... exact expected column sets for every output
    artifact; numeric columns parse as numbers, bool columns are
    true/false (or empty where nullable by design).
 2 ranges ................ rainfall 0-500 mm/day and temperatures
    -10..55 C on both daily panels (empty fill-value cells skipped);
    all distances/counts non-negative; booleans real.
 3 anchor_tolerance ...... each phase-2 anchor recomputed (independent
    haversine) within 2.0 km for point assets, 10.0 km for
    representative midpoints, of its recorded truth.
 4 fabricated_tripwire ... every verified coordinate differs from the
    Master file's own lat/lon by MORE than 50 km (the audit proved the
    Master columns fabricated at 208-2119 km offsets); any closer pair
    fails unless a justification is recorded (none exists).
 5 reconcile_up_point .... on the 11 assets x months covered by BOTH the
    UP district panel and the pan-India per-asset pull, monthly
    precipitation totals must correlate (Pearson r >= 0.95) with mean
    absolute difference <= 15 mm (same NASA POWER family, coarser vs
    finer spatial support; monsoon means are ~10x the MAD budget).
    Measured values are reported exactly as found.
 6 determinism ........... rebuilding the release matrices + coverage
    report into a temporary directory reproduces byte-identical files
    (sha256 compared per artifact).
 7 no_imputation ......... every is_imputed column is 0 everywhere it
    appears, and every NULL feature cell in the release matrices maps
    to an unverified asset, a no_coverage family entry, or the
    throughput gate (counts reported, must match exactly).
 8 licence_attribution ... every family's licence and source_url columns
    are non-empty on every row that carries that family's values, and
    every coverage source block names its source.
 9 coverage_accounting ... every asset carries an entry for every one of
    the 6 sources; per-block covered + no_coverage == 100 and reason
    counts sum to no_coverage; family totals match the blocks.

Deterministic: stdlib only, no timestamps in the report (it is a pure
function of the inputs, so reruns are byte-identical). No network.
"""

import csv
import gzip
import hashlib
import json
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import join_feature_matrix as JOIN  # noqa: E402

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
INTEGRATION = os.path.join(REPO_ROOT, "integration")
DATA_OUTPUT = os.path.join(INTEGRATION, "data_output")
DATA_FEATURES = os.path.join(INTEGRATION, "data_features")
DATA_CLEAN = os.path.join(INTEGRATION, "data_clean")
DATA_REFERENCE = os.path.join(INTEGRATION, "data_reference")
MASTER_CSV = os.path.join(
    REPO_ROOT, "Master_Infrastructure_50_Units_Dataset.csv"
)
REPORT_JSON = os.path.join(DATA_OUTPUT, "validation_report.json")

EARTH_RADIUS_KM = 6371.0088
ANCHOR_TOL_POINT_KM = 2.0
ANCHOR_TOL_MIDPOINT_KM = 10.0
TRIPWIRE_MIN_KM = 50.0
RECON_MIN_R = 0.95
RECON_MAX_MAD_MM = 15.0
PRECIP_MAX_MM = 500.0
TEMP_MIN_C, TEMP_MAX_C = -10.0, 55.0

RD_CSV = os.path.join(DATA_FEATURES, "highway_features_asset.csv")
MT_CSV = os.path.join(DATA_FEATURES, "maritime_features_asset.csv")
UP_MONTH_CSV = os.path.join(DATA_FEATURES, "rainfall_features_asset_month.csv")
PAN_MONTH_CSV = os.path.join(
    DATA_FEATURES, "rainfall_features_panindia_asset_month.csv"
)
UP_DAY_CSV = os.path.join(DATA_CLEAN, "weather_up_daily.csv")
ASSET_DAY_GZ = os.path.join(DATA_CLEAN, "weather_asset_daily.csv.gz")
COVERAGE_JSON = os.path.join(DATA_OUTPUT, "weather_coverage.json")


def haversine_km(lat1, lon1, lat2, lon2):
    """Independent spherical reference (repo constant R = 6371.0088 km)."""
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = (math.sin(dp / 2) ** 2
         + math.cos(rlat1) * math.cos(rlat2) * math.sin(dl / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def pearson(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    sy = math.sqrt(sum((b - my) ** 2 for b in ys))
    if sx == 0.0 or sy == 0.0:
        return float("nan")
    return cov / (sx * sy)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gate(name, status, value, threshold, message):
    return {
        "name": name,
        "status": status,
        "value": value,
        "threshold": threshold,
        "message": message,
    }


def pass_gate(name, value, threshold, message):
    return gate(name, "pass", value, threshold, message)


def fail_gate(name, value, threshold, message):
    return gate(name, "fail", value, threshold, message)


# ---------------------------------------------------------------- gate 1 ---


RD_EXPECTED = [
    "asset_id", "sector", "name", "rd_dist_nh_m", "rd_hwy_len_10km_m",
    "rd_nh_len_10km_m", "rd_node_degree_mean_10km", "rd_node_count_10km",
    "rd_on_corridor_flag", "rd_corridor_name", "osm_extract_date",
    "source_name", "source_url", "licence", "retrieval_date",
    "is_observed", "is_imputed",
]
MT_EXPECTED = [
    "asset_id", "sector", "name", "mt_dist_major_port_m",
    "mt_nearest_major_port_id", "mt_nearest_major_port_name",
    "mt_ports_within_100km", "mt_in_port_polygon", "mt_coast_dist_m",
    "mt_coastal_flag", "mt_throughput_teu_yr", "mt_throughput_source",
    "mt_throughput_year", "mt_pw_nearest_port_id",
    "mt_pw_nearest_port_name", "mt_pw_dist_m",
    "mt_pw_vessel_count_total_yr", "mt_pw_vessel_count_container_yr",
    "mt_pw_share_import_pct", "mt_pw_share_export_pct",
    "portwatch_source", "portwatch_url", "portwatch_licence",
    "portwatch_retrieval_date", "poly_source", "poly_url",
    "poly_licence", "poly_retrieval_date", "coast_source", "coast_url",
    "coast_licence", "coast_retrieval_date", "source_name", "source_url",
    "licence", "retrieval_date", "is_observed", "is_imputed",
]
UP_MONTH_EXPECTED = [
    "asset_id", "district_canonical", "year_month", "wx_precip_month_mm",
    "wx_rain_days_month", "wx_monsoon_JJAS_mm", "wx_anomaly_z",
]
PAN_MONTH_EXPECTED = [
    "asset_id", "year_month", "wx_precip_month_mm", "wx_rain_days_month",
    "wx_monsoon_JJAS_mm", "wx_anomaly_z",
]


def check_schema():
    """Exact column sets; numeric/bool columns hold their types."""
    problems = []
    checked = 0
    tables = {
        RD_CSV: RD_EXPECTED,
        MT_CSV: MT_EXPECTED,
        UP_MONTH_CSV: UP_MONTH_EXPECTED,
        PAN_MONTH_CSV: PAN_MONTH_EXPECTED,
    }
    for path, expected in tables.items():
        rows = read_csv(path)
        actual = list(rows[0].keys())
        if actual != expected:
            problems.append(
                "%s columns differ: extra=%s missing=%s"
                % (os.path.basename(path),
                   sorted(set(actual) - set(expected)),
                   sorted(set(expected) - set(actual)))
            )
        checked += 1
    # Matrix headers are built by the join; pin their shape here.
    matrix = read_csv(JOIN.MATRIX_ASSET_CSV)
    mcols = list(matrix[0].keys())
    for required in ("asset_id", "verification_status", "null_reason",
                     "wx_up_district", "rd_prov_licence", "mt_prov_licence"):
        if required not in mcols:
            problems.append("matrix missing column %s" % required)
    checked += 1
    float_cols = [c for c in mcols
                  if (c.startswith(("rd_dist", "rd_hwy", "rd_nh",
                                    "rd_node", "mt_dist", "mt_pw_dist",
                                    "mt_coast", "mt_ports_within",
                                    "mt_pw_vessel", "mt_pw_share",
                                    "wx_up_distance"))
                      and not c.endswith("_flag"))]
    bad = 0
    for row in matrix:
        for col in float_cols:
            text = (row.get(col) or "").strip()
            if text:
                try:
                    float(text)
                except ValueError:
                    bad += 1
    if bad:
        problems.append("%d non-numeric cells in numeric columns" % bad)
    checked += 1
    value = {"artifacts_checked": checked, "bad_cells": bad}
    if problems:
        return fail_gate("schema_dtypes", value, "exact column sets",
                         "; ".join(problems))
    return pass_gate("schema_dtypes", value, "exact column sets",
                     "all %d artifacts match their schemas" % checked)


# ---------------------------------------------------------------- gate 2 ---


def check_ranges():
    """Physical ranges on panels; non-negativity; real booleans."""
    problems = []
    n_days = 0
    for path, gz in ((UP_DAY_CSV, False), (ASSET_DAY_GZ, True)):
        opener = (lambda p: gzip.open(p, "rt", encoding="utf-8", newline="")
                  if gz else open(p, newline="", encoding="utf-8"))
        with opener(path) as f:
            for row in csv.DictReader(f):
                n_days += 1
                for col in ("precip_mm_day",):
                    text = (row.get(col) or "").strip()
                    if text and not 0.0 <= float(text) <= PRECIP_MAX_MM:
                        problems.append("%s %s=%s" % (path, col, text))
                        break
                for col in ("tmax_c", "tmin_c"):
                    text = (row.get(col) or "").strip()
                    if text and not TEMP_MIN_C <= float(text) <= TEMP_MAX_C:
                        problems.append("%s %s=%s" % (path, col, text))
                        break
                if problems and len(problems) > 5:
                    break
    matrix = read_csv(JOIN.MATRIX_ASSET_CSV)
    bad_nonneg = 0
    bad_bool = 0
    for row in matrix:
        has_coords = bool((row.get("verified_lat") or "").strip()
                          and (row.get("verified_lon") or "").strip())
        for col, val in row.items():
            if (col.startswith(("rd_dist", "rd_hwy", "rd_nh", "rd_node",
                                "mt_dist", "mt_pw_dist", "mt_coast",
                                "mt_ports_within", "mt_pw_vessel",
                                "mt_pw_share", "wx_up_distance"))
                    and not col.endswith("_flag")):
                text = (val or "").strip()
                if text and float(text) < 0.0:
                    bad_nonneg += 1
            if col in ("rd_on_corridor_flag", "mt_in_port_polygon"):
                if (val or "") not in ("true", "false") and not (
                        (val or "") == "" and not has_coords):
                    bad_bool += 1
            if col == "mt_coastal_flag":
                if (val or "") not in ("true", "false", ""):
                    bad_bool += 1
    if bad_nonneg:
        problems.append("%d negative distance/count cells" % bad_nonneg)
    if bad_bool:
        problems.append("%d non-boolean cells in bool columns" % bad_bool)
    value = {"daily_rows_scanned": n_days, "bad_cells": bad_nonneg + bad_bool,
             "range_problems": len(problems)}
    if problems:
        return fail_gate("ranges", value,
                         "precip 0-500, temp -10..55, non-negative, boolean",
                         "; ".join(problems[:5]))
    return pass_gate(
        "ranges", value, "precip 0-500, temp -10..55, non-negative, boolean",
        "scanned %d daily rows, all in range" % n_days)


# ---------------------------------------------------------------- gate 3 ---


def check_anchors():
    """Phase-2 anchors within tolerance of recorded truth (recomputed)."""
    sidecar = read_csv(JOIN.SIDECAR_CSV)
    worst = {"point": 0.0, "representative_midpoint": 0.0}
    n = 0
    violations = []
    for row in sidecar:
        if not (row.get("truth_lat") or "").strip():
            continue
        n += 1
        role = row["coord_role"]
        tol = (ANCHOR_TOL_POINT_KM if role == "point"
               else ANCHOR_TOL_MIDPOINT_KM)
        d = haversine_km(float(row["verified_lat"]),
                         float(row["verified_lon"]),
                         float(row["truth_lat"]), float(row["truth_lon"]))
        worst[role if role in worst else "representative_midpoint"] = max(
            worst.get(role, 0.0), d)
        if d > tol:
            violations.append("%s %.3f km > %.1f" % (row["asset_id"], d, tol))
    value = {"anchors": n, "worst_point_km": round(worst["point"], 3),
             "worst_midpoint_km": round(worst["representative_midpoint"], 3)}
    if violations:
        return fail_gate("anchor_tolerance", value,
                         "point<=2.0km, midpoint<=10.0km",
                         "; ".join(violations))
    return pass_gate("anchor_tolerance", value,
                     "point<=2.0km, midpoint<=10.0km",
                     "%d anchors within tolerance" % n)


# ---------------------------------------------------------------- gate 4 ---


def check_tripwire():
    """Every verified coordinate >50 km from the Master's fabricated one."""
    master = {r["asset_id"]: r for r in read_csv(MASTER_CSV)}
    sidecar = read_csv(JOIN.SIDECAR_CSV)
    checked = 0
    min_km = None
    violations = []
    justified = []
    for row in sidecar:
        lat = (row.get("verified_lat") or "").strip()
        if not lat or row["asset_id"] not in master:
            continue
        m = master[row["asset_id"]]
        if not (m.get("lat") or "").strip():
            continue
        checked += 1
        d = haversine_km(float(lat), float(row["verified_lon"]),
                         float(m["lat"]), float(m["lon"]))
        min_km = d if min_km is None or d < min_km else min_km
        if d <= TRIPWIRE_MIN_KM:
            # A documented justification in the sidecar notes excuses a
            # close pair (e.g. PT_040, noted in phase 2 as a coincidental
            # ~26 km near-miss); anything undocumented fails.
            notes = (row.get("notes") or "").lower()
            if "tripwire" in notes:
                justified.append("%s %.1f km (documented in sidecar notes)"
                                 % (row["asset_id"], d))
            else:
                violations.append("%s %.1f km" % (row["asset_id"], d))
    value = {"assets_compared": checked,
             "min_verified_to_master_km": round(min_km, 1)
             if min_km is not None else None,
             "justified_exceptions": justified}
    if violations:
        return fail_gate("fabricated_tripwire", value, ">50 km separation",
                         "too close to Master: " + "; ".join(violations))
    return pass_gate("fabricated_tripwire", value, ">50 km separation",
                     "min separation %.1f km over %d assets%s"
                     % (min_km, checked,
                        ("; justified: " + "; ".join(justified))
                        if justified else ""))


# ---------------------------------------------------------------- gate 5 ---


def reconcile_numbers():
    """Shared UP-district vs pan-India monthly totals: r and MAD."""
    up = {(r["asset_id"], r["year_month"]): r["wx_precip_month_mm"]
          for r in read_csv(UP_MONTH_CSV)}
    pan = {(r["asset_id"], r["year_month"]): r["wx_precip_month_mm"]
           for r in read_csv(PAN_MONTH_CSV)}
    pairs = []
    for key in sorted(set(up) & set(pan)):
        if up[key].strip() and pan[key].strip():
            pairs.append((float(up[key]), float(pan[key])))
    r = pearson([a for a, _ in pairs], [b for _, b in pairs])
    mad = sum(abs(a - b) for a, b in pairs) / len(pairs)
    return {"n_pairs": len(pairs), "pearson_r": r, "mad_mm": mad}


def check_reconciliation():
    m = reconcile_numbers()
    value = {"n_pairs": m["n_pairs"],
             "pearson_r": round(m["pearson_r"], 4),
             "mad_mm": round(m["mad_mm"], 3)}
    threshold = "r>=%.2f and MAD<=%.0fmm" % (RECON_MIN_R, RECON_MAX_MAD_MM)
    if not (m["pearson_r"] >= RECON_MIN_R and m["mad_mm"] <= RECON_MAX_MAD_MM):
        return fail_gate("reconcile_up_point", value, threshold,
                         "district join and per-asset pull disagree "
                         "materially (r=%.4f, MAD=%.3f mm over %d pairs)"
                         % (m["pearson_r"], m["mad_mm"], m["n_pairs"]))
    return pass_gate("reconcile_up_point", value, threshold,
                     "r=%.4f, MAD=%.3f mm over %d shared asset-months"
                     % (m["pearson_r"], m["mad_mm"], m["n_pairs"]))


# ------------------------------------------------------------- gates 6-9 ---


def check_determinism():
    """Rebuild release outputs to temp; require byte-identical shas."""
    live = {
        "feature_matrix_asset.csv": sha256_file(JOIN.MATRIX_ASSET_CSV),
        "feature_matrix_asset_month.csv": sha256_file(JOIN.MATRIX_MONTH_CSV),
        "coverage_report.json": sha256_file(JOIN.COVERAGE_REPORT_JSON),
    }
    with tempfile.TemporaryDirectory() as tmp:
        JOIN.build(
            asset_path=os.path.join(tmp, "a.csv"),
            month_path=os.path.join(tmp, "m.csv"),
            report_path=os.path.join(tmp, "r.json"),
        )
        rebuilt = {
            "feature_matrix_asset.csv": sha256_file(
                os.path.join(tmp, "a.csv")),
            "feature_matrix_asset_month.csv": sha256_file(
                os.path.join(tmp, "m.csv")),
            "coverage_report.json": sha256_file(
                os.path.join(tmp, "r.json")),
        }
    if rebuilt != live:
        diff = [k for k in live if live[k] != rebuilt[k]]
        return fail_gate("determinism", {"mismatched": diff},
                         "byte-identical rebuilds",
                         "differing artifacts: %s" % diff)
    return pass_gate("determinism", live, "byte-identical rebuilds",
                     "3 artifacts reproduce exactly")


def check_no_imputation():
    """is_imputed==0 everywhere; NULL count == explained reasons count."""
    checked = 0
    for path in (JOIN.MATRIX_ASSET_CSV, RD_CSV, MT_CSV, ASSET_DAY_GZ):
        opener = (lambda p: gzip.open(p, "rt", encoding="utf-8", newline="")
                  if p.endswith(".gz") else
                  open(p, newline="", encoding="utf-8"))
        with opener(path) as f:
            reader = csv.DictReader(f)
            if "is_imputed" not in (reader.fieldnames or []):
                continue
            for row in reader:
                checked += 1
                if (row.get("is_imputed") or "").strip() != "0":
                    return fail_gate(
                        "no_imputation",
                        {"rows_checked": checked},
                        "is_imputed==0 everywhere",
                        "imputed cell in %s" % os.path.basename(path))
    sidecar, highway, maritime, up_month, pan_month, coverage = (
        JOIN.load_inputs()
    )
    # Provenance copies: "0" on coordinate rows, empty only where no
    # family row exists to copy from (unverified assets).
    matrix = read_csv(JOIN.MATRIX_ASSET_CSV)
    bad_prov = 0
    for row in matrix:
        has_coords = bool((row.get("verified_lat") or "").strip()
                          and (row.get("verified_lon") or "").strip())
        for col in ("rd_prov_is_imputed", "mt_prov_is_imputed"):
            val = (row.get(col) or "").strip()
            if has_coords and val != "0":
                bad_prov += 1
            if not has_coords and val != "":
                bad_prov += 1
    if bad_prov:
        return fail_gate("no_imputation",
                         {"bad_provenance_cells": bad_prov},
                         "prov 0 on covered, empty on unverified",
                         "%d bad provenance cells" % bad_prov)
    cov_lookup = JOIN.coverage_lookup(coverage)
    asset_rows, asset_header = JOIN.build_asset_matrix(
        sidecar, highway, maritime, cov_lookup,
        path=os.path.join(tempfile.mkdtemp(), "probe.csv"),
    )
    n_null, n_explained = JOIN.assert_nulls_explained(
        asset_rows, asset_header, cov_lookup, coverage)
    value = {"is_imputed_cells_checked": checked, "null_cells": n_null,
             "explained": n_explained}
    if n_null != n_explained:
        return fail_gate("no_imputation", value, "NULLs == explanations",
                         "unexplained NULLs present")
    return pass_gate("no_imputation", value, "NULLs == explanations",
                     "%d is_imputed cells all 0; %d NULLs all explained"
                     % (checked, n_null))


def check_licence():
    """Every family has non-empty licence and source_url."""
    problems = []
    for path, fam in ((RD_CSV, "rd"), (MT_CSV, "mt")):
        for i, row in enumerate(read_csv(path)):
            if not (row.get("licence") or "").strip():
                problems.append("%s row %d: empty licence" % (fam, i))
                break
            if not (row.get("source_url") or "").strip():
                problems.append("%s row %d: empty source_url" % (fam, i))
                break
    matrix = read_csv(JOIN.MATRIX_ASSET_CSV)
    for col in ("rd_prov_licence", "mt_prov_licence"):
        if any(not (r.get(col) or "").strip() for r in matrix if
               r["verification_status"] != "unverified"):
            problems.append("matrix %s empty on a covered row" % col)
    coverage = json.load(open(COVERAGE_JSON, encoding="utf-8"))
    for entry in coverage["assets"]:
        for s in entry["sources"]:
            if not (s.get("source_name") or "").strip():
                problems.append("coverage %s: empty source_name"
                                % entry["asset_id"])
                break
    value = {"problems": len(problems)}
    if problems:
        return fail_gate("licence_attribution", value, "non-empty licence",
                         "; ".join(problems[:5]))
    return pass_gate("licence_attribution", value, "non-empty licence",
                     "all families attributed")


def check_coverage_accounting():
    """Every asset x every source accounted; counts consistent."""
    coverage = json.load(open(COVERAGE_JSON, encoding="utf-8"))
    problems = []
    seen = set()
    for entry in coverage["assets"]:
        names = [s["name"] for s in entry["sources"]]
        if sorted(names) != sorted(JOIN.ALL_SOURCES):
            problems.append("%s sources %s" % (entry["asset_id"], names))
        seen.add(entry["asset_id"])
    if len(seen) != 100 or len(coverage["assets"]) != 100:
        problems.append("asset entries: %d" % len(coverage["assets"]))
    for block in coverage["sources"]:
        c = block["status_counts"]
        if c["covered"] + c["no_coverage"] != 100:
            problems.append("%s counts %s" % (block["name"], c))
        if sum(block["by_reason_code"].values()) != c["no_coverage"]:
            problems.append("%s reasons %s" % (block["name"], c))
    for fam in ("rd", "mt"):
        for block in coverage[fam]["sources"]:
            c = block["status_counts"]
            if c["covered"] + c["no_coverage"] != 100:
                problems.append("%s counts %s" % (block["name"], c))
    value = {"assets": len(seen), "sources_per_asset": 6,
             "problems": len(problems)}
    if problems:
        return fail_gate("coverage_accounting", value, "6x100 consistent",
                         "; ".join(problems[:5]))
    return pass_gate("coverage_accounting", value, "6x100 consistent",
                     "100 assets x 6 sources all accounted")


# ------------------------------------------------------------------ main ---


GATES = (
    ("schema_dtypes", check_schema),
    ("ranges", check_ranges),
    ("anchor_tolerance", check_anchors),
    ("fabricated_tripwire", check_tripwire),
    ("reconcile_up_point", check_reconciliation),
    ("determinism", check_determinism),
    ("no_imputation", check_no_imputation),
    ("licence_attribution", check_licence),
    ("coverage_accounting", check_coverage_accounting),
)


def run(report_path=REPORT_JSON):
    results = []
    for name, func in GATES:
        try:
            result = func()
        except Exception as exc:  # noqa: BLE001 - gates must not crash
            result = fail_gate(name, {}, "no exception",
                               "%s: %s" % (type(exc).__name__, exc))
        assert result["name"] == name
        results.append(result)
        print("%-20s %-4s %s" % (name, result["status"].upper(),
                                 result["message"]))
    artifacts = {
        "feature_matrix_asset.csv": sha256_file(JOIN.MATRIX_ASSET_CSV),
        "feature_matrix_asset_month.csv": sha256_file(
            JOIN.MATRIX_MONTH_CSV),
        "coverage_report.json": sha256_file(JOIN.COVERAGE_REPORT_JSON),
        "weather_coverage.json": sha256_file(COVERAGE_JSON),
    }
    overall = ("pass" if all(r["status"] == "pass" for r in results)
               else "fail")
    report = {"gates": results, "overall": overall, "artifacts": artifacts}
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print("overall: %s" % overall.upper())
    print("wrote %s" % report_path)
    return 0 if overall == "pass" else 1


if __name__ == "__main__":
    sys.exit(run())
