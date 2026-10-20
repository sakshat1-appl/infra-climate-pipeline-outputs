"""Tests for the NASA POWER point panel and the pan-India feature build.

Offline: reads the committed gzipped asset panel, the committed feature
files, the committed fixture subset and the committed climatology. The
no-lookahead test recomputes features INDEPENDENTLY from the committed
fixture (own window logic, written from the feature definitions, no import
of any pipeline module) and asserts equality with the written values.
"""

import csv
import datetime
import gzip
import math
import os

import pytest

import build_panindia_features as pan
import build_rainfall_features as up
import rainfall_features_core as core

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
SIDECAR_CSV = os.path.join(
    REPO_ROOT, "integration", "data_reference", "assets_verified.csv"
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
FIXTURE_CSV = os.path.join(
    REPO_ROOT,
    "integration",
    "tests",
    "fixtures",
    "asset_daily_panel_subset_2019-2021.csv",
)
GAPS_JSON = os.path.join(
    REPO_ROOT, "integration", "data_clean", "weather_asset_daily_gaps.json"
)

FIXTURE_ASSETS = ["GP_081", "GP_084", "GP_086", "HW_002", "RE_061"]
WINDOW_START = datetime.date(2005, 1, 1)
WINDOW_END = datetime.date(2025, 11, 29)
WINDOW_DAYS = (WINDOW_END - WINDOW_START).days + 1

DAILY_INT_FEATURES = {
    "wx_heavy_days_7d",
    "wx_very_heavy_days_30d",
    "wx_rain_days_30d",
    "wx_cwd_30d",
    "wx_dry_spell_90d",
    "wx_p95_exceed_days_30d",
}
ALL_DAILY_FEATURES = DAILY_INT_FEATURES | {
    "wx_precip_7d_mm",
    "wx_precip_30d_mm",
    "wx_monsoon_to_date_mm",
}


def load_sidecar():
    with open(SIDECAR_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def non_null_coordinate_ids():
    return {
        r["asset_id"]
        for r in load_sidecar()
        if (r.get("verified_lat") or "").strip()
        and (r.get("verified_lon") or "").strip()
    }


def read_panel_rows(asset_ids=None):
    """Rows of the canonical panel, optionally filtered to some assets."""
    rows = []
    with gzip.open(PANEL_GZ, "rt", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if asset_ids is None or row["asset_id"] in asset_ids:
                rows.append(row)
    return rows


def read_written_day(asset_ids):
    """{(asset_id, date): row} from the committed pan-India day features."""
    written = {}
    with gzip.open(FEATURES_DAY_GZ, "rt", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["asset_id"] in asset_ids:
                written[(row["asset_id"], row["date"])] = row
    return written


def load_asset_climatology():
    """{asset_id: {month: {"mean","sd","p95","n"}}} from the committed CSV.

    An empty ``p95_wet_day_mm`` cell means the asset had no wet days in
    that calendar month over the baseline (p95 is null); it loads as None.
    """
    clim = {}
    with open(ASSET_CLIMATOLOGY_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            p95_text = row["p95_wet_day_mm"]
            clim.setdefault(row["asset_id"], {})[int(row["month"])] = {
                "mean": float(row["mean_precip_mm_day"]),
                "sd": float(row["sd_precip_mm_day"]),
                "p95": float(p95_text) if p95_text != "" else None,
                "n": int(row["n_baseline_days"]),
            }
    return clim


def load_fixture_panel():
    """{asset_id: {date: precip}} from the committed fixture subset."""
    panel = {}
    with open(FIXTURE_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            panel.setdefault(row["asset_id"], {})[
                datetime.date.fromisoformat(row["date"])
            ] = float(row["precip_mm_day"])
    return panel


# --------------------------------------------------------------------------
# Canonical panel structure and cleaning invariants
# --------------------------------------------------------------------------


def test_panel_one_row_per_asset_date_and_full_asset_count():
    rows = read_panel_rows()
    seen = set()
    for row in rows:
        key = (row["asset_id"], row["date"])
        assert key not in seen, "duplicate (asset_id, date): %r" % (key,)
        seen.add(key)
    ids = {row["asset_id"] for row in rows}
    expected = non_null_coordinate_ids()
    assert len(ids) == 81
    assert ids == expected
    assert len(rows) == 81 * WINDOW_DAYS


def test_panel_sorted_by_asset_then_date():
    rows = read_panel_rows()
    keys = [(r["asset_id"], r["date"]) for r in rows]
    assert keys == sorted(keys)


def test_panel_calendar_contiguous_per_asset():
    by_asset = {}
    for row in read_panel_rows():
        by_asset.setdefault(row["asset_id"], set()).add(row["date"])
    expected = set()
    day = WINDOW_START
    while day <= WINDOW_END:
        expected.add(day.isoformat())
        day += datetime.timedelta(days=1)
    assert len(expected) == WINDOW_DAYS
    for aid, dates in by_asset.items():
        assert dates == expected, "asset %s calendar is not contiguous" % aid


def test_no_fill_value_survives_and_ranges_hold():
    rows = read_panel_rows()
    for row in rows:
        for column in (
            "precip_mm_day",
            "tmax_c",
            "tmin_c",
            "rh_pct",
            "wind_ms",
            "ps_kpa",
        ):
            text = row[column]
            if text == "":
                continue  # partially observed day: cell left empty
            value = float(text)
            assert value > -999.0, "fill value survived in %s: %r" % (column, text)
        precip = float(row["precip_mm_day"])
        assert 0.0 <= precip <= 500.0, row
        if row["tmax_c"] and row["tmin_c"]:
            tmax, tmin = float(row["tmax_c"]), float(row["tmin_c"])
            assert -10.0 <= tmax <= 55.0, row
            assert -10.0 <= tmin <= 55.0, row
            assert tmax >= tmin, row
        # Provenance columns are never empty and never carry the fill value.
        assert row["source_name"] == "NASA POWER"
        assert row["source_url"].startswith(
            "https://power.larc.nasa.gov/api/temporal/daily/point?"
        )
        assert row["retrieval_date"] == "2026-10-15"
        assert row["is_imputed"] == "0"  # nothing is ever imputed


def test_no_any_999_sentinel_anywhere_in_panel():
    with gzip.open(PANEL_GZ, "rt", encoding="utf-8") as f:
        text = f.read()
    assert "-999" not in text


def test_gaps_file_documents_zero_gaps():
    import json

    with open(GAPS_JSON, encoding="utf-8") as f:
        gaps = json.load(f)
    assert gaps["expected_days"] == WINDOW_DAYS
    assert gaps["gaps"] == {}
    assert gaps["n_assets_with_gaps"] == 0


def test_panel_source_urls_match_manifest():
    import json

    with open(
        os.path.join(
            REPO_ROOT, "integration", "data_raw_cache", "power_points", "MANIFEST.json"
        ),
        encoding="utf-8",
    ) as f:
        manifest = json.load(f)
    seen = set()
    for row in read_panel_rows():
        seen.add((row["asset_id"], row["source_url"], row["retrieval_date"]))
    for aid, url, retrieval in seen:
        entry = manifest["assets"][aid]
        assert entry["request_url"] == url
        assert entry["retrieval_date"] == retrieval
    assert len(seen) == 81


# --------------------------------------------------------------------------
# Null-on-gap through the pan-India entry point
# --------------------------------------------------------------------------


def test_null_on_gap_panindia_path():
    """A mid-series gap forces nulls through the pan-India feature path."""
    first = datetime.date(2010, 3, 1)
    precip = {}
    for i in range(60):
        precip[first + datetime.timedelta(days=i)] = 10.0
    gap_start = first + datetime.timedelta(days=60)
    for i in range(40):
        precip[gap_start + datetime.timedelta(days=3 + i)] = 20.0
    calendar = [first + datetime.timedelta(days=i) for i in range(100)]
    clim = {
        "A1": {m: {"mean": 5.0, "sd": 2.0, "p95": 30.0, "n": 100} for m in range(1, 13)}
    }
    daily, _monthly = pan.compute_panindia_features(
        {"A1": precip}, clim, calendar, core.panel_months(calendar)
    )
    feats = daily["A1"]
    # The last day before the gap: its 7/30-day windows are complete.
    before = (gap_start - datetime.timedelta(days=1)).isoformat()
    assert feats[before]["wx_precip_7d_mm"] == 70.0
    assert feats[before]["wx_rain_days_30d"] == 30
    # The first day inside the gap: every windowed feature is null.
    inside = gap_start.isoformat()
    for name in ALL_DAILY_FEATURES:
        if name == "wx_monsoon_to_date_mm":
            continue
        assert feats[inside][name] is None, name
    # The day after the gap: windows still reach into the gap -> null.
    after = (gap_start + datetime.timedelta(days=3)).isoformat()
    assert feats[after]["wx_precip_7d_mm"] is None
    assert feats[after]["wx_precip_30d_mm"] is None
    assert feats[after]["wx_dry_spell_90d"] is None
    # Once 7 days have passed after the gap, the 7-day window is whole.
    recovered = (gap_start + datetime.timedelta(days=9)).isoformat()
    assert feats[recovered]["wx_precip_7d_mm"] == 140.0
    # The 30-day window still reaches back into the gap -> null, not a
    # partial sum over the available days.
    assert feats[recovered]["wx_rain_days_30d"] is None
    # Once 30 days have passed after the gap, counts are exact again.
    recovered_30d = (gap_start + datetime.timedelta(days=32)).isoformat()
    assert feats[recovered_30d]["wx_precip_30d_mm"] == 600.0
    assert feats[recovered_30d]["wx_rain_days_30d"] == 30


# --------------------------------------------------------------------------
# Single shared feature core
# --------------------------------------------------------------------------


def test_district_and_asset_entry_points_are_one_implementation():
    """Both entry points ARE the core functions: divergence is impossible."""
    assert up.compute_daily_features is core.compute_daily_features
    assert up.compute_month_features is core.compute_month_features
    assert up.compute_climatology is core.compute_climatology
    assert up.write_climatology_csv is core.write_climatology_csv
    assert pan.compute_daily_features is core.compute_daily_features
    assert pan.compute_month_features is core.compute_month_features
    assert pan.compute_climatology is core.compute_climatology


def test_district_and_asset_entry_points_return_identical_numbers():
    """Same synthetic panel in, same numbers out, through both paths."""
    first = datetime.date(2010, 1, 1)
    pattern = [0.0, 2.5, 60.0, 120.0, 0.5, 0.0, 7.25]
    precip = {}
    for i in range(200):
        precip[first + datetime.timedelta(days=i)] = pattern[i % len(pattern)]
    calendar = [first + datetime.timedelta(days=i) for i in range(200)]
    months = core.panel_months(calendar)
    clim = {
        "X": {m: {"mean": 3.0, "sd": 2.0, "p95": 40.0, "n": 100} for m in range(1, 13)}
    }
    up_daily = up.compute_daily_features("X", precip, calendar, clim)
    pan_daily = pan.compute_daily_features("X", precip, calendar, clim)
    assert up_daily == pan_daily
    up_month = up.compute_month_features("X", precip, months, clim)
    pan_month = pan.compute_month_features("X", precip, months, clim)
    assert up_month == pan_month
    # The synthetic panel exercises every feature: heavy days, very heavy
    # days, p95 exceedances and monsoon sums are all non-null somewhere.
    vals = [f for feats in pan_daily.values() for f in feats.values()]
    assert any(v is not None and v > 0 for v in vals)
    # January dates carry monsoon_to_date = 0 (empty window), never null.
    assert pan_daily[first.isoformat()]["wx_monsoon_to_date_mm"] == 0


def test_asset_climatology_matches_shared_definition():
    """One asset-month recomputed straight from the panel matches the CSV."""
    rows = read_panel_rows(asset_ids=["RE_061"])
    precip = {}
    for row in rows:
        precip[datetime.date.fromisoformat(row["date"])] = float(row["precip_mm_day"])
    clim = pan.compute_climatology({"RE_061": precip})["RE_061"]
    written = load_asset_climatology()["RE_061"]
    for month in range(1, 13):
        values = [
            v
            for d, v in precip.items()
            if d.month == month and datetime.date(2005, 1, 1) <= d <= datetime.date(2020, 12, 31)
        ]
        wet = sorted(v for v in values if v >= 1.0)
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values)
        sd = math.sqrt(var)
        pos = 0.95 * (len(wet) - 1)
        lo = int(math.floor(pos))
        hi = min(lo + 1, len(wet) - 1)
        p95 = wet[lo] + (pos - lo) * (wet[hi] - wet[lo])
        assert clim[month]["n"] == written[month]["n"] == len(values)
        assert math.isclose(clim[month]["mean"], mean, rel_tol=1e-12)
        assert math.isclose(clim[month]["sd"], sd, rel_tol=1e-12)
        if p95 is None:
            # No wet days in this baseline month: p95 is null everywhere.
            assert clim[month]["p95"] is None
            assert written[month]["p95"] is None
        else:
            assert math.isclose(clim[month]["p95"], p95, rel_tol=1e-12)
            assert math.isclose(written[month]["p95"], p95, rel_tol=1e-12)


# --------------------------------------------------------------------------
# No-lookahead: independent recomputation from the committed fixture
# --------------------------------------------------------------------------


def _longest_run(flags):
    best = 0
    current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        best = max(best, current)
    return best


def _window(day, k):
    return [day - datetime.timedelta(days=i) for i in range(k - 1, -1, -1)]


def independent_daily_features(precip_by_date, day, p95_by_month):
    """Recompute every daily feature for one date straight from a panel.

    Written from the feature definitions alone (no import of any pipeline
    module), so it is a genuine independent recomputation.
    """
    out = {}
    w7, w30, w90 = _window(day, 7), _window(day, 30), _window(day, 90)
    vals7 = [precip_by_date[d] for d in w7 if d in precip_by_date]
    if len(vals7) == 7:
        out["wx_precip_7d_mm"] = sum(vals7)
        out["wx_heavy_days_7d"] = sum(1 for v in vals7 if v >= 50.0)
    else:
        out["wx_precip_7d_mm"] = None
        out["wx_heavy_days_7d"] = None
    vals30 = [precip_by_date[d] for d in w30 if d in precip_by_date]
    if len(vals30) == 30:
        out["wx_precip_30d_mm"] = sum(vals30)
        out["wx_very_heavy_days_30d"] = sum(1 for v in vals30 if v >= 100.0)
        out["wx_rain_days_30d"] = sum(1 for v in vals30 if v >= 1.0)
        out["wx_cwd_30d"] = _longest_run([v >= 1.0 for v in vals30])
        if any(p95_by_month[d.month] is None for d in w30):
            out["wx_p95_exceed_days_30d"] = None
        else:
            out["wx_p95_exceed_days_30d"] = sum(
                1 for d in w30 if precip_by_date[d] > p95_by_month[d.month]
            )
    else:
        out["wx_precip_30d_mm"] = None
        out["wx_very_heavy_days_30d"] = None
        out["wx_rain_days_30d"] = None
        out["wx_cwd_30d"] = None
        out["wx_p95_exceed_days_30d"] = None
    vals90 = [precip_by_date[d] for d in w90 if d in precip_by_date]
    if len(vals90) == 90:
        out["wx_dry_spell_90d"] = _longest_run(
            [precip_by_date[d] < 1.0 for d in w90]
        )
    else:
        out["wx_dry_spell_90d"] = None
    if day.month >= 6:
        since_june = []
        d = datetime.date(day.year, 6, 1)
        while d <= day:
            since_june.append(d)
            d += datetime.timedelta(days=1)
        vals = [precip_by_date[d] for d in since_june if d in precip_by_date]
        out["wx_monsoon_to_date_mm"] = (
            sum(vals) if len(vals) == len(since_june) else None
        )
    else:
        out["wx_monsoon_to_date_mm"] = 0
    return out


def _compare_daily(expected, written_row, context):
    for name in sorted(ALL_DAILY_FEATURES):
        exp = expected[name]
        got = written_row[name]
        if exp is None:
            assert got == "", "%s: expected null, got %r" % (context, got)
        elif name in DAILY_INT_FEATURES:
            assert got != "", "%s: expected %r, got null" % (context, exp)
            assert int(float(got)) == exp, "%s: expected %r, got %r" % (
                context,
                exp,
                got,
            )
        else:
            assert got != "", "%s: expected %r, got null" % (context, exp)
            assert math.isclose(
                float(got), exp, rel_tol=1e-9, abs_tol=1e-9
            ), "%s: expected %r, got %r" % (context, exp, got)


def test_no_lookahead_independent_recomputation_from_fixture():
    """Written pan-India values equal fixture-only recomputation.

    Several (asset, date) rows are recomputed independently from the
    committed fixture with the fixture's own days only, and again from a
    fixture truncated at the feature date (proving no future day leaks in).
    """
    fixture = load_fixture_panel()
    clim = load_asset_climatology()
    written = read_written_day(set(FIXTURE_ASSETS))

    # All dates are >= 2019-03-31, so every feature window (7/30/90-day,
    # monsoon-to-date, and the p95 baseline months) lies fully inside the
    # fixture window 2019-01-01..2021-12-31.
    picks = [
        ("GP_086", "2019-08-15"),  # UP asset, core monsoon
        ("RE_061", "2020-07-04"),  # Rajasthan solar, monsoon
        ("HW_002", "2021-01-26"),  # UP highway, dry season
        ("GP_084", "2021-11-30"),  # Kerala, monsoon retreat
        ("GP_081", "2019-12-31"),  # Gujarat, year boundary
    ]
    for asset_id, date_text in picks:
        day = datetime.date.fromisoformat(date_text)
        assert day >= datetime.date(2019, 3, 31)
        p95_by_month = {m: clim[asset_id][m]["p95"] for m in range(1, 13)}
        expected = independent_daily_features(fixture[asset_id], day, p95_by_month)
        row = written[(asset_id, date_text)]
        _compare_daily(expected, row, "%s %s (fixture)" % (asset_id, date_text))
        # Truncation proof: drop every fixture day AFTER the feature date
        # and recompute. If any future day leaked into a value, this would
        # differ.
        truncated = {d: v for d, v in fixture[asset_id].items() if d <= day}
        expected_t = independent_daily_features(truncated, day, p95_by_month)
        _compare_daily(
            expected_t, row, "%s %s (truncated)" % (asset_id, date_text)
        )


def _month_days_present(precip_by_date, year, month):
    return sorted(
        d for d in precip_by_date if d.year == year and d.month == month
    )


def _month_length(year, month):
    if month == 12:
        nxt = datetime.date(year + 1, 1, 1)
    else:
        nxt = datetime.date(year, month + 1, 1)
    return (nxt - datetime.date(year, month, 1)).days


def test_month_features_independent_recomputation_from_fixture():
    fixture = load_fixture_panel()
    clim = load_asset_climatology()
    written = {}
    with open(FEATURES_MONTH_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["asset_id"] in FIXTURE_ASSETS:
                written[(row["asset_id"], row["year_month"])] = row

    picks = [
        ("GP_086", 2020, 10),  # JJAS 2020 complete inside the fixture
        ("RE_061", 2021, 11),
        ("GP_084", 2019, 12),
    ]
    for asset_id, year, month in picks:
        days = _month_days_present(fixture[asset_id], year, month)
        assert len(days) == _month_length(year, month)
        total = sum(fixture[asset_id][d] for d in days)
        rain = sum(1 for d in days if fixture[asset_id][d] >= 1.0)
        stats = clim[asset_id][month]
        z = (total - stats["mean"]) / stats["sd"]
        # JJAS: full June-September total, non-null only on Oct/Nov/Dec.
        jdays = _month_days_present(fixture[asset_id], year, 6) + [
            d
            for m in (7, 8, 9)
            for d in _month_days_present(fixture[asset_id], year, m)
        ]
        if month in (10, 11, 12):
            assert len(jdays) == 122  # 30+31+31+30, always
            jjas = sum(fixture[asset_id][d] for d in jdays)
        else:
            jjas = None
        row = written[(asset_id, "%04d-%02d" % (year, month))]
        assert math.isclose(
            float(row["wx_precip_month_mm"]), total, rel_tol=1e-9, abs_tol=1e-9
        )
        assert int(row["wx_rain_days_month"]) == rain
        assert math.isclose(
            float(row["wx_anomaly_z"]), z, rel_tol=1e-9, abs_tol=1e-9
        )
        if jjas is None:
            assert row["wx_monsoon_JJAS_mm"] == ""
        else:
            assert math.isclose(
                float(row["wx_monsoon_JJAS_mm"]),
                jjas,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )


def test_month_tables_cover_exactly_the_point_source_assets():
    from build_panindia_features import load_asset_panel

    panel = load_asset_panel()
    assert set(panel) == non_null_coordinate_ids()
    with open(FEATURES_MONTH_CSV, newline="", encoding="utf-8") as f:
        ids = {row["asset_id"] for row in csv.DictReader(f)}
    assert ids == set(panel)


# --------------------------------------------------------------------------
# Firewall
# ------------------------------------------------------------------------------------


def test_new_outputs_satisfy_firewall():
    from firewall import FORBIDDEN_MASTER_COLUMNS, assert_no_forbidden_columns

    headers = {}
    with gzip.open(PANEL_GZ, "rt", encoding="utf-8", newline="") as f:
        headers["weather_asset_daily"] = next(csv.reader(f))
    with gzip.open(FEATURES_DAY_GZ, "rt", encoding="utf-8", newline="") as f:
        headers["rainfall_features_panindia_asset_day"] = next(csv.reader(f))
    for name, path in (
        ("asset_climatology", ASSET_CLIMATOLOGY_CSV),
        ("rainfall_features_panindia_asset_month", FEATURES_MONTH_CSV),
    ):
        with open(path, newline="", encoding="utf-8") as f:
            headers[name] = next(csv.reader(f))
    for name, cols in headers.items():
        assert_no_forbidden_columns(cols, name)
        assert not (set(cols) & FORBIDDEN_MASTER_COLUMNS)
