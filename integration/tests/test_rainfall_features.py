"""Tests for the rainfall climatology and asset-level feature build.

Offline: reads the committed cleaned panel, climatology and feature files.
The no-lookahead test recomputes features independently from the cleaned
panel (own window logic, no shared helper) and additionally recomputes
them from a panel truncated at the feature date, which proves no future
day can influence a value.
"""

import csv
import datetime
import math
import os

import pytest

from build_rainfall_features import compute_daily_features

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
PANEL_CSV = os.path.join(
    REPO_ROOT, "integration", "data_clean", "weather_up_daily.csv"
)
REFERENCE_CSV = os.path.join(
    REPO_ROOT, "integration", "data_clean", "district_reference.csv"
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
SOURCE_CSV = os.path.join(REPO_ROOT, "UP_rainfall_dataset.csv")

BASELINE_START = datetime.date(2005, 1, 1)
BASELINE_END = datetime.date(2020, 12, 31)

FLOAT_FEATURES = {
    "wx_precip_7d_mm",
    "wx_precip_30d_mm",
    "wx_p95_exceed_days_30d",
    "wx_monsoon_to_date_mm",
}
INT_FEATURES = {
    "wx_heavy_days_7d",
    "wx_very_heavy_days_30d",
    "wx_rain_days_30d",
    "wx_cwd_30d",
    "wx_dry_spell_90d",
}
ALL_FEATURES = FLOAT_FEATURES | INT_FEATURES


def load_panel():
    """{district_canonical: {date: precip_mm_day}} from the cleaned panel."""
    panel = {}
    with open(PANEL_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            panel.setdefault(row["district_canonical"], {})[
                datetime.date.fromisoformat(row["date"])
            ] = float(row["precip_mm_day"])
    return panel


def load_features_day():
    with open(FEATURES_DAY_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_climatology():
    """{district: {month: {"mean","sd","p95","n"}}}."""
    clim = {}
    with open(CLIMATOLOGY_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            clim.setdefault(row["district_canonical"], {})[int(row["month"])] = {
                "mean": float(row["mean_precip_mm_day"]),
                "sd": float(row["sd_precip_mm_day"]),
                "p95": float(row["p95_wet_day_mm"]),
                "n": int(row["n_baseline_days"]),
            }
    return clim


def independent_daily_features(precip_by_date, day, p95_by_month):
    """Recompute every daily feature for one date straight from the panel.

    Written from the feature definitions alone (no import of the build
    module), so it is a genuine independent recomputation. ``p95_by_month``
    maps calendar month -> baseline p95 of wet days (or None).
    """
    out = {}

    def window(k):
        return [day - datetime.timedelta(days=i) for i in range(k - 1, -1, -1)]

    def present(days):
        return all(d in precip_by_date for d in days)

    w7, w30, w90 = window(7), window(30), window(90)
    if present(w7):
        vals = [precip_by_date[d] for d in w7]
        out["wx_precip_7d_mm"] = sum(vals)
        out["wx_heavy_days_7d"] = sum(1 for v in vals if v >= 50.0)
    else:
        out["wx_precip_7d_mm"] = None
        out["wx_heavy_days_7d"] = None
    if present(w30):
        vals = [precip_by_date[d] for d in w30]
        out["wx_precip_30d_mm"] = sum(vals)
        out["wx_very_heavy_days_30d"] = sum(1 for v in vals if v >= 100.0)
        out["wx_rain_days_30d"] = sum(1 for v in vals if v >= 1.0)
        out["wx_cwd_30d"] = _longest_run([v >= 1.0 for v in vals])
        if any(p95_by_month[d.month] is None for d in w30):
            out["wx_p95_exceed_days_30d"] = None
        else:
            out["wx_p95_exceed_days_30d"] = sum(
                1
                for d in w30
                if precip_by_date[d] > p95_by_month[d.month]
            )
    else:
        out["wx_precip_30d_mm"] = None
        out["wx_very_heavy_days_30d"] = None
        out["wx_rain_days_30d"] = None
        out["wx_cwd_30d"] = None
        out["wx_p95_exceed_days_30d"] = None
    if present(w90):
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
        out["wx_monsoon_to_date_mm"] = (
            sum(precip_by_date[d] for d in since_june)
            if present(since_june)
            else None
        )
    else:
        out["wx_monsoon_to_date_mm"] = 0
    return out


def _longest_run(flags):
    best = 0
    current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        best = max(best, current)
    return best


def _compare_features(expected, actual, context):
    """Compare a written feature row against an independent recomputation.

    Counts and durations compare exactly; float sums compare with a tight
    tolerance because float addition is not associative (the independent
    summation order need not match the builder's prefix-sum order).
    """
    for name in ALL_FEATURES:
        written = actual[name]
        recomputed = expected[name]
        if recomputed is None:
            assert written == "", "%s: expected null, got %r" % (context, written)
        elif name in INT_FEATURES:
            assert written != "", "%s: expected %r, got null" % (context, recomputed)
            assert int(written) == recomputed, (
                "%s: expected %r, got %r" % (context, recomputed, written)
            )
        else:
            assert written != "", "%s: expected %r, got null" % (context, recomputed)
            assert math.isclose(
                float(written), recomputed, rel_tol=1e-9, abs_tol=1e-9
            ), "%s: expected %r, got %r" % (context, recomputed, written)


def test_no_lookahead_independent_recomputation():
    panel = load_panel()
    clim = load_climatology()
    features = load_features_day()
    written = {(r["asset_id"], r["date"]): r for r in features}

    # Several (asset, date) rows across different districts and seasons.
    picks = [
        ("HW_007", "2020-08-15"),  # Ghaziabad, core monsoon
        ("TL_053", "2016-07-04"),  # Mirzpur, monsoon
        ("GP_086", "2025-08-31"),  # Kanpur, end of panel
        ("HW_002", "2011-01-26"),  # Mathura, dry season
        ("TL_049", "2008-09-30"),  # Gautam Buddha Nagar, monsoon retreat
        ("HW_014", "2019-06-02"),  # Sultanpur, monsoon onset
    ]
    for asset_id, date_text in picks:
        row = written[(asset_id, date_text)]
        district = row["district_canonical"]
        day = datetime.date.fromisoformat(date_text)
        p95_by_month = {
            m: clim[district][m]["p95"] for m in range(1, 13)
        }
        expected = independent_daily_features(
            panel[district], day, p95_by_month
        )
        _compare_features(
            expected, row, "%s %s (full panel)" % (asset_id, date_text)
        )
        # Truncation proof: drop every panel day AFTER the feature date and
        # recompute. If any future day leaked into a value, this would
        # differ. Counts/sums over past days are unchanged.
        truncated = {
            d: v for d, v in panel[district].items() if d <= day
        }
        expected_truncated = independent_daily_features(
            truncated, day, p95_by_month
        )
        _compare_features(
            expected_truncated,
            row,
            "%s %s (truncated panel)" % (asset_id, date_text),
        )


def test_null_on_gap_synthetic():
    """A mid-series gap forces nulls; nothing is filled or interpolated."""
    # 60 contiguous days, then a 3-day gap, then 40 more days.
    first = datetime.date(2010, 3, 1)
    precip = {}
    for i in range(60):
        precip[first + datetime.timedelta(days=i)] = 10.0
    gap_start = first + datetime.timedelta(days=60)
    for i in range(40):
        precip[gap_start + datetime.timedelta(days=3 + i)] = 20.0
    calendar = [first + datetime.timedelta(days=i) for i in range(100)]
    clim = {
        "X": {
            m: {"mean": 5.0, "sd": 2.0, "p95": 30.0, "n": 100}
            for m in range(1, 13)
        }
    }
    feats = compute_daily_features("X", precip, calendar, clim)
    # The last day before the gap: its 7/30/90-day windows are complete.
    before = (gap_start - datetime.timedelta(days=1)).isoformat()
    assert feats[before]["wx_precip_7d_mm"] == 70.0
    assert feats[before]["wx_rain_days_30d"] == 30
    # The first day inside the gap: every windowed feature is null, and the
    # null is not a filled/interpolated value.
    inside = gap_start.isoformat()
    for name in ALL_FEATURES:
        if name == "wx_monsoon_to_date_mm":
            continue
        assert feats[inside][name] is None, name
    # The day after the gap: windows still reach into the gap -> null.
    after = (gap_start + datetime.timedelta(days=3)).isoformat()
    assert feats[after]["wx_precip_7d_mm"] is None
    assert feats[after]["wx_precip_30d_mm"] is None
    assert feats[after]["wx_dry_spell_90d"] is None
    # Once 7 days have passed after the gap, the 7-day window is complete
    # again and the value is a real sum over the post-gap days only.
    recovered = (gap_start + datetime.timedelta(days=9)).isoformat()
    assert feats[recovered]["wx_precip_7d_mm"] == 140.0
    # The 30-day window still reaches back into the gap -> null, not a
    # partial sum over the available days.
    assert feats[recovered]["wx_rain_days_30d"] is None
    # Once 30 days have passed after the gap, the 30-day window is whole
    # again and counts exactly the post-gap wet days.
    recovered_30d = (gap_start + datetime.timedelta(days=32)).isoformat()
    assert feats[recovered_30d]["wx_precip_30d_mm"] == 600.0
    assert feats[recovered_30d]["wx_rain_days_30d"] == 30


def test_null_policy_matches_window_completeness():
    """On real data the null mask equals independent window completeness."""
    panel = load_panel()
    features = load_features_day()
    # Check one full asset (HW_007 -> Ghaziabad) across the whole calendar.
    rows = [r for r in features if r["asset_id"] == "HW_007"]
    assert rows, "HW_007 rows missing"
    district = rows[0]["district_canonical"]
    series = panel[district]
    for row in rows:
        day = datetime.date.fromisoformat(row["date"])
        for window, names in (
            (7, {"wx_precip_7d_mm", "wx_heavy_days_7d"}),
            (
                30,
                {
                    "wx_precip_30d_mm",
                    "wx_very_heavy_days_30d",
                    "wx_rain_days_30d",
                    "wx_cwd_30d",
                    "wx_p95_exceed_days_30d",
                },
            ),
            (90, {"wx_dry_spell_90d"}),
        ):
            complete = all(
                (day - datetime.timedelta(days=i)) in series
                for i in range(window)
            )
            for name in names:
                if complete:
                    assert row[name] != "", (
                        "HW_007 %s: %s should be non-null" % (row["date"], name)
                    )
                else:
                    assert row[name] == "", (
                        "HW_007 %s: %s should be null (incomplete window)"
                        % (row["date"], name)
                    )
    # Early-panel rows: the first 6/29/89 days of 2005 carry nulls.
    early = {r["date"]: r for r in rows}
    assert early["2005-01-01"]["wx_precip_7d_mm"] == ""
    assert early["2005-01-06"]["wx_precip_7d_mm"] == ""
    assert early["2005-01-07"]["wx_precip_7d_mm"] != ""
    assert early["2005-01-29"]["wx_precip_30d_mm"] == ""
    assert early["2005-01-30"]["wx_precip_30d_mm"] != ""
    assert early["2005-03-30"]["wx_dry_spell_90d"] == ""
    assert early["2005-03-31"]["wx_dry_spell_90d"] != ""
    # January-May monsoon-to-date is 0 (empty window), never null.
    assert early["2005-01-15"]["wx_monsoon_to_date_mm"] == "0.0"


def test_climatology_baseline_period():
    # Recompute one district-month straight from the raw source CSV,
    # restricted to baseline days, and compare with the committed file.
    district_raw = "Agra"
    district_canonical = "Agra"
    values = {m: [] for m in range(1, 13)}
    with open(SOURCE_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["DISTRICT"] != district_raw:
                continue
            day = datetime.date(
                int(row["YEAR"]), int(row["MO"]), int(row["DY"])
            )
            if BASELINE_START <= day <= BASELINE_END:
                values[day.month].append(float(row["PRECTOTCORR"]))
    clim = load_climatology()
    for month in range(1, 13):
        vals = values[month]
        wet = sorted(v for v in vals if v >= 1.0)
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        sd = math.sqrt(var)
        pos = 0.95 * (len(wet) - 1)
        lo = int(math.floor(pos))
        hi = min(lo + 1, len(wet) - 1)
        p95 = wet[lo] + (pos - lo) * (wet[hi] - wet[lo])
        got = clim[district_canonical][month]
        assert got["n"] == len(vals)
        assert math.isclose(got["mean"], mean, rel_tol=1e-12)
        assert math.isclose(got["sd"], sd, rel_tol=1e-12)
        assert math.isclose(got["p95"], p95, rel_tol=1e-12)
    # The 2021-2025 holdout is real: including it changes the baseline.
    all_values = {m: [] for m in range(1, 13)}
    with open(SOURCE_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["DISTRICT"] != district_raw:
                continue
            day = datetime.date(
                int(row["YEAR"]), int(row["MO"]), int(row["DY"])
            )
            all_values[day.month].append(float(row["PRECTOTCORR"]))
    changed = sum(
        1
        for m in range(1, 13)
        if math.isclose(
            sum(all_values[m]) / len(all_values[m]),
            clim[district_canonical][m]["mean"],
            rel_tol=1e-6,
        )
        is False
    )
    assert changed >= 10, (
        "including 2021-2025 should change most district-month means; "
        "only %d/12 changed" % changed
    )


def test_climatology_table_shape_and_baseline_columns():
    clim = load_climatology()
    assert len(clim) == 73
    with open(CLIMATOLOGY_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 73 * 12
    for row in rows:
        assert row["baseline_start"] == "2005-01-01"
        assert row["baseline_end"] == "2020-12-31"
        month = int(row["month"])
        # 2005-2020 holds 16 years; every district baseline is complete.
        leap_years = 4  # 2008, 2012, 2016, 2020
        if month == 2:
            expected_n = 12 * 28 + leap_years * 29
        else:
            expected_n = 16 * (30 if month in (4, 6, 9, 11) else 31)
        assert int(row["n_baseline_days"]) == expected_n, row


def test_month_features_independent_recomputation():
    panel = load_panel()
    clim = load_climatology()
    with open(FEATURES_MONTH_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    written = {(r["asset_id"], r["year_month"]): r for r in rows}
    picks = [
        ("HW_007", "2020-08"),
        ("TL_053", "2016-07"),
        ("GP_086", "2025-10"),
        ("HW_002", "2011-01"),
        ("HW_014", "2019-06"),
    ]
    for asset_id, ym in picks:
        row = written[(asset_id, ym)]
        district = row["district_canonical"]
        year, month = int(ym[:4]), int(ym[5:7])
        series = panel[district]
        days = []
        d = datetime.date(year, month, 1)
        while d.month == month:
            days.append(d)
            d += datetime.timedelta(days=1)
        complete = all(d in series for d in days)
        if complete:
            total = sum(series[d] for d in days)
            rain = sum(1 for d in days if series[d] >= 1.0)
            assert math.isclose(
                float(row["wx_precip_month_mm"]), total, rel_tol=1e-9
            )
            assert int(row["wx_rain_days_month"]) == rain
            stats = clim[district][month]
            expected_z = (total - stats["mean"]) / stats["sd"]
            assert math.isclose(
                float(row["wx_anomaly_z"]), expected_z, rel_tol=1e-9
            )
        else:
            assert row["wx_precip_month_mm"] == ""
            assert row["wx_rain_days_month"] == ""
            assert row["wx_anomaly_z"] == ""
        # JJAS: non-null only on Oct/Nov/Dec rows, equal to the full
        # June-September total of that year.
        if month in (10, 11, 12):
            jjas_days = []
            d = datetime.date(year, 6, 1)
            while d <= datetime.date(year, 9, 30):
                jjas_days.append(d)
                d += datetime.timedelta(days=1)
            if all(d in series for d in jjas_days):
                expected_jjas = sum(series[d] for d in jjas_days)
                assert math.isclose(
                    float(row["wx_monsoon_JJAS_mm"]),
                    expected_jjas,
                    rel_tol=1e-9,
                )
            else:
                assert row["wx_monsoon_JJAS_mm"] == ""
        else:
            assert row["wx_monsoon_JJAS_mm"] == "", (
                "%s %s: JJAS must be null outside Oct-Dec" % (asset_id, ym)
            )


def test_feature_outputs_satisfy_firewall():
    from firewall import assert_no_forbidden_columns

    with open(FEATURES_DAY_CSV, newline="", encoding="utf-8") as f:
        day_cols = next(csv.reader(f))
    with open(FEATURES_MONTH_CSV, newline="", encoding="utf-8") as f:
        month_cols = next(csv.reader(f))
    assert_no_forbidden_columns(day_cols, "rainfall_features_asset_day")
    assert_no_forbidden_columns(month_cols, "rainfall_features_asset_month")
