"""Shared climatology, feature and join logic for BOTH rainfall sources.

This module is the single implementation of the rainfall feature definitions.
Both sources feed it a canonical daily panel - a mapping
``{series_key: {date: precip_mm_day}}`` where ``series_key`` is a
``district_canonical`` for the Uttar Pradesh district panel or an
``asset_id`` for the pan-India asset panel - plus the matching climatology
(``{series_key: {month: {"mean", "sd", "p95", "n"}}}``). The feature
functions never learn which kind of series they are serving, so a
district-based and an asset-based entry point cannot diverge: they are the
same functions.

Feature definitions (identical for both sources; fixed, do not redesign):

  * All windows are trailing and backward-looking: a feature dated D uses
    days up to and including D, never after. If ANY day required by a
    trailing window is absent from the panel, that feature value is null
    for that row. Never forward-fill, never interpolate, never impute.
  * ``wx_cwd_30d`` / ``wx_dry_spell_90d`` are longest runs of consecutive
    wet (precip >= 1 mm) / dry (precip < 1 mm) clipped to the window.
  * ``wx_p95_exceed_days_30d`` compares each day's precip against the
    baseline p95 of wet days for that day's own calendar month.
  * ``wx_monsoon_to_date_mm`` sums precip since 1 June of the same
    calendar year up to and including D. For dates in January-May the
    window is empty and the value is 0 (no days are required, so the null
    policy does not apply).
  * ``wx_monsoon_JJAS_mm`` is the full June-September total of that
    calendar year. It is reported only on rows keyed October, November or
    December - the first month rows at which the whole season window is
    complete and observable without lookahead. Rows keyed January to
    September carry null.
  * ``wx_anomaly_z`` = (month total - baseline mean for that calendar
    month) / baseline population sd for that month; null when the month
    total is null or the baseline sd is null or zero.

Climatology baseline: 2005-01-01 to 2020-12-31. The years 2021-2025 are
held out and never enter the baseline. Baselines are per series per
calendar month: mean and population standard deviation of daily
precip_mm_day over all baseline days of that month, and the 95th
percentile (linear interpolation between closest ranks) of wet-day
(precip >= 1 mm) precipitation over baseline days of that month.

Join rule for the UP district source (fixed, do not redesign): for each
asset with non-null coordinates whose ``verification_status`` is not
``unverified``, find the nearest UP district centroid by haversine
distance (mean Earth radius 6371.0088 km). Join that district's daily
series ONLY IF the asset lies inside the Uttar Pradesh bounding box
(latitude 23.5..31.0, longitude 77.0..85.0) AND the distance is <= 75 km.
Every other asset receives no weather from this source. Never join UP
rainfall to a non-UP asset.

Deterministic: stdlib only, fixed constants, sorted iteration, LF line
endings. Byte-identical across runs. No network access.
"""

import csv
import datetime
import math
import os
import statistics

EARTH_RADIUS_KM = 6371.0088
UP_BOX_LAT_MIN, UP_BOX_LAT_MAX = 23.5, 31.0
UP_BOX_LON_MIN, UP_BOX_LON_MAX = 77.0, 85.0
MAX_JOIN_DISTANCE_KM = 75.0
BASELINE_START = datetime.date(2005, 1, 1)
BASELINE_END = datetime.date(2020, 12, 31)
SOURCE_NAME = "NASA POWER"
# Fixed constant (not datetime.today()) so repeated runs are byte-identical.
# Matches the retrieval date used by the phase-2 coordinate sidecar.
RETRIEVAL_DATE = "2026-10-14"

# Feature value thresholds (mm/day), fixed by the feature definitions.
HEAVY_DAY_MM = 50.0
VERY_HEAVY_DAY_MM = 100.0
RAIN_DAY_MM = 1.0

# Feature column names (shared by both sources; the key columns
# district_canonical / asset_id and date / year_month are added by the
# callers).
DAILY_FEATURE_COLUMNS = [
    "wx_precip_7d_mm",
    "wx_precip_30d_mm",
    "wx_heavy_days_7d",
    "wx_very_heavy_days_30d",
    "wx_rain_days_30d",
    "wx_cwd_30d",
    "wx_dry_spell_90d",
    "wx_p95_exceed_days_30d",
    "wx_monsoon_to_date_mm",
]

MONTH_FEATURE_COLUMNS = [
    "wx_precip_month_mm",
    "wx_rain_days_month",
    "wx_monsoon_JJAS_mm",
    "wx_anomaly_z",
]


def fmt_number(value):
    """Deterministic float formatting (repr round-trips, stable bytes)."""
    if value is None:
        return ""
    return repr(float(value))


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km, mean Earth radius 6371.0088 km."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def percentile_linear(sorted_values, p):
    """p-th percentile (0..100) of pre-sorted values.

    Linear interpolation between closest ranks (the common "linear"
    method): position = (p/100) * (n-1). Returns None for empty input.
    """
    if not sorted_values:
        return None
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    pos = (p / 100.0) * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_values[lo] + frac * (sorted_values[hi] - sorted_values[lo])


def load_panel(path):
    """Read the cleaned UP panel -> {district_canonical: {date: precip_mm_day}}."""
    precip = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            district = row["district_canonical"]
            day = datetime.date.fromisoformat(row["date"])
            precip.setdefault(district, {})[day] = float(row["precip_mm_day"])
    return precip


def load_centroids(path):
    """Read district_reference.csv -> {district_canonical: (lat, lon)}."""
    centroids = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            centroids[row["district_canonical"]] = (
                float(row["centroid_lat"]),
                float(row["centroid_lon"]),
            )
    return centroids


def load_assets(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def compute_climatology(precip_by_series):
    """Baseline stats per series (district or asset) per calendar month.

    ``precip_by_series`` maps a series key (district_canonical or asset_id)
    to ``{date: precip_mm_day}``. Returns
    ``{series_key: {month: {"mean":, "sd":, "p95":, "n":}}}`` computed from
    baseline days (2005-01-01..2020-12-31) only.
    """
    climatology = {}
    for series in sorted(precip_by_series):
        by_month = {m: [] for m in range(1, 13)}
        for day, value in precip_by_series[series].items():
            if BASELINE_START <= day <= BASELINE_END:
                by_month[day.month].append(value)
        months = {}
        for month in range(1, 13):
            values = by_month[month]
            wet = sorted(v for v in values if v >= RAIN_DAY_MM)
            months[month] = {
                "mean": statistics.fmean(values),
                "sd": statistics.pstdev(values),
                "p95": percentile_linear(wet, 95),
                "n": len(values),
            }
        climatology[series] = months
    return climatology


def write_climatology_csv(climatology, path, key_name="district_canonical"):
    """Write the per-series climatology table (CSV, LF endings).

    ``key_name`` is the first column header: ``district_canonical`` for the
    UP district source, ``asset_id`` for the pan-India asset source. The
    remaining columns and the row order (sorted by series, then month) are
    identical for both sources.
    """
    columns = [
        key_name,
        "month",
        "mean_precip_mm_day",
        "sd_precip_mm_day",
        "p95_wet_day_mm",
        "baseline_start",
        "baseline_end",
        "n_baseline_days",
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for series in sorted(climatology):
            for month in range(1, 13):
                stats = climatology[series][month]
                writer.writerow(
                    {
                        key_name: series,
                        "month": month,
                        "mean_precip_mm_day": fmt_number(stats["mean"]),
                        "sd_precip_mm_day": fmt_number(stats["sd"]),
                        "p95_wet_day_mm": fmt_number(stats["p95"]),
                        "baseline_start": BASELINE_START.isoformat(),
                        "baseline_end": BASELINE_END.isoformat(),
                        "n_baseline_days": stats["n"],
                    }
                )


def _prefix_sums(values):
    """Prefix sums with a leading zero: out[i] = sum(values[:i])."""
    out = [0.0] * (len(values) + 1)
    for i, v in enumerate(values):
        out[i + 1] = out[i] + v
    return out


def _prefix_counts(flags):
    out = [0] * (len(flags) + 1)
    for i, v in enumerate(flags):
        out[i + 1] = out[i] + (1 if v else 0)
    return out


class SeriesIndex:
    """Prefix-sum index over one series' panel for O(1) window queries.

    Serves a district panel and an asset panel alike: it only ever sees one
    series' ``{date: precip}`` mapping plus the calendar.
    """

    def __init__(self, precip_by_date, calendar_dates):
        self.calendar = list(calendar_dates)
        self.index = {d: i for i, d in enumerate(self.calendar)}
        precip = [precip_by_date.get(d) for d in self.calendar]
        self.precip = precip
        self.sum_pre = _prefix_sums([v if v is not None else 0.0 for v in precip])
        self.cnt_pre = _prefix_counts([v is not None for v in precip])
        self.heavy_pre = _prefix_counts(
            [v is not None and v >= HEAVY_DAY_MM for v in precip]
        )
        self.vheavy_pre = _prefix_counts(
            [v is not None and v >= VERY_HEAVY_DAY_MM for v in precip]
        )
        self.rain_pre = _prefix_counts(
            [v is not None and v >= RAIN_DAY_MM for v in precip]
        )
        # Run lengths of consecutive wet / dry days ending at each index.
        # An absent day breaks both runs.
        self.wet_run = [0] * len(self.calendar)
        self.dry_run = [0] * len(self.calendar)
        for i, v in enumerate(precip):
            if v is not None and v >= RAIN_DAY_MM:
                self.wet_run[i] = (self.wet_run[i - 1] if i else 0) + 1
            if v is not None and v < RAIN_DAY_MM:
                self.dry_run[i] = (self.dry_run[i - 1] if i else 0) + 1

    def window_complete(self, ia, ib):
        """True iff every calendar day in [ia, ib] is present in the panel."""
        return self.cnt_pre[ib + 1] - self.cnt_pre[ia] == ib - ia + 1

    def window_sum(self, ia, ib):
        return self.sum_pre[ib + 1] - self.sum_pre[ia]

    def window_count(self, pre, ia, ib):
        return pre[ib + 1] - pre[ia]

    def longest_run(self, run_lengths, ia, ib):
        """Longest run in [ia, ib], clipped at the window start."""
        best = 0
        for i in range(ia, ib + 1):
            clipped = min(run_lengths[i], i - ia + 1)
            if clipped > best:
                best = clipped
        return best


def compute_daily_features(series_key, precip_by_date, calendar_dates, climatology):
    """Daily rainfall features for one series over ``calendar_dates``.

    ``series_key`` is a ``district_canonical`` or an ``asset_id``; it only
    selects the climatology row. Returns ``{date_iso: {feature: value}}``
    where value is a float, int or None (null). Pure function of the panel:
    no lookahead, no filling. ``climatology`` is the output of
    :func:`compute_climatology`.
    """
    series = SeriesIndex(precip_by_date, calendar_dates)
    p95_by_month = {
        m: climatology[series_key][m]["p95"] for m in range(1, 13)
    }
    p95_none_months = {m for m, v in p95_by_month.items() if v is None}
    # Per-day flag: precip exceeds the baseline p95 of its own month.
    exceed_flags = []
    for i, v in enumerate(series.precip):
        day = series.calendar[i]
        exceeds = (
            v is not None
            and day.month not in p95_none_months
            and v > p95_by_month[day.month]
        )
        exceed_flags.append(exceeds)
    exceed_pre = _prefix_counts(exceed_flags)

    out = {}
    for i, day in enumerate(series.calendar):
        feats = {}
        # --- 7-day window ---
        ia = i - 6
        if ia >= 0 and series.window_complete(ia, i):
            feats["wx_precip_7d_mm"] = series.window_sum(ia, i)
            feats["wx_heavy_days_7d"] = series.window_count(
                series.heavy_pre, ia, i
            )
        else:
            feats["wx_precip_7d_mm"] = None
            feats["wx_heavy_days_7d"] = None
        # --- 30-day window ---
        ia = i - 29
        if ia >= 0 and series.window_complete(ia, i):
            feats["wx_precip_30d_mm"] = series.window_sum(ia, i)
            feats["wx_very_heavy_days_30d"] = series.window_count(
                series.vheavy_pre, ia, i
            )
            feats["wx_rain_days_30d"] = series.window_count(
                series.rain_pre, ia, i
            )
            feats["wx_cwd_30d"] = series.longest_run(series.wet_run, ia, i)
            if day.month in p95_none_months or any(
                series.calendar[j].month in p95_none_months
                for j in range(ia, i + 1)
            ):
                feats["wx_p95_exceed_days_30d"] = None
            else:
                feats["wx_p95_exceed_days_30d"] = series.window_count(
                    exceed_pre, ia, i
                )
        else:
            feats["wx_precip_30d_mm"] = None
            feats["wx_very_heavy_days_30d"] = None
            feats["wx_rain_days_30d"] = None
            feats["wx_cwd_30d"] = None
            feats["wx_p95_exceed_days_30d"] = None
        # --- 90-day window ---
        ia = i - 89
        if ia >= 0 and series.window_complete(ia, i):
            feats["wx_dry_spell_90d"] = series.longest_run(
                series.dry_run, ia, i
            )
        else:
            feats["wx_dry_spell_90d"] = None
        # --- monsoon to date ---
        if day.month >= 6:
            june1 = datetime.date(day.year, 6, 1)
            ja = series.index[june1]
            if series.window_complete(ja, i):
                feats["wx_monsoon_to_date_mm"] = series.window_sum(ja, i)
            else:
                feats["wx_monsoon_to_date_mm"] = None
        else:
            # January-May: the "since 1 June of that calendar year" window
            # is empty; no days are required, so the value is 0, not null.
            feats["wx_monsoon_to_date_mm"] = 0
        out[day.isoformat()] = feats
    return out


def compute_month_features(series_key, precip_by_date, months, climatology):
    """Month features for one series over ``months`` (list of (year, month)).

    Returns ``{(year, month): {feature: value}}``.
    """
    series = SeriesIndex(precip_by_date, _month_calendar(months))
    out = {}
    for year, month in months:
        first = datetime.date(year, month, 1)
        last = _month_end(year, month)
        ia = series.index[first]
        ib = series.index[last]
        complete = series.window_complete(ia, ib)
        if complete:
            total = series.window_sum(ia, ib)
            rain_days = series.window_count(series.rain_pre, ia, ib)
        else:
            total = None
            rain_days = None
        # Full June-September total of this calendar year, reported only on
        # Oct/Nov/Dec rows (first rows at which the season is complete).
        if month in (10, 11, 12):
            jja = series.index[datetime.date(year, 6, 1)]
            jjb = series.index[datetime.date(year, 9, 30)]
            if series.window_complete(jja, jjb):
                jjas = series.window_sum(jja, jjb)
            else:
                jjas = None
        else:
            jjas = None
        stats = climatology[series_key][month]
        sd = stats["sd"]
        if total is not None and sd is not None and sd > 0:
            anomaly_z = (total - stats["mean"]) / sd
        else:
            anomaly_z = None
        out[(year, month)] = {
            "wx_precip_month_mm": total,
            "wx_rain_days_month": rain_days,
            "wx_monsoon_JJAS_mm": jjas,
            "wx_anomaly_z": anomaly_z,
        }
    return out


def _month_calendar(months):
    """Contiguous calendar covering every (year, month) in ``months``."""
    days = []
    for year, month in months:
        day = datetime.date(year, month, 1)
        while day.month == month:
            days.append(day)
            day += datetime.timedelta(days=1)
    return days


def _month_end(year, month):
    if month == 12:
        return datetime.date(year, 12, 31)
    return datetime.date(year, month + 1, 1) - datetime.timedelta(days=1)


def join_assets(assets, centroids):
    """Apply the UP district join gate to every asset.

    Returns (entries, covered) where ``entries`` is a list of coverage
    dicts sorted by asset_id and ``covered`` maps asset_id ->
    (district_canonical, distance_km) for covered assets.

    Reason-code precedence: an unverified asset coordinate outranks the
    bounding box and distance gates (such an asset can never be joined,
    wherever it lies).
    """
    entries = []
    covered = {}
    for asset in sorted(assets, key=lambda r: r["asset_id"]):
        asset_id = asset["asset_id"]
        status = (asset.get("verification_status") or "").strip()
        lat_text = (asset.get("verified_lat") or "").strip()
        lon_text = (asset.get("verified_lon") or "").strip()
        lat = float(lat_text) if lat_text else None
        lon = float(lon_text) if lon_text else None
        entry = {
            "asset_id": asset_id,
            "family": "wx",
            "status": None,
            "reason_code": None,
            "nearest_district": None,
            "distance_km": None,
            "source_name": SOURCE_NAME,
            "retrieval_date": RETRIEVAL_DATE,
        }
        if status == "unverified" or lat is None or lon is None:
            entry["status"] = "no_coverage"
            entry["reason_code"] = "asset_coordinate_unverified"
        elif not (
            UP_BOX_LAT_MIN <= lat <= UP_BOX_LAT_MAX
            and UP_BOX_LON_MIN <= lon <= UP_BOX_LON_MAX
        ):
            entry["status"] = "no_coverage"
            entry["reason_code"] = "non_UP_asset"
        else:
            nearest = None
            nearest_dist = None
            for district in sorted(centroids):
                clat, clon = centroids[district]
                dist = haversine_km(lat, lon, clat, clon)
                if nearest_dist is None or dist < nearest_dist:
                    nearest = district
                    nearest_dist = dist
            entry["nearest_district"] = nearest
            entry["distance_km"] = round(nearest_dist, 3)
            if nearest_dist <= MAX_JOIN_DISTANCE_KM:
                entry["status"] = "covered"
                covered[asset_id] = (nearest, nearest_dist)
            else:
                entry["status"] = "no_coverage"
                entry["reason_code"] = "no_district_within_75km"
        entries.append(entry)
    return entries, covered


def full_calendar(precip_by_series):
    """Every calendar date from the panel's min to max date."""
    all_dates = set()
    for series in precip_by_series.values():
        all_dates.update(series)
    first, last = min(all_dates), max(all_dates)
    days = []
    day = first
    while day <= last:
        days.append(day)
        day += datetime.timedelta(days=1)
    return days


def panel_months(calendar_dates):
    """Every (year, month) present in the calendar, chronological."""
    months = []
    for day in calendar_dates:
        ym = (day.year, day.month)
        if not months or months[-1] != ym:
            months.append(ym)
    return months
