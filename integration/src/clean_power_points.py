"""Clean cached NASA POWER point responses into the canonical asset panel.

Reads (never the network - this script works purely from the local cache):
  * ``integration/data_raw_cache/power_points/<asset_id>.json`` (one raw
    response per asset, written by ``ingest_power_points.py``)
  * ``integration/data_raw_cache/power_points/MANIFEST.json`` (provenance:
    per-asset request URL and retrieval date)
  * ``integration/data_reference/assets_verified.csv`` (asset sidecar)

Writes:
  * ``integration/data_clean/weather_asset_daily.csv.gz`` (canonical asset
    daily panel; gzip because the 81 x 7638-day table exceeds the 20 MB
    committed-artifact limit - see README for the read path)
  * ``integration/data_clean/weather_asset_daily_gaps.json`` (per-asset
    calendar gaps; empty when every asset covers the full window)

Cleaning rules (deterministic, stdlib only):
  * The asset set is exactly the sidecar assets with non-null coordinates
    (81). The cache/manifest must cover all of them; anything else aborts.
  * ``date`` is derived from each YYYYMMDD key; an impossible date aborts
    the run loudly.
  * NASA POWER's fill value is -999 (any value <= -999 counts as fill for
    these parameters). A fill value never becomes a real measurement: if
    PRECTOTCORR is fill for a day, the day is omitted from the panel and
    recorded in the gap list (downstream features null-out via the
    null-on-gap policy). If PRECTOTCORR is valid but another parameter is
    fill, the row is emitted with that cell empty and ``is_observed=0``.
  * ``is_observed`` = 1 iff all seven parameters are non-fill for the day;
    ``is_imputed`` is always 0 - nothing is ever forward-filled,
    interpolated or imputed.
  * Per asset the calendar is asserted against the expected window
    2005-01-01..2025-11-29 and any gap is emitted (printed and written to
    the gaps file). Zero gaps is the expected, verified state.
  * Units are stated, never rescaled: precip_mm_day is PRECTOTCORR in
    mm/day, tmax_c/tmin_c are T2M_MAX/T2M_MIN in degC, rh_pct is RH2M in
    %, wind_ms is WS50M in m/s, ps_kpa is PS in kPa. (T2M is fetched and
    cached but the canonical panel carries the same columns as the UP
    district panel plus provenance, so T2M is not emitted.)

Panel columns: asset_id, date, precip_mm_day, tmax_c, tmin_c, rh_pct,
wind_ms, ps_kpa, source_name, source_url, retrieval_date, is_observed,
is_imputed. Sorted by (asset_id, date). gzip with a fixed header
(mtime=0, no embedded filename) so repeated runs are byte-identical.
"""

import csv
import datetime
import gzip
import io
import json
import os
import sys

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
SIDECAR_CSV = os.path.join(
    REPO_ROOT, "integration", "data_reference", "assets_verified.csv"
)
MANIFEST_JSON = os.path.join(
    REPO_ROOT, "integration", "data_raw_cache", "power_points", "MANIFEST.json"
)
CACHE_DIR = os.path.join(
    REPO_ROOT, "integration", "data_raw_cache", "power_points"
)
OUT_DIR = os.path.join(REPO_ROOT, "integration", "data_clean")
PANEL_GZ = os.path.join(OUT_DIR, "weather_asset_daily.csv.gz")
GAPS_JSON = os.path.join(OUT_DIR, "weather_asset_daily_gaps.json")

PANEL_COLUMNS = [
    "asset_id",
    "date",
    "precip_mm_day",
    "tmax_c",
    "tmin_c",
    "rh_pct",
    "wind_ms",
    "ps_kpa",
    "source_name",
    "source_url",
    "retrieval_date",
    "is_observed",
    "is_imputed",
]

WINDOW_START = datetime.date(2005, 1, 1)
WINDOW_END = datetime.date(2025, 11, 29)
PARAM_NAMES = ("PRECTOTCORR", "T2M_MAX", "T2M_MIN", "RH2M", "WS50M", "PS", "T2M")
# NASA POWER daily fill value is -999; anything at or below it is missing.
FILL_THRESHOLD = -999.0
SOURCE_NAME = "NASA POWER"


def fmt_number(value):
    """Deterministic float formatting (repr round-trips, stable bytes)."""
    return repr(float(value))


def is_fill(value):
    return value <= FILL_THRESHOLD


def expected_calendar():
    days = []
    day = WINDOW_START
    while day <= WINDOW_END:
        days.append(day)
        day += datetime.timedelta(days=1)
    return days


def load_sidecar_assets(path=SIDECAR_CSV):
    """[(asset_id, lat_text, lon_text)] with non-null coordinates, sorted."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for row in rows:
        lat_text = (row.get("verified_lat") or "").strip()
        lon_text = (row.get("verified_lon") or "").strip()
        if lat_text and lon_text:
            out.append((row["asset_id"], lat_text, lon_text))
    out.sort(key=lambda t: t[0])
    return out


def read_panel_row(panel, asset_id, day, values_by_param, source_url, retrieval_date):
    """Build one panel row dict; returns None when precip is a fill value.

    ``values_by_param`` maps parameter name -> value for this day.
    """
    precip = values_by_param["PRECTOTCORR"]
    if is_fill(precip):
        return None
    row = {
        "asset_id": asset_id,
        "date": day.isoformat(),
        "source_name": SOURCE_NAME,
        "source_url": source_url,
        "retrieval_date": retrieval_date,
        "is_imputed": 0,
    }
    complete = True
    column_param = (
        ("precip_mm_day", "PRECTOTCORR"),
        ("tmax_c", "T2M_MAX"),
        ("tmin_c", "T2M_MIN"),
        ("rh_pct", "RH2M"),
        ("wind_ms", "WS50M"),
        ("ps_kpa", "PS"),
    )
    for column, param in column_param:
        value = values_by_param[param]
        if is_fill(value):
            row[column] = ""
            complete = False
        else:
            row[column] = fmt_number(value)
    row["is_observed"] = 1 if complete else 0
    return row


def clean(cache_dir=CACHE_DIR, manifest_path=MANIFEST_JSON, panel_gz=PANEL_GZ,
          gaps_json=GAPS_JSON):
    assets = load_sidecar_assets()
    if not assets:
        raise ValueError("sidecar has no assets with coordinates")
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    manifest_entries = manifest.get("assets") or {}

    calendar = expected_calendar()
    expected_by_key = {d.strftime("%Y%m%d"): d for d in calendar}

    panel_rows = []
    gaps = {}
    seen_keys = set()

    for asset_id, _lat, _lon in assets:
        entry = manifest_entries.get(asset_id)
        if not entry:
            raise ValueError(
                "asset %s has coordinates but no MANIFEST.json entry; run "
                "integration/src/ingest_power_points.py first" % asset_id
            )
        cache_file = os.path.join(cache_dir, "%s.json" % asset_id)
        if not os.path.exists(cache_file):
            raise ValueError(
                "asset %s has no cached response at %s; run "
                "integration/src/ingest_power_points.py first"
                % (asset_id, cache_file)
            )
        with open(cache_file, "rb") as f:
            payload = json.loads(f.read().decode("utf-8"))
        params = (payload.get("properties") or {}).get("parameter")
        if not isinstance(params, dict):
            raise ValueError("cached response for %s has no properties.parameter" % asset_id)
        for name in PARAM_NAMES:
            if not isinstance(params.get(name), dict) or not params[name]:
                raise ValueError(
                    "cached response for %s is missing parameter %s" % (asset_id, name)
                )

        source_url = entry["request_url"]
        retrieval_date = entry["retrieval_date"]
        asset_gaps = []
        emitted = 0

        for key in sorted(params["PRECTOTCORR"]):
            if key not in expected_by_key:
                raise ValueError(
                    "impossible date key %r for %s (outside the requested "
                    "window)" % (key, asset_id)
                )
            day = expected_by_key[key]
            try:
                datetime.date(int(key[:4]), int(key[4:6]), int(key[6:8]))
            except ValueError:
                raise ValueError(
                    "impossible date %r for %s" % (key, asset_id)
                )
            values_by_param = {}
            for name in PARAM_NAMES:
                values_by_param[name] = params[name].get(key)
                if values_by_param[name] is None:
                    raise ValueError(
                        "parameter %s missing day %s for %s" % (name, key, asset_id)
                    )
            row = read_panel_row(
                panel_rows, asset_id, day, values_by_param, source_url, retrieval_date
            )
            if row is None:
                # PRECTOTCORR fill: the day is not a measurement. Omit it
                # and record the gap; features null-out downstream.
                asset_gaps.append(day.isoformat())
                continue
            key_pair = (asset_id, day.isoformat())
            if key_pair in seen_keys:
                raise ValueError(
                    "duplicate (asset_id, date) %r for %s" % (key_pair, asset_id)
                )
            seen_keys.add(key_pair)
            panel_rows.append(row)
            emitted += 1

        # Days expected but absent from the response are gaps too.
        present = {
            datetime.date.fromisoformat(r["date"])
            for r in panel_rows
            if r["asset_id"] == asset_id
        }
        for day in calendar:
            if day not in present and day.isoformat() not in asset_gaps:
                asset_gaps.append(day.isoformat())
        if asset_gaps:
            gaps[asset_id] = sorted(asset_gaps)

    panel_rows.sort(key=lambda r: (r["asset_id"], r["date"]))

    os.makedirs(OUT_DIR, exist_ok=True)
    # gzip with a fixed header (mtime=0, no embedded filename) so repeated
    # runs are byte-identical.
    with open(panel_gz, "wb") as raw:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, mtime=0
        ) as gz:
            with io.TextIOWrapper(gz, encoding="utf-8", newline="") as text:
                writer = csv.DictWriter(
                    text, fieldnames=PANEL_COLUMNS, lineterminator="\n"
                )
                writer.writeheader()
                for row in panel_rows:
                    writer.writerow(row)

    with open(gaps_json, "w", encoding="utf-8", newline="\n") as f:
        json.dump(
            {
                "expected_window": [
                    WINDOW_START.isoformat(),
                    WINDOW_END.isoformat(),
                ],
                "expected_days": len(calendar),
                "gaps": {aid: gaps[aid] for aid in sorted(gaps)},
                "n_assets_with_gaps": len(gaps),
            },
            f,
            indent=2,
        )
        f.write("\n")

    print("assets: %d" % len(assets))
    print("panel rows: %d" % len(panel_rows))
    print("assets with calendar gaps: %d" % len(gaps))
    for aid in sorted(gaps):
        print(
            "  gaps: %s missing %d day(s), e.g. %s"
            % (aid, len(gaps[aid]), ", ".join(gaps[aid][:5]))
        )
    contiguous = len(gaps) == 0
    if contiguous:
        print("calendar contiguous per asset: OK (%d days each)" % len(calendar))
    print("wrote %s" % panel_gz)
    print("wrote %s" % gaps_json)
    return {
        "n_assets": len(assets),
        "n_rows": len(panel_rows),
        "n_asset_with_gaps": len(gaps),
        "gaps": gaps,
    }


if __name__ == "__main__":
    sys.exit(clean() and 0)
