"""Clean the NASA POWER UP daily rainfall CSV to a canonical district panel.

Reads : root ``UP_rainfall_dataset.csv`` (real NASA POWER daily district
weather for Uttar Pradesh) plus ``integration/config/districts_up.yaml``
(the district normalisation map). Writes the canonical panel
``integration/data_clean/weather_up_daily.csv`` and the per-district audit
table ``integration/data_clean/district_reference.csv``.

Cleaning rules (deterministic, stdlib only):
  * District labels are normalised ONLY through the YAML aliases map. A raw
    value with no entry aborts the run (it is an error, not a new district).
  * Exact-duplicate (district_raw, date) rows are dropped keeping the first;
    the per-district drop count is printed (known blemish: ``shamli``).
  * ``date`` is derived from YEAR/MO/DY; impossible dates abort the run.
  * The panel is checked monotonic in date per district and the gap list is
    printed; ``n_gap_days`` per district is recorded in the reference table.
  * Units are converted explicitly and nothing is silently rescaled:
    PRECTOTCORR is already mm/day, PS is already kPa, WS50M is already m/s,
    T2M_MAX/T2M_MIN are already degC, RH2M is already %.

Panel columns: district_canonical, district_raw, date, precip_mm_day,
tmax_c, tmin_c, rh_pct, wind_ms, ps_kpa. Sorted by (district_canonical,
date). CSV only; no parquet/pandas dependency. Byte-identical across runs.
"""

import csv
import datetime
import os
import sys

import yaml

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
SOURCE_CSV = os.path.join(REPO_ROOT, "UP_rainfall_dataset.csv")
DISTRICTS_YAML = os.path.join(REPO_ROOT, "integration", "config", "districts_up.yaml")
OUT_DIR = os.path.join(REPO_ROOT, "integration", "data_clean")
PANEL_CSV = os.path.join(OUT_DIR, "weather_up_daily.csv")
REFERENCE_CSV = os.path.join(OUT_DIR, "district_reference.csv")

PANEL_COLUMNS = [
    "district_canonical",
    "district_raw",
    "date",
    "precip_mm_day",
    "tmax_c",
    "tmin_c",
    "rh_pct",
    "wind_ms",
    "ps_kpa",
]

REFERENCE_COLUMNS = [
    "district_canonical",
    "district_raw_variants",
    "centroid_lat",
    "centroid_lon",
    "n_days",
    "date_min",
    "date_max",
    "n_gap_days",
]


def fmt_number(value):
    """Format a float deterministically (repr round-trips, stable bytes)."""
    return repr(float(value))


def load_district_map(path):
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    aliases = data.get("aliases") or {}
    canonical = data.get("canonical") or []
    if set(aliases.values()) != set(canonical):
        raise ValueError(
            "districts_up.yaml is inconsistent: alias targets do not equal "
            "the canonical list"
        )
    return aliases


def clean():
    aliases = load_district_map(DISTRICTS_YAML)

    with open(SOURCE_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {
            "YEAR", "MO", "DY", "RH2M", "PS", "WS50M", "PRECTOTCORR",
            "T2M_MAX", "T2M_MIN", "DISTRICT", "LATITUDE", "LONGITUDE",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "source CSV is missing required columns: %s" % sorted(missing)
            )
        source_rows = list(reader)

    n_source = len(source_rows)
    panel_rows = []
    seen_keys = set()
    dropped_dupes = {}
    rejected_dates = []
    unmatched_districts = set()

    for row in source_rows:
        raw = row["DISTRICT"]
        if raw not in aliases:
            unmatched_districts.add(raw)
            continue
        canonical = aliases[raw]
        # Derive date from YEAR/MO/DY; impossible dates are rejected, loudly.
        try:
            day = datetime.date(
                int(row["YEAR"]), int(row["MO"]), int(row["DY"])
            )
        except ValueError:
            rejected_dates.append(
                (raw, row["YEAR"], row["MO"], row["DY"])
            )
            continue
        key = (raw, day.isoformat())
        if key in seen_keys:
            dropped_dupes[raw] = dropped_dupes.get(raw, 0) + 1
            continue
        seen_keys.add(key)
        # Explicit unit handling (no silent rescaling):
        #   PRECTOTCORR is mm/day, PS is kPa, WS50M is m/s,
        #   T2M_MAX/T2M_MIN are degC, RH2M is %.
        panel_rows.append(
            {
                "district_canonical": canonical,
                "district_raw": raw,
                "date": day.isoformat(),
                "precip_mm_day": fmt_number(row["PRECTOTCORR"]),
                "tmax_c": fmt_number(row["T2M_MAX"]),
                "tmin_c": fmt_number(row["T2M_MIN"]),
                "rh_pct": fmt_number(row["RH2M"]),
                "wind_ms": fmt_number(row["WS50M"]),
                "ps_kpa": fmt_number(row["PS"]),
                "_lat": row["LATITUDE"],
                "_lon": row["LONGITUDE"],
            }
        )

    if unmatched_districts:
        raise ValueError(
            "unmatched raw DISTRICT values with no YAML alias (zero "
            "tolerance): %s" % sorted(unmatched_districts)
        )
    if rejected_dates:
        raise ValueError(
            "rejected %d impossible dates, e.g. %s"
            % (len(rejected_dates), rejected_dates[:5])
        )

    panel_rows.sort(key=lambda r: (r["district_canonical"], r["date"]))

    # Per-district monotonicity check + gap list.
    by_district = {}
    for row in panel_rows:
        by_district.setdefault(row["district_canonical"], []).append(row)
    gap_list = []
    for district in sorted(by_district):
        rows = by_district[district]
        dates = [datetime.date.fromisoformat(r["date"]) for r in rows]
        for prev, cur in zip(dates, dates[1:]):
            if cur <= prev:
                raise ValueError(
                    "panel is not monotonic in date for %s at %s"
                    % (district, cur.isoformat())
                )
        day = dates[0]
        have = set(dates)
        while day <= dates[-1]:
            if day not in have:
                gap_list.append((district, day.isoformat()))
            day += datetime.timedelta(days=1)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(PANEL_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=PANEL_COLUMNS, lineterminator="\n"
        )
        writer.writeheader()
        for row in panel_rows:
            writer.writerow({c: row[c] for c in PANEL_COLUMNS})

    with open(REFERENCE_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=REFERENCE_COLUMNS, lineterminator="\n"
        )
        writer.writeheader()
        for district in sorted(by_district):
            rows = by_district[district]
            raw_variants = sorted({r["district_raw"] for r in rows})
            lats = {r["_lat"] for r in rows}
            lons = {r["_lon"] for r in rows}
            if len(lats) != 1 or len(lons) != 1:
                raise ValueError(
                    "district %s has more than one centroid in the source: "
                    "%s %s" % (district, sorted(lats), sorted(lons))
                )
            dmin = rows[0]["date"]
            dmax = rows[-1]["date"]
            span = (
                datetime.date.fromisoformat(dmax)
                - datetime.date.fromisoformat(dmin)
            ).days + 1
            writer.writerow(
                {
                    "district_canonical": district,
                    "district_raw_variants": "|".join(raw_variants),
                    "centroid_lat": rows[0]["_lat"],
                    "centroid_lon": rows[0]["_lon"],
                    "n_days": len(rows),
                    "date_min": dmin,
                    "date_max": dmax,
                    "n_gap_days": span - len(rows),
                }
            )

    total_dropped = sum(dropped_dupes.values())
    print("source rows: %d" % n_source)
    print("panel rows: %d" % len(panel_rows))
    print("exact-duplicate (district_raw, date) rows dropped: %d" % total_dropped)
    for raw in sorted(dropped_dupes):
        print("  dropped %d duplicates for raw district %r" % (dropped_dupes[raw], raw))
    print("districts: %d" % len(by_district))
    print("gap days in panel: %d" % len(gap_list))
    for district, missing_day in gap_list:
        print("  gap: %s missing %s" % (district, missing_day))
    print("wrote %s" % PANEL_CSV)
    print("wrote %s" % REFERENCE_CSV)
    return {
        "n_source": n_source,
        "n_panel": len(panel_rows),
        "dropped_dupes": dropped_dupes,
        "n_gaps": len(gap_list),
    }


if __name__ == "__main__":
    sys.exit(clean() and 0)
