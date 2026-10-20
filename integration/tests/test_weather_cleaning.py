"""Tests for the deterministic weather cleaning step.

Offline: reads the committed source CSV, the YAML district map and the
committed cleaned artifacts only. No network access.
"""

import csv
import datetime
import os

import yaml

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
SOURCE_CSV = os.path.join(REPO_ROOT, "UP_rainfall_dataset.csv")
DISTRICTS_YAML = os.path.join(
    REPO_ROOT, "integration", "config", "districts_up.yaml"
)
PANEL_CSV = os.path.join(
    REPO_ROOT, "integration", "data_clean", "weather_up_daily.csv"
)
REFERENCE_CSV = os.path.join(
    REPO_ROOT, "integration", "data_clean", "district_reference.csv"
)

# Ground truth established by the phase-1 audit of UP_rainfall_dataset.csv.
EXPECTED_DISTINCT_RAW = 73
EXPECTED_DISTRICTS = 73
EXPECTED_SOURCE_ROWS = 565210
EXPECTED_DUPLICATE_DROPS = 7638  # shamli exact-duplicate rows
EXPECTED_PANEL_ROWS = EXPECTED_SOURCE_ROWS - EXPECTED_DUPLICATE_DROPS
PANEL_MAX_DATE = datetime.date(2025, 11, 29)
# The two blemished districts are each missing the final panel day.
EXPECTED_DATE_MAX_EXCEPTIONS = {"Lakhimpur", "Mau"}


def load_yaml_map():
    with open(DISTRICTS_YAML, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["canonical"], data["aliases"]


def raw_district_values():
    with open(SOURCE_CSV, newline="", encoding="utf-8") as f:
        return {row["DISTRICT"] for row in csv.DictReader(f)}


def load_panel():
    with open(PANEL_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_district_normalisation_is_total():
    canonical, aliases = load_yaml_map()
    raws = raw_district_values()
    unmatched = sorted(r for r in raws if r not in aliases)
    assert not unmatched, (
        "raw DISTRICT values with no YAML alias (zero tolerance): %s"
        % unmatched
    )
    assert len(raws) == EXPECTED_DISTINCT_RAW, (
        "expected %d distinct raw DISTRICT values, found %d: %s"
        % (EXPECTED_DISTRICTS, len(raws), sorted(raws))
    )
    # The map is internally consistent and surjective onto the data.
    assert set(aliases.values()) == set(canonical)
    assert {aliases[r] for r in raws} == set(canonical)


def test_duplicate_drop_count_and_panel_size():
    panel = load_panel()
    assert len(panel) == EXPECTED_PANEL_ROWS
    n_shamli_raw = 0
    with open(SOURCE_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["DISTRICT"] == "shamli":
                n_shamli_raw += 1
    n_shamli_panel = sum(1 for r in panel if r["district_canonical"] == "Shamli")
    assert n_shamli_raw == 15276
    assert n_shamli_panel == 7638
    assert n_shamli_raw - n_shamli_panel == EXPECTED_DUPLICATE_DROPS


def test_no_duplicate_panel_keys():
    seen = set()
    for row in load_panel():
        key = (row["district_canonical"], row["date"])
        assert key not in seen, "duplicate (district_canonical, date): %s" % (key,)
        seen.add(key)


def test_panel_sorted_by_district_then_date():
    keys = [
        (r["district_canonical"], r["date"]) for r in load_panel()
    ]
    assert keys == sorted(keys)


def test_date_max_and_known_gaps():
    panel = load_panel()
    panel_max = max(datetime.date.fromisoformat(r["date"]) for r in panel)
    assert panel_max == PANEL_MAX_DATE
    by_district = {}
    for row in panel:
        by_district.setdefault(row["district_canonical"], set()).add(
            datetime.date.fromisoformat(row["date"])
        )
    assert len(by_district) == EXPECTED_DISTRICTS
    exceptions = set()
    for district, days in by_district.items():
        if max(days) != PANEL_MAX_DATE:
            exceptions.add(district)
            # The known gap is the final panel day, missing entirely.
            assert PANEL_MAX_DATE not in days, (
                "district %s: panel max date absent" % district
            )
            assert max(days) == PANEL_MAX_DATE - datetime.timedelta(days=1)
    assert exceptions == EXPECTED_DATE_MAX_EXCEPTIONS, (
        "districts whose date_max differs from the panel max: %s"
        % sorted(exceptions)
    )
    # Every non-blemished district really does reach the panel max date.
    for district in by_district:
        if district not in EXPECTED_DATE_MAX_EXCEPTIONS:
            assert max(by_district[district]) == PANEL_MAX_DATE


def test_value_ranges():
    for row in load_panel():
        precip = float(row["precip_mm_day"])
        tmax = float(row["tmax_c"])
        tmin = float(row["tmin_c"])
        rh = float(row["rh_pct"])
        wind = float(row["wind_ms"])
        ps = float(row["ps_kpa"])
        assert 0.0 <= precip <= 500.0, row
        assert -10.0 <= tmax <= 55.0, row
        assert -10.0 <= tmin <= 55.0, row
        assert tmax >= tmin, row
        assert 0.0 <= rh <= 100.0, row
        assert 0.0 <= wind <= 60.0, row
        assert 70.0 <= ps <= 105.0, row


def test_district_reference_table():
    with open(REFERENCE_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == EXPECTED_DISTRICTS
    for row in rows:
        assert row["date_min"] == "2005-01-01"
        if row["district_canonical"] in EXPECTED_DATE_MAX_EXCEPTIONS:
            assert row["date_max"] == "2025-11-28"
            assert int(row["n_days"]) == 7637
        else:
            assert row["date_max"] == "2025-11-29"
            assert int(row["n_days"]) == 7638
    assert sum(int(r["n_days"]) for r in rows) == EXPECTED_PANEL_ROWS
