"""Tests for the fabricated-coordinate firewall guard."""

import pytest

from firewall import (
    FORBIDDEN_MASTER_COLUMNS,
    assert_no_forbidden_columns,
    load_verified_assets,
)


def test_denylist_is_frozen():
    assert FORBIDDEN_MASTER_COLUMNS == frozenset(
        {
            "lat",
            "lon",
            "mean_temp_c",
            "max_temp_c",
            "mean_wind_speed",
            "total_precipitation",
            "mean_solar_irradiance",
            "cost_deviation_pct",
            "schedule_delay_days",
            "risk_score",
            "om_to_ebitda_ratio",
            "climate_stress_index",
        }
    )


def test_raises_on_lat():
    with pytest.raises(ValueError) as exc:
        assert_no_forbidden_columns(
            ["asset_id", "sector", "name", "lat", "lon"], "test-context"
        )
    message = str(exc.value)
    assert "lat" in message and "test-context" in message


def test_raises_on_climate_stress_index():
    with pytest.raises(ValueError):
        assert_no_forbidden_columns(
            ["asset_id", "climate_stress_index"], "test-context"
        )


def test_passes_on_verified_lat_lon():
    # The sidecar's deliberately renamed columns must not trip the denylist.
    assert_no_forbidden_columns(
        ["asset_id", "sector", "verified_lat", "verified_lon"], "test-context"
    )


def test_load_verified_assets_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_verified_assets(str(tmp_path / "does_not_exist.csv"))


def test_load_verified_assets_malformed_raises(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("asset_id,sector\nHW_001,Highways\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_verified_assets(str(bad))


def test_load_verified_assets_returns_only_coordinate_rows(tmp_path):
    path = tmp_path / "sidecar.csv"
    path.write_text(
        "asset_id,sector,name,verified_lat,verified_lon,coord_role,"
        "verification_status,provenance_source,source_url\n"
        "A1,S,Port X,18.9,72.9,point,verified,Wikipedia,https://example.com\n"
        "A2,S,Unknown,, ,point,unverified,,\n",
        encoding="utf-8",
    )
    rows = load_verified_assets(str(path))
    assert [r["asset_id"] for r in rows] == ["A1"]
