# Release note - infrastructure climate data-integration layer

## What this layer contains

Joined, audited external features for 100 Indian infrastructure assets
(`integration/data_output/feature_matrix_asset.csv`, one row per asset,
and `integration/data_output/feature_matrix_asset_month.csv`, 20,331
asset-month rows), with machine-readable honesty artifacts
(`weather_coverage.json`, `coverage_report.json`, `validation_report.json`):

- **Verified coordinates** for 81 assets (36 point-verified, 45
  representative midpoints) replacing the Master file's fabricated
  lat/lon, which are quarantined by `src/firewall.py` and never used.
  19 assets could not be verified and carry NULL coordinates.
- **Rainfall features** from two NASA POWER sources: the cleaned Uttar
  Pradesh district panel (11 assets near UP districts) and the per-asset
  point-API pull (all 81 coordinate-bearing assets, 2005-2025). The two
  paths share one implementation and agree with r = 0.982, MAD 8.1 mm on
  2,750 shared asset-months.
- **Highway exposure** from OpenStreetMap via Overpass (25 km tiles):
  distance to the nearest national highway, highway lengths within
  10 km, road-graph node degree, corridor flags (78 of 81 covered).
- **Maritime exposure** from IMF PortWatch via HDX (nearest of 37 India
  ports; 11 activity-defined majors), HOT Sea Ports of India polygons
  and the Natural Earth coastline (81 of 81 covered; 2 assets inside a
  mapped port polygon).
- **Nine validation gates** (`src/validate_layer.py`, exit non-zero on
  any failure): schema, physical ranges over 1,176,250 daily rows,
  anchor tolerances, a fabricated-coordinate tripwire, cross-source
  reconciliation, byte-identical determinism, zero imputation,
  licence attribution and coverage accounting. All pass; see
  `validation_report.json` for measured values.

## What it does NOT contain

- No cost, delay, O&M or EBITDA modelling. The Master file's
  `cost_deviation_pct`, `schedule_delay_days`, `risk_score`,
  `om_to_ebitda_ratio` and `climate_stress_index` are formula-derived
  and quarantined; nothing here reads them.
- No official port throughput (TEU): `mt_throughput_teu_yr` is NULL on
  all 81 rows (reason `throughput_gated`) because no licenced
  programmatic source exists.
- No line geometry for pipelines/transmission corridors: linear assets
  are represented by their sidecar midpoint coordinate only.
- No weather for the 19 unverified-coordinate assets and no UP-district
  weather outside the UP join gate; NULLs are explained per row in
  `null_reason` and per source in coverage - never filled.

## What the data cannot support

- Asset-day rainfall at phases below the district/point resolution;
  corridor flags near the 5 km boundary (straight-line approximations);
  node-degree means over a handful of nodes; any statement about the 19
  unverified assets beyond their documented reasons.
- The single tripwire exception: PT_040's Master coordinate lies 26 km
  from truth by coincidence (documented in the sidecar notes); every
  other verified coordinate is >50 km from its Master value.

## Boundary needing gated data

Project-level cost, delay, O&M and official port throughput require the
authoritative (gated) sources listed in SOURCES.md - principally the
Indian Ports Association / Sagarmala throughput tables (HTML/PDF-only,
not programmatically obtainable) and audited project accounts. This
layer deliberately stops at that boundary.

## Artifacts

All under `integration/`: `data_output/feature_matrix_asset.csv`
(100 rows), `data_output/feature_matrix_asset_month.csv` (20,331 rows),
`data_output/coverage_report.json`, `data_output/validation_report.json`,
`data_output/weather_coverage.json`, `data_output/verification_summary.json`,
`data_features/` (rainfall/highway/maritime tables),
`data_reference/assets_verified.csv`,
`data_raw_cache/*/MANIFEST.json` (raw responses are gitignored).

## Reproduction

```sh
python3 integration/src/verify_assets.py
python3 integration/src/clean_weather.py
python3 integration/src/ingest_power_points.py --offline
python3 integration/src/clean_power_points.py
python3 integration/src/build_rainfall_features.py
python3 integration/src/build_panindia_features.py
python3 integration/src/ingest_osm_highways.py --offline
python3 integration/src/build_highway_features.py
python3 integration/src/ingest_ports.py --offline
python3 integration/src/build_maritime_features.py
python3 integration/src/join_feature_matrix.py
python3 integration/src/validate_layer.py
python3 -m pytest integration/tests -q
```

Network steps use `--offline` against the committed MANIFESTs plus the
gitignored caches; cold rebuilds omit `--offline`. `--offline` only works
where the caches were already fetched: on a fresh clone the caches are
absent, so the three `--offline` ingest steps exit 2 (see RUN.md for the
exact errors). Every step is
byte-identical on rerun (sha256 recorded in `validation_report.json`).

## Acceptance checks (no code needed)

1. `feature_matrix_asset.csv` has 100 rows, one per asset_id, sorted.
2. `validation_report.json` says `"overall": "pass"` with 9 gates.
3. `grep -c asset_coordinate_unverified` reasoning covers every empty
   feature cell; `null_reason` is non-empty exactly where features are.
4. No file under `integration/data_output/` exceeds 20 MB.
5. `coverage_report.json` accounts 100 assets x 6 sources.

## Source registers

- [AUDIT.md](../AUDIT.md) - what was wrong with the Master file.
- [SOURCES.md](../SOURCES.md) - every external source and licence.
- [RETIRED.md](../RETIRED.md) - retired approaches and why.
