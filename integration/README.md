# Integration layer

A data-integration layer for the 100-asset Indian infrastructure study. It
joins authoritative external references to the Master asset file's identity
columns and quarantines everything else.

## What this layer is

- A verified asset-coordinate reference table (`data_reference/`) that
  replaces the Master file's fabricated `lat`/`lon` columns. The audit
  established that those coordinates are 208-2119 km from the real named
  assets, so no downstream join may ever use them.
- A hard validation gate (`src/firewall.py`) that raises whenever a
  quarantined Master column appears in an integration input.
- Machine-readable honesty artifacts (`data_output/verification_summary.json`)
  reporting exactly what could and could not be verified.
- Rainfall features for the assets from **two sources** (see "Rainfall"
  below): the cleaned NASA POWER district panel for Uttar Pradesh, and
  NASA POWER point-API weather fetched at each asset's own coordinate for
  all of India.

## Directory layout

| Path | Contents |
| --- | --- |
| `config/asset_coordinates_reference.yaml` | The human-auditable source of every coordinate: one block per asset_id with lat, lon, role, status, source URL and note. Review this file to audit any coordinate. |
| `src/verify_assets.py` | Deterministic generator: joins the YAML reference to the Master identity columns and writes the sidecar, the log and the summary. Reads asset_id, sector, name only. |
| `src/firewall.py` | Reusable guard (`FORBIDDEN_MASTER_COLUMNS`, `assert_no_forbidden_columns`, `load_verified_assets`) for this and later phases. |
| `data_reference/assets_verified.csv` | The sidecar: 100 rows keyed by asset_id with verified coordinates, roles, statuses, anchor truth values and provenance. |
| `data_reference/asset_verification_log.csv` | Per-asset audit trail of lookups, unverified markings and anchor cross-checks. |
| `data_output/verification_summary.json` | Counts by sector and status, unverified asset_ids with reasons, anchor errors, generation date. |
| `config/districts_up.yaml` | District normalisation map: the 73 canonical names derived from the rainfall file, every raw alias, and the cross-check against the official 75-district list of Uttar Pradesh. |
| `data_clean/weather_up_daily.csv` | Canonical daily rainfall panel: one row per (district_canonical, date), 2005-01-01 to 2025-11-29. |
| `data_clean/district_reference.csv` | Per-district audit table: raw variants, centroid, day counts, gaps. |
| `data_clean/district_climatology.csv` | Per-district per-calendar-month baseline stats, 2005-01-01 to 2020-12-31 only. |
| `data_features/rainfall_features_asset_day.csv` | Daily rainfall features per UP-covered asset, keyed (asset_id, date). |
| `data_features/rainfall_features_asset_month.csv` | Monthly rainfall features per UP-covered asset, keyed (asset_id, year_month). |
| `data_raw_cache/power_points/` | Raw NASA POWER point-API responses (one JSON per asset, gitignored) plus the committed `MANIFEST.json` provenance record. |
| `data_clean/weather_asset_daily.csv.gz` | Canonical asset daily panel: one row per (asset_id, date), 2005-01-01 to 2025-11-29, fetched at each asset's own coordinate. Gzipped because the 81 x 7638-day table exceeds the 20 MB committed-artifact limit. |
| `data_clean/weather_asset_daily_gaps.json` | Per-asset calendar gaps for the asset panel (empty: every asset covers the full window). |
| `data_clean/asset_climatology.csv` | Per-asset per-calendar-month baseline stats, 2005-01-01 to 2020-12-31 only. |
| `data_features/rainfall_features_panindia_asset_day.csv.gz` | Daily rainfall features for all 81 coordinate-bearing assets, keyed (asset_id, date). Gzipped (size rule). |
| `data_features/rainfall_features_panindia_asset_month.csv` | Monthly rainfall features for all 81 coordinate-bearing assets, keyed (asset_id, year_month). |
| `data_output/weather_coverage.json` | Unified machine-readable coverage report: one entry per asset with a `sources` list covering both sources, per-source totals and an overall total. |
| `config/corridors.yaml` | Named highway-corridor polylines (endpoints, waypoints, source URL, notes) for the corridor flag; straight-line approximations, not surveyed alignments. |
| `data_raw_cache/osm_highways/` | Raw Overpass highway tiles (one JSON per asset, gitignored) plus the committed `MANIFEST.json` provenance record (exact query, endpoint, bbox, dates, sha256, way/node counts). |
| `data_features/highway_features_asset.csv` | Highway exposure features for the 81 coordinate-bearing assets, one row per asset_id. |
| `data_raw_cache/ports/` | Raw maritime source files (PortWatch CSV, HOT ports zip, Natural Earth coastline zip - all gitignored) plus the committed `MANIFEST.json` provenance record (exact URL, retrieval date, sha256, byte size, record counts, licence as observed). |
| `data_features/maritime_features_asset.csv` | Maritime / port exposure features for the 81 coordinate-bearing assets, one row per asset_id. |
| `tests/fixtures/portwatch_india.csv` | The 37 verbatim PortWatch India rows (header + rows) so feature computation re-derives offline. |
| `tests/fixtures/hot_ports_reference.geojson` | The 65 verbatim HOT reference port features (points + polygons) so feature computation re-derives offline. |
| `tests/fixtures/ne_coast_india_segments.json` | Every Natural Earth coastline segment in the India window (lon/lat pairs) so coastal distances re-derive offline. |
| `tests/fixtures/osm_*.json` | Verbatim raw Overpass responses for a few small-footprint assets so feature computation re-derives offline. |
| `tests/fixtures/` | A small clearly labelled subset of the asset daily panel (5 assets, 2019-2021) so tests recompute features offline. |
| `tests/` | Pytest suite. Offline: tests only read committed files, never the network. |

All artifacts are CSV/JSON so diffs are reviewable. There is deliberately no
parquet dependency; do not introduce one.

## How to run the tests

From the repository root (use `python3`; there is no `python` binary on
this machine):

```sh
python3 -m pytest integration/tests -q
```

The suite needs no network. It reads committed files only, except four
power-ingest tests (`test_power_ingest.py`) that read the gitignored raw
cache (`data_raw_cache/power_points/*.json`); those four skip with a clear
reason when the cache is absent (fresh clone: 104 passed, 4 skipped) and
run once the cache is fetched (108 passed).

To regenerate the derived files (byte-identical on every run), in order:

```sh
python3 integration/src/verify_assets.py           # phase 2: asset coordinates
python3 integration/src/clean_weather.py           # phase 3: cleaned UP rainfall panel
python3 integration/src/ingest_power_points.py     # phase 4: fetch + cache NASA POWER point data
python3 integration/src/clean_power_points.py      # phase 4: canonical asset daily panel
python3 integration/src/build_rainfall_features.py # phase 3: UP climatology + features
python3 integration/src/build_panindia_features.py # phase 4: asset climatology + pan-India features + unified coverage
python3 integration/src/ingest_osm_highways.py     # phase 5: fetch + cache Overpass highway tiles
python3 integration/src/build_highway_features.py  # phase 5: highway features + highway coverage
python3 integration/src/ingest_ports.py            # phase 6: fetch + cache maritime sources
python3 integration/src/build_maritime_features.py # phase 6: maritime features + maritime coverage
```

`ingest_power_points.py` is idempotent: assets already cached with a
matching manifest URL and sha256 are skipped, so re-running issues zero
network requests. `--offline` makes it fail clearly (exit 2) instead of
fetching when any cache entry is missing or stale.

## Coordinate policy

Never invent a coordinate. Accuracy over coverage: an honest `unverified`
row with a reason is a correct result; a guessed coordinate is a defect.
`verification_status` is one of `verified` (authoritative source found),
`approximate_midpoint` (linear asset, midpoint derived from known endpoints)
or `unverified` (could not be determined; coordinates null, reason in notes).

## Rainfall

The only legitimate weather data on disk is `UP_rainfall_dataset.csv`,
real NASA POWER daily district weather for Uttar Pradesh (565,210 rows,
2005-01-01 to 2025-11-29), plus - from phase 4 - NASA POWER point-API
responses fetched at each asset's own coordinate. The Master file's own
climate columns remain quarantined; nothing here reads them.

### Two sources, one feature implementation

**Source 1 - UP district panel** (`up_district_panel`). The cleaned panel
`data_clean/weather_up_daily.csv` is at the canonical grain of one row per
(district_canonical, date): 73 districts, 557,572 rows after dropping 7,638
exact-duplicate `shamli` rows. District labels are normalised only through
`config/districts_up.yaml` (73 raw spellings mapped to 73 canonical names).
Two known blemishes remain visible in `data_clean/district_reference.csv`:
`Lakhimpur` and `Mau` are each missing the final day 2025-11-29.

**Join rule (fixed).** Each asset with non-null coordinates whose
`verification_status` is not `unverified` is joined to the nearest UP
district centroid by haversine distance (mean Earth radius 6371.0088 km)
ONLY IF the asset lies inside the Uttar Pradesh bounding box (latitude
23.5-31.0, longitude 77.0-85.0) AND the distance is <= 75 km. Every other
asset receives no weather from this source. UP rainfall is never joined to
a non-UP asset.

**Source 2 - NASA POWER point API** (`nasa_power_point`). Daily weather is
fetched directly at each asset's own verified coordinate from
`https://power.larc.nasa.gov/api/temporal/daily/point` (community AG,
parameters PRECTOTCORR,T2M_MAX,T2M_MIN,RH2M,WS50M,PS,T2M, window
2005-01-01..2025-11-29 - the same window as the UP panel, one request per
asset). This covers **every** asset with non-null coordinates (81 of 100),
whereas the district join covers only the 11 assets near UP district
centroids. The 19 assets with unverified coordinates can never be fetched
and are reported with reason `asset_coordinate_unverified`.

**Why two sources.** The district panel is the audited, already-cleaned
UP dataset; the point API extends the same physical variables to every
asset in the country. Both feed ONE feature implementation
(`src/rainfall_features_core.py`), so the two paths cannot diverge: the
district path and the asset path are the same functions over a canonical
daily panel, and a test asserts both entry points are literally the same
code object and return identical numbers on a synthetic panel.

### Ingest, cache and manifest

`src/ingest_power_points.py` downloads one full-window response per asset
into `integration/data_raw_cache/power_points/<asset_id>.json` (gitignored,
~0.9 MB per asset, ~72 MB total). Requests are strictly sequential with a
0.3 s delay, retry with exponential backoff (2,4,8,16,30,30 s) on transient
failures, and every raw response is validated (exact 7638-day window,
all seven parameters) before caching. The committed
`integration/data_raw_cache/power_points/MANIFEST.json` records per
asset_id: the exact request URL, latitude, longitude, retrieval date,
response sha256 and number of days returned, plus cumulative totals
(81 requests, 71,999,880 bytes). Re-running with a complete cache issues
zero requests and rewrites the manifest byte-identically; `--offline`
fails clearly instead of fetching.

### Cleaning and fill values

`src/clean_power_points.py` turns the cache into
`data_clean/weather_asset_daily.csv.gz` (618,678 rows = 81 assets x
7638 days, sorted by asset_id, date). NASA POWER's fill value (-999, or
any value <= -999) never becomes a measurement: a day whose PRECTOTCORR is
fill is omitted and recorded in `weather_asset_daily_gaps.json` (empty -
every asset covers the full window contiguously); a day with valid
precip but a fill secondary parameter is emitted with that cell empty and
`is_observed=0`. `is_imputed` is always 0 - nothing is ever filled.

### Baseline and holdout

Climatology baselines are computed per series (district OR asset) per
calendar month over 2005-01-01 to 2020-12-31 only. The years 2021-2025 are
held out and never enter the baseline. Each baseline row carries the mean
and population standard deviation of daily precipitation and the 95th
percentile (linear interpolation) of wet-day (precip >= 1 mm)
precipitation, plus `n_baseline_days`. Baselines live in
`data_clean/district_climatology.csv` (73 districts) and
`data_clean/asset_climatology.csv` (81 assets).

### Features

All windows are trailing and backward-looking: a feature dated D uses
days up to and including D, never after. If ANY day required by a trailing
window is absent from the panel, the feature is null for that row -
never forward-filled, interpolated or imputed. There are no composite
indices and no invented weights; every feature is a count, sum, duration
or standardised anomaly in physical units. The definitions are identical
for both sources:

Daily (`rainfall_features_asset_day.csv` for the 11 UP-covered assets,
`rainfall_features_panindia_asset_day.csv.gz` for all 81 coordinate-bearing
assets; keyed asset_id + date):
`wx_precip_7d_mm` / `wx_precip_30d_mm` (trailing sums),
`wx_heavy_days_7d` (days >= 50 mm in 7 d), `wx_very_heavy_days_30d`
(days >= 100 mm in 30 d), `wx_rain_days_30d` (days >= 1 mm in 30 d),
`wx_cwd_30d` (longest wet run within 30 d), `wx_dry_spell_90d` (longest
dry run within 90 d), `wx_p95_exceed_days_30d` (days exceeding the
series's baseline wet-day p95 for that calendar month), and
`wx_monsoon_to_date_mm` (sum since 1 June of the same year; 0 for
January-May, where the window is empty).

Monthly (`rainfall_features_asset_month.csv` /
`rainfall_features_panindia_asset_month.csv`; keyed asset_id + year_month):
`wx_precip_month_mm`, `wx_rain_days_month`, `wx_monsoon_JJAS_mm` (the
full June-September total of that year, reported only on October-December
rows - the first rows at which the whole season is observable without
lookahead), and `wx_anomaly_z` ((month total - baseline mean) / baseline
population sd for that calendar month).

### Coverage

`data_output/weather_coverage.json` is a unified per-asset report. Each
asset entry carries a `sources` list with one entry per source
(`up_district_panel`, `nasa_power_point`), each with `status`
(`covered`/`no_coverage`), `reason_code`, `nearest_district` or
`nearest_asset_coordinate` as applicable, `distance_km`, `source_name`
and `retrieval_date`. Totals are per source and overall.

**Before phase 4:** 11 of 100 assets covered (UP district join only); 68
`non_UP_asset`, 19 `asset_coordinate_unverified`, 2 `no_district_within_75km`.

**After phase 4:** 81 of 100 assets covered from at least one source - 11
from the UP district panel and 81 from the NASA POWER point API (the 11
UP-covered assets are covered by both). The remaining 19 assets have
unverified coordinates and no coverage from either source, with reason
`asset_coordinate_unverified`. This is reported as-is; coverage is not a
success metric.

### Size rule

No artifact introduced by phase 4 exceeds 20 MB. The two large tables are
committed gzipped (`weather_asset_daily.csv.gz`, 10.9 MB;
`rainfall_features_panindia_asset_day.csv.gz`, 8.0 MB) with gzip headers
pinned to mtime=0 for byte-identical reruns. Read path:

```python
import csv, gzip
with gzip.open("integration/data_clean/weather_asset_daily.csv.gz", "rt",
               encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f):
        ...
```

(The pre-existing `data_clean/weather_up_daily.csv` from phase 3 is 34.6 MB
and remains as committed; it is frozen by the phase-3 byte-identical
requirement.)

### Offline test fixture

`tests/fixtures/asset_daily_panel_subset_2019-2021.csv` is a labelled subset
of the committed asset panel (5 assets - GP_081, GP_084, GP_086, HW_002,
RE_061 - spanning 2019-01-01..2021-12-31). Tests recompute features
independently from it and compare against the committed feature files, so
the no-lookahead and null-on-gap guarantees are verified against real
fetched data without network access. See `tests/fixtures/README.md`.

## Highways

Highway-network exposure features for the 81 assets with non-null
coordinates, from OpenStreetMap road data fetched per asset through the
Overpass API.

### Source, licence and attribution

- **Data:** OpenStreetMap ways tagged `highway` in {motorway, trunk,
  primary, secondary}, fetched through the Overpass API endpoint
  `https://overpass-api.de/api/interpreter` (no key required). The
  feature table's `licence` column carries
  `ODbL 1.0, (c) OpenStreetMap contributors` on every row, and any reuse
  of the data must preserve that attribution (ODbL requirement).
- **Dependency:** `pyproj>=3.6` (WGS84 geodesy). Install with
  `pip install --user --break-system-packages pyproj` (verified 3.8.0).

### Query strategy (small bounding boxes, and why)

One query per asset (81 requests) over that asset's own bounding box -
the asset coordinate plus a 25 km buffer - instead of whole-state
queries, because the public instance kills heavy queries:

- a whole-state `way["highway"=...]` query times out;
- even a 0.45-degree box with a tag-**regex** filter returns HTTP 504
  (`Dispatcher_Client ... timeout. The server is probably too busy`);
- a union of four **exact** `highway=` matches over the same box succeeds
  (3 s rural, ~10 s densest Delhi tile; verified 2026-10-15).

The exact query per tile is recorded in the manifest. Requests are
strictly sequential with a 1.5 s polite delay, a descriptive `User-Agent`
(required by the Overpass usage policy), a 175 s per-request timeout, and
bounded retries with
exponential backoff (5..60 s, 5 attempts) on transient 429/504. Do NOT
use the overpass.kumi.systems mirror (timed out in reconnaissance). A
1.71 GB Geofabrik India extract exists but is deliberately NOT used -
per-asset boxes are the auditable approach.

### Caching and MANIFEST layout

`src/ingest_osm_highways.py` caches each raw response body as
`integration/data_raw_cache/osm_highways/<asset_id>.json` (gitignored;
KBs rural to ~6 MB densest urban) plus the committed `MANIFEST.json`
with per tile: exact Overpass query string, endpoint, bounding box
`[south, west, north, east]`, retrieval_date, response sha256, `n_ways`,
`n_nodes` (distinct node ids), and the OSM extract timestamp
(`osm_extract_date`, from the response's `osm3s.timestamp_osm_base`).
Tiles that still fail after all attempts are recorded under the
manifest's `failures` map (exit 1, manifest still written) so the builder
emits honest nulls instead of guessing. Idempotent: cached tiles whose
query and sha256 match are skipped (zero requests on re-run); `--offline`
fails clearly (exit 2) when any tile is missing or stale. A 200 body
carrying a server-side timeout/truncation remark is rejected and never
cached as valid.

### Feature definitions

Per asset with non-null coordinates (81 rows, sorted by asset_id, in
`data_features/highway_features_asset.csv`):

- `rd_dist_nh_m`: minimum geodesic distance (m) from the asset point to
  the nearest way tagged as a national highway.
- `rd_hwy_len_10km_m`: total geodesic length (m) of all fetched ways
  within 10 km, each way's geometry clipped to that radius (dual
  carriageways mapped as two ways count twice - mapped-way metres, not
  centreline km).
- `rd_nh_len_10km_m`: the same, restricted to national-highway ways
  (always <= `rd_hwy_len_10km_m`).
- `rd_node_degree_mean_10km` / `rd_node_count_10km`: mean node degree over
  graph nodes within 10 km, with the node count it was computed over.
- `rd_on_corridor_flag` (true/false) / `rd_corridor_name`: true when the
  asset is within 5 km of a `config/corridors.yaml` polyline.

**NH `ref` regex** (exact, pinned by test): each `;`-separated token of
the way's `ref` tag, stripped and upper-cased, must fully match
`^NH[\s\-]*\d{1,3}[A-Z]?$`. Matches `NH 44`, `NH44`, `NH-44B`; rejects
`SH 44`, `NHDP`, `NE 4`, `NH 44 bypass`. Expressway refs (NE/NS/E series)
are deliberately excluded. Matched NH ways across all tiles: 13,548
(reported by the builder on every run and stored in the test output).

**Node-degree definition** (pinned by test): the graph is built from the
fetched ways' `nodes` arrays; each consecutive node pair is one undirected
segment and a node's degree is its incident-segment count (shared nodes
accrue segments from every way). Only nodes within 10 km of the asset
enter the mean. Small samples are reported WITH their node count, not
silently: a mean over a handful of nodes is unstable. Zero nodes gives
null with reason `no_nodes_in_radius`.

**Geodesy:** everything via `pyproj.Geod` (WGS84) on EPSG:4326
coordinates. Point-to-segment distances and circle clipping run in a local
azimuthal-equidistant plane centred on the asset (true geodesic azimuth +
distance per endpoint; centimetre-accurate at these radii); sub-segment
lengths are measured with Geod. Corridor polylines are densified to <= 5 km
legs before matching. Never planar arithmetic on degrees (validated by
`tests/test_geodesy.py` against WGS84 degree-arc constants and a
haversine cross-check within 1 %).

**Corridors** (`config/corridors.yaml`, 17 entries with endpoints, short
name and source URL each): straight-line approximations between documented
places, NOT surveyed alignments - flags near the 5 km boundary can be
wrong either way (documented limitation). Corridor hits: 15 of the 81
coordinate-bearing assets (all other assets keep `false`/empty).

**Missing data:** `is_imputed` is 0 everywhere. Nulls (empty cells) appear
only with a matching `no_coverage` entry: `asset_coordinate_unverified`
(19 assets, no row), `osm_tile_not_cached` / `osm_fetch_failed`,
`no_nh_in_25km`, `no_nodes_in_radius`. Lengths are 0.0 - never null - on a
successfully fetched empty neighbourhood. Highway coverage: 78
of 81 coordinate-bearing assets covered (19 `asset_coordinate_unverified`
with no row emitted, 3 `no_nh_in_25km`: GP_094, GP_097, RE_062).

### Coverage

`data_output/weather_coverage.json` keeps its `wx` content byte-identical
in meaning and gains the highway family **alongside** (same file, so no
consumer breaks): each asset's `sources` list gains an `osm_highways`
entry (`status`, `reason_code`, `source_name`, `retrieval_date`,
`distance_km` = `rd_dist_nh_m`/1000 where applicable), plus a top-level
`rd` block with the source description, join rule and totals.

### Size rule

No committed artifact exceeds 20 MB. Committed: `highway_features_asset.csv`
(18,316 bytes), `MANIFEST.json` (70,834 bytes), three verbatim
fixture tiles under `tests/fixtures/` (48,614 + 96,818 + 147,569 bytes).
Raw tiles stay
gitignored. Re-fetch with the two phase-5 commands above; re-derive
offline from the fixtures via `python3 -m pytest integration/tests -q`.

## Maritime

Maritime / port exposure features for the 81 assets with non-null
coordinates, from three freely available sources: IMF PortWatch global
ports (via HDX), HOT Sea Ports of India (via HDX) and the Natural Earth
1:10m coastline. No composite indices, no invented weights, no
modelling; throughput is gated, never estimated.

### Sources, licences and attribution

- **PortWatch Ports via HDX** (`portwatch_ports.csv`, 2,065 rows, 37 for
  India; columns portid/portname/country/ISO3/lat/lon/vessel_count_* /
  industry_top* / share_country_maritime_* / LOCODE/pageid). Dataset
  source: UN Global Platform satellite AIS data; IMF PortWatch.
  Vessel-call counts and country-maritime-trade shares are carried as
  published (the file states no reference year; the snapshot was
  retrieved 2026-10-15). They are activity figures, explicitly NOT
  throughput. Licence (recorded exactly as HDX states it, verified via
  the HDX package API - it is NOT a PortWatch/IMF open licence):
  `Other (HDX); see IMF copyright and terms
  https://www.imf.org/en/about/copyright-and-terms`. There is NO
  capacity/class field and NO throughput field in this file.
- **HOT Sea Ports of India** (GeoJSON inside a zip, 772 features: 630
  points, 140 polygons, 1 line, 1 multipolygon; properties include
  name/amenity/industrial/harbour). Licence **ODbL** (HDX licence id
  `hdx-odc-odbl`, verified via the HDX package API; the file's own
  metadata records `ODbL-1.0` over a 2026-10-09 Geofabrik extract).
  Attribution `(c) OpenStreetMap contributors` is carried in the feature
  table. Caveat recorded by HOT: crowd-sourced OSM data, coverage
  varies; 625 of the 630 points are `amenity=ferry_terminal` jetties and
  are excluded from the reference set (see below).
- **Natural Earth 1:10m coastline** (SHP bundle, 4,133 PolyLine records,
  410,957 vertices; fetched from the official S3 distribution). Licence:
  **public domain** (Natural Earth data policy). Only the 13,232
  segments in the India window (lon 60-100, lat 0-40) enter the build.
- The feature table's `licence` column carries all three on every row;
  per-source `portwatch_licence` / `poly_licence` / `coast_licence`
  columns (with matching `_source`, `_url`, `_retrieval_date` columns)
  keep each value traceable to exactly one source. A test asserts the
  per-source licence and URL strings equal the committed MANIFEST
  record exactly.

### Caching and MANIFEST layout

`src/ingest_ports.py` caches the three source files byte-for-byte as
`integration/data_raw_cache/ports/` (gitignored; 451,506 + 87,467 +
3,069,451 bytes) plus the committed `MANIFEST.json` with per file: exact
download URL, endpoint host, retrieval date, response sha256, byte size,
the licence string as verified, and record counts (2,065 CSV rows / 37
India rows; 772 GeoJSON features; 4,133 SHP records). Sequential requests
with a 1.5 s polite delay, a descriptive `User-Agent`, a 120 s
per-request timeout and bounded retries with exponential backoff
(5..60 s, 5 attempts) on transient 429/5xx; shape validation per file
(CSV columns + India-row floor, zip magic + member list, SHP magic +
record walk) so a truncated body is never cached as valid. Idempotent
(skips files whose sha256 matches) with incremental manifest writes;
`--offline` fails clearly (exit 2) when any file is missing or stale.

### Feature definitions

Per asset with non-null coordinates (81 rows, sorted by asset_id, in
`data_features/maritime_features_asset.csv`):

- `mt_dist_major_port_m`: minimum geodesic distance (m) from the asset
  point to the nearest feature of the reference port set: 42 reference
  points (all 37 PortWatch India rows - every PortWatch-tracked
  commercial port in India, with no activity cutoff - plus 5 HOT points
  tagged `harbour=yes`) and 60 HOT Polygon/MultiPolygon features with a
  port-identifying tag (`industrial=port`, `harbour=yes`, or a non-null
  `port` tag). Zero when the asset lies inside a reference polygon.
  Deliberately excluded: 625 ferry-terminal points, 80 non-port polygons
  (56 ferry terminals, 24 untagged), and 1 `harbour=yes` LineString.
  Membership is by source inclusion and OSM tags only - there are no
  numeric thresholds and no activity cutoffs anywhere in the definition.
  Known consequence: one predicate polygon is the inland container depot
  `ICD CONCOR Kanpur` (tagged `industrial=port`); it enters the set as
  tagged.
- `mt_nearest_major_port_id` / `mt_nearest_major_port_name`: the nearest
  reference feature (PortWatch `portid`/`portname`, OSM `id` for HOT
  features; the name falls back to the id where OSM carries no name).
- `mt_ports_within_100km` (count): reference points within 100 km plus
  reference polygons whose boundary (or interior) lies within 100 km.
- `mt_in_port_polygon` (true/false): whether the asset coordinate falls
  inside a reference port Polygon/MultiPolygon (ray casting with holes;
  topological on degrees, no metric involved). Only assets physically
  inside a mapped port estate are true (2 of 81: PT_032 Mormugao and
  PT_039 Gangavaram); every other asset, port-sector or not, is false.
- `mt_coast_dist_m` / `mt_coastal_flag`: geodesic distance (m) to the
  nearest Natural Earth coastline segment in the India window;
  coastal iff within 50 km. At 1:10m resolution the line is adequate for
  a 50 km flag and kilometre-scale distances, not for metre-scale
  setback claims.
- `mt_pw_nearest_port_id` / `mt_pw_nearest_port_name` / `mt_pw_dist_m`:
  the nearest PortWatch India row specifically (identified separately
  because only PortWatch rows carry activity figures), with
  `mt_pw_vessel_count_total_yr` / `mt_pw_vessel_count_container_yr`
  (vessel-call counts as published) and `mt_pw_share_import_pct` /
  `mt_pw_share_export_pct` (country-maritime-trade shares, percent
  scale). Activity, never throughput.
- `mt_throughput_teu_yr/source/year`: ALWAYS NULL with reason
  `throughput_gated` (recorded on every portwatch coverage entry and in
  the `mt` coverage block). Throughput is never modelled or imputed. No
  licenced programmatic throughput source exists: official IPA/Sagarmala
  tables are HTML/PDF-only and both hosts time out from this network
  (verified 2026-10-15).
- Geodesy: `pyproj.Geod` (WGS84) throughout; point-to-segment distances
  in a local azimuthal-equidistant plane (same construction as phase 5);
  polygon boundaries measured with the same machinery. Never planar
  arithmetic on degrees.
- Geometric (observed) versus gated: every column except the three
  `mt_throughput_*` columns is geometrically observed from the sources
  above (`is_observed=1`, `is_imputed=0` on all 81 rows); the throughput
  triple is gated NULL on all 81 rows.

**Linear assets** (pipelines, transmission corridors) have no line
geometry in the sidecar - only an `approximate_midpoint` coordinate. All
maritime distances are computed from that representative midpoint; a
coastal midpoint does not imply the whole line is coastal and vice
versa. This is a documented limitation, not a line-based analysis.

**Coverage:** 81 of 81 coordinate-bearing assets covered from all three
maritime sources; the 19 unverified-coordinate assets receive no
maritime features (`asset_coordinate_unverified`, no row emitted).
Throughput gated on all 81 rows (`throughput_gated`).

### Size rule

No committed artifact exceeds 20 MB. Committed:
`tests/fixtures/portwatch_india.csv` (8,156 bytes: the 37 verbatim
India rows), `tests/fixtures/hot_ports_reference.geojson` (222,703
bytes: the 65 verbatim HOT reference features),
`tests/fixtures/ne_coast_india_segments.json` (1,044,117 bytes: the
13,232 India-window coastline segments),
`data_features/maritime_features_asset.csv` (81 rows), ports
`MANIFEST.json` (2,715 bytes). Raw sources stay gitignored.
Re-fetch with `python3 integration/src/ingest_ports.py`; re-derive
offline from the fixtures via `python3 -m pytest integration/tests -q`.
The exact test command (from the repository root; `python3`, not
`python`):

```sh
python3 -m pytest integration/tests -q
```

## Release

The release join (`src/join_feature_matrix.py`) produces
`data_output/feature_matrix_asset.csv` (100 rows: identity +
`coord_role`/`verification_status`/coordinates, every `rd_`/`mt_`
feature verbatim, wx join metadata, per-family provenance and
per-source coverage status/reason) and
`data_output/feature_matrix_asset_month.csv` (20,331 rows = 81 assets x
251 months: pan-India monthly columns everywhere, UP-district columns
where the district join covers, else NULL with empty
`district_canonical`). Rainfall measurements live in the month matrix;
the asset matrix carries wx join metadata only. `null_reason` explains
every row's NULLs. `coverage_report.json` consolidates
`weather_coverage.json` per asset per family (the latter is untouched).

`src/validate_layer.py` runs nine gates and writes
`data_output/validation_report.json` (exit non-zero on any failure):
schema_dtypes (exact columns), ranges (rainfall 0-500 mm/day, temp
-10..55 C, non-negative distances/counts, real booleans),
anchor_tolerance (2.0 km point / 10.0 km midpoint),
fabricated_tripwire (>50 km from Master lat/lon; PT_040's 26 km
coincidence is documented in the sidecar notes),
reconcile_up_point (r >= 0.95, MAD <= 15 mm; measured r = 0.9818,
MAD = 8.147 mm over 2,750 shared asset-months), determinism
(byte-identical rebuilds with sha256), no_imputation (all is_imputed 0;
684 NULLs all explained), licence_attribution, coverage_accounting
(100 x 6). See `integration/RELEASE.md` for the handover note.
Reproduce with the command list in RELEASE.md, then
`python3 -m pytest integration/tests -q`.
