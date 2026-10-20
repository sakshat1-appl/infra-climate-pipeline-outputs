# Test fixture: pan-India asset daily panel subset

`asset_daily_panel_subset_2019-2021.csv` is a **read-only subset of the
committed canonical panel** `integration/data_clean/weather_asset_daily.csv.gz`
(written by `integration/src/clean_power_points.py` from the cached NASA
POWER point-API responses). It exists so the test suite can independently
recompute pan-India rainfall features **offline**, directly from daily
weather, without parsing the 11 MB gzipped panel or touching the network.

## Contents

- **Assets (5):** GP_081 (Gujarat, gas pipeline), GP_084 (Kerala, gas
  pipeline), GP_086 (UP, gas pipeline), HW_002 (UP, highway), RE_061
  (Rajasthan, solar). Two are also covered by the UP district panel
  (GP_086, HW_002); three are outside Uttar Pradesh and are covered ONLY
  by the NASA POWER point source.
- **Window:** 2019-01-01 to 2021-12-31 (1096 days per asset, 5480 rows).
- **Columns:** identical to `weather_asset_daily.csv.gz` (asset_id, date,
  precip_mm_day, tmax_c, tmin_c, rh_pct, wind_ms, ps_kpa, source_name,
  source_url, retrieval_date, is_observed, is_imputed).

## Why this window

Every feature window used by the test suite (7/30/90-day trailing windows,
monsoon-to-date, and the June-September JJAS total reported on October-
December rows) lies fully inside 2019-01-01..2021-12-31 for the dates the
tests recompute (all recomputed dates are >= 2019-03-31, so even the 90-day
window never reaches before the fixture start). Values recomputed from this
fixture therefore equal the values in the committed feature files, which
were computed from the full 2005-01-01..2025-11-29 panel.

## Regenerating

Do not edit by hand. To regenerate from the committed panel:

```sh
python3 - <<'EOF'
import csv, gzip, datetime
FIXTURE_ASSETS = ["GP_084", "GP_086", "GP_081", "HW_002", "RE_061"]
START, END = datetime.date(2019, 1, 1), datetime.date(2021, 12, 31)
with gzip.open("integration/data_clean/weather_asset_daily.csv.gz", "rt",
               encoding="utf-8", newline="") as fin, \
     open("integration/tests/fixtures/asset_daily_panel_subset_2019-2021.csv",
          "w", encoding="utf-8", newline="") as fout:
    reader = csv.DictReader(fin)
    writer = csv.DictWriter(fout, fieldnames=reader.fieldnames,
                            lineterminator="\n")
    writer.writeheader()
    for row in reader:
        if row["asset_id"] in FIXTURE_ASSETS and \
                START <= datetime.date.fromisoformat(row["date"]) <= END:
            writer.writerow(row)
EOF
```

---

## OSM highway tiles (`osm_<asset_id>.json`)

Each file is a **complete, unmodified** raw Overpass API response for one
asset's 25 km-buffer tile, copied verbatim from the gitignored
`integration/data_raw_cache/osm_highways/<asset_id>.json`. They are
committed here so that
`tests/test_highway_features.py::test_fixture_recomputation_matches_committed_csv`
can re-derive `rd_dist_nh_m` and `rd_hwy_len_10km_m` from real fetched data
with test-local code and compare against
`integration/data_features/highway_features_asset.csv` - fully offline.

Do not hand-edit these files: they must stay byte-identical to the cached
tile whose sha256 is recorded in
`integration/data_raw_cache/osm_highways/MANIFEST.json`, otherwise the
recomputation test compares against data the pipeline never saw.

---

## Maritime fixtures (`portwatch_india.csv`, `hot_ports_reference.geojson`, `ne_coast_india_segments.json`)

Offline recomputation fixtures for `tests/test_maritime_features.py`.
They carry the **full reference port set** - not an asset-centric cut -
so every feature for every one of the 81 coordinate-bearing assets
re-derives exactly, fully offline:

- `portwatch_india.csv`: the 37 PortWatch India rows, verbatim
  (same header as the cached `portwatch_ports.csv`).
- `hot_ports_reference.geojson`: the 65 HOT reference port features
  (5 `harbour=yes` points + 60 port-tagged polygons), verbatim, as a
  sorted FeatureCollection.
- `ne_coast_india_segments.json`: every Natural Earth coastline
  segment in the India window (13,232 `[lon, lat]` pairs) plus a
  `provenance` header recording the source file and window constants.

Do not hand-edit. To regenerate from the gitignored raw cache:

```sh
python3 - <<'EOF'
import csv, json, zipfile, sys
sys.path.insert(0, 'integration/src')
import build_maritime_features as B
rows = list(csv.DictReader(open(
    'integration/data_raw_cache/ports/portwatch_ports.csv',
    encoding='utf-8')))
ind = [r for r in rows if (r.get('ISO3') or '').strip() == 'IND']
with open('integration/tests/fixtures/portwatch_india.csv', 'w',
          newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys(), lineterminator='\n')
    w.writeheader(); w.writerows(ind)
z = zipfile.ZipFile(
    'integration/data_raw_cache/ports/hotosm_ind_sea_ports_geojson.zip')
d = json.loads(z.read('sea_ports.geojson').decode('utf-8'))
pts, polys = B.load_hotosm_reference()
ref_ids = set(p['ref_id'] for p in pts) | set(p['ref_id'] for p in polys)
keep = [f for f in d['features']
        if (f.get('properties') or {}).get('id') in ref_ids]
fc = {'type': 'FeatureCollection', 'name': 'hot_ports_reference',
      'provenance': 'verbatim features of '
      'integration/data_raw_cache/ports/hotosm_ind_sea_ports_geojson.zip '
      'admitted by the reference-port predicate (see '
      'integration/src/build_maritime_features.py)',
      'features': keep}
with open('integration/tests/fixtures/hot_ports_reference.geojson', 'w',
          encoding='utf-8', newline='\n') as f:
    json.dump(fc, f, indent=1, sort_keys=True); f.write('\n')
segs = B.load_coastline_segments()
out = {'provenance': {
    'source': 'integration/data_raw_cache/ports/ne_10m_coastline.zip',
    'window': {'lon_min': B.COAST_LON_MIN, 'lon_max': B.COAST_LON_MAX,
               'lat_min': B.COAST_LAT_MIN, 'lat_max': B.COAST_LAT_MAX}},
    'segments': [[[s['a'][1], s['a'][0]], [s['b'][1], s['b'][0]]]
                 for s in segs]}
with open('integration/tests/fixtures/ne_coast_india_segments.json', 'w',
          encoding='utf-8', newline='\n') as f:
    json.dump(out, f, separators=(',', ':'), sort_keys=True); f.write('\n')
EOF
```
