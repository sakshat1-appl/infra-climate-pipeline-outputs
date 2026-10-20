"""Download and cache raw maritime source files, one file per source.

Sources (all verified reachable and licensed during reconnaissance):
  * PortWatch global ports CSV via HDX (2,065 rows, 37 for India; columns
    portid/portname/country/ISO3/lat/lon/vessel_count_*/industry_top*/
    share_country_maritime_*/LOCODE/pageid). Licence: HDX "Other" with
    licence_other "https://www.imf.org/en/about/copyright-and-terms"
    (NOT a PortWatch/IMF open licence - recorded exactly, never assumed).
    Vessel counts are vessel-call counts as published (the file states no
    reference year); they are activity figures, never throughput.
  * HOT (Humanitarian OpenStreetMap Team) Sea Ports of India, GeoJSON
    inside a zip (772 features: points, polygons, lines; properties
    include name/amenity/industrial/harbour). Licence: ODbL
    (HDX licence_id "hdx-odc-odbl", verified via the HDX package API).
    Caveat recorded by HOT: crowd-sourced OSM data, coverage varies.
  * Natural Earth 1:10m coastline zip (SHP bundle, ~6.8 MB .shp).
    Licence: public domain (verified on the Natural Earth terms page;
    AWS-hosted since 2021, which is the official distribution).

Files are cached as ``integration/data_raw_cache/ports/<name>`` byte for
byte (zips stay zipped; the builder reads them with stdlib ``zipfile``).
A committed ``MANIFEST.json`` in the same directory records, per file:
exact download URL, endpoint host, retrieval date, response sha256,
byte size, licence string as verified, and record counts.

Politeness and reliability:
  * Strictly sequential requests with a polite delay between them.
  * A descriptive User-Agent header on every request.
  * Transient failures (connection error, timeout, HTTP 429/5xx, a body
    that fails shape validation) retried with exponential backoff
    (5, 10, 20, 40, 60 s) up to 5 attempts; any other 4xx fails fast.
  * Shape validation per file (CSV columns + minimum India rows; zip
    magic + member list + GeoJSON feature count + SHP magic after
    unzip). A truncated/invalid body is NEVER cached as valid.

Idempotency: a file is skipped when it exists AND its sha256 matches the
manifest record AND the URL matches. Re-running with a complete cache
issues zero network requests and rewrites the manifest byte-identically.
``--offline`` never touches the network: exit 2 listing missing/stale
files.

Deterministic: fixed file order, fixed constants, LF line endings.
"""

import argparse
import csv
import datetime
import hashlib
import io
import json
import os
import struct
import sys
import time
import zipfile

import requests

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
CACHE_DIR = os.path.join(REPO_ROOT, "integration", "data_raw_cache", "ports")
MANIFEST_JSON = os.path.join(CACHE_DIR, "MANIFEST.json")

USER_AGENT = (
    "infra-climate-valuation-integration/1.0 "
    "(academic research data pipeline; https://github.com/VariableThe/infra-climate-valuation)"
)
DEFAULT_DELAY = 1.5
MAX_ATTEMPTS = 5
BACKOFF_S = (5.0, 10.0, 20.0, 40.0, 60.0)
HTTP_TIMEOUT_S = 120

# PortWatch Ports (HDX resource). Licence recorded exactly as HDX states
# it: "Other" -> IMF copyright and terms. Vessel counts are vessel-call
# counts as published (no reference year in the file).
PORTWATCH_URL = (
    "https://data.humdata.org/dataset/6cd84ca4-63e9-43b8-8dd9-32d4a0f3d128"
    "/resource/eddb3a76-29c1-45cb-bf7f-0fc62a375336/download/ports.csv"
)
PORTWATCH_FILE = "portwatch_ports.csv"
PORTWATCH_LICENCE = (
    "Other (HDX); see IMF copyright and terms "
    "https://www.imf.org/en/about/copyright-and-terms; "
    "source: UN Global Platform; IMF PortWatch"
)
PORTWATCH_REQUIRED_COLUMNS = {
    "portid",
    "portname",
    "country",
    "ISO3",
    "lat",
    "lon",
    "vessel_count_total",
    "LOCODE",
}
PORTWATCH_MIN_INDIA_ROWS = 30

# HOT Sea Ports of India (HDX). Licence ODbL (HDX "hdx-odc-odbl").
HOT_PORTS_URL = (
    "https://production-raw-data-api.s3.amazonaws.com/ISO3/IND/sea_ports/"
    "hotosm_ind_sea_ports_osm_geojson.zip"
)
HOT_PORTS_FILE = "hotosm_ind_sea_ports_geojson.zip"
HOT_PORTS_GEOJSON_MEMBER = "sea_ports.geojson"
HOT_PORTS_LICENCE = "ODbL 1.0 (HDX licence hdx-odc-odbl); (c) OpenStreetMap contributors"
HOT_PORTS_MIN_FEATURES = 100

# Natural Earth 1:10m coastline (official S3 distribution). Public domain.
NE_COAST_URL = (
    "https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_coastline.zip"
)
NE_COAST_FILE = "ne_10m_coastline.zip"
NE_COAST_SHP_MEMBER = "ne_10m_coastline.shp"
NE_COAST_LICENCE = "Public domain (Natural Earth)"
NE_COAST_MIN_SEGMENTS = 100

SOURCES = (
    {
        "key": "portwatch",
        "url": PORTWATCH_URL,
        "filename": PORTWATCH_FILE,
        "licence": PORTWATCH_LICENCE,
    },
    {
        "key": "hotosm_ports",
        "url": HOT_PORTS_URL,
        "filename": HOT_PORTS_FILE,
        "licence": HOT_PORTS_LICENCE,
    },
    {
        "key": "ne_coastline",
        "url": NE_COAST_URL,
        "filename": NE_COAST_FILE,
        "licence": NE_COAST_LICENCE,
    },
)


class OfflineCacheError(RuntimeError):
    """Raised by --offline when the cache is incomplete or stale."""


class ResponseShapeError(ValueError):
    """Raised when a downloaded body fails shape validation."""


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def load_manifest(path=MANIFEST_JSON):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def cache_is_current(key, url, cache_dir, payload):
    """True iff the cached file exists and matches the manifest record."""
    if not payload:
        return False
    entry = (payload.get("files") or {}).get(key)
    if not entry or entry.get("url") != url:
        return False
    path = os.path.join(cache_dir, entry.get("filename", ""))
    if not os.path.exists(path):
        return False
    with open(path, "rb") as f:
        return sha256_bytes(f.read()) == entry.get("response_sha256")


def validate_portwatch(body):
    """CSV parses, required columns present, India rows present."""
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ResponseShapeError("portwatch CSV is not UTF-8: %s" % exc)
    try:
        rows = list(csv.DictReader(io.StringIO(text)))
    except csv.Error as exc:
        raise ResponseShapeError("portwatch CSV does not parse: %s" % exc)
    if not rows:
        raise ResponseShapeError("portwatch CSV has no data rows")
    missing = PORTWATCH_REQUIRED_COLUMNS - set(rows[0].keys())
    if missing:
        raise ResponseShapeError(
            "portwatch CSV missing columns: %s" % sorted(missing)
        )
    n_india = sum(1 for r in rows if (r.get("ISO3") or "").strip() == "IND")
    if n_india < PORTWATCH_MIN_INDIA_ROWS:
        raise ResponseShapeError(
            "portwatch CSV has only %d India rows" % n_india
        )
    return {"n_rows": len(rows), "n_india_rows": n_india}


def validate_hotosm_zip(body):
    """Real zip with the GeoJSON member; features list non-trivial."""
    if body[:2] != b"PK":
        raise ResponseShapeError("hotosm body is not a zip (no PK magic)")
    try:
        archive = zipfile.ZipFile(io.BytesIO(body))
    except zipfile.BadZipFile as exc:
        raise ResponseShapeError("hotosm zip does not open: %s" % exc)
    if HOT_PORTS_GEOJSON_MEMBER not in archive.namelist():
        raise ResponseShapeError(
            "hotosm zip lacks %s (has %s)"
            % (HOT_PORTS_GEOJSON_MEMBER, archive.namelist())
        )
    try:
        payload = json.loads(
            archive.read(HOT_PORTS_GEOJSON_MEMBER).decode("utf-8")
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise ResponseShapeError("hotosm GeoJSON is not valid JSON: %s" % exc)
    features = payload.get("features")
    if not isinstance(features, list) or len(features) < HOT_PORTS_MIN_FEATURES:
        raise ResponseShapeError(
            "hotosm GeoJSON has too few features: %s"
            % (len(features) if isinstance(features, list) else type(features))
        )
    return {"n_features": len(features)}


def validate_ne_zip(body):
    """Real zip with the .shp member; SHP magic + segment count sane."""
    if body[:2] != b"PK":
        raise ResponseShapeError("coastline body is not a zip (no PK magic)")
    try:
        archive = zipfile.ZipFile(io.BytesIO(body))
    except zipfile.BadZipFile as exc:
        raise ResponseShapeError("coastline zip does not open: %s" % exc)
    if NE_COAST_SHP_MEMBER not in archive.namelist():
        raise ResponseShapeError(
            "coastline zip lacks %s (has %s)"
            % (NE_COAST_SHP_MEMBER, archive.namelist())
        )
    shp = archive.read(NE_COAST_SHP_MEMBER)
    if len(shp) < 104 or struct.unpack(">i", shp[:4])[0] != 9994:
        raise ResponseShapeError("coastline .shp has a bad header")
    # Walk the record headers to count records (the header's length
    # field is a byte count, not a record count).
    n_records = 0
    offset = 100
    while offset + 8 <= len(shp):
        content_words = struct.unpack(">i", shp[offset + 4 : offset + 8])[0]
        if content_words < 2:
            raise ResponseShapeError("coastline .shp has a short record")
        offset += 8 + content_words * 2
        n_records += 1
    if offset != len(shp):
        raise ResponseShapeError("coastline .shp has trailing bytes")
    if n_records < NE_COAST_MIN_SEGMENTS:
        raise ResponseShapeError(
            "coastline .shp has only %d records" % n_records
        )
    return {"n_shp_records": n_records}


VALIDATORS = {
    "portwatch": validate_portwatch,
    "hotosm_ports": validate_hotosm_zip,
    "ne_coastline": validate_ne_zip,
}


def fetch_url(session, url, validator, max_attempts=MAX_ATTEMPTS):
    """GET one URL with backoff; returns (body, counts)."""
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = session.get(url, timeout=HTTP_TIMEOUT_S)
        except requests.RequestException as exc:
            last_error = "request failed: %s" % exc
        else:
            if resp.status_code == 200:
                try:
                    counts = validator(resp.content)
                except ResponseShapeError as exc:
                    last_error = "invalid body: %s" % exc
                else:
                    return resp.content, counts
            elif resp.status_code == 429 or resp.status_code >= 500:
                last_error = "HTTP %d" % resp.status_code
            else:
                raise RuntimeError(
                    "HTTP %d for %s (not retryable)" % (resp.status_code, url)
                )
        if attempt < max_attempts:
            time.sleep(BACKOFF_S[attempt - 1])
    raise RuntimeError(
        "giving up on %s after %d attempts: %s" % (url, max_attempts, last_error)
    )


def write_manifest(path, entries):
    """Deterministic manifest: fixed source order, LF endings."""
    payload = {
        "sources": [
            {
                "key": src["key"],
                "url": src["url"],
                "endpoint": src["url"].split("/")[2],
                "filename": src["filename"],
                "licence": src["licence"],
            }
            for src in SOURCES
        ],
        "files": {
            src["key"]: entries[src["key"]]
            for src in SOURCES
            if src["key"] in entries
        },
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def fetch_all(
    cache_dir=CACHE_DIR,
    manifest_path=MANIFEST_JSON,
    offline=False,
    delay=DEFAULT_DELAY,
    session=None,
    today=None,
):
    """Fetch (or reuse) every source file. Returns download stats."""
    if today is None:
        today = datetime.date.today
    if session is None:
        session = requests.Session()
        session.headers.update({"User-Agent": USER_AGENT})

    prior = load_manifest(manifest_path) or {}
    prior_files = prior.get("files") or {}
    entries = dict(prior_files)
    n_requests = 0
    n_bytes = 0
    n_from_cache = 0

    if offline:
        missing = [
            src["key"]
            for src in SOURCES
            if not cache_is_current(src["key"], src["url"], cache_dir, prior)
        ]
        if missing:
            raise OfflineCacheError(
                "--offline: %d of %d source files lack a current cached "
                "copy (missing or stale cache/manifest): %s"
                % (len(missing), len(SOURCES), ", ".join(missing))
            )

    for src in SOURCES:
        key, url = src["key"], src["url"]
        if cache_is_current(key, url, cache_dir, prior):
            n_from_cache += 1
            continue
        if offline:
            continue  # unreachable: offline validation above already failed
        body, counts = fetch_url(session, url, VALIDATORS[key])
        n_requests += 1
        n_bytes += len(body)
        path = os.path.join(cache_dir, src["filename"])
        os.makedirs(cache_dir, exist_ok=True)
        with open(path, "wb") as f:
            f.write(body)
        entry = {
            "url": url,
            "endpoint": url.split("/")[2],
            "filename": src["filename"],
            "licence": src["licence"],
            "retrieval_date": today().isoformat(),
            "response_sha256": sha256_bytes(body),
            "bytes_downloaded": len(body),
        }
        entry.update(counts)
        entries[key] = entry
        print("fetched %s (%s, %.1f KB)" % (key, counts, len(body) / 1024.0))
        # Incremental manifest write so a killed run keeps provenance.
        write_manifest(manifest_path, entries)
        if delay:
            time.sleep(delay)

    print("files: %d" % len(entries))
    print("requests issued this run: %d" % n_requests)
    print("bytes downloaded this run: %d" % n_bytes)
    print("files served from cache: %d" % n_from_cache)
    print("wrote %s" % manifest_path)
    return {
        "files": len(entries),
        "requests_issued": n_requests,
        "bytes_downloaded": n_bytes,
        "files_from_cache": n_from_cache,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Download and cache maritime source files "
        "(PortWatch ports, HOT sea ports, Natural Earth coastline)."
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="never touch the network; fail clearly if any source file "
        "is missing or stale",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        help="seconds to wait between requests (default %(default)s)",
    )
    args = parser.parse_args(argv)
    try:
        fetch_all(offline=args.offline, delay=args.delay)
    except OfflineCacheError as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
