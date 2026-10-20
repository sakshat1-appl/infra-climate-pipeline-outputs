"""Download and cache raw NASA POWER point-API responses, one JSON per asset.

For every asset in the verified sidecar with non-null coordinates (81 of
100), fetch daily weather directly at the asset's OWN coordinate from the
NASA POWER point API and cache the raw response body as
``integration/data_raw_cache/power_points/<asset_id>.json``. A committed
``MANIFEST.json`` in the same directory records, per asset_id, the exact
request URL, latitude, longitude, retrieval date, response sha256 and
number of days returned.

Request pattern (no API key required; verified accessible):

  https://power.larc.nasa.gov/api/temporal/daily/point
    ?community=AG
    &parameters=PRECTOTCORR,T2M_MAX,T2M_MIN,RH2M,WS50M,PS,T2M
    &start=20050101&end=20251129
    &latitude=<asset lat>&longitude=<asset lon>
    &format=JSON

The window 2005-01-01..2025-11-29 matches the existing UP district panel
exactly so the two sources align day for day. One request per asset covers
the full range.

Politeness and reliability:
  * Strictly sequential requests with a small delay between them.
  * Transient failures (connection error, timeout, HTTP 429/5xx, body that
    is not valid JSON) are retried with exponential backoff (2, 4, 8, 16,
    30, 30 s) up to 6 attempts; a 4xx other than 429 fails immediately.
  * Every raw response is cached before anything is written downstream.

Idempotency: an asset is skipped when its cache file exists AND the
manifest entry matches the recomputed request URL AND the cached body's
sha256 matches the manifest record. Re-running with a complete cache
issues zero network requests and rewrites the manifest byte-identically.

``--offline`` never touches the network: it fails clearly (exit 2) when any
asset's cache entry is missing or stale, listing the offending asset_ids.

Response shape (verified 2026-10-14 against the live API):
``{"type":"Feature", "geometry":..., "properties":{"parameter":{
"<PARAM>":{"YYYYMMDD": value, ...}, ...}}, "header":..., "messages":[],
"parameters":..., "times":...}``. Every daily parameter map must cover
exactly the 7638 requested days; otherwise the response is rejected.

Deterministic: sorted asset iteration, fixed constants, LF line endings.
"""

import argparse
import csv
import datetime
import hashlib
import json
import os
import sys
import time

import requests

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
SIDECAR_CSV = os.path.join(
    REPO_ROOT, "integration", "data_reference", "assets_verified.csv"
)
CACHE_DIR = os.path.join(
    REPO_ROOT, "integration", "data_raw_cache", "power_points"
)
MANIFEST_JSON = os.path.join(CACHE_DIR, "MANIFEST.json")

ENDPOINT = "https://power.larc.nasa.gov/api/temporal/daily/point"
COMMUNITY = "AG"
PARAMETERS = "PRECTOTCORR,T2M_MAX,T2M_MIN,RH2M,WS50M,PS,T2M"
REQUIRED_PARAMS = ("PRECTOTCORR", "T2M_MAX", "T2M_MIN", "RH2M", "WS50M", "PS", "T2M")
START = "20050101"
END = "20251129"
WINDOW_START = datetime.date(2005, 1, 1)
WINDOW_END = datetime.date(2025, 11, 29)
WINDOW_DAYS = (WINDOW_END - WINDOW_START).days + 1
DEFAULT_DELAY = 0.3
MAX_ATTEMPTS = 6
BACKOFF_BASE_S = 2.0
BACKOFF_CAP_S = 30.0
HTTP_TIMEOUT_S = 120
USER_AGENT = "infra-climate-valuation-integration/1.0 (offline research pipeline)"

SOURCE_NAME = "NASA POWER"


class OfflineCacheError(RuntimeError):
    """Raised by --offline when the cache is incomplete or stale."""


class ResponseShapeError(ValueError):
    """Raised when a 200 response is not the expected NASA POWER shape."""


def expected_days():
    days = []
    day = WINDOW_START
    while day <= WINDOW_END:
        days.append(day)
        day += datetime.timedelta(days=1)
    return days


def build_request_url(lat_text, lon_text):
    """Exact request URL for one asset coordinate (sidecar strings verbatim)."""
    return (
        "%s?community=%s&parameters=%s&start=%s&end=%s"
        "&latitude=%s&longitude=%s&format=JSON"
        % (ENDPOINT, COMMUNITY, PARAMETERS, START, END, lat_text, lon_text)
    )


def select_assets(sidecar_rows):
    """[(asset_id, lat_text, lon_text)] for assets with non-null coordinates.

    Sidecar strings are stripped and used verbatim in the request URL so the
    manifest records exactly what was sent.
    """
    out = []
    for row in sidecar_rows:
        lat_text = (row.get("verified_lat") or "").strip()
        lon_text = (row.get("verified_lon") or "").strip()
        if lat_text and lon_text:
            out.append((row["asset_id"], lat_text, lon_text))
    out.sort(key=lambda t: t[0])
    return out


def load_sidecar(path=SIDECAR_CSV):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def cache_path(cache_dir, asset_id):
    return os.path.join(cache_dir, "%s.json" % asset_id)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def load_manifest(path=MANIFEST_JSON):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def manifest_entry(payload, asset_id):
    if not payload:
        return None
    assets = payload.get("assets") or {}
    return assets.get(asset_id)


def cache_is_current(asset_id, url, cache_dir, payload):
    """True iff the cached body exists and matches the manifest provenance."""
    entry = manifest_entry(payload, asset_id)
    if not entry or entry.get("request_url") != url:
        return False
    path = cache_path(cache_dir, asset_id)
    if not os.path.exists(path):
        return False
    with open(path, "rb") as f:
        return sha256_bytes(f.read()) == entry.get("response_sha256")


def validate_response(body):
    """Parse and validate one raw response body.

    Returns the number of daily entries (all parameter maps must agree).
    Raises ResponseShapeError on any deviation from the verified shape.
    """
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ResponseShapeError("response body is not valid JSON: %s" % exc)
    params = (payload.get("properties") or {}).get("parameter")
    if not isinstance(params, dict):
        raise ResponseShapeError("response has no properties.parameter object")
    for name in REQUIRED_PARAMS:
        series = params.get(name)
        if not isinstance(series, dict) or not series:
            raise ResponseShapeError("response parameter %r is missing or empty" % name)
    n_days = {len(params[name]) for name in REQUIRED_PARAMS}
    if len(n_days) != 1:
        raise ResponseShapeError(
            "parameter maps disagree on day count: %s"
            % sorted((name, len(params[name])) for name in REQUIRED_PARAMS)
        )
    n_days = n_days.pop()
    if n_days != WINDOW_DAYS:
        raise ResponseShapeError(
            "expected %d daily entries, got %d" % (WINDOW_DAYS, n_days)
        )
    expected = {d.strftime("%Y%m%d") for d in expected_days()}
    for name in REQUIRED_PARAMS:
        if set(params[name]) != expected:
            raise ResponseShapeError(
                "parameter %r does not cover the exact requested window" % name
            )
    return n_days


def fetch_url(session, url, max_attempts=MAX_ATTEMPTS):
    """GET one URL with exponential backoff on transient failures.

    Returns the raw response body. Raises RuntimeError after the last
    attempt, or ResponseShapeError for a well-formed-but-wrong 200 body.
    """
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = session.get(url, timeout=HTTP_TIMEOUT_S)
        except requests.RequestException as exc:
            last_error = "request failed: %s" % exc
        else:
            if resp.status_code == 200:
                validate_response(resp.content)
                return resp.content
            if resp.status_code == 429 or resp.status_code >= 500:
                last_error = "HTTP %d" % resp.status_code
            else:
                raise RuntimeError(
                    "HTTP %d for %s (not retryable)" % (resp.status_code, url)
                )
        if attempt < max_attempts:
            sleep_s = min(BACKOFF_BASE_S * (2 ** (attempt - 1)), BACKOFF_CAP_S)
            time.sleep(sleep_s)
    raise RuntimeError(
        "giving up on %s after %d attempts: %s" % (url, max_attempts, last_error)
    )


def write_manifest(path, entries, totals):
    """Deterministic manifest: sorted assets, fixed key order, LF endings."""
    payload = {
        "source": "%s point API (daily)" % SOURCE_NAME,
        "endpoint": ENDPOINT,
        "community": COMMUNITY,
        "parameters": list(REQUIRED_PARAMS),
        "start": START,
        "end": END,
        "window_days": WINDOW_DAYS,
        "assets": {aid: entries[aid] for aid in sorted(entries)},
        "totals": totals,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def fetch_all(
    sidecar_path=SIDECAR_CSV,
    cache_dir=CACHE_DIR,
    manifest_path=MANIFEST_JSON,
    offline=False,
    delay=DEFAULT_DELAY,
    session=None,
    today=None,
):
    """Fetch (or reuse) every asset's response. Returns download stats.

    ``session`` and ``today`` are injectable for tests. ``today`` must be a
    zero-arg callable returning a ``datetime.date``.
    """
    if today is None:
        today = datetime.date.today
    assets = select_assets(load_sidecar(sidecar_path))
    urls = {aid: build_request_url(lat, lon) for aid, lat, lon in assets}
    if not assets:
        raise ValueError("sidecar %s has no assets with coordinates" % sidecar_path)
    if session is None:
        session = requests.Session()
        session.headers.update({"User-Agent": USER_AGENT})

    prior = load_manifest(manifest_path) or {}
    prior_entries = prior.get("assets") or {}

    entries = dict(prior_entries)
    n_requests = 0
    n_bytes = 0
    n_from_cache = 0

    if offline:
        missing = [
            aid
            for aid in sorted(urls)
            if not cache_is_current(aid, urls[aid], cache_dir, prior)
        ]
        if missing:
            raise OfflineCacheError(
                "--offline: %d of %d assets lack a current cached response "
                "(missing or stale cache/manifest): %s"
                % (len(missing), len(urls), ", ".join(missing))
            )

    for asset_id, url in sorted(urls.items()):
        if cache_is_current(asset_id, url, cache_dir, prior):
            n_from_cache += 1
            continue
        if offline:
            continue  # unreachable: offline validation above already failed
        body = fetch_url(session, url)
        n_requests += 1
        n_bytes += len(body)
        n_days = validate_response(body)
        path = cache_path(cache_dir, asset_id)
        os.makedirs(cache_dir, exist_ok=True)
        with open(path, "wb") as f:
            f.write(body)
        entries[asset_id] = {
            "request_url": url,
            "latitude": float(url.split("latitude=")[1].split("&")[0]),
            "longitude": float(url.split("longitude=")[1].split("&")[0]),
            "retrieval_date": today().isoformat(),
            "response_sha256": sha256_bytes(body),
            "n_days_returned": n_days,
            "requests_issued": 1,
            "bytes_downloaded": len(body),
        }
        print(
            "fetched %s (%d days, %.1f KB)" % (asset_id, n_days, len(body) / 1024.0)
        )
        if delay:
            time.sleep(delay)

    # Totals are cumulative provenance over the manifest's lifetime, so the
    # manifest bytes are stable across re-runs with a complete cache.
    # (assets_from_cache is per-run state and is only printed, never stored.)
    totals = {
        "assets": len(entries),
        "requests_issued": sum(e.get("requests_issued", 0) for e in entries.values()),
        "bytes_downloaded": sum(e.get("bytes_downloaded", 0) for e in entries.values()),
    }
    write_manifest(manifest_path, entries, totals)
    print("assets: %d" % totals["assets"])
    print("requests issued this run: %d" % n_requests)
    print("bytes downloaded this run: %d" % n_bytes)
    print("assets served from cache: %d" % n_from_cache)
    print(
        "manifest totals: %d requests, %d bytes over %d assets"
        % (totals["requests_issued"], totals["bytes_downloaded"], totals["assets"])
    )
    print("wrote %s" % manifest_path)
    return {
        "assets": totals["assets"],
        "requests_issued": n_requests,
        "bytes_downloaded": n_bytes,
        "assets_from_cache": n_from_cache,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Download and cache NASA POWER point-API responses "
        "for every asset with non-null coordinates."
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="never touch the network; fail clearly if any asset's "
        "cached response is missing or stale",
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
