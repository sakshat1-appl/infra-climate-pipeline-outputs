"""Download and cache raw OpenStreetMap highway responses, one JSON per asset.

For every asset in the verified sidecar with non-null coordinates (81 of
100), fetch all ways tagged ``highway`` in {motorway, trunk, primary,
secondary} inside the asset's bounding box (asset coordinate plus a 25 km
buffer) from the Overpass API and cache the raw response body as
``integration/data_raw_cache/osm_highways/<asset_id>.json``. A committed
``MANIFEST.json`` in the same directory records, per asset_id, the exact
Overpass query string, endpoint, bounding box, retrieval date, response
sha256 and the counts of ways and nodes returned.

Query strategy (why small boxes, not whole-state queries):
  * The public Overpass instance rejects heavy queries: a whole-state
    ``way["highway"=...]`` query times out, and even a 0.45-degree box with
    a tag-regex filter returns HTTP 504 ("server is probably too busy").
    Verified 2026-10-15: one query per asset as a UNION of four exact
    ``highway=`` matches over that asset's own 25 km-buffer box succeeds
    (3 s on a rural tile, ~10 s on the densest Delhi tile).
  * One query per asset (81 requests), strictly sequential with a polite
    delay, ``out geom`` so each way carries its tags, node references and
    inline geometry in a single response (no ``>`` recursion pass needed).

Request pattern (POST, no API key required; verified accessible):

  https://overpass-api.de/api/interpreter
    body: data=<overpass QL query, see build_query()>

Overpass data is ODbL and requires attribution
"(c) OpenStreetMap contributors" (carried into the feature table licence
column by build_highway_features.py).

Politeness and reliability:
  * Strictly sequential requests with a polite delay between them
    (default 1.5 s; the instance sheds load under bursts).
  * A descriptive User-Agent header is REQUIRED by the Overpass usage
    policy and is set on every request.
  * Transient failures (connection error, timeout, HTTP 429/504, a 200
    body that is not valid JSON, or a well-formed body carrying a
    timeout/truncation remark) are retried with exponential backoff
    (5, 10, 20, 40, 60 s) up to 5 attempts; any other 4xx fails
    immediately. A truncated response is NEVER cached as valid.
  * A tile that still fails after all attempts does NOT abort the run:
    it is recorded under the manifest's ``failures`` map (asset_id ->
    error string) and the process exits 1 after writing the manifest, so
    build_highway_features.py can emit honest nulls with reason
    ``osm_fetch_failed`` instead of guessing.

Idempotency: an asset is skipped when its cache file exists AND the
manifest entry's query matches the recomputed query AND the cached body's
sha256 matches the manifest record. Re-running with a complete cache
issues zero network requests and rewrites the manifest byte-identically
(failure entries are re-tried, not carried over as success).

``--offline`` never touches the network: it fails clearly (exit 2) when any
asset's cache entry is missing or stale, listing the offending asset_ids.

Response shape: ``{"version":..., "generator":..., "osm3s": {
"timestamp_osm_base": ...}, "elements": [{"type":"way", "id":...,
"bounds":..., "nodes":[...], "geometry":[{lat,lon},...], "tags":
{"highway":..., ...}}, ...]}``. Only ``type == "way"`` elements are
counted; anything else is ignored by the builder.

Deterministic: sorted asset iteration, fixed constants, LF line endings.
"""

import argparse
import csv
import datetime
import hashlib
import json
import math
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
    REPO_ROOT, "integration", "data_raw_cache", "osm_highways"
)
MANIFEST_JSON = os.path.join(CACHE_DIR, "MANIFEST.json")

ENDPOINT = "https://overpass-api.de/api/interpreter"
OVERPASS_TIMEOUT_STMT = 170
BBOX_BUFFER_KM = 25.0
HIGHWAY_CLASSES = ("motorway", "trunk", "primary", "secondary")
DEFAULT_DELAY = 1.5
MAX_ATTEMPTS = 5
BACKOFF_S = (5.0, 10.0, 20.0, 40.0, 60.0)
HTTP_TIMEOUT_S = 175
USER_AGENT = (
    "infra-climate-valuation-integration/1.0 "
    "(academic research data pipeline; https://github.com/VariableThe/infra-climate-valuation)"
)

SOURCE_NAME = "OpenStreetMap via Overpass API"


class OfflineCacheError(RuntimeError):
    """Raised by --offline when the cache is incomplete or stale."""


class ResponseShapeError(ValueError):
    """Raised when a 200 response is not the expected Overpass shape."""


def bbox_for_point(lat, lon, buffer_km=BBOX_BUFFER_KM):
    """(south, west, north, east) for a point plus a km buffer.

    1 degree of latitude ~= 111.32 km; longitude degrees shrink with
    cos(latitude). This box is the FETCH scope only; the builder applies
    the exact radii.
    """
    dlat = buffer_km / 111.32
    dlon = buffer_km / (111.32 * math.cos(math.radians(lat)))
    return (lat - dlat, lon - dlon, lat + dlat, lon + dlon)


def build_query(south, west, north, east):
    """Exact Overpass QL query for one tile.

    A union of four EXACT highway matches (not a tag regex: regex filters
    over a large box are killed by the server with HTTP 504, verified
    2026-10-15). ``out geom`` returns each way's tags, node references
    AND inline geometry, so no recursion pass is needed.
    """
    scope = "%.6f,%.6f,%.6f,%.6f" % (south, west, north, east)
    members = "".join(
        'way["highway"="%s"](%s);' % (cls, scope) for cls in HIGHWAY_CLASSES
    )
    return "[out:json][timeout:%d];(%s);out geom;" % (
        OVERPASS_TIMEOUT_STMT,
        members,
    )


def select_assets(sidecar_rows):
    """[(asset_id, lat, lon)] for assets with non-null coordinates."""
    out = []
    for row in sidecar_rows:
        lat_text = (row.get("verified_lat") or "").strip()
        lon_text = (row.get("verified_lon") or "").strip()
        if lat_text and lon_text:
            out.append((row["asset_id"], float(lat_text), float(lon_text)))
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
    tiles = payload.get("tiles") or {}
    return tiles.get(asset_id)


def cache_is_current(asset_id, query, cache_dir, payload):
    """True iff the cached body exists and matches the manifest provenance."""
    entry = manifest_entry(payload, asset_id)
    if not entry or entry.get("query") != query:
        return False
    path = cache_path(cache_dir, asset_id)
    if not os.path.exists(path):
        return False
    with open(path, "rb") as f:
        return sha256_bytes(f.read()) == entry.get("response_sha256")


def validate_response(body):
    """Parse and validate one raw Overpass response body.

    Returns ``(n_ways, n_distinct_nodes, osm_extract_date)``. Raises
    ResponseShapeError on any deviation, including a server-side
    timeout/truncation remark (such a body must never be cached as valid).
    """
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ResponseShapeError("response body is not valid JSON: %s" % exc)
    elements = payload.get("elements")
    if not isinstance(elements, list):
        raise ResponseShapeError("response has no elements list")
    remark = payload.get("remark") or ""
    lowered = remark.lower()
    if "timeout" in lowered or "truncat" in lowered:
        raise ResponseShapeError(
            "server-side truncation remark (not cacheable): %r" % remark[:200]
        )
    for el in elements:
        if not isinstance(el, dict) or el.get("type") != "way":
            raise ResponseShapeError(
                "unexpected non-way element in response: %r"
                % sorted(el.keys() if isinstance(el, dict) else [])
            )
        if not isinstance(el.get("nodes"), list) or not isinstance(
            el.get("geometry"), list
        ):
            raise ResponseShapeError(
                "way element %r lacks nodes/geometry (need 'out geom')"
                % el.get("id")
            )
    n_ways = len(elements)
    node_ids = set()
    for el in elements:
        node_ids.update(el["nodes"])
    osm3s = payload.get("osm3s") or {}
    return n_ways, len(node_ids), osm3s.get("timestamp_osm_base")


def fetch_query(session, query, max_attempts=MAX_ATTEMPTS):
    """POST one query with exponential backoff on transient failures.

    Returns the raw response body. Raises RuntimeError after the last
    attempt, or ResponseShapeError for a well-formed-but-wrong 200 body
    (only after the truncation-remark check passes as retryable: a remark
    means the server shed load, so it IS retried; a structurally wrong
    body raises immediately via validate_response on the last attempt).
    """
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = session.post(
                ENDPOINT, data={"data": query}, timeout=HTTP_TIMEOUT_S
            )
        except requests.RequestException as exc:
            last_error = "request failed: %s" % exc
        else:
            if resp.status_code == 200:
                try:
                    validate_response(resp.content)
                except ResponseShapeError as exc:
                    message = str(exc)
                    if "not cacheable" in message or "not valid JSON" in message:
                        last_error = message
                    else:
                        raise
                else:
                    return resp.content
            elif resp.status_code == 429 or resp.status_code >= 500:
                last_error = "HTTP %d" % resp.status_code
            else:
                raise RuntimeError(
                    "HTTP %d (not retryable)" % resp.status_code
                )
        if attempt < max_attempts:
            time.sleep(BACKOFF_S[attempt - 1])
    raise RuntimeError(
        "giving up after %d attempts: %s" % (max_attempts, last_error)
    )


def write_manifest(path, entries, failures, totals):
    """Deterministic manifest: sorted tiles, fixed key order, LF endings."""
    payload = {
        "source": "%s (highways)" % SOURCE_NAME,
        "endpoint": ENDPOINT,
        "bbox_buffer_km": BBOX_BUFFER_KM,
        "highway_classes": list(HIGHWAY_CLASSES),
        "overpass_timeout_s": OVERPASS_TIMEOUT_STMT,
        "tiles": {aid: entries[aid] for aid in sorted(entries)},
        "failures": {aid: failures[aid] for aid in sorted(failures)},
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
    """Fetch (or reuse) every asset's tile. Returns download stats.

    ``session`` and ``today`` are injectable for tests. ``today`` must be a
    zero-arg callable returning a ``datetime.date``.
    """
    if today is None:
        today = datetime.date.today
    assets = select_assets(load_sidecar(sidecar_path))
    if not assets:
        raise ValueError("sidecar %s has no assets with coordinates" % sidecar_path)
    plan = {}
    for asset_id, lat, lon in assets:
        bbox = bbox_for_point(lat, lon)
        plan[asset_id] = {
            "query": build_query(*bbox),
            "bbox": list(bbox),
            "lat": lat,
            "lon": lon,
        }
    if session is None:
        session = requests.Session()
        session.headers.update({"User-Agent": USER_AGENT})

    prior = load_manifest(manifest_path) or {}
    prior_entries = prior.get("tiles") or {}

    entries = dict(prior_entries)
    failures = {}
    n_requests = 0
    n_bytes = 0
    n_from_cache = 0

    if offline:
        missing = [
            aid
            for aid in sorted(plan)
            if not cache_is_current(aid, plan[aid]["query"], cache_dir, prior)
        ]
        if missing:
            raise OfflineCacheError(
                "--offline: %d of %d assets lack a current cached tile "
                "(missing or stale cache/manifest): %s"
                % (len(missing), len(plan), ", ".join(missing))
            )

    for asset_id in sorted(plan):
        query = plan[asset_id]["query"]
        if cache_is_current(asset_id, query, cache_dir, prior):
            n_from_cache += 1
            continue
        if offline:
            continue  # unreachable: offline validation above already failed
        try:
            body = fetch_query(session, query)
        except (RuntimeError, ResponseShapeError) as exc:
            failures[asset_id] = str(exc)
            entries.pop(asset_id, None)
            print("FAILED %s: %s" % (asset_id, exc))
            continue
        n_requests += 1
        n_bytes += len(body)
        n_ways, n_nodes, extract_date = validate_response(body)
        path = cache_path(cache_dir, asset_id)
        os.makedirs(cache_dir, exist_ok=True)
        with open(path, "wb") as f:
            f.write(body)
        entries[asset_id] = {
            "query": query,
            "endpoint": ENDPOINT,
            "bbox": plan[asset_id]["bbox"],
            "bbox_buffer_km": BBOX_BUFFER_KM,
            "retrieval_date": today().isoformat(),
            "osm_extract_date": extract_date,
            "response_sha256": sha256_bytes(body),
            "n_ways": n_ways,
            "n_nodes": n_nodes,
            "requests_issued": 1,
            "bytes_downloaded": len(body),
        }
        print(
            "fetched %s (%d ways, %d nodes, %.1f KB)"
            % (asset_id, n_ways, n_nodes, len(body) / 1024.0),
            flush=True,
        )
        # Incremental manifest write: a killed run still leaves provenance
        # for every tile fetched so far (the manifest is rewritten
        # byte-identically once the cache is complete).
        write_manifest(
            manifest_path,
            entries,
            failures,
            {
                "tiles": len(entries),
                "requests_issued": sum(
                    e.get("requests_issued", 0) for e in entries.values()
                ),
                "bytes_downloaded": sum(
                    e.get("bytes_downloaded", 0) for e in entries.values()
                ),
            },
        )
        if delay:
            time.sleep(delay)

    # Totals are cumulative provenance over the manifest's lifetime, so the
    # manifest bytes are stable across re-runs with a complete cache.
    # (assets_from_cache is per-run state and is only printed, never stored.)
    totals = {
        "tiles": len(entries),
        "requests_issued": sum(e.get("requests_issued", 0) for e in entries.values()),
        "bytes_downloaded": sum(e.get("bytes_downloaded", 0) for e in entries.values()),
    }
    write_manifest(manifest_path, entries, failures, totals)
    print("tiles: %d" % totals["tiles"])
    print("requests issued this run: %d" % n_requests)
    print("bytes downloaded this run: %d" % n_bytes)
    print("tiles served from cache: %d" % n_from_cache)
    print("tiles failed this run: %d" % len(failures))
    print(
        "manifest totals: %d requests, %d bytes over %d tiles"
        % (totals["requests_issued"], totals["bytes_downloaded"], totals["tiles"])
    )
    print("wrote %s" % manifest_path)
    return {
        "tiles": totals["tiles"],
        "requests_issued": n_requests,
        "bytes_downloaded": n_bytes,
        "tiles_from_cache": n_from_cache,
        "failures": dict(failures),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Download and cache Overpass highway tiles "
        "for every asset with non-null coordinates."
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="never touch the network; fail clearly if any asset's "
        "cached tile is missing or stale",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        help="seconds to wait between requests (default %(default)s)",
    )
    args = parser.parse_args(argv)
    try:
        stats = fetch_all(offline=args.offline, delay=args.delay)
    except OfflineCacheError as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 2
    return 1 if stats["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
