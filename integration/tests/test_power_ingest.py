"""Tests for the NASA POWER point-API ingest and its manifest.

Offline: reads the committed MANIFEST.json and the local raw cache, and
exercises the fetch/retry/skip logic with stub sessions - never the network.
"""

import datetime
import hashlib
import json
import os

import pytest

from ingest_power_points import (
    MAX_ATTEMPTS,
    OfflineCacheError,
    build_request_url,
    cache_is_current,
    fetch_all,
    fetch_url,
    load_manifest,
    select_assets,
    validate_response,
)

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

REQUIRED_ENTRY_FIELDS = {
    "request_url",
    "latitude",
    "longitude",
    "retrieval_date",
    "response_sha256",
    "n_days_returned",
}
EXPECTED_PARAMS = "PRECTOTCORR,T2M_MAX,T2M_MIN,RH2M,WS50M,PS,T2M"
WINDOW_DAYS = 7638  # 2005-01-01..2025-11-29


def load_sidecar():
    import csv

    with open(SIDECAR_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def expected_assets():
    return select_assets(load_sidecar())


def _power_cache_complete():
    """True when every expected asset has a current cached response.

    The raw cache (integration/data_raw_cache/power_points/*.json) is
    gitignored, so a fresh clone holds only MANIFEST.json. Cache-needing
    tests skip instead of failing in that case.
    """
    try:
        manifest = load_manifest(MANIFEST_JSON)
        if not manifest:
            return False
        return all(
            cache_is_current(aid, build_request_url(lat, lon), CACHE_DIR, manifest)
            for aid, lat, lon in expected_assets()
        )
    except Exception:
        return False


requires_power_cache = pytest.mark.skipif(
    not _power_cache_complete(),
    reason="needs the fetched NASA POWER cache "
    "(integration/data_raw_cache/power_points/*.json are gitignored); "
    "run integration/src/ingest_power_points.py with network access first",
)


def make_payload(n_days=WINDOW_DAYS, first="20050101"):
    """A minimal valid NASA POWER point response body."""
    params = {}
    for name in ("PRECTOTCORR", "T2M_MAX", "T2M_MIN", "RH2M", "WS50M", "PS", "T2M"):
        series = {}
        day = datetime.datetime.strptime(first, "%Y%m%d").date()
        for _ in range(n_days):
            series[day.strftime("%Y%m%d")] = 1.0
            day += datetime.timedelta(days=1)
        params[name] = series
    return json.dumps({"properties": {"parameter": params}}).encode("utf-8")


class StubResponse:
    def __init__(self, status_code=200, body=b""):
        self.status_code = status_code
        self.content = body


class StubSession:
    """Scripted session: queues responses or raises if called unscripted."""

    def __init__(self, script=None, fail_with=None):
        self.script = list(script or [])
        self.fail_with = fail_with
        self.calls = []

    def get(self, url, timeout=None):
        self.calls.append((url, timeout))
        if self.fail_with is not None:
            raise self.fail_with
        if not self.script:
            raise AssertionError("unexpected network call: %s" % url)
        outcome = self.script.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


# --------------------------------------------------------------------------
# Manifest and cache integrity (committed state)
# --------------------------------------------------------------------------


@requires_power_cache
def test_manifest_covers_every_asset_with_coordinates():
    manifest = load_manifest(MANIFEST_JSON)
    assert manifest is not None
    entries = manifest["assets"]
    expected = expected_assets()
    assert len(entries) == len(expected) == 81
    assert set(entries) == {aid for aid, _, _ in expected}
    for aid, lat_text, lon_text in expected:
        entry = entries[aid]
        assert set(REQUIRED_ENTRY_FIELDS) <= set(entry)
        assert entry["n_days_returned"] == WINDOW_DAYS
        assert entry["retrieval_date"] == "2026-10-15"
        assert entry["response_sha256"] == hashlib.sha256(
            open(os.path.join(CACHE_DIR, "%s.json" % aid), "rb").read()
        ).hexdigest()


def test_manifest_urls_match_sidecar_coordinates_and_window():
    manifest = load_manifest(MANIFEST_JSON)
    by_id = {r["asset_id"]: r for r in load_sidecar()}
    for aid, entry in sorted(manifest["assets"].items()):
        url = entry["request_url"]
        assert url.startswith(
            "https://power.larc.nasa.gov/api/temporal/daily/point?"
        )
        assert "community=AG" in url
        assert "parameters=%s" % EXPECTED_PARAMS in url
        assert "start=20050101" in url and "end=20251129" in url
        assert "format=JSON" in url
        row = by_id[aid]
        assert "latitude=%s" % row["verified_lat"] in url
        assert "longitude=%s" % row["verified_lon"] in url
        assert entry["latitude"] == float(row["verified_lat"])
        assert entry["longitude"] == float(row["verified_lon"])


def test_manifest_totals_match_entries():
    manifest = load_manifest(MANIFEST_JSON)
    entries = manifest["assets"]
    assert manifest["totals"]["assets"] == len(entries)
    assert manifest["totals"]["requests_issued"] == sum(
        e.get("requests_issued", 0) for e in entries.values()
    )
    assert manifest["totals"]["bytes_downloaded"] == sum(
        e.get("bytes_downloaded", 0) for e in entries.values()
    )


def test_manifest_json_is_deterministic_bytes():
    """Re-serialising the manifest reproduces the committed bytes exactly."""
    with open(MANIFEST_JSON, encoding="utf-8") as f:
        raw = f.read()
    payload = json.loads(raw)
    assert json.dumps(payload, indent=2) + "\n" == raw


# --------------------------------------------------------------------------
# Idempotency and --offline
# --------------------------------------------------------------------------


@requires_power_cache
def test_rerun_with_complete_cache_issues_zero_requests(tmp_path):
    """Re-running against a full cache must not touch the network."""
    manifest_path = tmp_path / "MANIFEST.json"
    manifest_path.write_text(
        open(MANIFEST_JSON, encoding="utf-8").read(), encoding="utf-8"
    )
    boom = StubSession(fail_with=AssertionError("network was touched"))
    stats = fetch_all(
        sidecar_path=SIDECAR_CSV,
        cache_dir=CACHE_DIR,
        manifest_path=str(manifest_path),
        delay=0.0,
        session=boom,
    )
    assert boom.calls == []
    assert stats["requests_issued"] == 0
    assert stats["assets_from_cache"] == 81
    # The rewritten manifest is byte-identical.
    assert manifest_path.read_text(encoding="utf-8") == open(
        MANIFEST_JSON, encoding="utf-8"
    ).read()


def test_offline_fails_clearly_when_cache_missing(tmp_path):
    with pytest.raises(OfflineCacheError) as excinfo:
        fetch_all(
            sidecar_path=SIDECAR_CSV,
            cache_dir=str(tmp_path / "empty_cache"),
            manifest_path=str(tmp_path / "MANIFEST.json"),
            offline=True,
            delay=0.0,
            session=StubSession(),
        )
    message = str(excinfo.value)
    assert "GP_081" in message and "81 of 81" in message


@requires_power_cache
def test_offline_succeeds_with_complete_cache(tmp_path):
    manifest_path = tmp_path / "MANIFEST.json"
    manifest_path.write_text(
        open(MANIFEST_JSON, encoding="utf-8").read(), encoding="utf-8"
    )
    stats = fetch_all(
        sidecar_path=SIDECAR_CSV,
        cache_dir=CACHE_DIR,
        manifest_path=str(manifest_path),
        offline=True,
        delay=0.0,
        session=StubSession(fail_with=AssertionError("network was touched")),
    )
    assert stats["requests_issued"] == 0


@requires_power_cache
def test_cache_is_current_detects_stale_sha(tmp_path):
    """A cache file whose sha no longer matches the manifest is stale."""
    manifest = load_manifest(MANIFEST_JSON)
    aid = "GP_086"
    url = build_request_url("26.4600", "79.9300")
    assert cache_is_current(aid, url, CACHE_DIR, manifest) is True
    # Corrupt the cache file on a copy: point CACHE_DIR at a tampered tree.
    import shutil

    tampered = tmp_path / "cache"
    shutil.copytree(CACHE_DIR, tampered)
    with open(os.path.join(tampered, "%s.json" % aid), "ab") as f:
        f.write(b"tampered")
    assert cache_is_current(aid, url, str(tampered), manifest) is False
    assert cache_is_current(aid, url, str(tmp_path / "nonexistent"), manifest) is False


# --------------------------------------------------------------------------
# Request building and response validation
# --------------------------------------------------------------------------


def test_build_request_url_exact():
    url = build_request_url("26.4600", "79.9300")
    assert url == (
        "https://power.larc.nasa.gov/api/temporal/daily/point"
        "?community=AG&parameters=%s&start=20050101&end=20251129"
        "&latitude=26.4600&longitude=79.9300&format=JSON" % EXPECTED_PARAMS
    )


def test_validate_response_rejects_bad_shapes():
    with pytest.raises(ValueError):
        validate_response(b"not json")
    with pytest.raises(ValueError):
        validate_response(json.dumps({"properties": {}}).encode("utf-8"))
    # Missing a parameter.
    payload = json.loads(make_payload())
    del payload["properties"]["parameter"]["PS"]
    with pytest.raises(ValueError):
        validate_response(json.dumps(payload).encode("utf-8"))
    # Wrong day count.
    with pytest.raises(ValueError):
        validate_response(make_payload(n_days=100).decode("utf-8").encode("utf-8"))
    # A day outside the requested window.
    payload = json.loads(make_payload())
    series = payload["properties"]["parameter"]["PRECTOTCORR"]
    series["20251201"] = 1.0
    with pytest.raises(ValueError):
        validate_response(json.dumps(payload).encode("utf-8"))


def test_validate_response_accepts_real_shape():
    assert validate_response(make_payload()) == WINDOW_DAYS


# --------------------------------------------------------------------------
# Fetch retry / backoff behaviour
# --------------------------------------------------------------------------


def test_fetch_url_retries_transient_then_succeeds(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    body = make_payload()
    session = StubSession(
        script=[
            StubResponse(status_code=500),
            StubResponse(status_code=429),
            StubResponse(status_code=200, body=body),
        ]
    )
    assert fetch_url(session, "http://example.invalid", max_attempts=3) == body
    assert len(session.calls) == 3


def test_fetch_url_gives_up_after_max_attempts(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    import requests

    session = StubSession(fail_with=requests.ConnectionError("down"))
    with pytest.raises(RuntimeError) as excinfo:
        fetch_url(session, "http://example.invalid", max_attempts=MAX_ATTEMPTS)
    assert "giving up" in str(excinfo.value)
    assert len(session.calls) == MAX_ATTEMPTS


def test_fetch_url_does_not_retry_hard_4xx():
    session = StubSession(script=[StubResponse(status_code=400)])
    with pytest.raises(RuntimeError) as excinfo:
        fetch_url(session, "http://example.invalid")
    assert "not retryable" in str(excinfo.value)
    assert len(session.calls) == 1


def test_fetch_url_rejects_bad_200_body():
    session = StubSession(script=[StubResponse(status_code=200, body=b"garbage")])
    with pytest.raises(ValueError):
        fetch_url(session, "http://example.invalid")
