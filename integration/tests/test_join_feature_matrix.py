"""Tests for the release join (matrices + coverage report).

Offline: reads committed inputs/outputs only; the rebuild test writes to
a temporary directory. Never touches the network.
"""

import csv
import json
import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

import join_feature_matrix as J
from firewall import FORBIDDEN_MASTER_COLUMNS, assert_no_forbidden_columns

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


def load_sidecar():
    with open(J.SIDECAR_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_matrix():
    with open(J.MATRIX_ASSET_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_month():
    with open(J.MATRIX_MONTH_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_coverage():
    with open(J.COVERAGE_JSON, encoding="utf-8") as f:
        return json.load(f)


def test_asset_matrix_has_all_100_assets_sorted_unique():
    sidecar = load_sidecar()
    rows = load_matrix()
    assert len(rows) == 100 == len(sidecar)
    assert [r["asset_id"] for r in rows] == sorted(r["asset_id"] for r in rows)
    assert len({r["asset_id"] for r in rows}) == 100
    assert {r["asset_id"] for r in rows} == {r["asset_id"] for r in sidecar}


def test_month_matrix_grain_and_coverage():
    rows = load_month()
    assert len(rows) == 20331
    assert len({r["asset_id"] for r in rows}) == 81
    assert len({r["year_month"] for r in rows}) == 251
    keys = [(r["asset_id"], r["year_month"]) for r in rows]
    assert len(keys) == len(set(keys))
    assert keys == sorted(keys)
    # Pan-India monthly precipitation is filled on every row except the
    # trailing partial month 2025-11 (panel ends 2025-11-29, so the
    # November total is unobservable without lookahead - null by the
    # phase-4 null-on-gap rule, one per asset).
    null_pan = [(r["asset_id"], r["year_month"]) for r in rows
                if not r["pan_wx_precip_month_mm"].strip()]
    assert len(null_pan) == 81
    assert {ym for _, ym in null_pan} == {"2025-11"}
    assert {aid for aid, _ in null_pan} == {r["asset_id"] for r in rows}
    with_up = [r for r in rows if r["district_canonical"].strip()]
    assert len(with_up) == 2761
    # UP columns are filled exactly where the district join covers,
    # except the same trailing partial month 2025-11 (11 rows).
    null_up = [r for r in with_up
               if not r["up_wx_precip_month_mm"].strip()]
    assert len(null_up) == 11
    assert {r["year_month"] for r in null_up} == {"2025-11"}
    assert all(r["up_wx_precip_month_mm"].strip() != ""
               for r in with_up if r not in null_up)
    assert all(r["up_wx_precip_month_mm"].strip() == ""
               for r in rows if not r["district_canonical"].strip())


def test_unverified_assets_present_with_nulls_not_dropped():
    sidecar = {r["asset_id"]: r for r in load_sidecar()}
    by_id = {r["asset_id"]: r for r in load_matrix()}
    unverified = [aid for aid, r in sidecar.items()
                  if not (r.get("verified_lat") or "").strip()]
    assert len(unverified) == 19
    feature_cols = [c for c in by_id[unverified[0]]
                    if c.startswith(("rd_dist", "rd_hwy", "rd_nh",
                                      "rd_node", "mt_dist", "mt_nearest",
                                      "mt_ports", "mt_pw", "mt_in",
                                      "mt_coast", "mt_throughput"))]
    assert feature_cols
    for aid in unverified:
        row = by_id[aid]
        assert all((row[c] or "") == "" for c in feature_cols), aid
        assert row["null_reason"] == "asset_coordinate_unverified", aid
        # No family row exists to copy provenance from: provenance cells
        # are empty (not "0") exactly on unverified rows.
        assert row["rd_prov_is_imputed"] == "", aid
        assert row["mt_prov_is_imputed"] == "", aid
    for aid, row in by_id.items():
        if aid not in unverified:
            assert row["rd_prov_is_imputed"] == "0", aid
            assert row["mt_prov_is_imputed"] == "0", aid


def test_every_null_feature_cell_explained():
    coverage = load_coverage()
    cov_lookup = J.coverage_lookup(coverage)
    rows = load_matrix()
    header = list(rows[0].keys())
    n_null, n_explained = J.assert_nulls_explained(
        rows, header, cov_lookup, coverage)
    print("NULL cells: %d, explained: %d" % (n_null, n_explained))
    assert n_null > 0
    assert n_null == n_explained


def test_firewall_on_matrices():
    for path in (J.MATRIX_ASSET_CSV, J.MATRIX_MONTH_CSV):
        with open(path, newline="", encoding="utf-8") as f:
            header = next(csv.reader(f))
        assert_no_forbidden_columns(header, path)
        assert not (set(header) & set(FORBIDDEN_MASTER_COLUMNS))


def test_rebuild_to_tmp_is_byte_identical(tmp_path):
    a = str(tmp_path / "a.csv")
    m = str(tmp_path / "m.csv")
    r = str(tmp_path / "r.json")
    J.build(asset_path=a, month_path=m, report_path=r)
    for live, fresh in ((J.MATRIX_ASSET_CSV, a), (J.MATRIX_MONTH_CSV, m),
                        (J.COVERAGE_REPORT_JSON, r)):
        with open(live, "rb") as f:
            live_bytes = f.read()
        with open(fresh, "rb") as f:
            assert f.read() == live_bytes, live
