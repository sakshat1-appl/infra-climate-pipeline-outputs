"""Tests for the validation gates.

Offline: the gates read committed files only (plus tmp copies for fault
injection). Never touches the network. Thresholds are pinned here to
their specified values so a weakened gate is caught; measured values are
asserted against the observed layer state.
"""

import csv
import json
import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

import validate_layer as V

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

EXPECTED_GATES = [
    "schema_dtypes",
    "ranges",
    "anchor_tolerance",
    "fabricated_tripwire",
    "reconcile_up_point",
    "determinism",
    "no_imputation",
    "licence_attribution",
    "coverage_accounting",
]


def load_report():
    with open(V.REPORT_JSON, encoding="utf-8") as f:
        return json.load(f)


def test_report_shape_and_all_gates_pass():
    report = load_report()
    assert [g["name"] for g in report["gates"]] == EXPECTED_GATES
    for gate in report["gates"]:
        assert set(gate) == {"name", "status", "value", "threshold",
                             "message"}, gate["name"]
        assert gate["status"] in ("pass", "fail"), gate["name"]
    assert report["overall"] == "pass"
    assert all(g["status"] == "pass" for g in report["gates"])
    assert set(report["artifacts"]) == {
        "feature_matrix_asset.csv", "feature_matrix_asset_month.csv",
        "coverage_report.json", "weather_coverage.json"}


def test_thresholds_are_the_specified_values():
    assert V.ANCHOR_TOL_POINT_KM == 2.0
    assert V.ANCHOR_TOL_MIDPOINT_KM == 10.0
    assert V.TRIPWIRE_MIN_KM == 50.0
    assert V.RECON_MIN_R == 0.95
    assert V.RECON_MAX_MAD_MM == 15.0
    assert V.PRECIP_MAX_MM == 500.0
    assert (V.TEMP_MIN_C, V.TEMP_MAX_C) == (-10.0, 55.0)


def test_reconciliation_numbers_as_measured():
    measured = V.reconcile_numbers()
    print("reconciliation: %s" % measured)
    assert measured["n_pairs"] == 2750
    assert abs(measured["pearson_r"] - 0.9818) < 0.001
    assert abs(measured["mad_mm"] - 8.147) < 0.05
    gate = V.check_reconciliation()
    assert gate["status"] == "pass"


def test_report_artifacts_match_files():
    report = load_report()
    paths = {
        "feature_matrix_asset.csv": V.JOIN.MATRIX_ASSET_CSV,
        "feature_matrix_asset_month.csv": V.JOIN.MATRIX_MONTH_CSV,
        "coverage_report.json": V.JOIN.COVERAGE_REPORT_JSON,
        "weather_coverage.json": V.COVERAGE_JSON,
    }
    for name, path in paths.items():
        assert report["artifacts"][name] == V.sha256_file(path), name


def copy_artifact_tree(tmp_path):
    """Mirror the input files a gate reads into tmp; return path map."""
    import shutil
    mirror = str(tmp_path / "mirror")
    os.makedirs(mirror)
    mapping = {}
    for src in (V.JOIN.MATRIX_ASSET_CSV, V.RD_CSV, V.MT_CSV, V.UP_MONTH_CSV,
                V.PAN_MONTH_CSV, V.UP_DAY_CSV, V.ASSET_DAY_GZ,
                V.COVERAGE_JSON, V.JOIN.SIDECAR_CSV, V.MASTER_CSV,
                V.JOIN.HIGHWAY_CSV, V.JOIN.MARITIME_CSV):
        dst = os.path.join(mirror, os.path.basename(src))
        shutil.copy(src, dst)
        mapping[src] = dst
    return mapping


def test_fault_injection_fails_gates_and_entrypoint(tmp_path, monkeypatch):
    mapping = copy_artifact_tree(tmp_path)
    mirror_matrix = mapping[V.JOIN.MATRIX_ASSET_CSV]
    mirror_up_day = mapping[V.UP_DAY_CSV]
    mirror_rd = mapping[V.RD_CSV]
    mirror_mt = mapping[V.MT_CSV]
    mirror_asset_gz = mapping[V.ASSET_DAY_GZ]

    def use_mirror(monkeypatch):
        monkeypatch.setattr(V.JOIN, "MATRIX_ASSET_CSV", mirror_matrix)
        monkeypatch.setattr(V, "RD_CSV", mirror_rd)
        monkeypatch.setattr(V, "MT_CSV", mirror_mt)
        monkeypatch.setattr(V, "UP_DAY_CSV", mirror_up_day)
        monkeypatch.setattr(V, "ASSET_DAY_GZ", mirror_asset_gz)

    # Fault 1: an imputed cell.
    use_mirror(monkeypatch)
    with open(mirror_matrix, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    header = list(rows[0].keys())
    rows[0]["rd_prov_is_imputed"] = "1"
    with open(mirror_matrix, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    assert V.check_no_imputation()["status"] == "fail"

    # Fault 2: out-of-range rainfall.
    use_mirror(monkeypatch)
    with open(mirror_up_day, newline="", encoding="utf-8") as f:
        day_rows = list(csv.DictReader(f))
    day_header = list(day_rows[0].keys())
    day_rows[0]["precip_mm_day"] = "9999"
    with open(mirror_up_day, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=day_header,
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(day_rows)
    assert V.check_ranges()["status"] == "fail"

    # The entrypoint reports overall fail and exit non-zero on the
    # broken tree (report redirected to tmp; real artifacts untouched).
    assert V.run(report_path=str(tmp_path / "rep.json")) == 1
    with open(str(tmp_path / "rep.json"), encoding="utf-8") as f:
        broken = json.load(f)
    assert broken["overall"] == "fail"
    by_name = {g["name"]: g for g in broken["gates"]}
    assert by_name["no_imputation"]["status"] == "fail"
    assert by_name["ranges"]["status"] == "fail"


def test_offline_operation_no_network(monkeypatch):
    """Gates and joins must work with the network disabled."""
    import socket

    def no_network(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(socket, "socket", no_network)
    assert V.check_schema()["status"] == "pass"
    assert V.check_reconciliation()["status"] == "pass"
