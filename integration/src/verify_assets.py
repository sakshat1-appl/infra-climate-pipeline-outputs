"""Build the verified asset-coordinate sidecar (Phase 2).

Reads ONLY the identity columns (asset_id, sector, name) of
Master_Infrastructure_50_Units_Dataset.csv and joins them to the
human-auditable coordinate reference at
integration/config/asset_coordinates_reference.yaml. Writes:

  integration/data_reference/assets_verified.csv
  integration/data_reference/asset_verification_log.csv
  integration/data_output/verification_summary.json

Deterministic: identical inputs always produce byte-identical outputs
(rows ordered by asset_id, fixed float formatting, fixed key order).

This script never reads the quarantined Master columns (lat, lon,
climate_stress_index, risk_score, cost_deviation_pct, om_to_ebitda_ratio,
schedule_delay_days, or any climate column); the firewall guard enforces it.
"""

import csv
import json
import math
import os
from collections import Counter

import yaml

from firewall import assert_no_forbidden_columns, load_verified_assets

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(BASE_DIR)
MASTER_PATH = os.path.join(
    REPO_ROOT, "Master_Infrastructure_50_Units_Dataset.csv"
)
REFERENCE_PATH = os.path.join(
    BASE_DIR, "config", "asset_coordinates_reference.yaml"
)
SIDECAR_PATH = os.path.join(BASE_DIR, "data_reference", "assets_verified.csv")
LOG_PATH = os.path.join(BASE_DIR, "data_reference", "asset_verification_log.csv")
SUMMARY_PATH = os.path.join(BASE_DIR, "data_output", "verification_summary.json")

# Identity columns: the ONLY Master columns this phase may touch.
IDENTITY_COLUMNS = ("asset_id", "sector", "name")

SIDECAR_COLUMNS = (
    "asset_id",
    "sector",
    "name",
    "verified_lat",
    "verified_lon",
    "coord_role",
    "verification_status",
    "truth_lat",
    "truth_lon",
    "truth_source_url",
    "error_km",
    "provenance_source",
    "source_url",
    "retrieval_date",
    "notes",
)

RETRIEVAL_DATE = "2026-10-14"
EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km (mean Earth radius 6371.0088 km)."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def fmt(value):
    """Fixed 4-decimal formatting for coordinates; '' for nulls."""
    if value is None:
        return ""
    return "%.4f" % value


def main():
    with open(MASTER_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Hard gate: the only Master columns this script may consume are the
        # identity columns. Fail before reading any row if that is violated.
        assert_no_forbidden_columns(
            IDENTITY_COLUMNS, "verify_assets:identity-columns"
        )
        for col in IDENTITY_COLUMNS:
            if col not in (reader.fieldnames or []):
                raise ValueError("Master file missing identity column: %s" % col)
        master_rows = [
            {c: row[c] for c in IDENTITY_COLUMNS} for row in reader
        ]

    with open(REFERENCE_PATH, encoding="utf-8") as f:
        reference = yaml.safe_load(f)

    master_ids = [r["asset_id"] for r in master_rows]
    if sorted(master_ids) != sorted(reference.keys()):
        missing = sorted(set(master_ids) - set(reference.keys()))
        extra = sorted(set(reference.keys()) - set(master_ids))
        raise ValueError(
            "reference asset_id set must equal Master asset_id set; "
            "missing=%s extra=%s" % (missing, extra)
        )

    sidecar_rows = []
    log_rows = []
    for row in sorted(master_rows, key=lambda r: r["asset_id"]):
        aid = row["asset_id"]
        ref = reference[aid]
        lat, lon = ref.get("lat"), ref.get("lon")
        status = ref["status"]
        tlat, tlon = ref.get("truth_lat"), ref.get("truth_lon")
        error = ""
        if tlat is not None and tlon is not None:
            if lat is None or lon is None:
                raise ValueError("anchor %s must have a coordinate" % aid)
            error = "%.3f" % haversine_km(lat, lon, tlat, tlon)
        sidecar_rows.append(
            {
                "asset_id": aid,
                "sector": row["sector"],
                "name": row["name"],
                "verified_lat": fmt(lat),
                "verified_lon": fmt(lon),
                "coord_role": ref["role"],
                "verification_status": status,
                "truth_lat": fmt(tlat),
                "truth_lon": fmt(tlon),
                "truth_source_url": ref.get("truth_source_url") or "",
                "error_km": error,
                "provenance_source": ref.get("provenance") or "",
                "source_url": ref.get("source_url") or "",
                "retrieval_date": RETRIEVAL_DATE,
                "notes": ref.get("note") or "",
            }
        )
        if status == "unverified":
            log_rows.append(
                {
                    "asset_id": aid,
                    "action": "marked_unverified",
                    "source_consulted": "web research (Wikipedia, MNRE/CEA/Powergrid/PNGRB records)",
                    "url": "",
                    "outcome": "unverified",
                    "reason": ref.get("note") or "",
                }
            )
        else:
            log_rows.append(
                {
                    "asset_id": aid,
                    "action": "coordinate_lookup",
                    "source_consulted": ref.get("provenance") or "",
                    "url": ref.get("source_url") or "",
                    "outcome": status,
                    "reason": ref.get("note") or "",
                }
            )
            if tlat is not None:
                log_rows.append(
                    {
                        "asset_id": aid,
                        "action": "anchor_crosscheck",
                        "source_consulted": "independent reference",
                        "url": ref.get("truth_source_url") or "",
                        "outcome": "error_km=%s" % error,
                        "reason": "hand-checked anchor; tolerance 2 km (point) / 10 km (midpoint)",
                    }
                )

    os.makedirs(os.path.dirname(SIDECAR_PATH), exist_ok=True)
    with open(SIDECAR_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(SIDECAR_COLUMNS), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(sidecar_rows)

    with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "asset_id",
                "action",
                "source_consulted",
                "url",
                "outcome",
                "reason",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(sorted(log_rows, key=lambda r: (r["asset_id"], r["action"])))

    # Sanity: the sidecar must survive its own firewall before we summarise it.
    usable = load_verified_assets(SIDECAR_PATH)
    by_sector = Counter(r["sector"] for r in sidecar_rows)
    by_status = Counter(r["verification_status"] for r in sidecar_rows)
    unverified = [
        {"asset_id": r["asset_id"], "reason": r["notes"]}
        for r in sidecar_rows
        if r["verification_status"] == "unverified"
    ]
    anchors = [
        {
            "asset_id": r["asset_id"],
            "coord_role": r["coord_role"],
            "error_km": float(r["error_km"]),
        }
        for r in sidecar_rows
        if r["error_km"] != ""
    ]
    summary = {
        "generated_date": RETRIEVAL_DATE,
        "total_assets": len(sidecar_rows),
        "usable_with_coordinates": len(usable),
        "counts_by_sector": dict(sorted(by_sector.items())),
        "counts_by_verification_status": dict(sorted(by_status.items())),
        "unverified_assets": unverified,
        "anchor_assets": sorted(anchors, key=lambda a: a["asset_id"]),
    }
    os.makedirs(os.path.dirname(SUMMARY_PATH), exist_ok=True)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    print(
        "wrote %d sidecar rows (%d with coordinates), %d log rows"
        % (len(sidecar_rows), len(usable), len(log_rows))
    )


if __name__ == "__main__":
    main()
