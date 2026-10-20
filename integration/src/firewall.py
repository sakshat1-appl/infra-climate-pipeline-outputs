"""Fabricated-coordinate firewall for the integration layer.

The Master asset file's ``lat``/``lon`` columns are fabricated (verified
offsets of 208-2119 km from the real named assets) and its
``climate_stress_index``/``risk_score``/``cost_deviation_pct`` columns are a
deterministic formula plus noise. Nothing in the integration layer may ever
read those columns as inputs. This module is the reusable guard, importable
by later phases.

The sidecar's own coordinate columns are deliberately named
``verified_lat``/``verified_lon`` (not ``lat``/``lon``) so the denylist below
can be applied to any downstream dataframe without false positives.
"""

import csv

# Frozen denylist: columns of Master_Infrastructure_50_Units_Dataset.csv that
# must never be read as inputs by the integration layer. Do not remove names
# from this set without a design review recorded in a later PR.
FORBIDDEN_MASTER_COLUMNS = frozenset(
    {
        "lat",
        "lon",
        "mean_temp_c",
        "max_temp_c",
        "mean_wind_speed",
        "total_precipitation",
        "mean_solar_irradiance",
        "cost_deviation_pct",
        "schedule_delay_days",
        "risk_score",
        "om_to_ebitda_ratio",
        "climate_stress_index",
    }
)


def assert_no_forbidden_columns(columns, context):
    """Raise ValueError if any forbidden Master column is present.

    ``columns`` is any iterable of column names (e.g. ``df.columns``).
    ``context`` is a short string naming the caller, included in the error
    so the violation is traceable.
    """
    cols = list(columns)
    offending = [c for c in cols if c in FORBIDDEN_MASTER_COLUMNS]
    if offending:
        raise ValueError(
            "forbidden Master column(s) %s in context %r; "
            "these columns are quarantined (fabricated coordinates / "
            "formula-derived values) and must never enter the "
            "integration layer" % (sorted(offending), context)
        )


def load_verified_assets(path):
    """Load the verified asset sidecar, returning only rows with coordinates.

    Raises FileNotFoundError if the file is missing, ValueError if it is
    malformed (missing required columns, zero usable rows).
    """
    required = {
        "asset_id",
        "sector",
        "name",
        "verified_lat",
        "verified_lon",
        "coord_role",
        "verification_status",
        "provenance_source",
        "source_url",
    }
    try:
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except FileNotFoundError:
        raise FileNotFoundError("verified asset sidecar not found: %s" % path)
    if not rows:
        raise ValueError("verified asset sidecar is empty: %s" % path)
    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(
            "verified asset sidecar %s is malformed; missing columns: %s"
            % (path, sorted(missing))
        )
    usable = [
        r
        for r in rows
        if (r.get("verified_lat") or "").strip()
        and (r.get("verified_lon") or "").strip()
    ]
    if not usable:
        raise ValueError(
            "verified asset sidecar %s has no rows with coordinates" % path
        )
    return usable
