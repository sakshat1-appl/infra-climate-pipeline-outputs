#!/usr/bin/env python3
"""Build a single-file, offline HTML dashboard from the combined pipeline outputs.

Reads the six combined-pipeline artefacts, validates them, and writes
outputs_combined/dashboard.html with inline CSS and inline SVG.  No network
resources, no JavaScript and no third-party packages are used.
"""

import csv
import html
import json
import math
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs_combined")

JSON_PATH = os.path.join(OUTPUTS_DIR, "combined_pipeline_results.json")
FORECAST_PATH = os.path.join(OUTPUTS_DIR, "forecast_quarterly.csv")
QUARTERLY_PATH = os.path.join(OUTPUTS_DIR, "quarterly_series.csv")
STATE_PATH = os.path.join(OUTPUTS_DIR, "future_projection_by_state_2026.csv")
MONTE_CARLO_PATH = os.path.join(OUTPUTS_DIR, "monte_carlo_projects.csv")
EXPOSURE_PATH = os.path.join(OUTPUTS_DIR, "state_exposure.csv")
OUT_PATH = os.path.join(OUTPUTS_DIR, "dashboard.html")

INPUT_FILES = [
    JSON_PATH,
    FORECAST_PATH,
    QUARTERLY_PATH,
    STATE_PATH,
    MONTE_CARLO_PATH,
    EXPOSURE_PATH,
]
REQUIRED_JSON_SECTIONS = ("regression", "arima", "monte_carlo", "future_projection_2026")


# ---------------------------------------------------------------------------
# Validation and formatting helpers
# ---------------------------------------------------------------------------

def fail(message):
    raise SystemExit("ERROR: " + message)


def esc(value):
    """Escape every value that is interpolated into the HTML document."""
    return html.escape(str(value), quote=True)


def require_inputs():
    missing = [path for path in INPUT_FILES if not os.path.isfile(path)]
    if missing:
        listing = "\n".join("  - " + path for path in missing)
        fail("missing required input file(s):\n" + listing)


def load_json():
    try:
        with open(JSON_PATH, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        fail("could not read combined_pipeline_results.json: " + str(exc))

    missing = [key for key in REQUIRED_JSON_SECTIONS if key not in payload]
    if missing:
        fail(
            "combined_pipeline_results.json is missing required section(s): "
            + ", ".join(missing)
        )
    return payload


def load_csv(path, required_columns):
    try:
        with open(path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = reader.fieldnames or []
            rows = list(reader)
    except OSError as exc:
        fail("could not read " + os.path.basename(path) + ": " + str(exc))

    missing = [column for column in required_columns if column not in columns]
    if missing:
        fail(
            os.path.basename(path)
            + " is missing required column(s): "
            + ", ".join(missing)
        )
    if not rows:
        fail(os.path.basename(path) + " contains no data rows")
    return rows


def as_float(row, column, source):
    try:
        return float(row[column])
    except (KeyError, TypeError, ValueError):
        fail("invalid numeric value in " + source + " column " + column)


def percentile(sorted_values, pct):
    """Linear-interpolated percentile on an already sorted list."""
    if not sorted_values:
        fail("cannot compute a percentile of an empty series")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * pct / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def nice_step(max_value, target_ticks=6):
    """A 1/2/2.5/5/10-step that covers max_value in roughly target_ticks steps."""
    if max_value <= 0:
        return 1.0
    rough = max_value / float(target_ticks)
    magnitude = 10 ** math.floor(math.log10(rough))
    for multiplier in (1, 2, 2.5, 5, 10):
        step = multiplier * magnitude
        if step >= rough:
            return step
    return 10 * magnitude


def money(value, unit="B", dp=1):
    scale = 1_000_000_000.0 if unit == "B" else 1_000_000.0
    return f"₹{value / scale:,.{dp}f} {unit}"


def money_range(low, high, unit="B", dp=1):
    return money(low, unit, dp) + " – " + money(high, unit, dp)


def number(value, dp=1):
    return f"{value:,.{dp}f}"


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

def build_quarterly():
    rows = load_csv(
        QUARTERLY_PATH,
        ["project_start_date", "total_project_cost_inr"],
    )
    observed = []
    for row in rows:
        date = (row["project_start_date"] or "").strip()
        try:
            year = int(date[0:4])
            month = int(date[5:7])
        except (TypeError, ValueError):
            fail("invalid project_start_date in quarterly_series.csv: " + date)
        quarter = (month - 1) // 3 + 1
        observed.append(
            {
                "date": date,
                "quarter": f"{year}-Q{quarter}",
                "value": as_float(row, "total_project_cost_inr", "quarterly_series.csv"),
            }
        )
    observed.sort(key=lambda item: item["date"])
    return observed


def build_forecast():
    rows = load_csv(
        FORECAST_PATH,
        [
            "quarter",
            "forecast_inr",
            "ci95_lower_inr",
            "ci95_upper_inr",
            "mc_p5_inr",
            "mc_p50_inr",
            "mc_p95_inr",
        ],
    )
    forecast = []
    for row in rows:
        forecast.append(
            {
                "quarter": (row["quarter"] or "").strip(),
                "forecast_inr": as_float(row, "forecast_inr", "forecast_quarterly.csv"),
                "ci95_lower_inr": as_float(
                    row, "ci95_lower_inr", "forecast_quarterly.csv"
                ),
                "ci95_upper_inr": as_float(
                    row, "ci95_upper_inr", "forecast_quarterly.csv"
                ),
                "mc_p5_inr": as_float(row, "mc_p5_inr", "forecast_quarterly.csv"),
                "mc_p50_inr": as_float(row, "mc_p50_inr", "forecast_quarterly.csv"),
                "mc_p95_inr": as_float(row, "mc_p95_inr", "forecast_quarterly.csv"),
            }
        )
    if not forecast:
        fail("forecast_quarterly.csv contains no forecast rows")
    return forecast


def build_states():
    rows = load_csv(
        STATE_PATH,
        [
            "state",
            "n_projects",
            "expected_total_cost_inr",
            "expected_project_cost_inr",
            "p5_total_inr",
            "p95_total_inr",
        ],
    )
    states = []
    for row in rows:
        states.append(
            {
                "state": (row["state"] or "").strip(),
                "n_projects": int(as_float(row, "n_projects", "future_projection_by_state_2026.csv")),
                "expected_total_cost_inr": as_float(
                    row, "expected_total_cost_inr", "future_projection_by_state_2026.csv"
                ),
                "expected_project_cost_inr": as_float(
                    row,
                    "expected_project_cost_inr",
                    "future_projection_by_state_2026.csv",
                ),
                "p5_total_inr": as_float(
                    row, "p5_total_inr", "future_projection_by_state_2026.csv"
                ),
                "p95_total_inr": as_float(
                    row, "p95_total_inr", "future_projection_by_state_2026.csv"
                ),
            }
        )
    states.sort(key=lambda item: item["expected_total_cost_inr"], reverse=True)
    return states


def build_costs():
    rows = load_csv(MONTE_CARLO_PATH, ["predicted_total_cost_inr"])
    costs = sorted(
        as_float(row, "predicted_total_cost_inr", "monte_carlo_projects.csv")
        for row in rows
    )
    if len(costs) < 2:
        fail("monte_carlo_projects.csv needs at least two rows")
    return costs


def build_exposure():
    rows = load_csv(EXPOSURE_PATH, ["state"])
    columns = list(rows[0].keys())
    if len(columns) < 2:
        fail("state_exposure.csv does not contain exposure feature columns")
    return columns, rows


# ---------------------------------------------------------------------------
# Inline SVG charts
# ---------------------------------------------------------------------------

def build_forecast_svg(observed, forecast):
    width, height = 1120, 470
    left, right, top, bottom = 78, 26, 26, 58
    plot_w = width - left - right
    plot_h = height - top - bottom
    n_points = len(observed) + len(forecast)
    step = plot_w / float(n_points)

    def sx(index):
        return left + step * (index + 0.5)

    def sy(value):
        return top + plot_h - (value / 1_000_000_000.0) / y_max * plot_h

    candidates = [item["value"] for item in observed]
    for item in forecast:
        candidates.extend(
            [
                item["forecast_inr"],
                item["ci95_upper_inr"],
                item["ci95_lower_inr"],
                item["mc_p5_inr"],
                item["mc_p95_inr"],
            ]
        )
    y_max = math.ceil((max(candidates) / 1_000_000_000.0) / 5.0) * 5.0
    if y_max <= 0:
        y_max = 5.0

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{esc("Quarterly observed cost and eight-quarter ARIMA forecast")}" '
        f'preserveAspectRatio="xMidYMid meet">'
    ]

    # Horizontal grid and y-axis ticks (INR billion).
    tick = 0
    while tick <= y_max + 1e-9:
        y = sy(tick * 1_000_000_000.0)
        parts.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" class="grid"/>'
        )
        parts.append(
            f'<text x="{left - 12}" y="{y + 4:.2f}" class="tick-label" '
            f'text-anchor="end">{esc(f"{tick:g}")}</text>'
        )
        tick += 5

    axis_mid = top + plot_h / 2.0
    parts.append(
        f'<text x="20" y="{axis_mid:.2f}" class="axis-title" text-anchor="middle" '
        f'transform="rotate(-90 20 {axis_mid:.2f})">{esc("INR billion")}</text>'
    )

    # Bands cover the forecast horizon only (2026-Q1 onwards).
    def band_polygon(high_key, low_key, css_class, title):
        high_points = []
        low_points = []
        for offset, item in enumerate(forecast):
            index = len(observed) + offset
            high_points.append(f"{sx(index):.2f},{sy(item[high_key]):.2f}")
            low_points.append(f"{sx(index):.2f},{sy(item[low_key]):.2f}")
        points = " ".join(high_points + list(reversed(low_points)))
        return (
            f'<polygon points="{points}" class="{css_class}">'
            f"<title>{esc(title)}</title></polygon>"
        )

    parts.append(
        band_polygon(
            "ci95_upper_inr",
            "ci95_lower_inr",
            "band-ci",
            "95% confidence interval for the ARIMA forecast",
        )
    )
    parts.append(
        band_polygon(
            "mc_p95_inr",
            "mc_p5_inr",
            "band-mc",
            "Monte Carlo p5–p95 band for forecast quarters",
        )
    )

    # Boundary between observed history and the forecast horizon.
    boundary = left + step * len(observed)
    parts.append(
        f'<line x1="{boundary:.2f}" y1="{top}" x2="{boundary:.2f}" '
        f'y2="{top + plot_h}" class="divider"/>'
    )
    parts.append(
        f'<text x="{boundary + 7:.2f}" y="{top + 14}" class="annotation">'
        f"{esc("Forecast horizon →")}</text>"
    )

    # Dashed connector from the final observation to the first forecast point.
    if observed and forecast:
        last_obs = observed[-1]["value"]
        first_fc = forecast[0]["forecast_inr"]
        parts.append(
            f'<line x1="{sx(len(observed) - 1):.2f}" y1="{sy(last_obs):.2f}" '
            f'x2="{sx(len(observed)):.2f}" y2="{sy(first_fc):.2f}" class="connector"/>'
        )

    observed_points = " ".join(
        f"{sx(index):.2f},{sy(item['value']):.2f}"
        for index, item in enumerate(observed)
    )
    if len(observed) > 1:
        parts.append(f'<polyline points="{observed_points}" class="line-obs"/>')

    forecast_points = " ".join(
        f"{sx(len(observed) + offset):.2f},{sy(item['forecast_inr']):.2f}"
        for offset, item in enumerate(forecast)
    )
    if len(forecast) > 1:
        parts.append(f'<polyline points="{forecast_points}" class="line-fc"/>')

    for index, item in enumerate(observed):
        title = f"{item['quarter']} observed: {money(item['value'])}"
        parts.append(
            f'<circle cx="{sx(index):.2f}" cy="{sy(item["value"]):.2f}" r="3.6" '
            f'class="pt-obs"><title>{esc(title)}</title></circle>'
        )

    for offset, item in enumerate(forecast):
        index = len(observed) + offset
        title = (
            f"{item['quarter']} forecast: {money(item['forecast_inr'])}; "
            f"95% CI {money_range(item['ci95_lower_inr'], item['ci95_upper_inr'])}; "
            f"MC p5–p95 {money_range(item['mc_p5_inr'], item['mc_p95_inr'])}"
        )
        parts.append(
            f'<circle cx="{sx(index):.2f}" cy="{sy(item["forecast_inr"]):.2f}" r="4.2" '
            f'class="pt-fc"><title>{esc(title)}</title></circle>'
        )

    base = top + plot_h
    parts.append(
        f'<line x1="{left}" y1="{base}" x2="{left + plot_w}" y2="{base}" class="axis-line"/>'
    )
    quarter_labels = [item["quarter"] for item in observed] + [
        item["quarter"] for item in forecast
    ]
    for index in range(n_points):
        parts.append(
            f'<line x1="{sx(index):.2f}" y1="{base}" x2="{sx(index):.2f}" '
            f'y2="{base + 4}" class="tick-line"/>'
        )
    for index, label in enumerate(quarter_labels):
        if label.endswith("-Q1"):
            parts.append(
                f'<text x="{sx(index):.2f}" y="{base + 23}" class="tick-label" '
                f'text-anchor="middle">{esc(label.split("-")[0])}</text>'
            )
    parts.append(
        f'<text x="{left + plot_w / 2.0:.2f}" y="{height - 12}" class="axis-title" '
        f'text-anchor="middle">{esc("Calendar quarter")}</text>'
    )

    parts.append("</svg>")
    return "".join(parts)


def build_histogram_svg(costs, p50, p95):
    width, height = 1120, 410
    left, right, top, bottom = 78, 26, 30, 62
    plot_w = width - left - right
    plot_h = height - top - bottom

    cap = percentile(costs, 99.5)
    low = costs[0]
    if cap <= low:
        cap = low + 1.0
    n_bins = 40
    bin_width = (cap - low) / float(n_bins)
    counts = [0] * n_bins
    for value in costs:
        if value <= low:
            index = 0
        else:
            index = int((value - low) / bin_width)
        if index >= n_bins:
            index = n_bins - 1
        counts[index] += 1
    above_cap = sum(1 for value in costs if value > cap)

    y_step = nice_step(max(counts), 5)
    y_max = y_step * math.ceil(max(counts) / y_step)
    if y_max <= 0:
        y_max = 1.0

    def sx(value):
        return left + (value / cap) * plot_w

    def sy(value):
        return top + plot_h - (value / y_max) * plot_h

    def fmt_bin(value):
        return f"₹{value / 1_000_000_000.0:,.2f} B"

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{esc("Histogram of 10,000 simulated project total costs")}" '
        f'preserveAspectRatio="xMidYMid meet">'
    ]

    # Horizontal grid and count ticks.
    tick = 0.0
    while tick <= y_max + 1e-9:
        y = sy(tick)
        parts.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" class="grid"/>'
        )
        parts.append(
            f'<text x="{left - 12}" y="{y + 4:.2f}" class="tick-label" '
            f'text-anchor="end">{esc(f"{tick:g}")}</text>'
        )
        tick += y_step

    axis_mid = top + plot_h / 2.0
    parts.append(
        f'<text x="20" y="{axis_mid:.2f}" class="axis-title" text-anchor="middle" '
        f'transform="rotate(-90 20 {axis_mid:.2f})">{esc("Projects (count)")}</text>'
    )

    # Bars. Draws above the p99.5 cap are folded into the final bin.
    for index, count in enumerate(counts):
        x1_value = low + index * bin_width
        x2_value = low + (index + 1) * bin_width
        x1 = sx(x1_value)
        x2 = sx(x2_value)
        bar_width = max(1.2, x2 - x1 - 1.0)
        y = sy(count)
        bar_height = sy(0) - y
        suffix = " (includes draws above the p99.5 cap)" if index == n_bins - 1 and above_cap else ""
        title = (
            f"{fmt_bin(x1_value)}–{fmt_bin(x2_value)}: "
            f"{count:,} projects{suffix}"
        )
        parts.append(
            f'<rect x="{x1 + 0.5:.2f}" y="{y:.2f}" width="{bar_width:.2f}" '
            f'height="{max(bar_height, 0):.2f}" class="bar">'
            f"<title>{esc(title)}</title></rect>"
        )

    base = top + plot_h
    parts.append(
        f'<line x1="{left}" y1="{base}" x2="{left + plot_w}" y2="{base}" class="axis-line"/>'
    )

    x_step = nice_step(cap, 7)
    tick = 0.0
    while tick <= cap + 1e-9:
        x = sx(tick)
        parts.append(
            f'<line x1="{x:.2f}" y1="{base}" x2="{x:.2f}" y2="{base + 4}" class="tick-line"/>'
        )
        parts.append(
            f'<text x="{x:.2f}" y="{base + 23}" class="tick-label" '
            f'text-anchor="middle">{esc(f"{tick:g}")}</text>'
        )
        tick += x_step

    parts.append(
        f'<text x="{left + plot_w / 2.0:.2f}" y="{height - 12}" class="axis-title" '
        f'text-anchor="middle">{esc("Predicted total project cost (INR billion)")}</text>'
    )

    # p50 and p95 markers.
    for pct_value, label, css_class, y_offset in (
        (p50, "p50", "marker-line", 14),
        (p95, "p95", "marker-line95", 30),
    ):
        x = sx(pct_value)
        parts.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{base}" class="{css_class}"/>'
        )
        parts.append(
            f'<text x="{x + 5:.2f}" y="{top + y_offset}" class="marker-label">'
            f"{esc(label + " " + money(pct_value, "M", 1))}</text>"
        )

    # Mark the p99.5 cap explicitly.
    cap_x = sx(cap)
    parts.append(
        f'<line x1="{cap_x:.2f}" y1="{top}" x2="{cap_x:.2f}" y2="{base}" class="cap-line"/>'
    )
    parts.append(
        f'<text x="{cap_x - 5:.2f}" y="{top + 12}" class="cap-label" text-anchor="end">'
        f"{esc("p99.5 cap = " + money(cap, "B", 2))}</text>"
    )

    parts.append("</svg>")
    return "".join(parts), cap, above_cap


def build_state_svg(states):
    width, height = 1120, 462
    left, right, top, bottom = 150, 30, 24, 58
    plot_w = width - left - right
    plot_h = height - top - bottom
    row_h = plot_h / float(len(states))

    x_max = math.ceil(max(item["p95_total_inr"] for item in states) / 50_000_000_000.0) * 50.0
    if x_max <= 0:
        x_max = 50.0

    def sx(value):
        return left + (value / 1_000_000_000.0) / x_max * plot_w

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{esc("Expected 2026 total cost by state with p5-p95 whiskers")}" '
        f'preserveAspectRatio="xMidYMid meet">'
    ]

    # Alternating row backgrounds.
    for index in range(len(states)):
        if index % 2 == 1:
            parts.append(
                f'<rect x="{left}" y="{top + index * row_h:.2f}" width="{plot_w}" '
                f'height="{row_h:.2f}" fill="#f7f9fc"/>'
            )

    # Vertical grid and x-axis ticks (INR billion).
    tick = 0.0
    while tick <= x_max + 1e-9:
        x = sx(tick * 1_000_000_000.0)
        parts.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" class="grid"/>'
        )
        parts.append(
            f'<text x="{x:.2f}" y="{top + plot_h + 21}" class="tick-label" '
            f'text-anchor="middle">{esc(f"{tick:g}")}</text>'
        )
        tick += 50

    axis_mid = left + plot_w / 2.0
    parts.append(
        f'<text x="{axis_mid:.2f}" y="{height - 10}" class="axis-title" '
        f'text-anchor="middle">{esc("Expected total cost, 2026 (INR billion)")}</text>'
    )

    base = top + plot_h
    parts.append(
        f'<line x1="{left}" y1="{base}" x2="{left + plot_w}" y2="{base}" class="axis-line"/>'
    )

    for index, item in enumerate(states):
        center = top + row_h * (index + 0.5)
        bar_y = center - 11
        bar_end = sx(item["expected_total_cost_inr"])
        p5_x = sx(item["p5_total_inr"])
        p95_x = sx(item["p95_total_inr"])

        parts.append(
            f'<rect x="{left}" y="{bar_y:.2f}" width="{max(bar_end - left, 1):.2f}" '
            f'height="22" rx="3" class="state-bar"/>'
        )
        # p5-p95 whisker, drawn on top of the bar.
        parts.append(
            f'<line x1="{p5_x:.2f}" y1="{center:.2f}" x2="{p95_x:.2f}" '
            f'y2="{center:.2f}" class="whisker"/>'
        )
        parts.append(
            f'<line x1="{p5_x:.2f}" y1="{center - 5:.2f}" x2="{p5_x:.2f}" '
            f'y2="{center + 5:.2f}" class="whisker-cap"/>'
        )
        parts.append(
            f'<line x1="{p95_x:.2f}" y1="{center - 5:.2f}" x2="{p95_x:.2f}" '
            f'y2="{center + 5:.2f}" class="whisker-cap"/>'
        )
        parts.append(
            f'<circle cx="{bar_end:.2f}" cy="{center:.2f}" r="3.2" class="expected-dot"/>'
        )
        parts.append(
            f'<text x="{left - 12}" y="{center + 4:.2f}" class="svg-label" '
            f'text-anchor="end">{esc(item["state"])}</text>'
        )
        parts.append(
            f'<text x="{bar_end - 8:.2f}" y="{center + 4:.2f}" class="value-label" '
            f'text-anchor="end">{esc(money(item["expected_total_cost_inr"]))}</text>'
        )
        title = (
            f"{item['state']}: expected {money(item['expected_total_cost_inr'])}; "
            f"p5 {money(item['p5_total_inr'])}; p95 {money(item['p95_total_inr'])}; "
            f"{item['n_projects']} projects"
        )
        # Invisible full-row hit area provides hover text for the row.
        parts.append(
            f'<rect x="{left}" y="{center - row_h / 2.0:.2f}" width="{plot_w}" '
            f'height="{row_h:.2f}" fill="transparent" pointer-events="all">'
            f"<title>{esc(title)}</title></rect>"
        )

    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# HTML section builders
# ---------------------------------------------------------------------------

def kpi_card(label, value, note, tone):
    return (
        f'<article class="kpi {tone}">'
        f'<p class="kpi-label">{esc(label)}</p>'
        f'<p class="kpi-value">{esc(value)}</p>'
        f'<p class="kpi-note">{esc(note)}</p>'
        f"</article>"
    )


def build_exposure_table(columns, rows):
    # Keep the exact CSV header text in the title attribute and wrap the visible
    # label after underscores so all twelve columns fit without horizontal
    # clipping. The <wbr> elements do not change the header's text content.
    head = "".join(
        f'<th title="{esc(column)}">'
        f"{esc(column).replace('_', '_<wbr>')}</th>"
        for column in columns
    )
    body_parts = []
    for row in rows:
        cells = []
        for column in columns:
            raw = row.get(column, "")
            try:
                rendered = f"{float(raw):.3f}"
            except (TypeError, ValueError):
                rendered = raw
            cells.append(f"<td>{esc(rendered)}</td>")
        body_parts.append("<tr>" + "".join(cells) + "</tr>")
    return (
        '<div class="table-wrap">'
        '<table class="exposure">'
        f"<thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(body_parts)}</tbody>"
        "</table></div>"
    )


def build_state_table(states):
    rows = []
    for item in states:
        rows.append(
            "<tr>"
            f"<td>{esc(item['state'])}</td>"
            f'<td class="num">{esc(f"{item['n_projects']:,}")}</td>'
            f'<td class="num">{esc(money(item["expected_total_cost_inr"]))}</td>'
            f'<td class="num">{esc(money(item["expected_project_cost_inr"], "M"))}</td>'
            f'<td class="num">{esc(money(item["p5_total_inr"]))}</td>'
            f'<td class="num">{esc(money(item["p95_total_inr"]))}</td>'
            "</tr>"
        )
    return (
        '<div class="table-wrap">'
        "<table>"
        "<thead><tr>"
        "<th>State</th>"
        '<th class="num">Projects</th>'
        '<th class="num">Expected total (INR B)</th>'
        '<th class="num">Expected per project (INR M)</th>'
        '<th class="num">p5 total (INR B)</th>'
        '<th class="num">p95 total (INR B)</th>'
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table></div>"
    )


def build_percentile_table(mc_summary):
    labels = ("p5", "p25", "p50", "p75", "p95")
    head = "".join(f"<th>{esc(label)}</th>" for label in labels)
    cells = "".join(
        f"<td>{esc(money(mc_summary[label], 'M'))}</td>" for label in labels
    )
    return (
        '<div class="table-wrap percentile-wrap">'
        "<table class=\"percentile\">"
        "<caption>Project total-cost percentiles (INR million; M = million)</caption>"
        f"<thead><tr>{head}</tr></thead>"
        f"<tbody><tr>{cells}</tr></tbody>"
        "</table></div>"
    )


CSS = """
:root {
  --bg: #f2f5f9;
  --card: #ffffff;
  --ink: #17212f;
  --muted: #5b6778;
  --line: #dfe6ef;
  --accent: #145da0;
  --teal: #0f766e;
  --warm: #b45309;
  --shadow: 0 1px 2px rgba(23, 33, 47, .04), 0 10px 30px rgba(23, 33, 47, .05);
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  font-size: 15px;
  line-height: 1.55;
  -webkit-font-smoothing: antialiased;
}
.wrap { max-width: 1200px; margin: 0 auto; padding: 36px 28px 48px; }
.hero { padding: 8px 0 24px; }
.eyebrow {
  margin: 0 0 8px;
  font-size: .78rem;
  letter-spacing: .13em;
  text-transform: uppercase;
  font-weight: 700;
  color: var(--accent);
}
h1 {
  margin: 0;
  font-size: clamp(2rem, 4vw, 2.9rem);
  letter-spacing: -.025em;
  line-height: 1.08;
}
.subtitle { margin: 12px 0 0; max-width: 860px; color: var(--muted); font-size: 1.03rem; }
.caveat {
  margin-top: 22px;
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 16px;
  padding: 15px 20px;
  background: #fff9eb;
  border: 1px solid #f0d49b;
  border-left: 5px solid #d99a1f;
  border-radius: 10px;
}
.caveat-title {
  font-size: .76rem;
  font-weight: 800;
  letter-spacing: .09em;
  text-transform: uppercase;
  color: #8a5a06;
  padding-top: 2px;
  white-space: nowrap;
}
.caveat ul { margin: 0; padding-left: 18px; color: #5f4a1e; font-size: .9rem; }
.caveat li + li { margin-top: 3px; }
.kpi-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 16px;
  margin: 0 0 26px;
}
.kpi {
  background: var(--card);
  border: 1px solid var(--line);
  border-top: 3px solid var(--accent);
  border-radius: 12px;
  padding: 17px 18px 15px;
  box-shadow: var(--shadow);
  display: flex;
  flex-direction: column;
  gap: 5px;
  min-height: 142px;
}
.kpi.tone-b { border-top-color: var(--teal); }
.kpi.tone-c { border-top-color: var(--warm); }
.kpi.tone-d { border-top-color: #6d4aa8; }
.kpi.tone-e { border-top-color: #2f7d4f; }
.kpi-label {
  margin: 0;
  font-size: .72rem;
  font-weight: 700;
  letter-spacing: .08em;
  text-transform: uppercase;
  color: var(--muted);
}
.kpi-value {
  margin: 2px 0 0;
  font-size: 1.85rem;
  line-height: 1.12;
  font-weight: 700;
  letter-spacing: -.02em;
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}
.kpi-note { margin: auto 0 0; font-size: .78rem; color: var(--muted); }
.card {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 24px;
  margin-bottom: 24px;
  box-shadow: var(--shadow);
}
.section-head {
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  gap: 20px;
  margin-bottom: 10px;
}
.section-kicker {
  margin: 0;
  font-size: .7rem;
  text-transform: uppercase;
  letter-spacing: .13em;
  font-weight: 800;
  color: var(--accent);
}
h2 { margin: 3px 0 0; font-size: 1.35rem; letter-spacing: -.01em; }
.section-note { margin: 0; color: var(--muted); font-size: .87rem; max-width: 540px; text-align: right; }
.legend {
  display: flex;
  flex-wrap: wrap;
  gap: 8px 20px;
  padding: 10px 2px 14px;
  font-size: .84rem;
  color: #3d4a5c;
}
.legend-item { display: inline-flex; align-items: center; gap: 7px; }
.sw { width: 24px; height: 4px; border-radius: 2px; display: inline-block; flex: 0 0 auto; }
.sw-obs { background: #145da0; }
.sw-fc { background: repeating-linear-gradient(90deg, #0f766e 0 5px, transparent 5px 9px); }
.sw-ci { height: 11px; border-radius: 2px; background: rgba(20, 93, 160, .15); border: 1px solid rgba(20, 93, 160, .28); }
.sw-mc { height: 11px; border-radius: 2px; background: rgba(217, 119, 6, .16); border: 1px solid rgba(217, 119, 6, .3); }
.sw-bar { height: 11px; background: #2f6fb0; border-radius: 2px; }
.sw-whisker {
  height: 12px;
  background: linear-gradient(#1e293b, #1e293b) center / 100% 1.5px no-repeat;
  border-left: 1px solid #1e293b;
  border-right: 1px solid #1e293b;
}
.chart { margin: 0 -4px; }
.chart svg { width: 100%; height: auto; display: block; overflow: visible; }
.chart svg text { font-family: inherit; }
.grid { stroke: #e4eaf2; stroke-width: 1; }
.axis-line { stroke: #9aa7b8; stroke-width: 1.2; }
.tick-line { stroke: #9aa7b8; stroke-width: 1; }
.tick-label { fill: #64748b; font-size: 12px; }
.axis-title { fill: #64748b; font-size: 12px; font-weight: 600; letter-spacing: .03em; }
.band-ci { fill: rgba(20, 93, 160, .10); stroke: rgba(20, 93, 160, .55); stroke-width: 1; }
.band-mc { fill: rgba(217, 119, 6, .12); stroke: rgba(217, 119, 6, .75); stroke-width: 1; }
.divider { stroke: #94a3b8; stroke-width: 1; stroke-dasharray: 3 4; }
.annotation { fill: #475569; font-size: 11.5px; font-style: italic; }
.line-obs { fill: none; stroke: #145da0; stroke-width: 2.4; stroke-linejoin: round; stroke-linecap: round; }
.line-fc { fill: none; stroke: #0f766e; stroke-width: 2.2; stroke-dasharray: 7 5; stroke-linejoin: round; }
.connector { fill: none; stroke: #0f766e; stroke-width: 1.6; stroke-dasharray: 4 5; opacity: .8; }
.pt-obs { fill: #ffffff; stroke: #145da0; stroke-width: 2; }
.pt-fc { fill: #0f766e; stroke: #ffffff; stroke-width: 2; }
.bar { fill: #2f6fb0; }
.bar:hover { fill: #145da0; }
.marker-line { stroke: #b45309; stroke-width: 1.4; stroke-dasharray: 5 4; }
.marker-line95 { stroke: #b91c1c; stroke-width: 1.4; stroke-dasharray: 5 4; }
.marker-label { fill: #7c2d12; font-size: 11.5px; font-weight: 700; }
.cap-line { stroke: #94a3b8; stroke-width: 1; stroke-dasharray: 2 4; }
.cap-label { fill: #64748b; font-size: 11px; font-style: italic; }
.state-bar { fill: #2f6fb0; }
.whisker { stroke: #1e293b; stroke-width: 1.4; }
.whisker-cap { stroke: #1e293b; stroke-width: 1.4; }
.expected-dot { fill: #ffffff; stroke: #1e293b; stroke-width: 1.5; }
.svg-label { fill: #334155; font-size: 12.5px; }
.value-label { fill: #ffffff; font-size: 11.5px; font-weight: 700; }
.caption { margin: 10px 2px 0; color: var(--muted); font-size: .82rem; }
.note {
  margin: 14px 0 0;
  background: #f7f9fc;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 10px 14px;
  font-size: .83rem;
  color: var(--muted);
}
.table-wrap { overflow-x: auto; margin-top: 16px; }
table { width: 100%; border-collapse: collapse; font-size: .86rem; font-variant-numeric: tabular-nums; }
th, td { padding: 8px 10px; border-bottom: 1px solid var(--line); text-align: left; white-space: nowrap; }
th {
  background: #f7f9fc;
  color: #445164;
  font-weight: 700;
  font-size: .74rem;
  text-transform: uppercase;
  letter-spacing: .04em;
}
tbody tr:hover { background: #f8fafc; }
.num { text-align: right; }
.exposure {
  width: 100%;
  table-layout: fixed;
  font-size: .72rem;
}
.exposure th,
.exposure td { padding: 6px; vertical-align: bottom; }
.exposure th {
  white-space: normal;
  overflow-wrap: normal;
  word-break: normal;
  line-height: 1.2;
  font-size: .68rem;
  letter-spacing: .02em;
}
.exposure td { white-space: normal; }
.exposure th:first-child,
.exposure td:first-child { width: 12%; }
.percentile-wrap { max-width: 760px; }
.percentile caption {
  caption-side: top;
  text-align: left;
  padding: 0 0 7px;
  color: var(--muted);
  font-size: .8rem;
}
.percentile th, .percentile td { text-align: right; }
.percentile th:first-child, .percentile td:first-child { text-align: left; }
.footer {
  margin-top: 30px;
  border-top: 1px solid var(--line);
  padding-top: 22px;
  display: grid;
  grid-template-columns: 1.25fr 1.25fr 1fr;
  gap: 26px;
  color: var(--muted);
  font-size: .84rem;
}
.footer h3 {
  margin: 0 0 7px;
  color: var(--ink);
  font-size: .76rem;
  letter-spacing: .09em;
  text-transform: uppercase;
}
.footer ul { margin: 0; padding-left: 17px; }
.footer li + li { margin-top: 4px; }
.footer code {
  background: #eef2f7;
  color: #334155;
  padding: 1px 5px;
  border-radius: 4px;
  font-size: .78rem;
}
.offline-note { margin: 12px 0 0; font-size: .8rem; }
@media (max-width: 760px) {
  .wrap { padding: 24px 14px 36px; }
  .card { padding: 18px 14px; }
  .section-head { display: block; }
  .section-note { text-align: left; margin-top: 8px; }
  .caveat { grid-template-columns: 1fr; gap: 6px; }
  .footer { grid-template-columns: 1fr; }
  .kpi-value { font-size: 1.6rem; }
}
"""


def main():
    require_inputs()
    results = load_json()
    observed = build_quarterly()
    forecast = build_forecast()
    states = build_states()
    costs = build_costs()
    exposure_columns, exposure_rows = build_exposure()

    regression = results["regression"]
    arima = results["arima"]
    monte_carlo = results["monte_carlo"]
    projection = results["future_projection_2026"]

    mc_summary = monte_carlo["total_cost_inr"]
    p50 = float(mc_summary["p50"])
    p95 = float(mc_summary["p95"])

    order = "(" + ", ".join(str(int(part)) for part in arima["best_order"]) + ")"
    forecast_svg = build_forecast_svg(observed, forecast)
    histogram_svg, cap, above_cap = build_histogram_svg(costs, p50, p95)
    state_svg = build_state_svg(states)

    generated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    caveats = results.get("caveats") or [
        "BOQ costs are synthetic, not real projects.",
        "State assignment is nearest-centroid; assets spanning states are approximated.",
        "Exposure features are state-level averages, not asset-specific.",
    ]
    caveat_items = "".join(f"<li>{esc(item)}</li>" for item in caveats)

    portfolio = float(projection["portfolio_expected_total_inr"])
    portfolio_p5 = float(projection["portfolio_p5_total_inr"])
    portfolio_p95 = float(projection["portfolio_p95_total_inr"])

    kpis = "".join(
        [
            kpi_card(
                "Regression fit (INR cost)",
                f"{float(regression['r2_cost']):.4f}",
                "R² on the INR cost scale; log-target R² "
                + f"{float(regression['r2_log']):.4f}"
                + f" (n={int(regression['n_test'])} test rows)",
                "tone-a",
            ),
            kpi_card(
                "MAPE",
                f"{float(regression['mape_pct']):.2f}%",
                "Mean absolute percentage error on the test set",
                "tone-c",
            ),
            kpi_card(
                "Best ARIMA order",
                order,
                f"AIC {float(arima['aic']):.2f} · selected by AIC",
                "tone-b",
            ),
            kpi_card(
                "Monte Carlo median project",
                money(p50, "M"),
                f"10,000 draws · p95 {money(p95, 'M')}",
                "tone-d",
            ),
            kpi_card(
                "2026 portfolio expected total",
                money(portfolio, "B"),
                "p5–p95: " + money_range(portfolio_p5, portfolio_p95),
                "tone-e",
            ),
        ]
    )

    source_files = [
        "combined_pipeline_results.json",
        "forecast_quarterly.csv",
        "quarterly_series.csv",
        "future_projection_by_state_2026.csv",
        "monte_carlo_projects.csv",
        "state_exposure.csv",
    ]
    source_items = "".join(f"<li><code>{esc(name)}</code></li>" for name in source_files)

    percentile_table = build_percentile_table(mc_summary)
    state_table = build_state_table(states)
    exposure_table = build_exposure_table(exposure_columns, exposure_rows)

    histogram_note = (
        f"Histogram uses 40 equal-width bins from {money(costs[0], 'M', 2)} to the "
        f"p99.5 cap of {money(cap, 'B', 2)}. {above_cap:,} of {len(costs):,} draws above "
        "the cap are folded into the final bin; the x-axis is capped at p99.5."
    )

    document = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc("Indian Construction Cost Pipeline")}</title>
<style>
{CSS}
</style>
</head>
<body>
<div class="wrap">
<header class="hero">
  <p class="eyebrow">{esc("Consolidated pipeline · 2026 outlook")}</p>
  <h1>{esc("Indian Construction Cost Pipeline")}</h1>
  <p class="subtitle">{esc(
      f"Regression, ARIMA and Monte Carlo results for {int(results['n_projects'])} synthetic "
      "BOQ projects, joined to state-level exposure features, with a 2026 state and "
      "portfolio projection."
  )}</p>
  <div class="caveat" role="note">
    <div class="caveat-title">{esc("Data caveats")}</div>
    <ul>{caveat_items}</ul>
  </div>
</header>

<main>
<section class="kpi-grid" aria-label="Key performance indicators">
{kpis}
</section>

<section class="card">
  <div class="section-head">
    <div>
      <p class="section-kicker">{esc("Section 3")}</p>
      <h2>{esc("Quarterly forecast")}</h2>
    </div>
    <p class="section-note">{esc("Observed 2021-Q1 to 2025-Q4; eight-quarter ARIMA forecast for 2026-Q1 to 2027-Q4. Amounts in INR billion.")}</p>
  </div>
  <div class="legend" aria-label="Chart legend">
    <span class="legend-item"><span class="sw sw-obs"></span>{esc("Observed quarterly cost")}</span>
    <span class="legend-item"><span class="sw sw-fc"></span>{esc("ARIMA forecast")}</span>
    <span class="legend-item"><span class="sw sw-ci"></span>{esc("95% confidence interval")}</span>
    <span class="legend-item"><span class="sw sw-mc"></span>{esc("Monte Carlo p5–p95 band")}</span>
  </div>
  <div class="chart">{forecast_svg}</div>
  <p class="caption">{esc("Hover any point for its exact value. The dashed connector links the last observed quarter to the first forecast quarter; bands cover the forecast horizon only.")}</p>
</section>

<section class="card">
  <div class="section-head">
    <div>
      <p class="section-kicker">{esc("Section 4")}</p>
      <h2>{esc("Monte Carlo project-cost distribution")}</h2>
    </div>
    <p class="section-note">{esc(f"{len(costs):,} simulated projects · 40 bins · x-axis capped at p99.5")}</p>
  </div>
  <div class="legend" aria-label="Chart legend">
    <span class="legend-item"><span class="sw sw-bar"></span>{esc("Simulated projects per bin")}</span>
    <span class="legend-item"><span class="sw sw-whisker"></span>{esc("p50 / p95 percentile markers")}</span>
  </div>
  <div class="chart">{histogram_svg}</div>
  <p class="caption">{esc(histogram_note)}</p>
  {percentile_table}
</section>

<section class="card">
  <div class="section-head">
    <div>
      <p class="section-kicker">{esc("Section 5")}</p>
      <h2>{esc("2026 state projection")}</h2>
    </div>
    <p class="section-note">{esc("Expected 2026 total cost by state, sorted high to low. Bars show the expected total; whiskers span p5–p95.")}</p>
  </div>
  <div class="legend" aria-label="Chart legend">
    <span class="legend-item"><span class="sw sw-bar"></span>{esc("Expected total")}</span>
    <span class="legend-item"><span class="sw sw-whisker"></span>{esc("p5–p95 interval")}</span>
  </div>
  <div class="chart">{state_svg}</div>
  <p class="caption">{esc("Hover a row for its state-level expected total, interval and project count. Amounts in INR billion.")}</p>
  {state_table}
</section>

<section class="card">
  <div class="section-head">
    <div>
      <p class="section-kicker">{esc("Section 6")}</p>
      <h2>{esc("State exposure features")}</h2>
    </div>
    <p class="section-note">{esc("All 10 states and all exposure columns from state_exposure.csv, rounded to 3 decimals.")}</p>
  </div>
  {exposure_table}
  <p class="note">{esc("Exposure features are state-level averages; exp_n_assets is a state asset count. These values are joined to BOQ projects by nearest-centroid state.")}</p>
</section>
</main>

<footer class="footer">
  <div>
    <h3>{esc("Data sources")}</h3>
    <ul>{source_items}</ul>
  </div>
  <div>
    <h3>{esc("Method summary")}</h3>
    <ul>
      <li>{esc("Join key = state (nearest-centroid assignment).")}</li>
      <li>{esc("Regression target = log1p(total cost).")}</li>
      <li>{esc("ARIMA order search by AIC.")}</li>
      <li>{esc("Monte Carlo: 10,000 draws.")}</li>
    </ul>
  </div>
  <div>
    <h3>{esc("Generation")}</h3>
    <p>{esc("Generated " + generated_at)}</p>
    <p>{esc("Units: M = INR million; B = INR billion. All amounts nominal INR.")}</p>
    <p class="offline-note">{esc("Static offline HTML with inline CSS and inline SVG. No network requests.")}</p>
  </div>
</footer>
</div>
</body>
</html>
"""

    with open(OUT_PATH, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(document)

    size = os.path.getsize(OUT_PATH)
    print("Wrote " + OUT_PATH)
    print("Size: " + f"{size:,}" + " bytes")


if __name__ == "__main__":
    main()
