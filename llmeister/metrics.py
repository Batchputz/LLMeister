"""LLMeister metrics — parse vLLM engine statistics from launch logs.

Extracts the periodic engine-statistics lines that vLLM prints (~every 5s):
  Engine 000: Avg prompt throughput: 1115.4 tokens/s, Avg generation throughput: 201.2 tokens/s,
  Running: 32 reqs, Waiting: 100 reqs, GPU KV cache usage: 61.2%,
  Prefix cache hit rate: 22.5%, MM cache hit rate: 74.5%

Returns structured JSON for the dashboard's Plotly chart modal.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

# Metric specs — key, display label, unit, chart color, Y-axis ('y'=left, 'y2'=right)
METRICS: list[dict[str, str]] = [
    {"key": "prompt",     "label": "Avg prompt throughput",     "unit": "tokens/s", "color": "#4f46e5", "axis": "y"},
    {"key": "generation", "label": "Avg generation throughput", "unit": "tokens/s", "color": "#2563eb", "axis": "y"},
    {"key": "running",    "label": "Running reqs",              "unit": "reqs",     "color": "#059669", "axis": "y2"},
    {"key": "waiting",    "label": "Waiting reqs",              "unit": "reqs",     "color": "#d97706", "axis": "y2"},
    {"key": "gpu_kv",     "label": "GPU KV cache usage",        "unit": "%",        "color": "#e11d48", "axis": "y2"},
    {"key": "prefix",     "label": "Prefix cache hit rate",     "unit": "%",        "color": "#0891b2", "axis": "y2"},
    {"key": "mm",         "label": "MM cache hit rate",         "unit": "%",        "color": "#7c3aed", "axis": "y2"},
]
METRIC_KEYS = [m["key"] for m in METRICS]

LINE_RE = re.compile(
    r"(?P<month>\d{2})-(?P<day>\d{2})\s+"
    r"(?P<hour>\d{2}):(?P<minute>\d{2}):(?P<second>\d{2})"
    r".*?Avg prompt throughput:\s*(?P<prompt>[\d.]+)\s*tokens/s"
    r".*?Avg generation throughput:\s*(?P<generation>[\d.]+)\s*tokens/s"
    r".*?Running:\s*(?P<running>\d+)\s*reqs"
    r".*?Waiting:\s*(?P<waiting>\d+)\s*reqs"
    r".*?GPU KV cache usage:\s*(?P<gpu_kv>[\d.]+)%"
    r".*?Prefix cache hit rate:\s*(?P<prefix>[\d.]+)%"
    r"(?:.*?MM cache hit rate:\s*(?P<mm>[\d.]+)%)?"  # MM is optional (not all models)
)


def parse_engine_stats(log_path: Path, max_points: int = 0) -> dict[str, Any]:
    """Parse vLLM engine-statistics lines from a launch log.

    Returns:
        {
            "timestamps": ["2026-07-14T10:00:00", ...],
            "metrics": [{"key", "label", "unit", "color", "axis", "values": [...]}, ...],
            "summary": {"key": {"latest", "min", "max", "avg"}, ...},
            "sample_count": int,
            "time_start": str | None,
            "time_end": str | None,
        }
    """
    points: list[tuple[str, dict[str, float]]] = []
    prev_month: int | None = None
    year = datetime.now().year

    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = LINE_RE.search(line)
                if not m:
                    continue
                month = int(m["month"])
                if prev_month is not None and month < prev_month and (prev_month - month) >= 6:
                    year += 1
                dt = datetime(year, month, int(m["day"]),
                              int(m["hour"]), int(m["minute"]), int(m["second"]))
                prev_month = month
                rec = {}
                for k in METRIC_KEYS:
                    val = m.group(k)
                    rec[k] = float(val) if val else 0.0
                points.append((dt.strftime("%Y-%m-%dT%H:%M:%S"), rec))
    except (OSError, IOError):
        return {"timestamps": [], "metrics": [], "summary": {}, "sample_count": 0,
                "time_start": None, "time_end": None}

    # Downsample if needed
    if max_points > 0 and len(points) > max_points:
        step = len(points) / max_points
        idxs = sorted({int(i * step) for i in range(max_points)} | {0, len(points) - 1})
        points = [points[i] for i in idxs]

    timestamps = [p[0] for p in points]
    metrics_out = []
    summary = {}
    for spec in METRICS:
        vals = [p[1][spec["key"]] for p in points]
        metrics_out.append({**spec, "values": vals})
        if vals:
            summary[spec["key"]] = {
                "latest": vals[-1],
                "min": min(vals),
                "max": max(vals),
                "avg": round(sum(vals) / len(vals), 1),
            }

    return {
        "timestamps": timestamps,
        "metrics": metrics_out,
        "summary": summary,
        "sample_count": len(points),
        "time_start": timestamps[0] if timestamps else None,
        "time_end": timestamps[-1] if timestamps else None,
    }
