#!/usr/bin/env python3
import csv
import math
from pathlib import Path


CSV_PATH = Path("/home/joyce/f1sim/results/parameter_study_results.csv")
OUT_PATH = Path("/home/joyce/f1sim/results/parameter_study_lap_time_chart.svg")
CAPTION_PATH = Path("/home/joyce/f1sim/results/parameter_study_lap_time_chart_caption.txt")


def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def y_of(value: float, top: int, height: int, min_v: int, max_v: int) -> float:
    span = max_v - min_v
    return top + (max_v - value) / span * height


def main() -> None:
    rows = []
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row["id"], row["title"], float(row["lap_time_text"])))

    baseline = next(value for group_id, _, value in rows if group_id == "g01_baseline_fast")
    best_id, _, best_value = min(rows, key=lambda item: item[2])

    width, height = 1400, 900
    margin_left, margin_right = 120, 60
    margin_top, margin_bottom = 90, 170
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom

    min_value = math.floor(min(value for _, _, value in rows) - 1)
    max_value = math.ceil(max(value for _, _, value in rows) + 1)
    bar_gap = 18
    bar_width = (chart_width - bar_gap * (len(rows) - 1)) / len(rows)

    parts = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
    )
    parts.append('<rect width="100%" height="100%" fill="white"/>')
    parts.append(
        '<text x="700" y="42" text-anchor="middle" '
        'font-family="Arial, Helvetica, sans-serif" font-size="28" font-weight="700">'
        'Lap Time Comparison for 10 Parameter Groups</text>'
    )
    parts.append(
        '<text x="700" y="70" text-anchor="middle" '
        'font-family="Arial, Helvetica, sans-serif" font-size="16" fill="#444">'
        'Spielberg map, one clean-restart run per group; lower is better</text>'
    )

    for tick in range(min_value, max_value + 1):
        y = y_of(tick, margin_top, chart_height, min_value, max_value)
        color = "#d8d8d8" if tick != baseline else "#c9b26b"
        stroke = 1 if tick != baseline else 2
        parts.append(
            f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" '
            f'stroke="{color}" stroke-width="{stroke}"/>'
        )
        parts.append(
            f'<text x="{margin_left - 14}" y="{y + 5:.1f}" text-anchor="end" '
            'font-family="Arial, Helvetica, sans-serif" font-size="15" fill="#333">'
            f"{tick}</text>"
        )

    parts.append(
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" '
        'stroke="#222" stroke-width="2"/>'
    )
    parts.append(
        f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" '
        'stroke="#222" stroke-width="2"/>'
    )
    axis_y = margin_top + chart_height / 2
    parts.append(
        f'<text x="38" y="{axis_y:.1f}" transform="rotate(-90 38 {axis_y:.1f})" '
        'text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="18" font-weight="700">'
        'Lap Time (s)</text>'
    )

    for index, (group_id, _, value) in enumerate(rows):
        x = margin_left + index * (bar_width + bar_gap)
        y = y_of(value, margin_top, chart_height, min_value, max_value)
        bar_height = height - margin_bottom - y

        fill = "#d1d5db"
        edge = "#6b7280"
        if group_id == "g01_baseline_fast":
            fill = "#eab308"
            edge = "#a16207"
        if group_id == best_id:
            fill = "#10b981"
            edge = "#047857"

        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" '
            f'fill="{fill}" stroke="{edge}" stroke-width="2" rx="4"/>'
        )
        parts.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{y - 10:.1f}" text-anchor="middle" '
            'font-family="Arial, Helvetica, sans-serif" font-size="16" font-weight="700" fill="#111">'
            f"{value:.2f}</text>"
        )
        label_x = x + bar_width / 2
        label_y = height - margin_bottom + 26
        parts.append(
            f'<text x="{label_x:.1f}" y="{label_y:.1f}" text-anchor="end" '
            f'transform="rotate(-35 {label_x:.1f} {label_y:.1f})" '
            'font-family="Arial, Helvetica, sans-serif" font-size="14" fill="#222">'
            f"{esc(group_id)}</text>"
        )

    legend_x, legend_y = width - 360, 100
    legend = [
        ("#10b981", "Best result"),
        ("#eab308", "Baseline"),
        ("#d1d5db", "Other groups"),
    ]
    for offset, (color, label) in enumerate(legend):
        y = legend_y + offset * 28
        parts.append(
            f'<rect x="{legend_x}" y="{y}" width="18" height="18" fill="{color}" '
            'stroke="#555" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{legend_x + 28}" y="{y + 14}" font-family="Arial, Helvetica, sans-serif" '
            f'font-size="15" fill="#222">{esc(label)}</text>'
        )

    baseline_y = y_of(baseline, margin_top, chart_height, min_value, max_value)
    parts.append(
        f'<text x="{width - margin_right - 8}" y="{baseline_y - 8:.1f}" text-anchor="end" '
        'font-family="Arial, Helvetica, sans-serif" font-size="14" fill="#8a6a00">'
        f"Baseline = {baseline:.2f}s</text>"
    )
    parts.append("</svg>")

    OUT_PATH.write_text("\n".join(parts), encoding="utf-8")
    CAPTION_PATH.write_text(
        "Figure. One-lap performance comparison of 10 race_planner parameter groups on the Spielberg map. "
        "Lower lap time indicates better performance. The baseline group is highlighted in gold, and the "
        "best-performing group is highlighted in green.",
        encoding="utf-8",
    )

    print(OUT_PATH)
    print(CAPTION_PATH)


if __name__ == "__main__":
    main()
