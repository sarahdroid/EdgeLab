"""edgelab.charts: dependency-free SVG chart rendering (standard library only).

Provides line charts (linear or log y-axis) and bar charts as standalone SVG
strings, plus small helpers for tick generation, number formatting and
downsampling long series so files stay small.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple
from xml.sax.saxutils import escape

PALETTE: Tuple[str, ...] = (
    "#2563eb",  # blue
    "#dc2626",  # red
    "#16a34a",  # green
    "#9333ea",  # purple
    "#ea580c",  # orange
    "#0891b2",  # cyan
    "#a16207",  # amber
    "#475569",  # slate
)
TEXT_COLOR = "#1f2937"
MUTED_TEXT_COLOR = "#6b7280"
GRID_COLOR = "#e5e7eb"
AXIS_COLOR = "#9ca3af"
FONT_FAMILY = "Helvetica, Arial, sans-serif"

ValueFormatter = Callable[[float], str]


@dataclass(frozen=True)
class ChartSeries:
    """One line on a line chart.

    Attributes:
        label: Legend text.
        x_values: X coordinates (same length as ``y_values``).
        y_values: Y coordinates.
        color: Optional CSS color; defaults to the palette by series index.
        mark_end: Draw an x marker on the final point (used to flag ruin).
    """

    label: str
    x_values: Sequence[float]
    y_values: Sequence[float]
    color: Optional[str] = None
    mark_end: bool = False


# ---------------------------------------------------------------------------
# Number formatting
# ---------------------------------------------------------------------------

def _trim_number(value: float, decimals: int = 1) -> str:
    """Format with up to ``decimals`` places, dropping trailing zeros."""
    text = f"{value:.{decimals}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def format_money_compact(value: float) -> str:
    """Compact dollar text: 950 -> $950, 1_500_000 -> $1.5M, 1e20 -> $1.0e+20."""
    sign = "-" if value < 0 else ""
    magnitude = abs(value)
    if magnitude >= 1e15:
        return f"{sign}${magnitude:.1e}"
    for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if magnitude >= threshold:
            return f"{sign}${_trim_number(magnitude / threshold)}{suffix}"
    return f"{sign}${_trim_number(magnitude, 2)}"


def format_count_compact(value: float) -> str:
    """Compact count text: 50_000 -> 50K, 2_500 -> 2.5K, 750 -> 750."""
    magnitude = abs(value)
    if magnitude >= 1e6:
        return f"{_trim_number(value / 1e6)}M"
    if magnitude >= 1e3:
        return f"{_trim_number(value / 1e3)}K"
    return _trim_number(value, 0)


def format_percent_axis(value: float) -> str:
    """Format a number that is already a percentage: 12.5 -> 12.5%."""
    return f"{_trim_number(value, 2)}%"


# ---------------------------------------------------------------------------
# Ticks, scales and downsampling
# ---------------------------------------------------------------------------

def nice_ticks(minimum: float, maximum: float, target_count: int = 6) -> List[float]:
    """Round-number tick values covering [minimum, maximum]."""
    if target_count < 2:
        raise ValueError("target_count must be at least 2.")
    if maximum < minimum:
        raise ValueError("maximum must not be below minimum.")
    if maximum == minimum:
        maximum = minimum + 1.0
    raw_step = (maximum - minimum) / target_count
    magnitude = 10.0 ** math.floor(math.log10(raw_step))
    step = magnitude * 10.0
    for multiplier in (1.0, 2.0, 2.5, 5.0, 10.0):
        if multiplier * magnitude >= raw_step:
            step = multiplier * magnitude
            break
    first_tick = math.floor(minimum / step) * step
    ticks: List[float] = []
    tick_index = 0
    while True:
        tick_value = round(first_tick + tick_index * step, 10)
        ticks.append(tick_value)
        if tick_value >= maximum - 1e-12 * step:
            break
        tick_index += 1
    return ticks


def log_tick_exponents(minimum: float, maximum: float, log_floor: float = 1.0) -> List[int]:
    """Decade exponents covering [minimum, maximum] on a base-10 log axis.

    Values at or below ``log_floor`` are treated as ``log_floor``. When the
    range spans many decades, exponents are thinned to at most about nine.
    """
    if log_floor <= 0:
        raise ValueError("log_floor must be positive.")
    clipped_minimum = max(minimum, log_floor)
    clipped_maximum = max(maximum, log_floor)
    low_exponent = math.floor(math.log10(clipped_minimum))
    high_exponent = math.ceil(math.log10(clipped_maximum))
    if high_exponent == low_exponent:
        high_exponent += 1
    stride = max(1, math.ceil((high_exponent - low_exponent) / 8))
    return list(range(low_exponent, high_exponent + 1, stride))


def downsample_series(
    x_values: Sequence[float],
    y_values: Sequence[float],
    maximum_points: int,
    keep_peaks: bool = False,
) -> Tuple[List[float], List[float]]:
    """Reduce a series to at most ``maximum_points`` points.

    The first and last points are always kept. With ``keep_peaks`` each bucket
    contributes its highest point (useful for drawdown so spikes survive);
    otherwise points are taken at an even stride.
    """
    if len(x_values) != len(y_values):
        raise ValueError("x_values and y_values must have the same length.")
    if maximum_points < 3:
        raise ValueError("maximum_points must be at least 3.")
    point_count = len(x_values)
    if point_count <= maximum_points:
        return list(x_values), list(y_values)

    selected_indices = {0, point_count - 1}
    if keep_peaks:
        bucket_count = maximum_points - 2
        for bucket_index in range(bucket_count):
            start = 1 + (bucket_index * (point_count - 2)) // bucket_count
            stop = 1 + ((bucket_index + 1) * (point_count - 2)) // bucket_count
            if stop <= start:
                continue
            peak_index = max(range(start, stop), key=lambda index: y_values[index])
            selected_indices.add(peak_index)
    else:
        for step_index in range(maximum_points):
            selected_indices.add(round(step_index * (point_count - 1) / (maximum_points - 1)))
    ordered_indices = sorted(selected_indices)
    return (
        [x_values[index] for index in ordered_indices],
        [y_values[index] for index in ordered_indices],
    )


@dataclass(frozen=True)
class _Axis:
    """Maps data values to pixel positions along one axis."""

    minimum: float  # data value (linear) or exponent (log)
    maximum: float
    pixel_start: float
    pixel_end: float
    is_log: bool = False
    log_floor: float = 1.0

    def position(self, value: float) -> float:
        """Pixel coordinate of a data value."""
        transformed = math.log10(max(value, self.log_floor)) if self.is_log else value
        fraction = (transformed - self.minimum) / (self.maximum - self.minimum)
        return self.pixel_start + fraction * (self.pixel_end - self.pixel_start)


def _build_axis(
    values: Sequence[float],
    is_log: bool,
    log_floor: float,
    pixel_start: float,
    pixel_end: float,
    include_zero: bool,
) -> Tuple[_Axis, List[float]]:
    """Create an axis spanning ``values`` and return it with its tick values."""
    if is_log:
        exponents = log_tick_exponents(min(values), max(values), log_floor)
        low_exponent = exponents[0]
        high_exponent = max(exponents[-1], math.ceil(math.log10(max(max(values), log_floor))))
        if high_exponent == low_exponent:
            high_exponent += 1
        axis = _Axis(low_exponent, high_exponent, pixel_start, pixel_end, True, log_floor)
        return axis, [10.0 ** exponent for exponent in exponents]
    low_value = min(values)
    high_value = max(values)
    if include_zero:
        low_value = min(low_value, 0.0)
        high_value = max(high_value, 0.0)
    ticks = nice_ticks(low_value, high_value)
    return _Axis(ticks[0], ticks[-1], pixel_start, pixel_end), ticks


# ---------------------------------------------------------------------------
# SVG assembly helpers
# ---------------------------------------------------------------------------

def _text(
    x: float,
    y: float,
    content: str,
    size: int = 12,
    anchor: str = "start",
    color: str = TEXT_COLOR,
    weight: str = "normal",
    rotate: Optional[float] = None,
) -> str:
    """Return an SVG <text> element with escaped content."""
    transform = f' transform="rotate({rotate} {x:.1f} {y:.1f})"' if rotate is not None else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" '
        f'fill="{color}" font-weight="{weight}"{transform}>{escape(content)}</text>'
    )


def _svg_header(width: int, height: int, title: str) -> List[str]:
    """Opening tags, background and accessible title."""
    return [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
            f'width="{width}" height="{height}" role="img" font-family="{FONT_FAMILY}">'
        ),
        f"<title>{escape(title)}</title>",
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
    ]


def _title_block(width: int, title: str, subtitle: str) -> List[str]:
    """Chart title and optional subtitle."""
    parts = [_text(30, 36, title, size=20, weight="bold")]
    if subtitle:
        parts.append(_text(30, 58, subtitle, size=13, color=MUTED_TEXT_COLOR))
    return parts


def _points_attribute(points: Sequence[Tuple[float, float]]) -> str:
    """Format polyline coordinates."""
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


# ---------------------------------------------------------------------------
# Public renderers
# ---------------------------------------------------------------------------

def render_line_chart(
    title: str,
    x_label: str,
    y_label: str,
    series_list: Sequence[ChartSeries],
    log_y: bool = False,
    log_floor: float = 1.0,
    y_formatter: ValueFormatter = format_money_compact,
    x_formatter: ValueFormatter = format_count_compact,
    subtitle: str = "",
    width: int = 960,
    height: int = 540,
) -> str:
    """Render a multi-series line chart as an SVG string."""
    if not series_list:
        raise ValueError("At least one series is required.")
    for series in series_list:
        if len(series.x_values) != len(series.y_values):
            raise ValueError(f"Series {series.label!r} has mismatched x/y lengths.")
        if len(series.x_values) == 0:
            raise ValueError(f"Series {series.label!r} is empty.")

    plot_left, plot_right = 90, width - 230
    plot_top, plot_bottom = 80, height - 70
    all_x_values = [value for series in series_list for value in series.x_values]
    all_y_values = [value for series in series_list for value in series.y_values]
    x_axis, x_ticks = _build_axis(all_x_values, False, 1.0, plot_left, plot_right, True)
    y_axis, y_ticks = _build_axis(
        all_y_values, log_y, log_floor, plot_bottom, plot_top, include_zero=not log_y
    )

    parts = _svg_header(width, height, title) + _title_block(width, title, subtitle)

    for tick_value in y_ticks:
        pixel_y = y_axis.position(tick_value)
        parts.append(
            f'<line x1="{plot_left}" y1="{pixel_y:.1f}" x2="{plot_right}" y2="{pixel_y:.1f}" '
            f'stroke="{GRID_COLOR}" stroke-width="1"/>'
        )
        parts.append(_text(plot_left - 10, pixel_y + 4, y_formatter(tick_value), anchor="end"))
    for tick_value in x_ticks:
        pixel_x = x_axis.position(tick_value)
        parts.append(
            f'<line x1="{pixel_x:.1f}" y1="{plot_top}" x2="{pixel_x:.1f}" y2="{plot_bottom}" '
            f'stroke="{GRID_COLOR}" stroke-width="1"/>'
        )
        parts.append(_text(pixel_x, plot_bottom + 20, x_formatter(tick_value), anchor="middle"))

    parts.append(
        f'<rect x="{plot_left}" y="{plot_top}" width="{plot_right - plot_left}" '
        f'height="{plot_bottom - plot_top}" fill="none" stroke="{AXIS_COLOR}" stroke-width="1"/>'
    )

    for series_index, series in enumerate(series_list):
        color = series.color or PALETTE[series_index % len(PALETTE)]
        pixel_points = [
            (x_axis.position(x_value), y_axis.position(y_value))
            for x_value, y_value in zip(series.x_values, series.y_values)
        ]
        parts.append(
            f'<polyline points="{_points_attribute(pixel_points)}" fill="none" stroke="{color}" '
            f'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        if series.mark_end:
            end_x, end_y = pixel_points[-1]
            parts.append(
                f'<path d="M {end_x - 5:.1f} {end_y - 5:.1f} L {end_x + 5:.1f} {end_y + 5:.1f} '
                f'M {end_x - 5:.1f} {end_y + 5:.1f} L {end_x + 5:.1f} {end_y - 5:.1f}" '
                f'stroke="{color}" stroke-width="2.5" fill="none"/>'
            )

    legend_x = plot_right + 24
    for series_index, series in enumerate(series_list):
        color = series.color or PALETTE[series_index % len(PALETTE)]
        legend_y = plot_top + 14 + series_index * 24
        parts.append(
            f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 26}" y2="{legend_y}" '
            f'stroke="{color}" stroke-width="3"/>'
        )
        parts.append(_text(legend_x + 34, legend_y + 4, series.label, size=13))

    parts.append(_text((plot_left + plot_right) / 2, height - 20, x_label, size=13, anchor="middle"))
    parts.append(
        _text(24, (plot_top + plot_bottom) / 2, y_label, size=13, anchor="middle", rotate=-90)
    )
    parts.append("</svg>")
    return "\n".join(parts)


def render_bar_chart(
    title: str,
    x_label: str,
    y_label: str,
    categories: Sequence[str],
    values: Sequence[float],
    log_y: bool = False,
    log_floor: float = 1.0,
    y_formatter: ValueFormatter = format_money_compact,
    value_formatter: Optional[ValueFormatter] = None,
    bar_colors: Optional[Sequence[str]] = None,
    subtitle: str = "",
    width: int = 960,
    height: int = 540,
) -> str:
    """Render a single-series bar chart as an SVG string.

    Linear axes include zero and support negative bars. Log axes need positive
    values; anything at or below ``log_floor`` is drawn at the floor.
    """
    if not categories or len(categories) != len(values):
        raise ValueError("categories and values must be non-empty and equal in length.")
    if bar_colors is not None and len(bar_colors) != len(values):
        raise ValueError("bar_colors must match the number of bars.")
    label_formatter = value_formatter if value_formatter is not None else y_formatter

    plot_left, plot_right = 90, width - 40
    plot_top, plot_bottom = 80, height - 90
    y_axis, y_ticks = _build_axis(
        values, log_y, log_floor, plot_bottom, plot_top, include_zero=not log_y
    )
    baseline_pixel = plot_bottom if log_y else y_axis.position(0.0)

    parts = _svg_header(width, height, title) + _title_block(width, title, subtitle)
    for tick_value in y_ticks:
        pixel_y = y_axis.position(tick_value)
        parts.append(
            f'<line x1="{plot_left}" y1="{pixel_y:.1f}" x2="{plot_right}" y2="{pixel_y:.1f}" '
            f'stroke="{GRID_COLOR}" stroke-width="1"/>'
        )
        parts.append(_text(plot_left - 10, pixel_y + 4, y_formatter(tick_value), anchor="end"))

    slot_width = (plot_right - plot_left) / len(values)
    bar_width = slot_width * 0.62
    for bar_index, (category, value) in enumerate(zip(categories, values)):
        color = bar_colors[bar_index] if bar_colors is not None else PALETTE[0]
        bar_left = plot_left + bar_index * slot_width + (slot_width - bar_width) / 2
        value_pixel = y_axis.position(value)
        bar_top = min(value_pixel, baseline_pixel)
        bar_height = abs(baseline_pixel - value_pixel)
        parts.append(
            f'<rect x="{bar_left:.1f}" y="{bar_top:.1f}" width="{bar_width:.1f}" '
            f'height="{bar_height:.1f}" fill="{color}" rx="2"/>'
        )
        label_y = value_pixel + 16 if (value < 0 and not log_y) else value_pixel - 6
        parts.append(
            _text(bar_left + bar_width / 2, label_y, label_formatter(value), anchor="middle")
        )
        parts.append(
            _text(bar_left + bar_width / 2, plot_bottom + 22, category, anchor="middle")
        )

    parts.append(
        f'<line x1="{plot_left}" y1="{baseline_pixel:.1f}" x2="{plot_right}" y2="{baseline_pixel:.1f}" '
        f'stroke="{AXIS_COLOR}" stroke-width="1.5"/>'
    )
    parts.append(
        f'<line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_bottom}" '
        f'stroke="{AXIS_COLOR}" stroke-width="1"/>'
    )
    parts.append(_text((plot_left + plot_right) / 2, height - 22, x_label, size=13, anchor="middle"))
    parts.append(
        _text(24, (plot_top + plot_bottom) / 2, y_label, size=13, anchor="middle", rotate=-90)
    )
    parts.append("</svg>")
    return "\n".join(parts)


def write_svg(path: str, svg_content: str) -> str:
    """Write SVG text to ``path`` (creating directories) and return the path."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as output_file:
        output_file.write(svg_content)
    return path
