"""Unit tests for edgelab.charts (SVG rendering helpers).

Run:  python3 -m unittest -v test_charts
"""

from __future__ import annotations

import math
import os
import tempfile
import unittest
import xml.etree.ElementTree as ElementTree

import charts
import main

SVG_NAMESPACE = "{http://www.w3.org/2000/svg}"


def parse_svg(svg_text: str) -> ElementTree.Element:
    """Parse SVG text, failing the test on malformed XML."""
    return ElementTree.fromstring(svg_text)


class FormattingTests(unittest.TestCase):
    """Compact number formatting."""

    def test_money_formatting_across_magnitudes(self) -> None:
        self.assertEqual(charts.format_money_compact(0), "$0")
        self.assertEqual(charts.format_money_compact(25), "$25")
        self.assertEqual(charts.format_money_compact(5.5), "$5.5")
        self.assertEqual(charts.format_money_compact(950), "$950")
        self.assertEqual(charts.format_money_compact(1_000), "$1K")
        self.assertEqual(charts.format_money_compact(1_500_000), "$1.5M")
        self.assertEqual(charts.format_money_compact(2_000_000_000), "$2B")
        self.assertEqual(charts.format_money_compact(3e12), "$3T")
        self.assertEqual(charts.format_money_compact(1.7e58), "$1.7e+58")
        self.assertEqual(charts.format_money_compact(-1_500), "-$1.5K")

    def test_count_formatting(self) -> None:
        self.assertEqual(charts.format_count_compact(0), "0")
        self.assertEqual(charts.format_count_compact(750), "750")
        self.assertEqual(charts.format_count_compact(2_500), "2.5K")
        self.assertEqual(charts.format_count_compact(50_000), "50K")
        self.assertEqual(charts.format_count_compact(2_000_000), "2M")

    def test_percent_axis_formatting(self) -> None:
        self.assertEqual(charts.format_percent_axis(12.5), "12.5%")
        self.assertEqual(charts.format_percent_axis(100.0), "100%")


class TickTests(unittest.TestCase):
    """Tick generation."""

    def test_nice_ticks_cover_range_with_round_steps(self) -> None:
        ticks = charts.nice_ticks(0, 97)
        self.assertEqual(ticks[0], 0)
        self.assertGreaterEqual(ticks[-1], 97)
        steps = {round(b - a, 9) for a, b in zip(ticks, ticks[1:])}
        self.assertEqual(len(steps), 1)

    def test_nice_ticks_handle_negative_ranges(self) -> None:
        ticks = charts.nice_ticks(-30, 70)
        self.assertLessEqual(ticks[0], -30)
        self.assertGreaterEqual(ticks[-1], 70)

    def test_nice_ticks_degenerate_range(self) -> None:
        ticks = charts.nice_ticks(5, 5)
        self.assertGreaterEqual(len(ticks), 2)

    def test_nice_ticks_validation(self) -> None:
        with self.assertRaises(ValueError):
            charts.nice_ticks(10, 1)
        with self.assertRaises(ValueError):
            charts.nice_ticks(0, 1, target_count=1)

    def test_log_ticks_are_decades(self) -> None:
        self.assertEqual(charts.log_tick_exponents(3, 4_500), [0, 1, 2, 3, 4])

    def test_log_ticks_thin_wide_ranges(self) -> None:
        exponents = charts.log_tick_exponents(1, 1e58)
        self.assertLessEqual(len(exponents), 10)
        self.assertEqual(exponents[0], 0)

    def test_log_ticks_clip_values_below_floor(self) -> None:
        self.assertEqual(charts.log_tick_exponents(0, 10), [0, 1])

    def test_log_ticks_reject_bad_floor(self) -> None:
        with self.assertRaises(ValueError):
            charts.log_tick_exponents(1, 10, log_floor=0)


class DownsampleTests(unittest.TestCase):
    """Series reduction."""

    def test_short_series_unchanged(self) -> None:
        x_values = [0.0, 1.0, 2.0]
        y_values = [5.0, 6.0, 7.0]
        self.assertEqual(charts.downsample_series(x_values, y_values, 10), (x_values, y_values))

    def test_stride_keeps_endpoints_and_limits_size(self) -> None:
        x_values = [float(i) for i in range(10_001)]
        sampled_x, sampled_y = charts.downsample_series(x_values, x_values, 500)
        self.assertLessEqual(len(sampled_x), 501)
        self.assertEqual(sampled_x[0], 0.0)
        self.assertEqual(sampled_x[-1], 10_000.0)
        self.assertEqual(sampled_x, sampled_y)
        self.assertEqual(sampled_x, sorted(sampled_x))

    def test_peak_mode_preserves_spike(self) -> None:
        x_values = [float(i) for i in range(5_000)]
        y_values = [0.0] * 5_000
        y_values[2_345] = 99.0
        _, sampled_y = charts.downsample_series(x_values, y_values, 100, keep_peaks=True)
        self.assertIn(99.0, sampled_y)

    def test_validation(self) -> None:
        with self.assertRaises(ValueError):
            charts.downsample_series([1.0], [1.0, 2.0], 10)
        with self.assertRaises(ValueError):
            charts.downsample_series([1.0], [1.0], 2)


class LineChartTests(unittest.TestCase):
    """Line chart rendering."""

    def _series(self) -> list:
        return [
            charts.ChartSeries("Alpha", [0, 50, 100], [1_000, 2_000, 4_000]),
            charts.ChartSeries("Beta & <Gamma>", [0, 50, 100], [1_000, 500, 1], mark_end=True),
        ]

    def test_output_is_valid_svg_with_one_polyline_per_series(self) -> None:
        root = parse_svg(charts.render_line_chart("T", "X", "Y", self._series()))
        self.assertEqual(root.tag, SVG_NAMESPACE + "svg")
        self.assertEqual(len(list(root.iter(SVG_NAMESPACE + "polyline"))), 2)

    def test_special_characters_are_escaped(self) -> None:
        svg_text = charts.render_line_chart("A & B", "x", "y", self._series())
        parse_svg(svg_text)  # Would raise on unescaped characters.
        self.assertIn("Beta &amp; &lt;Gamma&gt;", svg_text)

    def test_ruin_marker_drawn_only_when_requested(self) -> None:
        root = parse_svg(charts.render_line_chart("T", "X", "Y", self._series()))
        self.assertEqual(len(list(root.iter(SVG_NAMESPACE + "path"))), 1)

    def test_log_axis_clips_zero_values(self) -> None:
        series = [charts.ChartSeries("Ruined", [0, 1, 2], [1_000, 10, 0])]
        svg_text = charts.render_line_chart("T", "X", "Y", series, log_y=True)
        self.assertNotIn("nan", svg_text.lower())
        self.assertNotIn("inf", svg_text.lower())
        parse_svg(svg_text)

    def test_all_coordinates_inside_canvas(self) -> None:
        root = parse_svg(charts.render_line_chart("T", "X", "Y", self._series(), log_y=True))
        for polyline in root.iter(SVG_NAMESPACE + "polyline"):
            for pair in polyline.get("points").split():
                x_text, y_text = pair.split(",")
                self.assertTrue(0 <= float(x_text) <= 960)
                self.assertTrue(0 <= float(y_text) <= 540)

    def test_validation_errors(self) -> None:
        with self.assertRaises(ValueError):
            charts.render_line_chart("T", "X", "Y", [])
        with self.assertRaises(ValueError):
            charts.render_line_chart("T", "X", "Y", [charts.ChartSeries("a", [1, 2], [1])])
        with self.assertRaises(ValueError):
            charts.render_line_chart("T", "X", "Y", [charts.ChartSeries("a", [], [])])


class BarChartTests(unittest.TestCase):
    """Bar chart rendering."""

    def test_one_bar_per_category_plus_background(self) -> None:
        svg_text = charts.render_bar_chart("T", "X", "Y", ["a", "b", "c"], [1.0, 2.0, 3.0])
        rectangles = list(parse_svg(svg_text).iter(SVG_NAMESPACE + "rect"))
        self.assertEqual(len(rectangles), 4)  # background + 3 bars

    def test_negative_values_draw_below_baseline(self) -> None:
        svg_text = charts.render_bar_chart("T", "X", "Y", ["up", "down"], [10.0, -10.0])
        bars = [r for r in parse_svg(svg_text).iter(SVG_NAMESPACE + "rect") if r.get("rx")]
        up_bar, down_bar = bars
        self.assertLess(float(up_bar.get("y")), float(down_bar.get("y")))

    def test_log_bars_scale_with_magnitude(self) -> None:
        svg_text = charts.render_bar_chart(
            "T", "X", "Y", ["small", "huge"], [10.0, 1e12], log_y=True
        )
        bars = [r for r in parse_svg(svg_text).iter(SVG_NAMESPACE + "rect") if r.get("rx")]
        self.assertGreater(float(bars[1].get("height")), float(bars[0].get("height")))
        self.assertNotIn("nan", svg_text.lower())

    def test_validation_errors(self) -> None:
        with self.assertRaises(ValueError):
            charts.render_bar_chart("T", "X", "Y", ["a"], [1.0, 2.0])
        with self.assertRaises(ValueError):
            charts.render_bar_chart("T", "X", "Y", [], [])
        with self.assertRaises(ValueError):
            charts.render_bar_chart("T", "X", "Y", ["a"], [1.0], bar_colors=["red", "blue"])


class FileOutputTests(unittest.TestCase):
    """Writing SVG files and the main.py chart hook."""

    def test_write_svg_creates_directories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = os.path.join(directory, "nested", "dir", "chart.svg")
            returned = charts.write_svg(target, "<svg xmlns='http://www.w3.org/2000/svg'/>")
            self.assertEqual(returned, target)
            self.assertTrue(os.path.exists(target))

    def test_strategy_chart_hook_writes_valid_svgs(self) -> None:
        config = main.SimulationConfig(number_of_hands=800, random_seed=5)
        results = [
            main.run_simulation("Flat", main.baseline_strategy_decision, main.FlatBetSizer(10.0), config),
            main.run_simulation("Kelly", main.advanced_strategy_decision, main.FractionalKellyBetSizer(), config),
        ]
        with tempfile.TemporaryDirectory() as directory:
            written_paths = main.write_strategy_charts(results, directory)
            self.assertEqual(len(written_paths), 2)
            for path in written_paths:
                ElementTree.parse(path)

    def test_drawdown_series_matches_max_drawdown(self) -> None:
        series = [1_000, 1_200, 600, 900, 1_500]
        drawdowns = main.calculate_drawdown_series(series)
        self.assertEqual(len(drawdowns), len(series))
        self.assertAlmostEqual(max(drawdowns), main.calculate_max_drawdown(series))
        self.assertEqual(drawdowns[0], 0.0)
        self.assertEqual(drawdowns[-1], 0.0)

    def test_cli_defaults(self) -> None:
        arguments = main.parse_arguments([])
        self.assertEqual(arguments.hands, 50_000)
        self.assertEqual(arguments.seed, 42)
        self.assertIsNone(arguments.charts)


if __name__ == "__main__":
    unittest.main()
