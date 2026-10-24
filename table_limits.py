"""edgelab.table_limits: what a casino bet cap does to Kelly sizing.

Kelly sizing is proportional: the wager is a fixed fraction f of the current
bankroll B, so winning lifts the next bet and growth is exponential
(log-growth per hand ~ f*mu - f^2*sigma^2/2 for edge mu and variance sigma^2).

A table maximum M truncates that rule to min(f*B, M). Define the crossover
bankroll B* = M / f. Below B* nothing changes. Above B* the bet is pinned at M,
so expected profit per hand is the constant M*mu, growth turns LINEAR
(B(n) ~ B* + M*mu*n), and the log-growth rate decays like M*mu / B.

This script runs the "upcard-informed" Kelly module (the variant with a real
statistical edge) under a ladder of table limits on identical shuffled decks,
prints a comparison table and writes SVG charts.

Run:  python3 table_limits.py [--seeds 4] [--hands 50000] [--out charts]
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import charts
import main

DEFAULT_TABLE_LIMITS: Tuple[Optional[float], ...] = (
    25.0, 50.0, 100.0, 250.0, 500.0, 1_000.0, 5_000.0, None,
)
BASE_SEED = 42
REPRESENTATIVE_UPCARDS: Tuple[int, ...] = (6, 7, 10)  # weak, neutral, strong
MAXIMUM_CHART_POINTS = 600
BINDING_TOLERANCE = 1e-9


@dataclass
class LimitTrial:
    """Outcome of one (table limit, seed) simulation."""

    table_limit: Optional[float]
    seed: int
    result: main.SimulationResult
    report: main.RiskReport
    log_growth_per_100_hands: float
    cap_binding_share: float
    first_binding_hand: Optional[int]
    bankroll_at_first_binding: Optional[float]
    peak_bankroll_fraction: float


@dataclass
class LimitAggregate:
    """Medians across seeds for one table limit."""

    label: str
    table_limit: Optional[float]
    trial_count: int
    median_final_bankroll: float
    median_log_growth_per_100_hands: float
    median_max_drawdown: float
    median_sharpe_equivalent: float
    ruin_rate: float
    median_cap_binding_share: float
    median_bankroll_at_first_binding: Optional[float]
    predicted_crossover_bankroll: Optional[float]


def limit_label(table_limit: Optional[float]) -> str:
    """Display label for a table limit (None means no limit)."""
    if table_limit is None:
        return "No limit"
    return f"${table_limit:,.0f} max"


def build_kelly_config_for_limit(table_limit: Optional[float]) -> main.KellyConfig:
    """Upcard-informed Kelly config with the given table maximum bet."""
    maximum_bet = math.inf if table_limit is None else table_limit
    return main.KellyConfig(
        condition_on_dealer_upcard=True,
        maximum_bet=maximum_bet,
    )


def find_binding_hand_indices(
    bet_history: Sequence[float], maximum_bet: float
) -> List[int]:
    """Indices of hands whose wager was pinned at the table maximum."""
    if math.isinf(maximum_bet):
        return []
    return [
        hand_index
        for hand_index, bet_amount in enumerate(bet_history)
        if bet_amount >= maximum_bet - BINDING_TOLERANCE
    ]


def calculate_log_growth_per_100_hands(
    starting_bankroll: float, final_bankroll: float, hands_played: int
) -> float:
    """Natural-log bankroll growth per 100 hands (0.05 means about +5.1%).

    A bankroll at or below one cent is floored at one cent so ruin gives a
    large finite negative number rather than an error.
    """
    if hands_played <= 0 or starting_bankroll <= 0:
        return 0.0
    floored_final = max(final_bankroll, 0.01)
    return math.log(floored_final / starting_bankroll) / hands_played * 100.0


def predicted_crossover_bankroll(
    maximum_bet: Optional[float], peak_bankroll_fraction: float
) -> Optional[float]:
    """Bankroll B* = M / f at which the most aggressive bet reaches the cap."""
    if maximum_bet is None or math.isinf(maximum_bet) or peak_bankroll_fraction <= 0:
        return None
    return maximum_bet / peak_bankroll_fraction


def _median_or_none(values: Sequence[Optional[float]]) -> Optional[float]:
    """Median of the non-None values, or None when there are none."""
    present_values = [value for value in values if value is not None]
    if not present_values:
        return None
    return statistics.median(present_values)


def run_single_limit_trial(
    table_limit: Optional[float],
    seed: int,
    number_of_hands: int,
    starting_bankroll: float,
) -> LimitTrial:
    """Simulate upcard-informed Kelly under one table limit and one seed."""
    kelly_config = build_kelly_config_for_limit(table_limit)
    bet_sizer = main.FractionalKellyBetSizer(kelly_config)
    simulation_config = main.SimulationConfig(
        number_of_hands=number_of_hands,
        starting_bankroll=starting_bankroll,
        random_seed=seed,
    )
    result = main.run_simulation(
        strategy_name=limit_label(table_limit),
        decision_function=main.advanced_strategy_decision,
        bet_sizer=bet_sizer,
        config=simulation_config,
    )
    report = main.build_risk_report(result, starting_bankroll)

    binding_indices = find_binding_hand_indices(result.bet_history, kelly_config.maximum_bet)
    cap_binding_share = len(binding_indices) / result.hands_played if result.hands_played else 0.0
    first_binding_hand = binding_indices[0] if binding_indices else None
    bankroll_at_first_binding = (
        result.bankroll_history[first_binding_hand] if first_binding_hand is not None else None
    )
    peak_bankroll_fraction = max(
        bet_sizer.kelly_bankroll_fraction(upcard) for upcard in REPRESENTATIVE_UPCARDS
    )
    return LimitTrial(
        table_limit=table_limit,
        seed=seed,
        result=result,
        report=report,
        log_growth_per_100_hands=calculate_log_growth_per_100_hands(
            starting_bankroll, result.final_bankroll, result.hands_played
        ),
        cap_binding_share=cap_binding_share,
        first_binding_hand=first_binding_hand,
        bankroll_at_first_binding=bankroll_at_first_binding,
        peak_bankroll_fraction=peak_bankroll_fraction,
    )


def run_table_limit_experiment(
    table_limits: Sequence[Optional[float]],
    seeds: Sequence[int],
    number_of_hands: int,
    starting_bankroll: float,
) -> Dict[str, List[LimitTrial]]:
    """Run every (limit, seed) pair. Same seed means same decks for all limits."""
    trials_by_label: Dict[str, List[LimitTrial]] = {}
    for table_limit in table_limits:
        trials_by_label[limit_label(table_limit)] = [
            run_single_limit_trial(table_limit, seed, number_of_hands, starting_bankroll)
            for seed in seeds
        ]
    return trials_by_label


def aggregate_trials(
    label: str, table_limit: Optional[float], trials: Sequence[LimitTrial]
) -> LimitAggregate:
    """Collapse trials for one limit into medians across seeds."""
    if not trials:
        raise ValueError("At least one trial is required.")
    median_peak_fraction = statistics.median(trial.peak_bankroll_fraction for trial in trials)
    return LimitAggregate(
        label=label,
        table_limit=table_limit,
        trial_count=len(trials),
        median_final_bankroll=statistics.median(trial.result.final_bankroll for trial in trials),
        median_log_growth_per_100_hands=statistics.median(
            trial.log_growth_per_100_hands for trial in trials
        ),
        median_max_drawdown=statistics.median(trial.report.max_drawdown for trial in trials),
        median_sharpe_equivalent=statistics.median(
            trial.report.sharpe_equivalent for trial in trials
        ),
        ruin_rate=sum(1 for trial in trials if trial.result.ruined) / len(trials),
        median_cap_binding_share=statistics.median(trial.cap_binding_share for trial in trials),
        median_bankroll_at_first_binding=_median_or_none(
            [trial.bankroll_at_first_binding for trial in trials]
        ),
        predicted_crossover_bankroll=predicted_crossover_bankroll(
            table_limit, median_peak_fraction
        ),
    )


def format_limit_table(
    aggregates: Sequence[LimitAggregate],
    number_of_hands: int,
    starting_bankroll: float,
) -> str:
    """Render the comparison as an aligned terminal table."""
    header = (
        f"{'Table limit':<13}{'Median final':>16}{'Log growth':>12}{'Max DD':>9}"
        f"{'Ruin':>7}{'At cap':>9}{'Cap first binds at':>22}{'Predicted B*':>15}"
    )
    rule = "-" * len(header)
    heavy_rule = "=" * len(header)
    lines = [
        heavy_rule,
        "EDGELAB | TABLE LIMITS vs FRACTIONAL KELLY (upcard-informed variant)".center(len(header)),
        heavy_rule,
        (
            f"{number_of_hands:,} hands per run | start ${starting_bankroll:,.2f} | "
            f"medians over {aggregates[0].trial_count} seeds, identical decks per seed"
        ),
        rule,
        header,
        rule,
    ]
    for aggregate in aggregates:
        observed = (
            charts.format_money_compact(aggregate.median_bankroll_at_first_binding)
            if aggregate.median_bankroll_at_first_binding is not None
            else "never"
        )
        predicted = (
            charts.format_money_compact(aggregate.predicted_crossover_bankroll)
            if aggregate.predicted_crossover_bankroll is not None
            else "n/a"
        )
        lines.append(
            f"{aggregate.label:<13}"
            f"{charts.format_money_compact(aggregate.median_final_bankroll):>16}"
            f"{aggregate.median_log_growth_per_100_hands * 100:>11.2f}%"
            f"{aggregate.median_max_drawdown * 100:>8.1f}%"
            f"{aggregate.ruin_rate * 100:>6.0f}%"
            f"{aggregate.median_cap_binding_share * 100:>8.1f}%"
            f"{observed:>22}"
            f"{predicted:>15}"
        )
    lines.extend(
        [
            rule,
            "Log growth = natural-log bankroll growth per 100 hands. At cap = share of hands",
            "whose bet was pinned at the limit. B* = limit / peak Kelly fraction: below it",
            "bets scale with bankroll (exponential growth); above it the bet is fixed, so",
            "profit per hand is constant (about limit x edge) and growth turns linear.",
            "Monte Carlo approximation under fixed seeds, not an analytical solution.",
            "The upcard-informed module bets after seeing the dealer upcard, an information",
            "advantage real tables do not allow; treat the absolute dollars as a stress test.",
            heavy_rule,
        ]
    )
    return "\n".join(lines)


def write_limit_charts(
    aggregates: Sequence[LimitAggregate],
    trials_by_label: Dict[str, List[LimitTrial]],
    output_directory: str,
) -> List[str]:
    """Write the four table-limit SVG charts and return their paths."""
    labels = [aggregate.label for aggregate in aggregates]
    first_seed = next(iter(trials_by_label.values()))[0].seed

    path_series: List[charts.ChartSeries] = []
    for aggregate in aggregates:
        representative_trial = trials_by_label[aggregate.label][0]
        history = representative_trial.result.bankroll_history
        hand_numbers = [float(index) for index in range(len(history))]
        sampled_x, sampled_y = charts.downsample_series(hand_numbers, history, MAXIMUM_CHART_POINTS)
        path_series.append(
            charts.ChartSeries(
                label=aggregate.label,
                x_values=sampled_x,
                y_values=sampled_y,
                mark_end=representative_trial.result.ruined,
            )
        )

    paths_svg = charts.render_line_chart(
        title="Bankroll paths under different table limits (log scale)",
        subtitle=(
            f"Seed {first_seed}, identical decks. Each curve bends from exponential "
            "toward linear once the cap binds."
        ),
        x_label="Hands played",
        y_label="Bankroll",
        series_list=path_series,
        log_y=True,
        log_floor=1.0,
    )
    final_svg = charts.render_bar_chart(
        title="Median final bankroll by table limit (log scale)",
        subtitle=f"Medians over {aggregates[0].trial_count} seeds. Each limit multiplies the payoff of the edge.",
        x_label="Table maximum bet",
        y_label="Final bankroll",
        categories=labels,
        values=[aggregate.median_final_bankroll for aggregate in aggregates],
        log_y=True,
        log_floor=1.0,
    )
    growth_svg = charts.render_bar_chart(
        title="Kelly growth rate by table limit (log scale)",
        subtitle="Natural-log bankroll growth per 100 hands. The cap throttles compounding.",
        x_label="Table maximum bet",
        y_label="Log growth per 100 hands",
        categories=labels,
        values=[aggregate.median_log_growth_per_100_hands * 100.0 for aggregate in aggregates],
        log_y=True,
        log_floor=0.1,
        y_formatter=charts.format_percent_axis,
    )
    drawdown_svg = charts.render_bar_chart(
        title="Median maximum drawdown by table limit",
        subtitle="Peak-to-trough decline of the bankroll.",
        x_label="Table maximum bet",
        y_label="Maximum drawdown",
        categories=labels,
        values=[aggregate.median_max_drawdown * 100.0 for aggregate in aggregates],
        y_formatter=charts.format_percent_axis,
        bar_colors=[charts.PALETTE[1]] * len(aggregates),
    )
    return [
        charts.write_svg(os.path.join(output_directory, "table_limits_bankroll_paths.svg"), paths_svg),
        charts.write_svg(os.path.join(output_directory, "table_limits_final_bankroll.svg"), final_svg),
        charts.write_svg(os.path.join(output_directory, "table_limits_growth_rate.svg"), growth_svg),
        charts.write_svg(os.path.join(output_directory, "table_limits_drawdown.svg"), drawdown_svg),
    ]


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        prog="edgelab-table-limits",
        description="Show how table maximum bets constrain fractional Kelly sizing.",
    )
    parser.add_argument("--seeds", type=int, default=4, help="number of seeds per limit")
    parser.add_argument("--hands", type=int, default=50_000, help="hands per run")
    parser.add_argument("--bankroll", type=float, default=1_000.0, help="starting bankroll")
    parser.add_argument("--out", default="charts", help="directory for SVG charts")
    return parser.parse_args(argv)


def run(argv: Optional[Sequence[str]] = None) -> None:
    """Run the experiment, print the table and write the charts."""
    arguments = parse_arguments(argv)
    if arguments.seeds < 1:
        raise SystemExit("--seeds must be at least 1")
    seeds = [BASE_SEED + offset for offset in range(arguments.seeds)]
    trials_by_label = run_table_limit_experiment(
        DEFAULT_TABLE_LIMITS, seeds, arguments.hands, arguments.bankroll
    )
    aggregates = [
        aggregate_trials(limit_label(table_limit), table_limit, trials_by_label[limit_label(table_limit)])
        for table_limit in DEFAULT_TABLE_LIMITS
    ]
    print(format_limit_table(aggregates, arguments.hands, arguments.bankroll))
    for written_path in write_limit_charts(aggregates, trials_by_label, arguments.out):
        print(f"Chart written: {written_path}")


if __name__ == "__main__":
    run()
