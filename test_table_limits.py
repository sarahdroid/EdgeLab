"""Unit tests for edgelab.table_limits (table-limit experiment).

Run:  python3 -m unittest -v test_table_limits
"""

from __future__ import annotations

import math
import os
import tempfile
import unittest
import xml.etree.ElementTree as ElementTree

import main
import table_limits


class HelperTests(unittest.TestCase):
    """Pure helper functions."""

    def test_limit_labels(self) -> None:
        self.assertEqual(table_limits.limit_label(None), "No limit")
        self.assertEqual(table_limits.limit_label(500.0), "$500 max")
        self.assertEqual(table_limits.limit_label(5_000.0), "$5,000 max")

    def test_config_for_limit(self) -> None:
        capped = table_limits.build_kelly_config_for_limit(100.0)
        self.assertEqual(capped.maximum_bet, 100.0)
        self.assertTrue(capped.condition_on_dealer_upcard)
        unlimited = table_limits.build_kelly_config_for_limit(None)
        self.assertTrue(math.isinf(unlimited.maximum_bet))

    def test_binding_indices(self) -> None:
        bets = [5.0, 25.0, 12.0, 25.0, 24.99]
        self.assertEqual(table_limits.find_binding_hand_indices(bets, 25.0), [1, 3])
        self.assertEqual(table_limits.find_binding_hand_indices(bets, math.inf), [])
        self.assertEqual(table_limits.find_binding_hand_indices([], 25.0), [])

    def test_log_growth_math(self) -> None:
        growth = table_limits.calculate_log_growth_per_100_hands(1_000.0, 1_000.0 * math.e, 100)
        self.assertAlmostEqual(growth, 1.0)
        self.assertAlmostEqual(
            table_limits.calculate_log_growth_per_100_hands(1_000.0, 1_000.0 * math.e, 200), 0.5
        )
        self.assertEqual(table_limits.calculate_log_growth_per_100_hands(1_000.0, 500.0, 0), 0.0)

    def test_log_growth_of_ruin_is_finite_and_negative(self) -> None:
        growth = table_limits.calculate_log_growth_per_100_hands(1_000.0, 0.0, 100)
        self.assertTrue(math.isfinite(growth))
        self.assertLess(growth, 0.0)

    def test_crossover_prediction(self) -> None:
        self.assertAlmostEqual(table_limits.predicted_crossover_bankroll(500.0, 0.05), 10_000.0)
        self.assertIsNone(table_limits.predicted_crossover_bankroll(None, 0.05))
        self.assertIsNone(table_limits.predicted_crossover_bankroll(math.inf, 0.05))
        self.assertIsNone(table_limits.predicted_crossover_bankroll(500.0, 0.0))


class ExperimentTests(unittest.TestCase):
    """Small end-to-end runs."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.limits = (25.0, 200.0, None)
        cls.trials = table_limits.run_table_limit_experiment(
            cls.limits, seeds=[7, 8], number_of_hands=6_000, starting_bankroll=1_000.0
        )

    def test_every_limit_and_seed_is_run(self) -> None:
        self.assertEqual(set(self.trials), {"$25 max", "$200 max", "No limit"})
        for trial_list in self.trials.values():
            self.assertEqual([trial.seed for trial in trial_list], [7, 8])

    def test_no_wager_exceeds_the_table_limit(self) -> None:
        for limit in (25.0, 200.0):
            for trial in self.trials[table_limits.limit_label(limit)]:
                self.assertLessEqual(max(trial.result.bet_history), limit + 1e-9)

    def test_bet_history_matches_hands_played(self) -> None:
        for trial_list in self.trials.values():
            for trial in trial_list:
                self.assertEqual(len(trial.result.bet_history), trial.result.hands_played)
                self.assertEqual(len(trial.result.bankroll_history), trial.result.hands_played + 1)

    def test_small_cap_binds_and_unlimited_never_does(self) -> None:
        for trial in self.trials["$25 max"]:
            self.assertGreater(trial.cap_binding_share, 0.0)
            self.assertIsNotNone(trial.first_binding_hand)
            self.assertIsNotNone(trial.bankroll_at_first_binding)
        for trial in self.trials["No limit"]:
            self.assertEqual(trial.cap_binding_share, 0.0)
            self.assertIsNone(trial.first_binding_hand)

    def test_bankroll_at_first_binding_is_consistent_with_history(self) -> None:
        for trial in self.trials["$25 max"]:
            hand_index = trial.first_binding_hand
            self.assertEqual(trial.bankroll_at_first_binding, trial.result.bankroll_history[hand_index])
            self.assertGreaterEqual(trial.result.bet_history[hand_index], 25.0 - 1e-9)

    def test_identical_decks_before_any_cap_binds(self) -> None:
        # Until the cap first binds, a capped run and the unlimited run bet the
        # same amounts on the same decks, so their histories must agree.
        capped = self.trials["$200 max"][0]
        unlimited = self.trials["No limit"][0]
        agree_until = capped.first_binding_hand
        if agree_until is None:
            agree_until = len(capped.result.bet_history)
        self.assertEqual(
            capped.result.bankroll_history[: agree_until + 1],
            unlimited.result.bankroll_history[: agree_until + 1],
        )

    def test_same_seed_is_deterministic(self) -> None:
        first = table_limits.run_single_limit_trial(100.0, 3, 1_500, 1_000.0)
        second = table_limits.run_single_limit_trial(100.0, 3, 1_500, 1_000.0)
        self.assertEqual(first.result.bankroll_history, second.result.bankroll_history)

    def test_aggregation_uses_medians(self) -> None:
        trial_list = self.trials["$25 max"]
        aggregate = table_limits.aggregate_trials("$25 max", 25.0, trial_list)
        finals = sorted(trial.result.final_bankroll for trial in trial_list)
        self.assertAlmostEqual(aggregate.median_final_bankroll, sum(finals) / 2)
        self.assertEqual(aggregate.trial_count, 2)
        self.assertIsNotNone(aggregate.predicted_crossover_bankroll)

    def test_aggregation_requires_trials(self) -> None:
        with self.assertRaises(ValueError):
            table_limits.aggregate_trials("x", 1.0, [])

    def test_unlimited_aggregate_has_no_crossover(self) -> None:
        aggregate = table_limits.aggregate_trials("No limit", None, self.trials["No limit"])
        self.assertIsNone(aggregate.predicted_crossover_bankroll)
        self.assertIsNone(aggregate.median_bankroll_at_first_binding)

    def test_table_and_charts_are_produced(self) -> None:
        aggregates = [
            table_limits.aggregate_trials(table_limits.limit_label(limit), limit,
                                          self.trials[table_limits.limit_label(limit)])
            for limit in self.limits
        ]
        text = table_limits.format_limit_table(aggregates, 6_000, 1_000.0)
        for expected in ("$25 max", "$200 max", "No limit", "never", "Monte Carlo"):
            self.assertIn(expected, text)
        with tempfile.TemporaryDirectory() as directory:
            written = table_limits.write_limit_charts(aggregates, self.trials, directory)
            self.assertEqual(len(written), 4)
            for path in written:
                self.assertTrue(os.path.exists(path))
                ElementTree.parse(path)

    def test_cli_argument_defaults(self) -> None:
        arguments = table_limits.parse_arguments([])
        self.assertEqual(arguments.seeds, 4)
        self.assertEqual(arguments.hands, 50_000)
        self.assertEqual(arguments.out, "charts")


if __name__ == "__main__":
    unittest.main()
