"""Unit tests for the Quantitative Capital Allocation & Risk Engine.

Run:  python3 -m unittest -v test_main
"""

from __future__ import annotations

import unittest
from typing import List, Sequence

import main
from main import (
    Action,
    Card,
    FlatBetSizer,
    FractionalKellyBetSizer,
    KellyConfig,
    Outcome,
    SimulationConfig,
)


def make_cards(*ranks: str) -> List[Card]:
    """Build a hand from rank strings (suit is irrelevant to the rules)."""
    return [Card(rank=rank, suit="Spades") for rank in ranks]


def stacked_deck(*draw_order: str) -> List[Card]:
    """Return a deck whose draws come out in exactly ``draw_order``."""
    return list(reversed(make_cards(*draw_order)))


def always_stand(player_cards: Sequence[Card], dealer_upcard_value: int) -> Action:
    """Test decision function that never hits."""
    return Action.STAND


def always_hit(player_cards: Sequence[Card], dealer_upcard_value: int) -> Action:
    """Test decision function that always hits."""
    return Action.HIT


def play_scripted_round(draw_order: Sequence[str], decision_function) -> Outcome:
    """Deal and resolve a round from a stacked deck.

    Draw order: player card 1, dealer upcard, player card 2, dealer hole
    card, then any hit cards in the order they are drawn.
    """
    deck = stacked_deck(*draw_order)
    player_cards, dealer_cards = main.deal_initial_hands(deck)
    return main.resolve_round(deck, player_cards, dealer_cards, decision_function)


class DeckTests(unittest.TestCase):
    """Deck construction and dealing."""

    def test_standard_deck_has_52_unique_cards(self) -> None:
        deck = main.build_standard_deck()
        self.assertEqual(len(deck), 52)
        self.assertEqual(len(set(deck)), 52)

    def test_deck_has_sixteen_ten_value_cards_and_four_aces(self) -> None:
        deck = main.build_standard_deck()
        ten_values = sum(1 for card in deck if main.card_base_value(card) == 10)
        aces = sum(1 for card in deck if card.rank == "A")
        self.assertEqual(ten_values, 16)
        self.assertEqual(aces, 4)

    def test_draw_from_empty_deck_raises(self) -> None:
        with self.assertRaises(IndexError):
            main.draw_card([])

    def test_deal_order_is_player_dealer_player_dealer(self) -> None:
        deck = stacked_deck("2", "3", "4", "5")
        player_cards, dealer_cards = main.deal_initial_hands(deck)
        self.assertEqual([card.rank for card in player_cards], ["2", "4"])
        self.assertEqual([card.rank for card in dealer_cards], ["3", "5"])

    def test_same_seed_gives_same_shuffle(self) -> None:
        import random

        first_deck = main.build_standard_deck()
        second_deck = main.build_standard_deck()
        main.shuffle_deck(first_deck, random.Random(7))
        main.shuffle_deck(second_deck, random.Random(7))
        self.assertEqual(first_deck, second_deck)


class HandValueTests(unittest.TestCase):
    """Card values and Ace inversion (11 -> 1)."""

    def test_face_cards_are_ten(self) -> None:
        for rank in ("J", "Q", "K", "10"):
            self.assertEqual(main.card_base_value(Card(rank, "Hearts")), 10)

    def test_number_cards_use_face_value(self) -> None:
        for value in range(2, 10):
            self.assertEqual(main.card_base_value(Card(str(value), "Hearts")), value)

    def test_ace_defaults_to_eleven(self) -> None:
        self.assertEqual(main.hand_value(make_cards("A", "6")), main.HandValue(17, True))

    def test_ace_and_king_is_soft_21(self) -> None:
        self.assertEqual(main.hand_value(make_cards("A", "K")), main.HandValue(21, True))

    def test_ace_inverts_to_one_to_avoid_bust(self) -> None:
        self.assertEqual(main.hand_value(make_cards("A", "K", "5")), main.HandValue(16, False))

    def test_ace_inverts_after_hit_on_soft_hand(self) -> None:
        self.assertEqual(main.hand_value(make_cards("A", "6", "10")), main.HandValue(17, False))

    def test_two_aces_are_soft_12(self) -> None:
        self.assertEqual(main.hand_value(make_cards("A", "A")), main.HandValue(12, True))

    def test_only_one_of_two_aces_inverts(self) -> None:
        self.assertEqual(main.hand_value(make_cards("A", "A", "9")), main.HandValue(21, True))

    def test_both_aces_invert_when_needed(self) -> None:
        self.assertEqual(main.hand_value(make_cards("A", "A", "9", "10")), main.HandValue(21, False))

    def test_four_aces_total_14_soft(self) -> None:
        self.assertEqual(main.hand_value(make_cards("A", "A", "A", "A")), main.HandValue(14, True))

    def test_hard_bust_total_exceeds_21(self) -> None:
        self.assertEqual(main.hand_value(make_cards("10", "9", "5")).total, 24)

    def test_natural_detection(self) -> None:
        self.assertTrue(main.is_natural_blackjack(make_cards("A", "K")))
        self.assertTrue(main.is_natural_blackjack(make_cards("10", "A")))
        self.assertFalse(main.is_natural_blackjack(make_cards("7", "7", "7")))
        self.assertFalse(main.is_natural_blackjack(make_cards("K", "9")))


class DealerRuleTests(unittest.TestCase):
    """Dealer hit/stand boundary conditions."""

    def test_dealer_hits_on_16(self) -> None:
        self.assertTrue(main.dealer_should_hit(make_cards("10", "6")))

    def test_dealer_stands_on_17(self) -> None:
        self.assertFalse(main.dealer_should_hit(make_cards("10", "7")))

    def test_dealer_hits_on_hard_12_through_16(self) -> None:
        for second_card in ("2", "3", "4", "5", "6"):
            self.assertTrue(main.dealer_should_hit(make_cards("10", second_card)))

    def test_dealer_stands_on_soft_17(self) -> None:
        self.assertFalse(main.dealer_should_hit(make_cards("A", "6")))

    def test_dealer_hits_on_soft_16(self) -> None:
        self.assertTrue(main.dealer_should_hit(make_cards("A", "5")))

    def test_dealer_hits_when_ace_demotes_to_hard_16(self) -> None:
        self.assertTrue(main.dealer_should_hit(make_cards("A", "5", "10")))

    def test_dealer_stands_on_21_and_bust(self) -> None:
        self.assertFalse(main.dealer_should_hit(make_cards("10", "5", "6")))
        self.assertFalse(main.dealer_should_hit(make_cards("10", "6", "9")))

    def test_dealer_draws_ace_as_one_to_reach_exactly_17(self) -> None:
        # Player 10+8 = 18; dealer 10+6 = 16, draws Ace -> 17 (not 27); stands.
        outcome = play_scripted_round(["10", "10", "8", "6", "A"], always_stand)
        self.assertIs(outcome, Outcome.WIN)

    def test_dealer_stands_on_exactly_17_and_pushes_player_17(self) -> None:
        outcome = play_scripted_round(["10", "10", "7", "7"], always_stand)
        self.assertIs(outcome, Outcome.PUSH)

    def test_dealer_soft_17_stands_in_full_round(self) -> None:
        outcome = play_scripted_round(["10", "A", "8", "6"], always_stand)
        self.assertIs(outcome, Outcome.WIN)

    def test_dealer_bust_is_player_win(self) -> None:
        outcome = play_scripted_round(["10", "10", "8", "6", "K"], always_stand)
        self.assertIs(outcome, Outcome.WIN)

    def test_dealer_higher_total_beats_player(self) -> None:
        outcome = play_scripted_round(["10", "10", "7", "9"], always_stand)
        self.assertIs(outcome, Outcome.LOSS)


class RoundResolutionTests(unittest.TestCase):
    """Naturals and player bust handling."""

    def test_player_natural_pays_blackjack(self) -> None:
        outcome = play_scripted_round(["A", "5", "K", "6"], always_stand)
        self.assertIs(outcome, Outcome.BLACKJACK)

    def test_both_naturals_push(self) -> None:
        outcome = play_scripted_round(["A", "A", "K", "K"], always_stand)
        self.assertIs(outcome, Outcome.PUSH)

    def test_dealer_natural_alone_is_loss(self) -> None:
        outcome = play_scripted_round(["10", "A", "9", "K"], always_stand)
        self.assertIs(outcome, Outcome.LOSS)

    def test_player_three_card_21_is_not_a_natural(self) -> None:
        # Player 7+7+7 = 21 (not natural); dealer 10+9 = 19.
        outcome = play_scripted_round(["7", "10", "7", "9", "7"], always_hit)
        self.assertIs(outcome, Outcome.WIN)

    def test_player_bust_is_immediate_loss_even_if_dealer_would_bust(self) -> None:
        # Player 10+6 hits K -> 26. Dealer 5+6 would not matter.
        outcome = play_scripted_round(["10", "5", "6", "6", "K", "K"], always_hit)
        self.assertIs(outcome, Outcome.LOSS)

    def test_player_stops_drawing_at_21(self) -> None:
        # Player 5+6 hits 10 -> 21 and must stop even with an always-hit policy.
        outcome = play_scripted_round(["5", "10", "6", "8", "10"], always_hit)
        self.assertIs(outcome, Outcome.WIN)


class PayoutTests(unittest.TestCase):
    """Payout arithmetic."""

    def test_win_pays_even_money(self) -> None:
        self.assertEqual(main.calculate_payout(Outcome.WIN, 10.0), 10.0)

    def test_loss_forfeits_bet(self) -> None:
        self.assertEqual(main.calculate_payout(Outcome.LOSS, 10.0), -10.0)

    def test_push_returns_zero(self) -> None:
        self.assertEqual(main.calculate_payout(Outcome.PUSH, 25.0), 0.0)

    def test_blackjack_pays_three_to_two(self) -> None:
        self.assertEqual(main.calculate_payout(Outcome.BLACKJACK, 10.0), 15.0)
        self.assertEqual(main.calculate_payout(Outcome.BLACKJACK, 5.0), 7.5)
        self.assertEqual(main.calculate_payout(Outcome.BLACKJACK, 11.0), 16.5)

    def test_payout_rounds_to_cents(self) -> None:
        self.assertEqual(main.calculate_payout(Outcome.BLACKJACK, 10.01), 15.02)

    def test_bankroll_math_over_sequence(self) -> None:
        bankroll = 1000.0
        for outcome, bet in (
            (Outcome.WIN, 10.0),
            (Outcome.BLACKJACK, 10.0),
            (Outcome.LOSS, 20.0),
            (Outcome.PUSH, 50.0),
        ):
            bankroll += main.calculate_payout(outcome, bet)
        self.assertEqual(bankroll, 1005.0)


class StrategyMatrixTests(unittest.TestCase):
    """Baseline and matrix strategies."""

    def test_baseline_stands_on_16_and_hits_on_15(self) -> None:
        self.assertIs(main.baseline_strategy_decision(make_cards("10", "6"), 10), Action.STAND)
        self.assertIs(main.baseline_strategy_decision(make_cards("10", "5"), 2), Action.HIT)

    def test_baseline_ignores_dealer_upcard(self) -> None:
        hand = make_cards("10", "6")
        decisions = {main.baseline_strategy_decision(hand, up) for up in range(2, 12)}
        self.assertEqual(decisions, {Action.STAND})

    def test_matrix_hard_12_depends_on_dealer(self) -> None:
        hand = make_cards("10", "2")
        self.assertIs(main.advanced_strategy_decision(hand, 2), Action.HIT)
        self.assertIs(main.advanced_strategy_decision(hand, 4), Action.STAND)
        self.assertIs(main.advanced_strategy_decision(hand, 7), Action.HIT)

    def test_matrix_hard_16_vs_ten_hits_and_vs_six_stands(self) -> None:
        hand = make_cards("10", "6")
        self.assertIs(main.advanced_strategy_decision(hand, 10), Action.HIT)
        self.assertIs(main.advanced_strategy_decision(hand, 6), Action.STAND)

    def test_matrix_soft_18_depends_on_dealer(self) -> None:
        hand = make_cards("A", "7")
        self.assertIs(main.advanced_strategy_decision(hand, 8), Action.STAND)
        self.assertIs(main.advanced_strategy_decision(hand, 9), Action.HIT)
        self.assertIs(main.advanced_strategy_decision(hand, 11), Action.HIT)

    def test_matrix_soft_17_always_hits_and_soft_19_always_stands(self) -> None:
        for upcard in range(2, 12):
            self.assertIs(main.advanced_strategy_decision(make_cards("A", "6"), upcard), Action.HIT)
            self.assertIs(main.advanced_strategy_decision(make_cards("A", "8"), upcard), Action.STAND)

    def test_matrix_covers_every_total_and_upcard(self) -> None:
        for total in range(4, 22):
            self.assertEqual(set(main.HARD_STRATEGY_MATRIX[total]), set(main.DEALER_UPCARD_VALUES))
        for total in range(12, 22):
            self.assertEqual(set(main.SOFT_STRATEGY_MATRIX[total]), set(main.DEALER_UPCARD_VALUES))

    def test_bad_strategy_row_length_raises(self) -> None:
        with self.assertRaises(ValueError):
            main._parse_strategy_row("HHS")


class BetSizingTests(unittest.TestCase):
    """Flat and fractional Kelly sizing."""

    def test_flat_bet_is_constant(self) -> None:
        sizer = FlatBetSizer(10.0)
        self.assertEqual(sizer.size_bet(1000.0, 6), 10.0)
        self.assertEqual(sizer.size_bet(55.0, 10), 10.0)

    def test_flat_bet_clipped_to_bankroll(self) -> None:
        self.assertEqual(FlatBetSizer(10.0).size_bet(4.5, 6), 4.5)
        self.assertEqual(FlatBetSizer(10.0).size_bet(0.0, 6), 0.0)

    def test_flat_bet_rejects_non_positive_amount(self) -> None:
        with self.assertRaises(ValueError):
            FlatBetSizer(0.0)

    def test_kelly_uses_minimum_bet_when_edge_negative(self) -> None:
        sizer = FractionalKellyBetSizer(KellyConfig(minimum_bet=5.0))
        self.assertEqual(sizer.size_bet(1000.0, 10), 5.0)

    def test_kelly_scales_with_bankroll_when_edge_positive(self) -> None:
        config = KellyConfig(kelly_fraction=0.5, max_bankroll_fraction=0.05,
                             minimum_bet=1.0, prior_weight=1.0, prior_mean=0.0,
                             prior_variance=1.0)
        sizer = FractionalKellyBetSizer(config)
        for _ in range(1000):
            sizer.record_outcome(5, 0.04)  # Constant +4% return per unit
        small_bet = sizer.size_bet(1000.0, 5)
        large_bet = sizer.size_bet(2000.0, 5)
        self.assertGreater(small_bet, 1.0)
        self.assertAlmostEqual(large_bet, 2 * small_bet, places=1)

    def test_kelly_bet_capped_at_max_fraction(self) -> None:
        config = KellyConfig(kelly_fraction=1.0, max_bankroll_fraction=0.02,
                             minimum_bet=1.0, prior_weight=1.0)
        sizer = FractionalKellyBetSizer(config)
        for _ in range(500):
            sizer.record_outcome(6, 1.0)  # Absurdly strong edge
        self.assertAlmostEqual(sizer.size_bet(1000.0, 6), 20.0, places=2)

    def test_kelly_bet_capped_at_table_maximum(self) -> None:
        config = KellyConfig(kelly_fraction=1.0, max_bankroll_fraction=1.0,
                             minimum_bet=1.0, maximum_bet=75.0, prior_weight=1.0)
        sizer = FractionalKellyBetSizer(config)
        for _ in range(500):
            sizer.record_outcome(6, 1.0)
        self.assertEqual(sizer.size_bet(1_000_000.0, 6), 75.0)

    def test_kelly_never_bets_more_than_bankroll(self) -> None:
        sizer = FractionalKellyBetSizer(KellyConfig(minimum_bet=5.0))
        self.assertEqual(sizer.size_bet(3.0, 6), 3.0)
        self.assertEqual(sizer.size_bet(0.0, 6), 0.0)

    def test_kelly_estimates_are_conditioned_on_hand_type(self) -> None:
        config = KellyConfig(prior_weight=1.0, prior_mean=0.0, prior_variance=1.0,
                             condition_on_dealer_upcard=True)
        sizer = FractionalKellyBetSizer(config)
        for _ in range(200):
            sizer.record_outcome(6, 0.5)
            sizer.record_outcome(10, -0.5)
        weak_edge, _ = sizer.estimate_edge_and_variance(5)
        strong_edge, _ = sizer.estimate_edge_and_variance(10)
        self.assertGreater(weak_edge, 0.0)
        self.assertLess(strong_edge, 0.0)

    def test_pooled_mode_ignores_dealer_upcard(self) -> None:
        config = KellyConfig(prior_weight=1.0, prior_mean=0.0, prior_variance=1.0,
                             condition_on_dealer_upcard=False)
        sizer = FractionalKellyBetSizer(config)
        for _ in range(100):
            sizer.record_outcome(6, 0.5)
            sizer.record_outcome(10, -0.5)
        self.assertEqual(sizer.estimate_edge_and_variance(5),
                         sizer.estimate_edge_and_variance(10))

    def test_upcard_classification(self) -> None:
        self.assertEqual(main.classify_dealer_upcard(2), "weak")
        self.assertEqual(main.classify_dealer_upcard(6), "weak")
        self.assertEqual(main.classify_dealer_upcard(7), "neutral")
        self.assertEqual(main.classify_dealer_upcard(8), "neutral")
        self.assertEqual(main.classify_dealer_upcard(9), "strong")
        self.assertEqual(main.classify_dealer_upcard(11), "strong")

    def test_invalid_kelly_config_raises(self) -> None:
        with self.assertRaises(ValueError):
            FractionalKellyBetSizer(KellyConfig(kelly_fraction=0.0))
        with self.assertRaises(ValueError):
            FractionalKellyBetSizer(KellyConfig(max_bankroll_fraction=1.5))


class DrawdownTests(unittest.TestCase):
    """Maximum drawdown math."""

    def test_classic_peak_to_trough(self) -> None:
        self.assertAlmostEqual(main.calculate_max_drawdown([1000, 1200, 600, 900]), 0.5)

    def test_monotonic_increase_has_zero_drawdown(self) -> None:
        self.assertEqual(main.calculate_max_drawdown([100, 110, 120, 150]), 0.0)

    def test_total_loss_is_full_drawdown(self) -> None:
        self.assertAlmostEqual(main.calculate_max_drawdown([1000, 500, 0]), 1.0)

    def test_largest_of_multiple_drawdowns_is_reported(self) -> None:
        series = [100, 90, 100, 200, 150, 200, 100]
        self.assertAlmostEqual(main.calculate_max_drawdown(series), 0.5)

    def test_drawdown_measured_from_running_peak_not_start(self) -> None:
        self.assertAlmostEqual(main.calculate_max_drawdown([100, 300, 240]), 0.2)

    def test_empty_and_single_point_series(self) -> None:
        self.assertEqual(main.calculate_max_drawdown([]), 0.0)
        self.assertEqual(main.calculate_max_drawdown([1000]), 0.0)

    def test_flat_series_has_zero_drawdown(self) -> None:
        self.assertEqual(main.calculate_max_drawdown([500, 500, 500]), 0.0)


class VolatilityTests(unittest.TestCase):
    """Block returns, standard deviation and Sharpe-equivalent."""

    def test_block_returns_use_block_start_value(self) -> None:
        series = [100.0] + [100.0] * 99 + [110.0] + [110.0] * 99 + [99.0]
        # series has 201 points -> 200 hands -> two complete blocks of 100.
        returns = main.calculate_block_returns(series, 100)
        self.assertEqual(len(returns), 2)
        self.assertAlmostEqual(returns[0], 0.10)
        self.assertAlmostEqual(returns[1], -0.10)

    def test_incomplete_block_excluded_by_default_and_included_on_request(self) -> None:
        series = [100.0] * 101 + [50.0] * 30
        self.assertEqual(len(main.calculate_block_returns(series, 100)), 1)
        with_tail = main.calculate_block_returns(series, 100, include_partial_tail=True)
        self.assertEqual(len(with_tail), 2)
        self.assertAlmostEqual(with_tail[1], -0.5)

    def test_invalid_block_size_raises(self) -> None:
        with self.assertRaises(ValueError):
            main.calculate_block_returns([1.0, 2.0], 0)

    def test_summary_statistics_match_manual_calculation(self) -> None:
        mean_return, standard_deviation, sharpe = main.summarize_block_returns([0.01, 0.03])
        self.assertAlmostEqual(mean_return, 0.02)
        self.assertAlmostEqual(standard_deviation, 0.014142135623730951)
        self.assertAlmostEqual(sharpe, 0.02 / 0.014142135623730951)

    def test_zero_variance_gives_zero_sharpe(self) -> None:
        self.assertEqual(main.summarize_block_returns([0.01, 0.01, 0.01])[2], 0.0)

    def test_empty_and_single_inputs(self) -> None:
        self.assertEqual(main.summarize_block_returns([]), (0.0, 0.0, 0.0))
        self.assertEqual(main.summarize_block_returns([0.05]), (0.05, 0.0, 0.0))


class SimulationTests(unittest.TestCase):
    """End-to-end simulation behaviour."""

    def _small_config(self, hands: int = 500) -> SimulationConfig:
        return SimulationConfig(number_of_hands=hands, random_seed=123)

    def test_simulation_is_deterministic_for_same_seed(self) -> None:
        config = self._small_config()
        first = main.run_simulation("a", main.baseline_strategy_decision, FlatBetSizer(10.0), config)
        second = main.run_simulation("a", main.baseline_strategy_decision, FlatBetSizer(10.0), config)
        self.assertEqual(first.bankroll_history, second.bankroll_history)

    def test_counts_sum_to_hands_played(self) -> None:
        result = main.run_simulation("a", main.advanced_strategy_decision,
                                     FractionalKellyBetSizer(), self._small_config())
        self.assertEqual(result.wins + result.losses + result.pushes, result.hands_played)
        self.assertEqual(len(result.bankroll_history), result.hands_played + 1)

    def test_flat_bankroll_change_matches_payout_total(self) -> None:
        result = main.run_simulation("a", main.baseline_strategy_decision,
                                     FlatBetSizer(10.0), self._small_config(200))
        if not result.ruined:
            # With a flat $10 bet each net change is -10, 0, +10 or +15.
            for previous, current in zip(result.bankroll_history, result.bankroll_history[1:]):
                self.assertIn(round(current - previous, 2), (-10.0, 0.0, 10.0, 15.0))

    def test_ruin_stops_simulation_at_zero(self) -> None:
        config = SimulationConfig(number_of_hands=5000, starting_bankroll=20.0,
                                  random_seed=1)
        result = main.run_simulation("a", main.baseline_strategy_decision,
                                     FlatBetSizer(10.0), config)
        self.assertTrue(result.ruined)
        self.assertEqual(result.final_bankroll, 0.0)
        self.assertLess(result.hands_played, 5000)

    def test_bankroll_never_negative(self) -> None:
        config = SimulationConfig(number_of_hands=3000, starting_bankroll=50.0,
                                  random_seed=9)
        result = main.run_simulation("a", main.baseline_strategy_decision,
                                     FlatBetSizer(10.0), config)
        self.assertGreaterEqual(min(result.bankroll_history), 0.0)

    def test_both_strategies_face_identical_decks(self) -> None:
        import random

        generator_one = random.Random(5)
        generator_two = random.Random(5)
        for _ in range(20):
            deck_one = main.build_standard_deck()
            deck_two = main.build_standard_deck()
            main.shuffle_deck(deck_one, generator_one)
            main.shuffle_deck(deck_two, generator_two)
            self.assertEqual(deck_one, deck_two)


class ReportTests(unittest.TestCase):
    """Report assembly and formatting."""

    def test_report_fields_and_win_rate(self) -> None:
        config = SimulationConfig(number_of_hands=1000, random_seed=11)
        result = main.run_simulation("flat", main.baseline_strategy_decision,
                                     FlatBetSizer(10.0), config)
        report = main.build_risk_report(result, config.starting_bankroll)
        self.assertAlmostEqual(report.win_rate, result.wins / result.hands_played)
        self.assertEqual(report.ruined, result.ruined)
        self.assertAlmostEqual(report.net_profit, result.final_bankroll - 1000.0, places=2)
        self.assertGreaterEqual(report.max_drawdown, 0.0)
        self.assertLessEqual(report.max_drawdown, 1.0)

    def test_formatted_report_contains_required_sections(self) -> None:
        config = SimulationConfig(number_of_hands=300, random_seed=3)
        reports = [
            main.build_risk_report(
                main.run_simulation("Flat", main.baseline_strategy_decision,
                                    FlatBetSizer(10.0), config), config.starting_bankroll),
            main.build_risk_report(
                main.run_simulation("Kelly", main.advanced_strategy_decision,
                                    FractionalKellyBetSizer(), config), config.starting_bankroll),
        ]
        text = main.format_comparison_report(reports, config)
        for expected in ("Wins", "Losses", "Pushes", "Overall Win Rate", "Final Bankroll",
                         "Maximum Drawdown", "Risk of Ruin", "Std Dev of Returns",
                         "Sharpe-Equivalent", "Monte Carlo approximations"):
            self.assertIn(expected, text)

    def test_report_lines_are_aligned(self) -> None:
        config = SimulationConfig(number_of_hands=300, random_seed=3)
        result = main.run_simulation("Flat", main.baseline_strategy_decision,
                                     FlatBetSizer(10.0), config)
        report = main.build_risk_report(result, config.starting_bankroll)
        text = main.format_comparison_report([report], config)
        row_lengths = {len(line) for line in text.splitlines()
                       if line.startswith(("Hands Played", "Losses", "Final Bankroll"))}
        self.assertEqual(len(row_lengths), 1)


if __name__ == "__main__":
    unittest.main()
