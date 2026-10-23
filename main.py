"""Quantitative Capital Allocation & Risk Engine.

Monte Carlo evaluation of position-sizing strategies (flat betting versus a
fractional Kelly Criterion module) in an imperfect-information environment
(Blackjack) over a finite capital pool.

Design notes
------------
* Python 3 standard library only.
* A fresh, fully shuffled 52-card deck is used for every hand. Each strategy
  gets its own ``random.Random`` seeded identically, and that generator is
  consumed ONLY by deck shuffles. Consequently both strategies face the exact
  same sequence of shuffled decks (hand N is identical for both), which makes
  the comparison paired and removes cross-strategy sampling noise.
* The player's choices are limited to HIT or STAND (no doubling, splitting,
  insurance or surrender), matching the rule set of the specification.
* Dealer peeks for a natural: a dealer natural ends the round immediately.
* Betting timing: a conventional table takes bets BEFORE any card is seen, so
  the default Kelly module ("blind") uses one pooled edge estimate. As an
  explicit what-if, ``KellyConfig.condition_on_dealer_upcard=True`` lets the
  module size its wager AFTER the dealer upcard is visible, with separate
  edge estimates per hand type (weak / neutral / strong upcard). That is
  NOT how a real table works; it models an information advantage.
* A table maximum bet (``KellyConfig.maximum_bet``) bounds every wager so
  compounding cannot produce unphysical bankrolls.

Run:  python3 main.py
"""

from __future__ import annotations

import random
import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUITS: Tuple[str, ...] = ("Clubs", "Diamonds", "Hearts", "Spades")
RANKS: Tuple[str, ...] = (
    "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A",
)
BLACKJACK_TARGET: int = 21
DEALER_STAND_THRESHOLD: int = 17
BLACKJACK_PAYOUT_MULTIPLIER: float = 1.5
HANDS_PER_VOLATILITY_BLOCK: int = 100
DEALER_UPCARD_VALUES: Tuple[int, ...] = tuple(range(2, 12))  # 2..10, Ace = 11


# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------

class Action(Enum):
    """A player decision."""

    HIT = "hit"
    STAND = "stand"


class Outcome(Enum):
    """Result of a single round from the player's perspective."""

    WIN = "win"
    LOSS = "loss"
    PUSH = "push"
    BLACKJACK = "blackjack"  # Player natural that beats the dealer (pays 3:2)


@dataclass(frozen=True)
class Card:
    """An immutable playing card."""

    rank: str
    suit: str


@dataclass(frozen=True)
class HandValue:
    """Best total of a hand and whether an Ace is still counted as 11."""

    total: int
    is_soft: bool


DecisionFunction = Callable[[Sequence[Card], int], Action]


# ---------------------------------------------------------------------------
# Deck / card state
# ---------------------------------------------------------------------------

def build_standard_deck() -> List[Card]:
    """Return an ordered standard 52-card deck (4 suits x 13 ranks)."""
    return [Card(rank=rank, suit=suit) for suit in SUITS for rank in RANKS]


def shuffle_deck(deck: List[Card], random_generator: random.Random) -> None:
    """Shuffle ``deck`` in place using the supplied random generator."""
    random_generator.shuffle(deck)


def draw_card(deck: List[Card]) -> Card:
    """Remove and return the top card of the deck (the end of the list).

    Raises:
        IndexError: if the deck is empty.
    """
    if not deck:
        raise IndexError("Cannot draw from an empty deck.")
    return deck.pop()


def deal_initial_hands(deck: List[Card]) -> Tuple[List[Card], List[Card]]:
    """Deal two cards each in casino order: player, dealer, player, dealer.

    Returns:
        (player_cards, dealer_cards). ``dealer_cards[0]`` is the upcard.
    """
    player_cards: List[Card] = []
    dealer_cards: List[Card] = []
    player_cards.append(draw_card(deck))
    dealer_cards.append(draw_card(deck))
    player_cards.append(draw_card(deck))
    dealer_cards.append(draw_card(deck))
    return player_cards, dealer_cards


# ---------------------------------------------------------------------------
# Game rules
# ---------------------------------------------------------------------------

def card_base_value(card: Card) -> int:
    """Return the face value of a card with Ace counted as 11."""
    if card.rank == "A":
        return 11
    if card.rank in ("J", "Q", "K"):
        return 10
    return int(card.rank)


def hand_value(cards: Sequence[Card]) -> HandValue:
    """Compute the best hand total, demoting Aces from 11 to 1 as needed.

    Every Ace starts at 11. While the total exceeds 21 and at least one Ace
    is still valued at 11, one Ace is reduced to 1 (subtract 10).
    """
    total = sum(card_base_value(card) for card in cards)
    aces_counted_as_eleven = sum(1 for card in cards if card.rank == "A")
    while total > BLACKJACK_TARGET and aces_counted_as_eleven > 0:
        total -= 10
        aces_counted_as_eleven -= 1
    return HandValue(total=total, is_soft=aces_counted_as_eleven > 0)


def is_natural_blackjack(cards: Sequence[Card]) -> bool:
    """True for an initial two-card 21."""
    return len(cards) == 2 and hand_value(cards).total == BLACKJACK_TARGET


def dealer_should_hit(dealer_cards: Sequence[Card]) -> bool:
    """Dealer hits strictly below 17 and stands on any 17+ (including soft 17)."""
    return hand_value(dealer_cards).total < DEALER_STAND_THRESHOLD


def resolve_round(
    deck: List[Card],
    player_cards: Sequence[Card],
    dealer_cards: Sequence[Card],
    decision_function: DecisionFunction,
) -> Outcome:
    """Play out one round from already-dealt hands and return the outcome.

    Order of resolution: naturals, player turn (bust is an immediate loss),
    dealer turn, then comparison of final totals.
    """
    player_hand: List[Card] = list(player_cards)
    dealer_hand: List[Card] = list(dealer_cards)

    player_has_natural = is_natural_blackjack(player_hand)
    dealer_has_natural = is_natural_blackjack(dealer_hand)
    if player_has_natural and dealer_has_natural:
        return Outcome.PUSH
    if player_has_natural:
        return Outcome.BLACKJACK
    if dealer_has_natural:
        return Outcome.LOSS

    dealer_upcard_value = card_base_value(dealer_hand[0])

    while True:
        current_total = hand_value(player_hand).total
        if current_total >= BLACKJACK_TARGET:
            break
        if decision_function(player_hand, dealer_upcard_value) is Action.STAND:
            break
        player_hand.append(draw_card(deck))

    player_total = hand_value(player_hand).total
    if player_total > BLACKJACK_TARGET:
        return Outcome.LOSS  # Player bust: dealer never plays.

    while dealer_should_hit(dealer_hand):
        dealer_hand.append(draw_card(deck))

    dealer_total = hand_value(dealer_hand).total
    if dealer_total > BLACKJACK_TARGET:
        return Outcome.WIN
    if player_total > dealer_total:
        return Outcome.WIN
    if player_total < dealer_total:
        return Outcome.LOSS
    return Outcome.PUSH


def calculate_payout(outcome: Outcome, bet_amount: float) -> float:
    """Net profit/loss of a round, rounded to cents.

    WIN: +bet. BLACKJACK: +1.5 x bet (3:2). PUSH: 0. LOSS: -bet.
    """
    if outcome is Outcome.WIN:
        net_result = bet_amount
    elif outcome is Outcome.BLACKJACK:
        net_result = bet_amount * BLACKJACK_PAYOUT_MULTIPLIER
    elif outcome is Outcome.PUSH:
        net_result = 0.0
    else:
        net_result = -bet_amount
    return round(net_result, 2)


# ---------------------------------------------------------------------------
# Playing strategies (hit / stand decision logic)
# ---------------------------------------------------------------------------

def baseline_strategy_decision(
    player_cards: Sequence[Card], dealer_upcard_value: int
) -> Action:
    """Fixed baseline: stand on 16+, hit otherwise. Ignores the dealer upcard."""
    del dealer_upcard_value  # Intentionally unused by the baseline.
    if hand_value(player_cards).total >= 16:
        return Action.STAND
    return Action.HIT


def _parse_strategy_row(pattern: str) -> Dict[int, Action]:
    """Convert an 'H'/'S' string (dealer 2..10, A) into an action mapping."""
    if len(pattern) != len(DEALER_UPCARD_VALUES):
        raise ValueError(
            f"Strategy row must have {len(DEALER_UPCARD_VALUES)} entries: {pattern!r}"
        )
    return {
        upcard: (Action.HIT if symbol == "H" else Action.STAND)
        for upcard, symbol in zip(DEALER_UPCARD_VALUES, pattern)
    }


def _build_hard_strategy_matrix() -> Dict[int, Dict[int, Action]]:
    """Hard-total matrix (hit/stand only), columns are dealer 2,3,...,10,A."""
    matrix: Dict[int, Dict[int, Action]] = {}
    for total in range(4, 22):
        if total <= 11:
            pattern = "HHHHHHHHHH"
        elif total == 12:
            pattern = "HHSSSHHHHH"
        elif total <= 16:
            pattern = "SSSSSHHHHH"
        else:
            pattern = "SSSSSSSSSS"
        matrix[total] = _parse_strategy_row(pattern)
    return matrix


def _build_soft_strategy_matrix() -> Dict[int, Dict[int, Action]]:
    """Soft-total matrix (Ace counted as 11), columns are dealer 2,...,10,A."""
    matrix: Dict[int, Dict[int, Action]] = {}
    for total in range(12, 22):
        if total <= 17:
            pattern = "HHHHHHHHHH"
        elif total == 18:
            pattern = "SSSSSSSHHH"
        else:
            pattern = "SSSSSSSSSS"
        matrix[total] = _parse_strategy_row(pattern)
    return matrix


HARD_STRATEGY_MATRIX: Dict[int, Dict[int, Action]] = _build_hard_strategy_matrix()
SOFT_STRATEGY_MATRIX: Dict[int, Dict[int, Action]] = _build_soft_strategy_matrix()


def advanced_strategy_decision(
    player_cards: Sequence[Card], dealer_upcard_value: int
) -> Action:
    """Matrix basic strategy keyed on (player total, dealer upcard)."""
    current_hand = hand_value(player_cards)
    if current_hand.total >= BLACKJACK_TARGET:
        return Action.STAND
    matrix = SOFT_STRATEGY_MATRIX if current_hand.is_soft else HARD_STRATEGY_MATRIX
    return matrix[current_hand.total][dealer_upcard_value]


# ---------------------------------------------------------------------------
# Bet-sizing algorithms
# ---------------------------------------------------------------------------

class BetSizer(ABC):
    """Interface for position-sizing models."""

    @abstractmethod
    def size_bet(self, current_bankroll: float, dealer_upcard_value: int) -> float:
        """Return the wager for the next hand (never above the bankroll)."""

    @abstractmethod
    def record_outcome(self, dealer_upcard_value: int, unit_return: float) -> None:
        """Observe the realised return per unit staked on a finished hand."""


class FlatBetSizer(BetSizer):
    """Constant wager, clipped to the remaining bankroll."""

    def __init__(self, flat_bet_amount: float = 10.0) -> None:
        if flat_bet_amount <= 0:
            raise ValueError("flat_bet_amount must be positive.")
        self.flat_bet_amount = flat_bet_amount

    def size_bet(self, current_bankroll: float, dealer_upcard_value: int) -> float:
        """Return the flat bet, or the whole bankroll if it is smaller."""
        del dealer_upcard_value
        if current_bankroll <= 0:
            return 0.0
        return round(min(self.flat_bet_amount, current_bankroll), 2)

    def record_outcome(self, dealer_upcard_value: int, unit_return: float) -> None:
        """Flat betting is non-adaptive, so nothing is learned."""
        del dealer_upcard_value, unit_return


@dataclass
class KellyConfig:
    """Parameters of the fractional Kelly module.

    Attributes:
        kelly_fraction: Multiplier on full Kelly (0.5 = half Kelly).
        max_bankroll_fraction: Hard cap on a bet as a share of the bankroll.
        minimum_bet: Table minimum wagered when estimated edge is not positive.
        maximum_bet: Table maximum; no wager ever exceeds it.
        prior_mean: Prior expected return per unit staked. Hit/stand-only
            basic strategy (no doubling, splitting or surrender) loses about
            2% to 2.5% per hand, so the prior starts there.
        prior_variance: Prior variance of the return per unit staked.
        prior_weight: Pseudo-observations behind the prior (Bayesian shrinkage).
        condition_on_dealer_upcard: If True, bet after seeing the dealer
            upcard using per-hand-type estimates (weak / neutral / strong).
            If False (default), bet blind using one pooled estimate.
    """

    kelly_fraction: float = 0.5
    max_bankroll_fraction: float = 0.05
    minimum_bet: float = 5.0
    maximum_bet: float = 500.0
    prior_mean: float = -0.02
    prior_variance: float = 1.3
    prior_weight: float = 500.0
    condition_on_dealer_upcard: bool = False


@dataclass
class OutcomeStatistics:
    """Running sums of per-unit returns for one hand type."""

    observation_count: int = 0
    total_return: float = 0.0
    total_squared_return: float = 0.0


def classify_dealer_upcard(dealer_upcard_value: int) -> str:
    """Bucket the dealer upcard into a hand type: weak, neutral or strong."""
    if 2 <= dealer_upcard_value <= 6:
        return "weak"
    if dealer_upcard_value in (7, 8):
        return "neutral"
    return "strong"


class FractionalKellyBetSizer(BetSizer):
    """Bankroll-proportional sizing from online edge and variance estimates.

    For a wager with expected return ``mu`` and return variance ``sigma^2`` per
    unit staked, the continuous-approximation Kelly fraction is
    ``f* = mu / sigma^2``. Blackjack has pushes and a 3:2 natural, so ``mu`` and
    ``sigma^2`` are estimated empirically (shrunk toward a prior) rather than
    assuming a binary payoff. The applied fraction is
    ``min(max(f*, 0) * kelly_fraction, max_bankroll_fraction)``.
    Non-positive estimated edge falls back to the table minimum.
    """

    HAND_TYPES: Tuple[str, ...] = ("weak", "neutral", "strong")

    def __init__(self, config: Optional[KellyConfig] = None) -> None:
        self.config = config if config is not None else KellyConfig()
        if not 0 < self.config.kelly_fraction <= 1:
            raise ValueError("kelly_fraction must be in (0, 1].")
        if not 0 < self.config.max_bankroll_fraction <= 1:
            raise ValueError("max_bankroll_fraction must be in (0, 1].")
        self._statistics_by_hand_type: Dict[str, OutcomeStatistics] = {
            hand_type: OutcomeStatistics() for hand_type in self.HAND_TYPES
        }

    def _pooled_statistics(self) -> OutcomeStatistics:
        """Combine all hand types into one set of statistics."""
        pooled = OutcomeStatistics()
        for statistics_entry in self._statistics_by_hand_type.values():
            pooled.observation_count += statistics_entry.observation_count
            pooled.total_return += statistics_entry.total_return
            pooled.total_squared_return += statistics_entry.total_squared_return
        return pooled

    def estimate_edge_and_variance(
        self, dealer_upcard_value: int
    ) -> Tuple[float, float]:
        """Posterior-style estimate of (edge, variance) for the next hand."""
        if self.config.condition_on_dealer_upcard:
            hand_type = classify_dealer_upcard(dealer_upcard_value)
            observed = self._statistics_by_hand_type[hand_type]
        else:
            observed = self._pooled_statistics()

        prior_weight = self.config.prior_weight
        total_weight = prior_weight + observed.observation_count
        estimated_edge = (
            prior_weight * self.config.prior_mean + observed.total_return
        ) / total_weight
        prior_second_moment = self.config.prior_variance + self.config.prior_mean ** 2
        estimated_second_moment = (
            prior_weight * prior_second_moment + observed.total_squared_return
        ) / total_weight
        estimated_variance = max(estimated_second_moment - estimated_edge ** 2, 1e-9)
        return estimated_edge, estimated_variance

    def kelly_bankroll_fraction(self, dealer_upcard_value: int) -> float:
        """Applied (fractional, capped, non-negative) share of bankroll to bet."""
        estimated_edge, estimated_variance = self.estimate_edge_and_variance(
            dealer_upcard_value
        )
        full_kelly_fraction = estimated_edge / estimated_variance
        scaled_fraction = max(full_kelly_fraction, 0.0) * self.config.kelly_fraction
        return min(scaled_fraction, self.config.max_bankroll_fraction)

    def size_bet(self, current_bankroll: float, dealer_upcard_value: int) -> float:
        """Return the dollar wager for the next hand."""
        if current_bankroll <= 0:
            return 0.0
        applied_fraction = self.kelly_bankroll_fraction(dealer_upcard_value)
        proportional_bet = current_bankroll * applied_fraction
        bet_amount = max(proportional_bet, self.config.minimum_bet)
        bet_amount = min(bet_amount, self.config.maximum_bet)
        bet_amount = min(bet_amount, current_bankroll)
        return round(bet_amount, 2)

    def record_outcome(self, dealer_upcard_value: int, unit_return: float) -> None:
        """Update the estimator for the hand type of the finished hand."""
        hand_type = classify_dealer_upcard(dealer_upcard_value)
        statistics_entry = self._statistics_by_hand_type[hand_type]
        statistics_entry.observation_count += 1
        statistics_entry.total_return += unit_return
        statistics_entry.total_squared_return += unit_return ** 2


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

@dataclass
class SimulationConfig:
    """Global simulation parameters shared by every strategy."""

    number_of_hands: int = 50_000
    starting_bankroll: float = 1_000.0
    random_seed: int = 42
    flat_bet_amount: float = 10.0
    kelly_config: KellyConfig = field(default_factory=KellyConfig)
    informed_kelly_config: KellyConfig = field(
        default_factory=lambda: KellyConfig(condition_on_dealer_upcard=True)
    )


@dataclass
class SimulationResult:
    """Raw output of one strategy run."""

    strategy_name: str
    hands_played: int
    wins: int
    losses: int
    pushes: int
    naturals: int
    final_bankroll: float
    ruined: bool
    bankroll_history: List[float]


def run_simulation(
    strategy_name: str,
    decision_function: DecisionFunction,
    bet_sizer: BetSizer,
    config: SimulationConfig,
) -> SimulationResult:
    """Run sequential hands until the hand budget is used or the bankroll is 0.

    The random generator is consumed only by shuffling, so any two strategies
    run with the same seed see identical decks on identical hand numbers.
    """
    random_generator = random.Random(config.random_seed)
    current_bankroll = round(config.starting_bankroll, 2)
    bankroll_history: List[float] = [current_bankroll]
    wins = 0
    losses = 0
    pushes = 0
    naturals = 0
    ruined = False
    hands_played = 0

    for _ in range(config.number_of_hands):
        deck = build_standard_deck()
        shuffle_deck(deck, random_generator)
        player_cards, dealer_cards = deal_initial_hands(deck)
        dealer_upcard_value = card_base_value(dealer_cards[0])

        bet_amount = bet_sizer.size_bet(current_bankroll, dealer_upcard_value)
        outcome = resolve_round(deck, player_cards, dealer_cards, decision_function)
        net_result = calculate_payout(outcome, bet_amount)

        current_bankroll = round(current_bankroll + net_result, 2)
        bet_sizer.record_outcome(dealer_upcard_value, net_result / bet_amount)
        bankroll_history.append(current_bankroll)
        hands_played += 1

        if outcome is Outcome.WIN:
            wins += 1
        elif outcome is Outcome.BLACKJACK:
            wins += 1
            naturals += 1
        elif outcome is Outcome.LOSS:
            losses += 1
        else:
            pushes += 1

        if current_bankroll <= 0:
            ruined = True
            break

    return SimulationResult(
        strategy_name=strategy_name,
        hands_played=hands_played,
        wins=wins,
        losses=losses,
        pushes=pushes,
        naturals=naturals,
        final_bankroll=current_bankroll,
        ruined=ruined,
        bankroll_history=bankroll_history,
    )


# ---------------------------------------------------------------------------
# Statistical aggregation
# ---------------------------------------------------------------------------

def calculate_max_drawdown(bankroll_series: Sequence[float]) -> float:
    """Largest peak-to-trough decline as a fraction of the peak (0.0 to 1.0).

    Example: [1000, 1200, 600, 900] -> peak 1200, trough 600 -> 0.5.
    """
    running_peak = float("-inf")
    maximum_drawdown = 0.0
    for bankroll_value in bankroll_series:
        if bankroll_value > running_peak:
            running_peak = bankroll_value
        if running_peak > 0:
            drawdown = (running_peak - bankroll_value) / running_peak
            if drawdown > maximum_drawdown:
                maximum_drawdown = drawdown
    return maximum_drawdown


def calculate_block_returns(
    bankroll_series: Sequence[float],
    block_size: int = HANDS_PER_VOLATILITY_BLOCK,
    include_partial_tail: bool = False,
) -> List[float]:
    """Fractional bankroll return over consecutive blocks of ``block_size`` hands.

    ``bankroll_series`` includes the starting value at index 0. A block's
    return is (end - start) / start. Blocks that begin at or below zero are
    skipped. An incomplete trailing block is included only on request.
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    hands_available = len(bankroll_series) - 1
    block_returns: List[float] = []
    block_start_index = 0
    while block_start_index + block_size <= hands_available:
        start_value = bankroll_series[block_start_index]
        end_value = bankroll_series[block_start_index + block_size]
        if start_value > 0:
            block_returns.append((end_value - start_value) / start_value)
        block_start_index += block_size
    if include_partial_tail and block_start_index < hands_available:
        start_value = bankroll_series[block_start_index]
        end_value = bankroll_series[hands_available]
        if start_value > 0:
            block_returns.append((end_value - start_value) / start_value)
    return block_returns


def summarize_block_returns(block_returns: Sequence[float]) -> Tuple[float, float, float]:
    """Return (mean, sample standard deviation, Sharpe-equivalent ratio).

    The Sharpe-equivalent is mean / standard deviation with a zero risk-free
    rate and no annualisation. Degenerate inputs yield 0.0 for the affected
    values.
    """
    if not block_returns:
        return 0.0, 0.0, 0.0
    mean_return = statistics.fmean(block_returns)
    if len(block_returns) < 2:
        return mean_return, 0.0, 0.0
    standard_deviation = statistics.stdev(block_returns)
    if standard_deviation == 0.0:
        return mean_return, 0.0, 0.0
    return mean_return, standard_deviation, mean_return / standard_deviation


# ---------------------------------------------------------------------------
# Risk reporting
# ---------------------------------------------------------------------------

@dataclass
class RiskReport:
    """Derived metrics for one strategy."""

    strategy_name: str
    hands_played: int
    wins: int
    losses: int
    pushes: int
    naturals: int
    win_rate: float
    final_bankroll: float
    net_profit: float
    max_drawdown: float
    ruined: bool
    mean_block_return: float
    block_return_stdev: float
    sharpe_equivalent: float


def build_risk_report(
    result: SimulationResult, starting_bankroll: float
) -> RiskReport:
    """Compute classical, drawdown, ruin and volatility metrics for a run."""
    win_rate = result.wins / result.hands_played if result.hands_played else 0.0
    block_returns = calculate_block_returns(
        result.bankroll_history,
        HANDS_PER_VOLATILITY_BLOCK,
        include_partial_tail=result.ruined,
    )
    mean_return, standard_deviation, sharpe_equivalent = summarize_block_returns(
        block_returns
    )
    return RiskReport(
        strategy_name=result.strategy_name,
        hands_played=result.hands_played,
        wins=result.wins,
        losses=result.losses,
        pushes=result.pushes,
        naturals=result.naturals,
        win_rate=win_rate,
        final_bankroll=result.final_bankroll,
        net_profit=round(result.final_bankroll - starting_bankroll, 2),
        max_drawdown=calculate_max_drawdown(result.bankroll_history),
        ruined=result.ruined,
        mean_block_return=mean_return,
        block_return_stdev=standard_deviation,
        sharpe_equivalent=sharpe_equivalent,
    )


def format_comparison_report(
    reports: Sequence[RiskReport], config: SimulationConfig
) -> str:
    """Render a cleanly aligned side-by-side terminal report."""
    label_width = 34
    column_width = 26
    total_width = label_width + column_width * len(reports)
    heavy_rule = "=" * total_width
    light_rule = "-" * total_width

    metric_rows: List[Tuple[str, Callable[[RiskReport], str]]] = [
        ("CLASSICAL METRICS", lambda report: ""),
        ("Hands Played", lambda report: f"{report.hands_played:,}"),
        ("Wins (incl. naturals)", lambda report: f"{report.wins:,}"),
        ("Losses", lambda report: f"{report.losses:,}"),
        ("Pushes", lambda report: f"{report.pushes:,}"),
        ("Natural Blackjacks", lambda report: f"{report.naturals:,}"),
        ("Overall Win Rate", lambda report: f"{report.win_rate * 100:.2f}%"),
        ("Final Bankroll", lambda report: f"${report.final_bankroll:,.2f}"),
        ("Net Profit / Loss", lambda report: f"${report.net_profit:,.2f}"),
        ("RISK METRICS", lambda report: ""),
        ("Maximum Drawdown", lambda report: f"{report.max_drawdown * 100:.2f}%"),
        ("Risk of Ruin (bankroll hit $0)", lambda report: "YES" if report.ruined else "NO"),
        ("VOLATILITY (per 100 hands)", lambda report: ""),
        ("Mean Return", lambda report: f"{report.mean_block_return * 100:.3f}%"),
        ("Std Dev of Returns", lambda report: f"{report.block_return_stdev * 100:.3f}%"),
        ("Sharpe-Equivalent Ratio", lambda report: f"{report.sharpe_equivalent:.4f}"),
    ]

    lines: List[str] = [
        heavy_rule,
        "QUANTITATIVE CAPITAL ALLOCATION & RISK ENGINE".center(total_width),
        heavy_rule,
        (
            f"Hands: {config.number_of_hands:,} | Seed: {config.random_seed} | "
            f"Starting Bankroll: ${config.starting_bankroll:,.2f}"
        ),
        light_rule,
        "Metric".ljust(label_width)
        + "".join(report.strategy_name.rjust(column_width) for report in reports),
        light_rule,
    ]
    for label, value_getter in metric_rows:
        if not value_getter(reports[0]) and label.isupper():
            lines.append(label)
            continue
        lines.append(
            label.ljust(label_width)
            + "".join(value_getter(report).rjust(column_width) for report in reports)
        )
    lines.append(light_rule)
    lines.append(
        "NOTE: These results are Monte Carlo approximations reflecting statistical"
    )
    lines.append(
        "variance under a fixed seed, not absolute analytical solutions."
    )
    lines.append(
        "Sharpe-equivalent = mean / sample std dev of 100-hand returns (risk-free = 0)."
    )
    lines.append(
        "Kelly (blind) bets before any card is seen. Kelly (upcard) bets after the dealer"
    )
    lines.append(
        "upcard is visible, which is an information advantage a real table does not give."
    )
    lines.append(heavy_rule)
    return "\n".join(lines)


def print_risk_report(reports: Sequence[RiskReport], config: SimulationConfig) -> None:
    """Print the formatted comparison report to the terminal."""
    print(format_comparison_report(reports, config))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run both strategies under identical seeds and print the risk report."""
    config = SimulationConfig()

    baseline_result = run_simulation(
        strategy_name="Baseline Flat $10",
        decision_function=baseline_strategy_decision,
        bet_sizer=FlatBetSizer(config.flat_bet_amount),
        config=config,
    )
    advanced_result = run_simulation(
        strategy_name="Matrix + Kelly (blind)",
        decision_function=advanced_strategy_decision,
        bet_sizer=FractionalKellyBetSizer(config.kelly_config),
        config=config,
    )
    informed_result = run_simulation(
        strategy_name="Matrix + Kelly (upcard)",
        decision_function=advanced_strategy_decision,
        bet_sizer=FractionalKellyBetSizer(config.informed_kelly_config),
        config=config,
    )

    reports = [
        build_risk_report(baseline_result, config.starting_bankroll),
        build_risk_report(advanced_result, config.starting_bankroll),
        build_risk_report(informed_result, config.starting_bankroll),
    ]
    print_risk_report(reports, config)


if __name__ == "__main__":
    main()
