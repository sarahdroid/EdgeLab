# EdgeLab

**A Quantitative Capital Allocation & Risk Engine.** EdgeLab is a Monte Carlo lab that asks one question: *given a finite bankroll and an imperfect-information game, how much should you bet, and what does that choice do to your risk?*

It compares flat betting against a fractional Kelly Criterion in Blackjack, measures the result with risk metrics a trading desk would recognize (drawdown, risk of ruin, Sharpe-equivalent), and then stress-tests Kelly against a real-world constraint: **casino table limits**.

- Python 3 **standard library only**. No NumPy, no Pandas, no Matplotlib.
- Charts are hand-built **SVG files**, so they open in any browser and render on GitHub.
- Deterministic: fixed seeds, and every strategy sees the *same shuffled decks* hand for hand.

---

## Quick start

```bash
python3 main.py                      # strategy comparison report
python3 main.py --charts charts      # ...and write SVG charts into ./charts
python3 table_limits.py              # table-limit experiment + charts (about 45 s)
python3 -m unittest discover -v      # 130 tests
```

Options: `main.py --hands N --seed N --charts DIR`, and `table_limits.py --seeds N --hands N --bankroll X --out DIR`.

## Project layout

| File | Purpose |
|---|---|
| `main.py` | The core engine: deck state, game rules, strategies, bet sizing, simulation, statistics, risk report. Runs standalone. |
| `table_limits.py` | The table-limit experiment: runs Kelly under a ladder of bet caps and charts the result. |
| `charts.py` | Dependency-free SVG line/bar chart renderer. |
| `test_main.py` | Engine tests: Ace inversion, dealer boundaries, payouts, drawdown, strategies, Kelly sizing. |
| `test_charts.py`, `test_table_limits.py` | Tests for charting and the experiment. |
| `charts/` | Generated SVG output. |

## The game

- A fresh, fully shuffled 52-card deck **every hand**. No counting is possible.
- 2 to 10 at face value, J/Q/K = 10, Ace = 11 that demotes to 1 whenever the hand would otherwise bust (each Ace is demoted independently).
- Dealer hits below 17 and stands on 17+ (including soft 17). A player bust is an immediate loss.
- Natural blackjack (two-card 21) pays 3:2. Both sides holding one is a push. A dealer natural ends the round at once.
- Player options are hit or stand only. There is no doubling, splitting, insurance or surrender, which is why the house edge here is larger than at a real table (roughly 2% to 2.5% per hand under good hit/stand play).

## Strategies

**Baseline.** Flat $10 per hand, stand on 16+, hit otherwise.

**Matrix + fractional Kelly.** Hit/stand decisions come from hard-total and soft-total matrices keyed on (player total, dealer upcard). Bet size comes from a fractional Kelly module:

```
f* = edge / variance                        (continuous Kelly fraction)
applied fraction = min( max(f*, 0) * 0.5,   half Kelly
                        5% of bankroll )    hard cap against ruin
bet = clamp(bankroll * fraction, $5 minimum, table maximum, bankroll)
```

Blackjack has pushes and a 3:2 natural, so a simple win-probability formula would be wrong. Instead the module learns the **expected return and variance per unit staked online**, shrunk toward a prior (Bayesian-style, 500 pseudo-observations, prior edge -2%), and keeps separate estimates per **hand type**: dealer upcard weak (2 to 6), neutral (7 to 8) or strong (9 to A).

Two variants run side by side, and the difference matters:

| Variant | When it bets | Realistic? |
|---|---|---|
| **Kelly (blind)** | Before any card is seen, using one pooled estimate | Yes, this is how a real table works |
| **Kelly (upcard)** | After the dealer upcard is visible, using per-hand-type estimates | **No.** It models an information advantage |

## Metrics

| Metric | Definition |
|---|---|
| Wins / Losses / Pushes / Win rate | Wins include natural blackjacks. Win rate is wins divided by hands played. |
| Maximum drawdown | Largest peak-to-trough fall of the bankroll, as a percentage of the peak. |
| Risk of ruin | Binary: did the bankroll reach $0? A ruined run stops there. |
| Std dev per 100 hands | Sample standard deviation of non-overlapping 100-hand bankroll returns. |
| Sharpe-equivalent | Mean divided by standard deviation of those 100-hand returns. Risk-free rate is 0 and nothing is annualized. |

## Results

50,000 hands, $1,000 start, seed 42:

| | Flat $10 | Kelly (blind) | Kelly (upcard) |
|---|---:|---:|---:|
| Hands survived | 1,426 | 8,401 | 50,000 |
| Final bankroll | $0.00 | $0.00 | $1,540,821.85 |
| Max drawdown | 100% | 100% | 45.3% |
| Risk of ruin | Yes | Yes | No |
| Std dev / 100 hands | 26.98% | 26.16% | 6.72% |
| Sharpe-equivalent | -0.74 | -0.13 | 0.25 |

![Bankroll over time](charts/strategy_bankroll_paths.svg)

![Drawdown over time](charts/strategy_drawdown.svg)

**How to read this honestly.**

1. **Kelly cannot create an edge.** In the realistic (blind) setting the edge is negative, so a correct Kelly sizer wants to bet *nothing*. The $5 table minimum forces a small wager, and the bankroll still bleeds out, just 6x slower than flat betting.
2. **Kelly shines only when an edge exists.** The upcard variant has one because weak dealer upcards favor the player. There Kelly produced the best returns *and* the smoothest ride.
3. **A single seed is one draw from a distribution.** Treat the numbers as Monte Carlo approximations of statistical variance, not analytical solutions.

---

## Table limits: when the casino caps your bet

Kelly sizing is *proportional*: bet a fixed fraction `f` of your current bankroll `B`. Wins raise the next bet, which is what makes growth exponential. A table maximum `M` breaks that feedback loop:

```
bet = min( f * B , M )
```

This creates a **crossover bankroll**:

```
B* = M / f
```

| Regime | Bet | Growth |
|---|---|---|
| `B < B*` (cap not binding) | `f * B`, scales with bankroll | Exponential. Log growth per hand is about `f*mu - f^2*sigma^2 / 2` |
| `B > B*` (cap binding) | Pinned at `M` | **Linear.** Profit per hand is about `M * mu`, so `B(n) ~ B* + M*mu*n` |

Above `B*` the effective fraction `M / B` keeps shrinking, so the log growth rate decays toward zero (about `M*mu / B`). The cap also *lowers relative risk*: a fixed bet becomes a smaller and smaller slice of a growing bankroll. Kelly is "underbetting" by design at that point, and the casino has quietly capped your compounding.

In this engine the peak fraction sits at the 5% hard cap, so `B* = 20 x M`.

### Experiment

`table_limits.py` runs the upcard-informed Kelly module (the variant that has an edge) under eight limits. Every limit sees identical decks for a given seed, and results are medians over 4 seeds of 50,000 hands each:

| Table limit | Median final | Log growth / 100 hands | Max drawdown | Ruin | Hands at cap | Cap first binds at | Predicted B* |
|---|---:|---:|---:|---:|---:|---:|---:|
| $25 | $62.3K | 0.83% | 29.1% | 0% | 52.6% | $1.1K | $500 |
| $50 | $143.6K | 0.99% | 29.1% | 0% | 52.4% | $1.6K | $1K |
| $100 | $305K | 1.14% | 34.8% | 0% | 52.1% | $2.8K | $2K |
| $250 | $783.8K | 1.33% | 37.7% | 0% | 51.5% | $6.2K | $5K |
| $500 | $1.6M | 1.47% | 37.7% | 0% | 51.2% | $12.1K | $10K |
| $1,000 | $3.1M | 1.61% | 39.6% | 0% | 50.9% | $22.8K | $20K |
| $5,000 | $15.6M | 1.93% | 55.4% | 0% | 50.0% | $105.8K | $100K |
| No limit | $1.7e+58 | 24.73% | 73.3% | 0% | 0% | never | n/a |

![Bankroll paths by table limit](charts/table_limits_bankroll_paths.svg)

![Median final bankroll](charts/table_limits_final_bankroll.svg)

![Growth rate](charts/table_limits_growth_rate.svg)

![Drawdown](charts/table_limits_drawdown.svg)

### What it shows

1. **The cap is the whole story.** With no limit, the same strategy compounds to about $10^58, which is absurd and exactly why casinos set limits. Even a $5,000 cap cuts the log growth rate by more than 12x.
2. **The bend is visible.** In the paths chart each capped curve follows the uncapped one until its crossover, then flattens from exponential into a near-straight line on the (log-scaled) axis.
3. **The measured crossover tracks the prediction.** The bankroll at which the cap first binds lands close to `M / 5%` for large limits. For small limits it binds later than predicted, most likely because the edge estimator needs time to learn, so early fractions sit below the end-of-run peak that the prediction uses.
4. **Final bankroll scales roughly with the limit; the growth rate barely moves.** Each step up the ladder multiplies the final bankroll by a similar factor to the limit (as `B* + M*mu*n` implies), while log growth per 100 hands creeps from 0.83% to 1.93% across a 200x range of limits.
5. **Tighter caps mean a gentler ride.** Median drawdown climbs from 29% at $25 to 73% uncapped. A plausible reason is that capped runs stake a shrinking share of their bankroll once the cap binds, while uncapped runs keep staking up to 5% on every hand.

### Caveats

- The upcard-informed module bets after seeing the dealer upcard, which a real table does not allow. Read the absolute dollars as a stress test of the *mechanism*, not a forecast.
- Medians over 4 seeds are still noisy. Run `python3 table_limits.py --seeds 20` for tighter numbers (about 4 minutes).
- The model has no table minimums beyond $5, no bet spread rules, no dealer-penetration effects and no trading costs.

## Design notes

- **Paired comparison.** The random generator is consumed *only* by deck shuffles, so hand N is identical for every strategy and every limit. Differences in outcome come from decisions, not luck.
- **Cents accounting.** Bets and bankrolls are rounded to cents, and a bet can never exceed the remaining bankroll.
- **Separation of concerns.** Deck state, rules, strategies, bet sizers, simulation, statistics and reporting are distinct functions or classes. `main.py` never imports `charts.py` unless `--charts` is used.

## Extending edgelab

- Add a sizer by subclassing `BetSizer` (`size_bet` and `record_outcome`).
- Add a playing strategy by writing a function `(player_cards, dealer_upcard_value) -> Action`.
- Add a limit to the ladder by editing `DEFAULT_TABLE_LIMITS` in `table_limits.py`.
- Try `KellyConfig(kelly_fraction=1.0)` (full Kelly) against half Kelly to see why practitioners fractionalize.
