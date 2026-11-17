# Trend following

## Why it can work
Volatile assets trend because information and positioning diffuse slowly and because leverage forces buyers and sellers to chase. Trend rules earn their return in a few large moves and give back small amounts in ranges, so win rates are low and the payoff is skewed.

## Construction notes
Slow filters such as close above a 150 to 250 bar average or a fast EMA above a slow EMA define the regime. Breakouts, measured as position near the top of a Donchian range, time the entry. Exits should be slower than entries: a trend rule that exits on the first pullback never holds the move that pays for it. Wide ATR stops of three multiples or more, or no stop with a regime exit, usually beat tight stops on 4-hour bars.

## Failure modes
Short lookbacks under ten bars mostly trade noise and costs. Long-only trend rules look excellent on a training segment that happened to rise; check that the rule goes flat, not just less long, when the regime filter turns.
