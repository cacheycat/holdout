# Mean reversion

## Why it can work
Forced selling and liquidations push price further than information justifies, and it snaps back. The edge is short-lived, measured in a handful of bars, and concentrated after volatility spikes.

## Construction notes
Buy oversold conditions such as RSI under 30 or a Bollinger z-score under minus two, but only inside an uptrend regime: dips in an uptrend recover, dips in a downtrend continue. Exit on reversion to the mean, such as z-score back above zero or RSI above 55, and cap the holding period, because a dip that has not recovered in twenty bars is a trend.

## Failure modes
Catching falling knives: without a regime filter the rule buys every leg of a bear market. Stops interact badly with mean reversion, since the best entries are the ones that look worst; a time exit is often better than a tight price stop. Shorting overbought conditions on an asset with upward drift has historically been much weaker than buying oversold ones.
