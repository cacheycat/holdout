# Overfitting

## Multiple testing
Every strategy you try is a draw from a distribution of Sharpe ratios. Try enough of them and the best one looks good by luck alone. The deflated Sharpe gate models this: the expected maximum Sharpe under the null grows with the number of trials, so a strategy found late must be stronger than one found early. Fewer, better-motivated trials beat a wide sweep.

## Parameter cliffs
A rule that works at RSI 27 and fails at RSI 30 has found a coincidence, not an edge. Robust rules sit on a plateau: nearby thresholds and lookbacks give similar results. Round numbers and wide tolerances survive jitter; three-decimal thresholds do not.

## Thin samples
A Sharpe ratio estimated from fifteen trades has an error bar wider than the estimate. Conditions that fire rarely, such as three filters that must all hold, produce attractive curves and unusable statistics. Loosen the rarest condition before adding another one.

## Train to validation decay
Some decay is normal. Validation Sharpe near zero or negative after a strong train result means the rule fitted noise in the training segment. Strategies that hold at least half their Sharpe out of sample are worth refining; the rest are not.
