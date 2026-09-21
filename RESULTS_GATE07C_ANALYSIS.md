# Gate 0.7c result — the preregistered horizon story does not survive

Date: 2026-09-21

Source receipt: `gate07c_temporal_footprint.json`

## Preregistered checks

Mean cumulative target margins across 17x24, 13x26 and 19x23:

| route | k=1 | k=2 | k=3 |
|---|---:|---:|---:|
| A | -0.00178 | -0.00573 | -0.00234 |
| B | +0.01236 | +0.00542 | +0.00383 |
| C | +0.01915 | +0.01157 | +0.01625 |

The preregistered shape checks therefore give:

- A short-horizon hypothesis: **fail**;
- B late-horizon hypothesis: **fail**;
- C persistent-positive hypothesis: **pass**;
- B true-recurrence-law beats matched decoys on a majority of problems: **pass (2/3)**.

The aggregate `all_shape_checks_pass` is false.

## The incremental view is even less like a scalar lifetime

Mean incremental target margins:

| route | transition 1 | transition 2 | transition 3 |
|---|---:|---:|---:|
| A | -0.00178 | -0.01134 | +0.00026 |
| B | +0.01236 | -0.00189 | -0.00085 |
| C | +0.01915 | -0.00369 | +0.02576 |

If these values are trustworthy, the causal footprint is not a simple monotone "direction lifetime." B is front-loaded, while C is strongest on the first and third transitions with a dip on the middle correction step.

## B recurrence attacker

The true B law beats the best matched decoy on:

- 17x24: +0.00523 margin;
- 13x26: +0.00075 margin;
- 19x23: -0.00798 margin.

So the generic-plus attacker does not fully explain B, but the effect is not robust across all held-out problems.

## Reproducibility problem discovered by this gate

Before interpreting any of those shapes, Gate 0.7c exposes a more basic inconsistency.

Gate 0.7 previously reported A as robustly positive on all three first-action problems and B as the first-action failure. Gate 0.7c's k=1 readout, which uses the same candidate strings and frozen route definitions, instead makes mean A negative and mean B positive.

Likewise, the k=3 values do not exactly reproduce Gate 0.7b.

That difference may be a low-precision/batch-shape numerical effect or a scoring-path bug. Either way, it is large enough relative to the tiny causal margins that the temporal interpretation must be suspended.

## Next gate

Gate 0.7d is therefore a scoring-invariance audit.

The same model load, prompt, route vectors and candidate strings are scored in the legacy three-row batches and inside the larger Gate-0.7c batch. If route-margin signs or route winners change under that harmless batching change, the current microscope is not precise enough for horizon claims.

This is a measurement gate, not a new cognitive hypothesis.
