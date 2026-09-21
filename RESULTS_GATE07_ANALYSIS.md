# Gate 0.7 result — first-action route steering is route-dependent

Date: 2026-09-21

Source receipt: `results/gate07_numeric_first_action.json`

## Outcome

Gate 0.7 does **not** satisfy its strong-pass criterion.

| problem | target wins | diagonal advantage | minimum target margin |
|---|---:|---:|---:|
| 17x24 | 2/3 | +0.00941 | -0.01657 |
| 13x26 | 3/3 | +0.01121 | +0.00026 |
| 19x23 | 2/3 | +0.00781 | -0.01208 |

Mean held-out target-win rate is 0.8333 and mean held-out diagonal advantage is +0.00951, but the all-problem minimum target margin is negative.

## A and C survive the no-language attack

Route A target margins:

    17x24: +0.00931
    13x26: +0.00983
    19x23: +0.01003

Route C target margins:

    17x24: +0.02110
    13x26: +0.01510
    19x23: +0.01777

These are strikingly stable across the held-out numbers.

Neither A nor C necessarily works by increasing its own move. In several rows the matching move is also suppressed, but competitors are suppressed more strongly. The route vector therefore behaves more like a local reshaping of future accessibility than like a semantic command vector.

## B is the failure

Route B target margins:

    17x24: -0.01657
    13x26: +0.00026
    19x23: -0.01208

Thus the bare first operation `a + a` is not a robust causal signature for B.

## Pre-registered next hypothesis

This failure suggests a structural asymmetry:

- A has a distinctive first transform: `a * round_chunk`;
- C has a distinctive first transform: `a * nearby_factor`;
- B is defined by a recurrence: `s_(k+1) = s_k + a`.

So B may require a **trajectory-level** readout to reveal its identity.

Gate 0.7b tests this directly with equation-only multi-step trajectories. No method-language is permitted.

## Scope discipline

The Gate-0.6 claim must be narrowed:

> method-level route steering transfers strongly, while method-language-free first-action steering is robust for A/C but not B.

Do not claim that all three route vectors already steer arbitrary concrete arithmetic actions.
