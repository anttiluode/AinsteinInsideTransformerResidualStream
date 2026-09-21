# Gate 0.7e — rise above the numerical floor

Gate 0.7d failed its semantic invariance check for the **first-action** readout.

The same frozen route vectors and identical candidate strings changed qualitative interpretation when unrelated longer candidates were added to the BF16 batch:

- A changed from positive target margin / A winner to negative margin / B winner;
- B changed from negative target margin / A winner to positive margin / B winner;
- C stayed stable.

Reversing the three-row batch order was bit-for-bit stable, while enlarging the padded batch changed the result. The trajectory readout remained semantically stable despite raw numerical drift.

That pattern points to **batch-shape-dependent low-precision numerics**, not ordinary stochasticity.

## Question

Is the route effect real but too close to the present numerical floor at scale 1?

A genuine causal direction should become easier to distinguish from low-precision execution drift when the same intervention is strengthened.

## Pre-registered sweep

Use the same model load, same source problem, same D_A/D_B/D_C vectors and same two batch contexts from Gate 0.7d.

Test intervention scales:

    0.5, 1.0, 2.0

The primary test is fixed in advance at:

    scale = 2.0

No scale is selected after seeing the result.

## Primary pass condition

At scale 2.0, between the legacy three-row batch and the larger Gate-0.7c batch:

1. every route's target-margin sign must agree;
2. every route's winning target must agree.

The receipt also records the cosine similarity of the two 3x3 lift matrices and their signal-to-disagreement ratio.

## Interpretation

A pass does **not** rescue the old scale-1 measurements.

It earns only:

> the frozen route intervention produces a semantic effect large enough to survive the current BF16 batch-shape perturbation at a prospectively chosen stronger scale.

If it passes, the next temporal-footprint run should use that stable scale and remain fixed there.

If it fails, stop increasing scale. The next move is a different measurement apparatus: candidate-serial scoring, a generation/KV-cache scorer, or higher precision downstream execution.

## Run

    python3.13 gate07e_numerical_floor.py
