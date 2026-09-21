# Gate 0.7b — numeric trajectory signature

Gate 0.7 removed method-language and scored one concrete arithmetic move per route.

It did **not** meet the strong-pass criterion:

- source 17x24: 2/3 target wins, diagonal advantage +0.00941;
- held-out 13x26: 3/3 target wins, diagonal advantage +0.01121;
- held-out 19x23: 2/3 target wins, diagonal advantage +0.00781;
- the minimum target margin was negative (-0.01657).

The failure is concentrated in route B. Route A and route C retain positive target margins on all three problems, while B fails on 17x24 and 19x23 and barely wins 13x26.

## New hypothesis

The first-action gate may be structurally unfair to B.

A and C have distinctive first transforms:

    A: a * round_chunk
    C: a * nearby_factor

B's identity is not one isolated addition. It is the recurrence:

    s_(k+1) = s_k + a

Thus B may be represented as a **temporal transformation law** rather than as a preference for one first edge.

This hypothesis is pre-registered before seeing Gate-0.7b results.

## Readout

Use the exact same frozen route vectors as Gates 0.6 and 0.7.

Score short, equation-only trajectories. No alphabetic characters are permitted in the candidate strings.

For 17x24:

    A: 17 * 20 = 340; 17 * 4 = 68; 340 + 68 = 408
    B: 17 + 17 = 34; 34 + 17 = 51; 51 + 17 = 68
    C: 17 * 25 = 425; 425 - 17 = 408; 408 = 17 * 24

The same templates are generated for held-out multiplication problems.

## Why this is different from Gate 0.7

Gate 0.7 asked whether D_B points toward the first addition.

Gate 0.7b asks whether D_B biases a **sequence governed by repeated application of the same operator**.

This is closer to the GAx / Temporal-Sihti notion that route identity can live in the transformation law across states rather than in one state.

## Strong pass

- source target-win rate >= 2/3;
- held-out mean target-win rate >= 2/3;
- all-problem minimum target margin > 0;
- route-B margin is positive on the source and every held-out problem;
- carrier/random controls do not reproduce a route-aligned diagonal.

## Partial pass

A and C remain robustly selective but B still fails.

Interpretation: two route directions have numeric causal content, while the earlier B result was likely more dependent on verbal-plan framing.

## Fail

The A/C effects also disappear when scored over equation-only trajectories.

Interpretation: Gate 0.7's numeric-action effects were too local/fragile to support a route-level claim.

## After a strong pass

Proceed to free rollout: one-shot injection, no forced candidate trajectories, then classify the generated arithmetic trace.

## Educated directions and accidents

This gate also motivates a broader scheduling distinction.

Scientific search should spend most compute along **educated directions** distilled from prior successful trajectories, but reserve a small exploration budget for directions poorly explained by the current route basis.

A useful exploratory proposal can be written schematically as:

    d = sum_i alpha_i D_i + eta

with eta containing an outside-basis component.

Novelty alone is not valuable. An accident earns more resolution only when it produces useful surprise or information:

    score(d) = expected_value + beta * information_gain
               + gamma * outside_basis_novelty - lambda * compute_cost

This is an architecture hypothesis, not a claim about biological cognition.