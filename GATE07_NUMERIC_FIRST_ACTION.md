# Gate 0.7 — numeric first-action steering

Gate 0.6 established a causal result: centered layer-30 route vectors derived only from 17x24 produced a 3/3 matching-route preference on the source problem and on both held-out multiplication problems, with nearly unchanged diagonal advantage.

That result still has one major alternative explanation:

> the edit may steer the model toward **language describing a method**, rather than toward the computation performed by that method.

Gate 0.7 removes that carrier.

## Question

Does the same frozen route vector selectively bias the **first concrete arithmetic operation** associated with its route when the candidate continuation contains no method words?

No route vector is retrained or refit.

## Route vectors

Exactly as in Gate 0.6:

    R_A, R_B, R_C = route centroids from two 17x24 paraphrases each
    C = mean(R_A, R_B, R_C)
    D_i = R_i - C

The primary edits are the centered route departures D_i. They are written once into the neutral prompt state entering block 31.

## Numeric first actions

For a neutral multiplication problem a x b, score only concrete equations.

Example for 17 x 24:

- A: `17 * 20 = 340`
- B: `17 + 17 = 34`
- C: `17 * 25 = 425`

For 13 x 26:

- A: `13 * 20 = 260`
- B: `13 + 13 = 26`
- C: `13 * 27 = 351`

The candidate strings contain none of the method labels used to define the routes. A guard aborts if forbidden strategy words appear.

All three equations are correct statements. The metric asks which **next computational move** becomes more likely, not which answer is correct.

## Measurement

    L(i,j) =
      avg_logp(numeric move j | +D_i)
      - avg_logp(numeric move j | baseline)

For each route row, the matching action should receive the highest lift.

Report target-win rate, diagonal/off-diagonal mean lift, diagonal advantage, per-route target margins, and the minimum target margin across problems.

## Transfer

Route vectors are still derived only from 17x24. Default held-out problems remain 13x26 and 19x23.

If the edit changes the numeric first move on unseen numbers, the result is substantially harder to explain as memorized wording from the source prompts.

## Source controls

Repeat on 17x24 with no edit, common carrier C, norm-matched random orthogonal direction, and raw uncentered R_i.

## Strong pass

At scale 1.0:

- source target-win rate >= 2/3;
- source diagonal advantage > 0;
- held-out mean target-win rate >= 2/3;
- held-out mean diagonal advantage > 0;
- minimum target margin across all route/problem rows > 0;
- carrier/random controls do not reproduce a route-aligned diagonal.

## Weak pass

The source numeric action is steerable, but held-out transfer fails.

## Fail

Route-specific preference disappears when method-language is removed.

## Consequence

A strong pass earns a narrower claim:

> a paraphrase-invariant layer-30 route direction causally and transferably biases the next concrete arithmetic operation.

It still does not prove an entire algorithm is encoded in one vector.

After a pass, the next attacker is **free rollout**: inject D_i once, let the model generate without forced continuations, and classify the actual arithmetic trace. Only after that should the nonlinear AInstein collision become the main experiment.