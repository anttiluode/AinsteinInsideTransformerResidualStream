# Gate 0.6 result — reusable causal route steering

Date: 2026-09-21

Source receipt: `results/gate06_causal_route_steering.json`

## Pre-registered gate result

Gate 0.6 passes strongly.

The centered route vectors were derived only from the two 17x24 paraphrases per strategy, with the shared carrier removed:

    C = mean(R_A, R_B, R_C)
    D_i = R_i - C

They were written once into the neutral prompt state entering block 31.

On the source problem 17x24:

- target-win rate: 3/3;
- diagonal mean lift: +0.05535 nats/token;
- off-diagonal mean lift: -0.03243;
- diagonal advantage: **+0.08778**.

On held-out 13x26:

- target-win rate: 3/3;
- diagonal advantage: **+0.08428**.

On held-out 19x23:

- target-win rate: 3/3;
- diagonal advantage: **+0.08416**.

Mean held-out diagonal advantage is **+0.08422**, essentially unchanged from the source problem.

## Route signatures

The three routes are not mechanistically identical.

Route A (decomposition) barely increases its own continuation and instead wins mainly by suppressing the alternatives. On 17x24:

    A -> A: +0.00193
    A -> B: -0.12825
    A -> C: -0.04094

The same inhibitory/selective pattern persists on both held-out problems.

Route B is positively excitatory toward the repeated-addition continuation:

    +0.08310 on 17x24
    +0.07260 on 13x26
    +0.08496 on 19x23

Route C is likewise positively excitatory toward the nearby-multiple continuation:

    +0.08101 on 17x24
    +0.09011 on 13x26
    +0.07119 on 19x23

This matters because a reusable route direction need not simply add semantic content. It can change the **relative accessibility of future trajectories** by suppressing competitors.

## Centering is causal, not cosmetic

The common carrier alone strongly favors route B on the source problem:

    carrier -> A: +0.00898
    carrier -> B: +0.08514
    carrier -> C: +0.01179

The raw uncentered C route therefore lifts B and C almost equally:

    raw R_C -> B: +0.09047
    raw R_C -> C: +0.08871

After subtracting the carrier, centered D_C becomes selective:

    D_C -> B: +0.02234
    D_C -> C: +0.08101

So the empirical decomposition

    R_i = C + D_i

has real behavioral meaning in this experiment:

- C carries a shared task/answer-preparation bias;
- D_i carries route-specific steering.

## Narrow claim earned

The current evidence supports the following bounded statement:

> Qwen3-8B contains a paraphrase-invariant layer-30 direction whose one-shot intervention causally biases which arithmetic method continuation is preferred, and the direction transfers from 17x24 to held-out multiplication problems.

This is not yet evidence that the vector encodes the full algorithm.

## Remaining language-carrier attacker

Gate 0.6 still scored prose continuations that explicitly described the methods.

Gate 0.7 therefore removes method-language entirely and scores only the first concrete arithmetic action. If the same D_i vectors selectively favor those numeric moves on held-out problems, the result becomes substantially harder to explain as verbal framing.
