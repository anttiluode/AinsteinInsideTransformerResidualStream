# Gate 0.6 — causal route steering

Gate 0.5d passed its pre-registered lexical-echo attacker at layer 30:

- raw within-strategy cosine: 0.811 versus between-strategy 0.509 (gap +0.302);
- Jacobian-transported: 0.818 versus 0.513 (gap +0.305);
- full J-lens delta-logit: 0.870 versus 0.676 (gap +0.194);
- all three negated-keyword decoys classified by intended strategy (3/3);
- no decoy was captured by the misleading strategy word (0/3).

This establishes a paraphrase-invariant, lexically robust **route representation** more strongly than the first readable-token experiment did. It still does not establish causality.

## Question

If the layer-30 route residue is part of the computation rather than a passive description of the instruction, does a one-shot linear edit of that route change which valid reasoning continuation the frozen transformer prefers?

## Route decomposition

From the two paraphrases of each strategy on the source problem 17 x 24:

    R_A, R_B, R_C = strategy centroid residues relative to the neutral anchor
    C = mean(R_A, R_B, R_C)
    D_i = R_i - C

`C` is the common branch carrier. `D_i` is the centered route departure.

The primary intervention is **D_i**, not the raw R_i. The previous J-lens receipt showed that Jacobian transport makes different raw branches reconverge, so a raw intervention risks steering mainly through the shared task/answer-preparation carrier.

## Intervention

Layer 30 is measured as the raw output of decoder block 30. The edit is therefore written once into the same token **entering block 31**:

    h_31(prompt boundary) <- h_31 + D_i

No permanent weights are changed.

The edit occurs only at the neutral workspace token. It is not repeatedly added to every candidate continuation token.

## Readout

For each neutral problem, score three *correct* forced continuations:

- A: split the second factor into a round tens chunk plus remainder;
- B: repeated/running addition;
- C: use the next nearby multiple and subtract once.

The primary quantity is lift over the unedited baseline:

    L(i,j) = logp(continuation_j | +D_i) - logp(continuation_j | baseline)

For a causal route code, the diagonal should dominate:

    L(A,A), L(B,B), L(C,C) > corresponding off-diagonal effects.

## Transfer test

Route vectors are derived **only from 17 x 24**.

They are then applied, without refitting, to neutral prompts for held-out multiplication problems (default 13 x 26 and 19 x 23).

This is the stronger test. A vector that merely stores problem-specific wording may steer 17 x 24 but should not systematically select the same algorithm on new numbers.

## Controls

On the source problem also test:

1. no edit;
2. common carrier C alone;
3. norm-matched random direction orthogonal to span(C,D_A,D_B,D_C);
4. raw uncentered R_A/R_B/R_C.

The centered route is favored scientifically if it is more selectively algorithmic than the common carrier/random controls.

## Strong pass

At natural scale 1.0:

- at least 2/3 centered route rows prefer their matching continuation on the source problem;
- diagonal mean lift > off-diagonal mean lift;
- held-out problems retain mean target-win rate >= 2/3 and positive diagonal advantage;
- random/common controls do not show comparable strategy-selective structure.

## Weak pass

The source problem is causally steerable but the route fails to transfer to held-out multiplication problems.

Interpretation: local planned-route state exists, but it is not yet a reusable algorithm vector.

## Fail

The centered route does not preferentially increase its own continuation, or a matched random/common-carrier edit performs similarly.

## Consequence for AInstein

If Gate 0.6 passes, **linear route steering becomes a mandatory attacker** for every later residue-collision claim.

A nonlinear composer does not earn anything by producing a behavior that one of these already-existing route directions can produce linearly.

The next constructive question would then be narrower:

> Can two individually insufficient causal route directions interact to produce a third useful trajectory that neither linear steering direction, their sum, nor a sparse J-coordinate edit can produce?
