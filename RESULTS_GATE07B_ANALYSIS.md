# Gate 0.7b result — trajectory length changes the route signature

Date: 2026-09-21

Source receipt: user-run `gate07b_numeric_trajectory.json`

## Outcome

Gate 0.7b is a mixed result and does not satisfy its strongest criterion.

Across the source problem and two held-out multiplication problems:

- source target-win rate: 2/3;
- held-out mean target-win rate: 2/3;
- all-problem minimum target margin: -0.00407.

The useful part is not the aggregate pass/fail label. The three frozen directions changed differently when the readout was extended from one arithmetic move to a short numeric trajectory.

## Route A: strong entrance, weak persistence

Gate 0.7 first-action margins:

    +0.00931, +0.00983, +0.01003

Gate 0.7b trajectory margins:

    -0.00407, +0.00165, -0.00093

Mean trajectory margin is approximately -0.00112.

The fixed A direction therefore looks more like an **entrance intervention** than a reusable whole-route algorithm.

## Route B: weak entrance, recovery over recurrence

Gate 0.7 first-action margins:

    -0.01657, +0.00026, -0.01208

Gate 0.7b trajectory margins:

    +0.01007, +0.00635, -0.00214

Mean changes from approximately -0.00946 at the isolated first move to +0.00476 over the short trajectory.

This is consistent with the pre-registered hypothesis that B is more trajectory-like than point-action-like, but one held-out problem remains negative.

## Route C: persistent short-route signature

Gate 0.7b trajectory margins:

    +0.00636, +0.00640, +0.00767

C therefore remains the cleanest direction so far: it was positive on the first-action gate and stays positive when the readout spans the short correction trajectory.

## New confound

The B trajectory contains repeated `+` operations. A route-B lift could therefore reflect generic addition syntax rather than the recurrence law

    s_(t+1) = s_t + a.

That confound must be attacked before calling B a transformation-law direction.

## Next gate

Gate 0.7c measures the full horizon profile k=1,2,3 and adds recurrence-matched decoys.

The new object of interest is not a global steering vector, but a **causal temporal footprint**:

    direction + support horizon.

This is an empirical description only. The experiment does not yet establish that the useful direction rotates with hidden state.
