# Gate 0.7c — causal temporal footprint

Gate 0.7 and Gate 0.7b exposed an asymmetry that a single steering score cannot describe.

- Route A had a strong equation-only first-action signature but weakened over the longer trajectory.
- Route B was weak at the first action and recovered when the readout included repeated additions.
- Route C remained positive at both scales.

The next question is therefore not merely whether a route vector works. It is:

> **For how long, and on which successive transitions, does one fixed route intervention remain causally useful?**

This gate is deliberately narrower than the broader "local vector field" idea. It does **not** claim that the useful direction itself has been measured at each hidden state. It measures the temporal support profile of one frozen intervention.

## Frozen intervention

Use the exact centered route vectors from Gates 0.6–0.7b:

    D_A, D_B, D_C

derived only from the source-problem strategy paraphrases at layer 30.

Each vector is written once into the neutral problem state before block 31. There is no refitting by problem, horizon, or trajectory step.

## Numeric-only horizons

For each multiplication problem, construct the same three equation-only routes as Gate 0.7b and score prefixes at:

    k = 1, 2, 3

For route i, define the cumulative causal target margin

    M_i(k)
      = lift(D_i -> target route prefix i at k)
        - max_j!=i lift(D_i -> route prefix j at k).

Also derive an **incremental** margin for the newly added transition at each k by subtracting cumulative token log-probability between adjacent prefixes.

This gives two views:

- cumulative: how much the route is preferred after k transitions;
- incremental: where along the trajectory the intervention is actually doing work.

## Pre-registered shape hypothesis

The qualitative prediction is:

    A: strongest early, then decays
    B: weak early, improves with recurrence depth
    C: positive and comparatively persistent

The minimal cross-problem shape checks are:

1. mean M_A(1) > 0 and mean M_A(1) > mean M_A(3);
2. mean M_B(3) > mean M_B(1);
3. mean M_C(k) > 0 for k = 1,2,3.

These are deliberately qualitative. Exact numbers are not fitted in advance.

## Route-B operator attacker

Gate 0.7b could be fooled if D_B merely likes repeated plus signs.

The true three-step route is:

    a+a
    previous+a
    previous+a

It must therefore compete against equation-only decoys with the same operator family:

### Same first edge, wrong continuation

The first transition is identical to B, then the recurrence switches to +1.

This is the strongest local attacker because the first arithmetic move is held fixed.

### Unit recurrence

Every step is +1.

This preserves addition and recurrence depth but destroys the specific transformation law.

### Other-operand recurrence

Every step adds b instead of a.

This preserves the constant-increment recurrent form while binding the wrong problem quantity to the operator.

The relevant statistic is the D_B lift of the true three-step recurrence minus the best decoy lift.

A strong result requires the true law to beat the matched decoys on at least two of the three source/held-out problems. Carrier and orthogonal-random controls are checked on the source problem.

## Interpretation boundary

Passing this gate earns only:

> a frozen route intervention has a route-specific causal horizon, and B contains information more specific than generic repeated addition if it survives the matched recurrence attacker.

It does **not** yet establish:

- a dynamically rotating local vector field D(h);
- a biological mood/state analogue;
- free-running algorithm execution;
- useful residue collision;
- compute savings.

Those remain later tests.

## Run

    python3.13 gate07c_temporal_footprint.py

The default 6 GiB GPU + 6 GiB CPU placement caps intentionally preserve the conservative disk-offload profile that survived Gate 0.7b on Windows.
