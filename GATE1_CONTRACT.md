# Gate 1 contract — computational foveation

## Question

Under a fixed total compute budget, can a transformer do better by first probing several latent futures at low resolution and then purchasing additional depth only for the futures that look useful, while preserving enough diversity to recover from an initially attractive dead end?

The claim is about allocation of computation, not about consciousness, human vision, or biological foveation.

## Core object

A current residual state S forks into candidate futures B_i. Every branch receives a coarse probe depth d0 and leaves a compact residue R_i(d0). A scheduler then allocates additional depth rho_i subject to a hard shared budget:

sum_i rho_i <= B.

The scheduler may use:

- current-state features;
- the coarse residue;
- provenance / branch role;
- a learned estimate of marginal task value;
- uncertainty or information gain;
- branch diversity.

The scheduler may not use the held-out answer or future branch states that have not yet been computed.

## Why this is different from Gate 0

Gate 0 asks whether separately insufficient branch residues can be composed into a useful temporary lens.

Gate 1 asks where the branch compute should be spent in the first place.

The two mechanisms are orthogonal:

1. foveation decides which trajectories deserve resolution;
2. residue collision decides whether selected trajectories should interact;
3. writeback decides whether the result changes the resumed main computation.

A Gate-1 pass does not establish Gate 0, and vice versa.

## Gate 1a — offline layer-trace experiment

The first implementation may cache complete hidden-state traces from a frozen transformer, then pretend that deeper layers had to be purchased one at a time. This is allowed only as an algorithmic test of the scheduler.

For each problem:

1. begin from one common parent state;
2. instantiate the same N branch proposals for every method;
3. expose all methods to the same shallow depth d0;
4. let each method allocate an identical number of additional branch-layer equivalents;
5. score the final answer or target representation using only the depths each method purchased.

This gate must report branch-layer equivalents, not wall-clock savings.

## Gate 1b — real partial execution

A compute-efficiency claim is earned only after branch execution actually stops at unpurchased depths.

Gate 1b must implement partial block execution or another mechanism that avoids the downstream FLOPs of abandoned branches, and must report:

- measured wall time;
- estimated FLOPs / block applications;
- peak memory;
- answer quality.

Cached full-pass traces do not count as a compute win.

## Required baselines

All methods use identical branch proposals and identical total branch-depth budget.

1. no rumination;
2. one deepest branch;
3. random allocation;
4. uniform allocation across branches;
5. greedy allocation with no diversity reserve;
6. adaptive allocation with diversity reserve;
7. oracle allocation using future payoff, as an upper bound only;
8. extra serial reasoning with a matched compute budget;
9. textual Tree-of-Thought style branching with matched compute where practical.

If the adaptive method uses a learned value head, include:

10. shuffled value-head outputs;
11. value head trained on wrong branch provenance;
12. answer-leaking oracle features as a positive control only.

## Late-bloomer attacker

The scheduler must face tasks where:

- an easy branch has high early utility but saturates;
- a different branch looks mediocre at the coarse probe;
- that second branch becomes decisive only after additional depth.

A purely greedy scheduler should fail some of these cases.

The diversity reserve is justified only if it recovers useful late bloomers often enough to improve held-out task reward under the same total compute.

## Purification versus collapse

A selected route may become sharp without deleting every alternative.

At any cycle, maintain:

C_t = current sharp route

and a small quiver of compressed alternatives:

Q_t = {R_i^gist, Z_i}.

An abandoned full trajectory may be discarded, but its gist/provenance can remain addressable. Re-expansion is allowed only by spending new compute.

This is the Temporal Sihti hypothesis in executable form:

- purification = sharpen one route while retaining compressed alternatives;
- collapse = sharpen one route by deleting alternatives.

The gate must measure the cost of keeping alternatives rather than assuming it is free.

## Measurements

Report at minimum:

- task accuracy / reward;
- target-token log probability where applicable;
- total branch-layer equivalents;
- allocation vector rho;
- normalized allocation entropy;
- number of distinct branches receiving more than d0 depth;
- oracle allocation regret;
- late-bloomer recovery rate;
- value-head calibration if learned;
- provenance-swap damage;
- wall time and FLOPs only for Gate 1b.

## Strong pass

Under equal real compute in Gate 1b, adaptive resolution improves held-out task performance or achieves the same performance with less compute than uniform, random, deepest-single, and matched serial reasoning, while surviving provenance and value-head controls.

## Weak pass

Adaptive allocation beats uniform/random in the offline layer-trace accounting of Gate 1a but the real partial-execution implementation does not yet show a wall-clock/FLOP advantage.

Interpretation: scheduler mechanism is useful; compute-efficiency claim remains unearned.

## Fail

The advantage disappears under equal branch proposals, equal budget, or a non-leaking value model; the scheduler simply selects the branch whose answer was already exposed; or uniform allocation performs as well.

## Kill

If repeated tasks show that the value of deeper computation is not predictable from cheaper probes well enough to beat uniform allocation, remove computational foveation from the architecture.

## Relation to J-space

Gurnee et al. (2026), arXiv:2607.15495, report a small workspace-like set of verbalizable representations in an intermediate layer band, with much of the model's other processing outside it. They also show planned future content in this workspace can causally redirect current generation.

This motivates, but does not establish, the Gate-1 design:

- limited workspace is not evidence for adaptive depth;
- a planned future representation is not a branch scheduler;
- the experiment here must still earn every compute-allocation claim directly.
