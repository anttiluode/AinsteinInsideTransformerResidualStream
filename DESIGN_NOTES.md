# Design note — present register, peripheral futures, and selection

## Attention is not a self

The human analogy that motivated this branch has a useful asymmetry:

- the current embodied self is persistently present;
- other objects enter and leave;
- past objects and past selves can be reinstated as objects relative to the current self.

A vanilla causal transformer does not obviously contain a persistent self-object. Every position can form attention queries. The closest existing computational privilege is the **autoregressive frontier**: the current decoding position integrates earlier context and is the state from which the next token is emitted.

If the analogy is useful, make the asymmetry explicit rather than claiming it already exists.

## Persistent present register

Introduce a compact state register S_t that is carried across internal reasoning cycles.

Peripheral branches fork from it:

S_t -> B_1, B_2, ..., B_k

but do not replace it.

Each branch receives:

- the same current register S_t;
- the same external context / KV history;
- a different temporary perturbation, hypothesis, role, or operator;
- a small compute budget.

A branch returns a stamped departure:

R_i = C(trajectory_i - S_t)

with provenance Z_i.

The central state therefore sees branches as temporary computational objects:

(S_t, R_i, Z_i).

## Do not choose by similarity alone

The useful branch is not necessarily the one most similar to the current state.

A selector should estimate something closer to **counterfactual utility**:

u_i = V(S_t, R_i, Z_i)

or for a collision:

u_ij = V_pair(S_t, R_i, Z_i, R_j, Z_j).

Possible training/evaluation signals include:

- verified task reward;
- reduction in answer uncertainty;
- expected information gain;
- improvement in prediction of a later observation;
- reduction in a known residual/error;
- a learned value model trained from eventual success.

This is where AnotherOddThing enters naturally: when collision budget is tiny, spend it on the branch or pair expected to reduce uncertainty about what matters.

## Mathematics analogy: proposal policy plus verifier

A mathematician does not uniformly enumerate all possible next thoughts.

Long experience changes which moves are proposed at all. Familiar structures, coordinate systems, invariants, lemmas and analogies become high-prior moves.

At a computational level the analogy is:

experience -> proposal policy

proposal -> cheap local test

survivors -> proof / stronger verifier

This is an **amortized search** view. It does not claim a particular biological mechanism.

The transformer version should therefore eventually learn both:

1. a branch proposal policy: which temporary lenses are worth trying here;
2. a branch/collision value model: which departures deserve more compute.

## Why several survivors matter

GAx supplies the warning against purifying too early.

If every cycle immediately keeps only argmax_i u_i, the mechanism collapses toward ordinary greedy reasoning.

Instead preserve a small basis of incompatible but still plausible computational modes:

Q_t = {R_i1, R_i2, ..., R_im}.

Only later context may reveal which mode is useful, or two modes may become useful only through interaction.

## Why this might differ from 10k extra tokens

A long text scratchpad can in principle emulate a great deal of this. The architectural claim is therefore about **inductive bias and compute**, not fundamental transformer expressivity.

Potential advantages that must be measured:

### 1. No forced language bottleneck

A latent branch need not choose a word at every internal step. Incompatible possibilities can remain continuous.

### 2. Breadth before commitment

Several futures can be explored from exactly the same parent state instead of one generated token changing the prefix for everything that follows.

### 3. Residue instead of transcript

The retained object may be a compact computational departure rather than hundreds of tokens describing the departure.

### 4. Cross-branch interaction

The merger is not restricted to selecting or averaging branch conclusions. A nonlinear interaction can create a temporary direction/operator outside the individual branch span.

### 5. Persistent effect

A synthesized temporary operator can affect many subsequent token computations without forcing every later token to reread a long textual scratchpad.

### 6. Provenance

Branch identity, role, assumption and simulated/actual status can remain explicitly bound to the residue. AInstein already showed that confident computation with wrong address binding can be useless.

## The strong attacker

If an ordinary model given the same FLOPs as extra serial tokens or textual Tree-of-Thought matches or beats the latent architecture, there is no reason to claim a new reasoning mechanism.

The useful result would be narrower and measurable:

> for some task class, preserving and colliding internal departures provides a better accuracy/compute or adaptation/compute tradeoff than verbalizing equivalent search.

That is the target.


## Computational foveation: depth is a budget

The next control variable is not only which branch survives, but how much resolution each branch is allowed to buy.

For a branch set Q_t, assign an integer or continuous depth budget rho_i:

sum_i rho_i <= B.

A cheap first pass should estimate whether spending another unit of compute on branch i is likely to improve the task. The scheduler then allocates additional resolution selectively.

This makes the architecture closer to a foveated sensor than to exhaustive search:

- many routes receive only gist-level processing;
- a few routes receive deeper computation;
- abandoned full trajectories can disappear while compressed residues and provenance remain addressable;
- later evidence can justify re-expanding an old gist by paying compute again.

The relevant failure mode is premature collapse. A route with the best early payoff can saturate while a weak-looking branch is a late bloomer. Gate 1 therefore includes a diversity reserve rather than pure greedy allocation.

### Purification is not deletion

Temporal Sihti suggests a useful distinction:

S -> C + {R_1, R_2, ...}

C is the currently sharpened route. The residues need not remain at full resolution; they can be compressed to gist plus provenance.

Purification means making C sharp while retaining enough compressed alternatives to re-open the search later.

Collapse means deleting the alternatives and forcing all later computation to continue from C.

The cost of retaining those alternatives must be measured. No free memory is assumed.

### Relation to the J-space paper

Gurnee et al., "Verbalizable Representations Form a Global Workspace in Language Models" (arXiv:2607.15495, 2026), report a small workspace-like set of representations in an intermediate band of transformer layers. Their J-space is selective, limited in capacity, and causally involved in internal reasoning; they also show planned future content that can redirect current generation.

That result is useful motivation for three engineering distinctions here:

1. the whole residual stream need not be treated as the current workspace;
2. a representation of a prospective future can bend the present trajectory;
3. limited workspace capacity makes selective resolution allocation plausible as a design target.

But the paper does not demonstrate adaptive branch depth, Temporal Sihti, or the foveation scheduler proposed here. Those remain hypotheses to test directly.

### Future field

A speculative branch is useful even when it never becomes the chosen answer if its coarse residue says something about reachability:

"structure is over there"
"this route dead-ends"
"these two regions may meet"

A collection of such low-resolution residues forms a crude future vector field around the present state. The scheduler's job is to decide where that field deserves sharper computation.

## Educated directions and an accident budget

For scientific reasoning, extra compute is only valuable if it is spent in useful regions of route space.

A good proposal direction should therefore be **educated**: shaped by prior successful trajectories, known invariants, useful decompositions, and learned route/value structure.

But restricting all search to the current educated basis risks scientific conservatism. Many discoveries begin as anomalies, accidental measurements, or combinations that the current model did not assign high prior value.

So the scheduler should reserve a small exploration budget for proposals with a component outside the currently favored route span:

    d = sum_i alpha_i D_i + eta

where eta contains an outside-basis component.

Novelty alone is not rewarded. An accidental direction buys deeper compute only if a cheap probe produces surprise, information gain, or task evidence:

    score(d) = expected_value
               + beta * information_gain
               + gamma * outside_basis_novelty
               - lambda * compute_cost

This makes accidents **cheap peripheral probes**, not expensive random rumination.

Most compute follows educated directions. A little compute maintains the possibility that the current basis is wrong.

That exploration reserve is the scientific analogue of the diversity reserve already used by computational foveation.
