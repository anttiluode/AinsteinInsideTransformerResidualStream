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
