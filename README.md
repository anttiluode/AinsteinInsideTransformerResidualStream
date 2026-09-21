# AInstein Inside the Transformer Residual Stream

**Can a transformer fork one internal computational state into several cheap latent futures, preserve what each future changed, collide those departures, and resume through a temporary operator that no branch contained alone?**

This repo moves the constructive-residue question from [AInstein](https://github.com/anttiluode/AInstein) *inside* a frozen transformer.

The claim is deliberately **not** that transformers cannot compose representations, branching is new, latent reasoning is new, or residual steering is new. The experiment is narrower:

~~~text
common residual state S
        |
        +---- latent branch A ----> stamped residue R_A
        |
        +---- latent branch B ----> stamped residue R_B
        |
        +---- latent branch C ----> stamped residue R_C

chosen residues
        |
        v
nonlinear provenance-sensitive collision H(R_i, R_j)
        |
        v
temporary operator / residual intervention dO
        |
        v
resume the original main stream
~~~

The central falsifiable question is whether this buys anything over spending the same compute on more ordinary reasoning tokens.

## Why this is not just 10,000 more reasoning tokens

Ordinary chain-of-thought serializes one trajectory. Tree-style search branches, but normally commits by selecting, voting, or rereading textual branches.

Here the desired object is:

~~~text
state
  -> branch A -> computational departure R_A
  -> branch B -> computational departure R_B

(R_A, provenance_A) x (R_B, provenance_B)
  -> new temporary lens
  -> resume state
~~~

The branch endpoint is not the answer. The branch is raw material.

A win requires the useful synthesized intervention to generalize to held-out branch pairings and to contain useful structure not explained by either branch alone, simple interpolation, pair lookup, or extra compute. If a matched serial or Tree-of-Thought baseline solves the same tasks with equal or less compute, the stronger architecture claim dies.

## Temporal Sihti

The image lineage gives a useful representation, not a biological claim.

SighImageSuper asked what survives repeated application of an operator. Sihti kept what disappeared at each depth:

R_j = x(d_j) - x(d_j+1).

A transformer residual stream instead accumulates departures:

r_(l+1) = r_l + Delta r_l.

This repo treats a short speculative branch as a temporal object whose useful product may be its **departure trajectory**, not its textual endpoint:

R_A = C(Delta r^A_(k+1), ..., Delta r^A_(k+m)).

Several cheap speculative futures can therefore coexist around one sharp present. "Peripheral thought field" or "subconscious" are useful visual metaphors only; the code must cash them out as tensors and controls.

## First backbone

The first real-model target is **Qwen/Qwen3-8B** through Hugging Face Transformers.

Reasons:

- approximately the requested 8B class;
- Apache-2.0 license;
- ordinary AutoModelForCausalLM loading;
- 36 transformer layers, hidden size 4096;
- Hugging Face exposes layer hidden states, attention outputs, and KV cache;
- easy to freeze while training only a tiny composer.

The first instrumentation code does **not** claim a result. It extracts branch residues at a chosen layer, measures geometry, and provides a hook for injecting a temporary last-token residual intervention.

## Gate 0

See [GATE0_CONTRACT.md](GATE0_CONTRACT.md).

The first benchmark family uses **composable function/operator tasks**. One branch induces operation A, another induces B, and held-out queries require a composition that was not present in either branch alone.

This is intentionally close to AInstein's held-out operator-pair discipline, but now the residues come from a frozen transformer's own hidden computation.

Attackers include:

1. base model / no extra thinking;
2. branch A only;
3. branch B only;
4. best convex branch blend;
5. additive residual steering;
6. linear concatenation/readout;
7. provenance erased or swapped;
8. same-compute serial chain-of-thought;
9. same-compute textual branch / Tree-of-Thought style search;
10. nonlinear latent composer.

The architecture only earns attention if the last mechanism contributes something after those controls.

## Current code

~~~bash
pip install -r requirements.txt

python inspect_qwen.py \
  --model Qwen/Qwen3-8B \
  --layer 18
~~~

This loads the model, compares two branch-induced residual departures from a common anchor, and prints their geometry.

It ends with an explicit warning: a nonlinear vector outside the branch span is **not** evidence of useful invention. Gate 0 must train/freeze the composer and demonstrate held-out behavior.

## Working architecture

S_t -> {B_1, ..., B_n}

B_i --L_i--> B'_i

R_i = ( C(B_i - B'_i, trajectory_i), Z_i )

Choose a collision under a budget:

(i,j)* = argmax EIG(R_i, R_j | S_t)

Synthesize:

M* = H(R_i, R_j)

Compile to a temporary operator/residual intervention:

Delta O* = C_op(M*)

and resume:

S_(t+1) = (O + Delta O*) S_t.

A low-rank ordered bilinear composer is the first nonlinear family:

z_AB = U_A R_A elementwise-multiplied-by U_B R_B

Delta r = W z_AB.

It is cheap, role-sensitive, and can leave the span of the individual branch vectors.


## Gate 1 — computational foveation

Gate 1 adds an explicit **resolution budget** per speculative branch:

rho_i = compute depth allocated to branch i

subject to

sum_i rho_i <= B.

Every branch receives a cheap coarse probe first. The scheduler then purchases additional depth only where the estimated marginal value justifies it, while reserving a small amount of compute for underexplored routes so an initially unimpressive late bloomer can still recover.

This turns the peripheral-field metaphor into a falsifiable compute-allocation question:

- **uniform depth:** every branch gets the same resolution;
- **greedy depth:** keep deepening whichever branch currently looks best;
- **adaptive foveation:** deepen promising branches but preserve a diversity reserve;
- **oracle:** use future payoff only as an upper bound.

The scheduler implementation lives in [foveation.py](foveation.py), and the pre-registered real-model test is [GATE1_CONTRACT.md](GATE1_CONTRACT.md).

A small deterministic receipt can be run without a model:

~~~bash
python gate1_allocator_receipt.py
~~~

That receipt is intentionally only a scheduler test. It includes a late-bloomer branch whose first probe is weak but whose deeper trajectory has the highest ceiling. A useful scheduler should not collapse permanently onto the easiest early route.

Gate 1a will use cached hidden-state traces from the frozen transformer and count branch-layer equivalents. Gate 1b must actually stop abandoned branches early and show a wall-clock/FLOP advantage before this repo can claim compute efficiency.

The new distinction is:

~~~text
purification = one route becomes sharp while alternatives remain compressed and addressable
collapse      = one route becomes sharp because alternatives were deleted
~~~

The hoped-for architecture is therefore not "think longer everywhere." It is "spend high-resolution computation only where a coarse future field says it matters."

## Nearest neighbors / boundaries

This project should be attacked against at least:

- Tree of Thoughts: textual branching, evaluation and search;
- Coconut / Chain of Continuous Thought: recurrent latent reasoning;
- activation addition / contrastive activation steering;
- function-vector and representation-steering work;
- ordinary test-time scaling with more generated reasoning tokens.

If we cannot isolate a useful difference after those attackers, the honest conclusion is that the idea is another form of test-time compute.

## Lineage

~~~text
SighImageSuper
    repeated operator / forgetting times
        |
        v
Sihti / Sihti2
    keep departures
    data can write the geometry through which it moves
        |
        v
GAx
    preserve alternative computational modes/futures
        |
        v
AInstein
    collide stamped residues
    seek an operator absent from either branch
        |
        v
AInsteinInsideTransformerResidualStream
    fork a transformer's internal trajectory
    collide branch departures
    compile the collision into the next temporary lens
~~~
