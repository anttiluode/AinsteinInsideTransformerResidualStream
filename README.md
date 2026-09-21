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


## Gate 0.5 — J-space audit

The Anthropic Jacobian-lens / global-workspace result changes what counts as a meaningful branch residue.

A raw residual departure can be large and geometrically novel while still living mostly in bookkeeping directions that downstream circuitry does not preferentially read or broadcast. Gate 0.5 therefore asks whether speculative branch material enters a **readable / workspace-like format** before constructive collision gets credit.

See [GATE05_JSPACE.md](GATE05_JSPACE.md).

Two practical changes follow immediately.

First, the depth microscope now captures **raw decoder-block outputs with forward hooks**. The first real Qwen3-8B run exposed an instrumentation trap: Hugging Face's last \`output_hidden_states\` slot on Qwen/Llama-style models is post-final-normalization, so the original apparent layer-35 norm collapse was not evidence of late purification. The first receipt and correction are recorded in [RESULTS_GATE1_FIRST_RUN.md](RESULTS_GATE1_FIRST_RUN.md).

Second, a new script reads the same speculative branches through a pre-fitted Qwen3-8B Jacobian lens:

~~~bash
python gate05_jspace_readout.py --layers 18,24,30
~~~

By default it discovers the Qwen3-8B lens in Neuronpedia's public \`neuronpedia/jacobian-lens\` repository, downloads it, and reports branch-specific J-lens logit deltas at the fitted layers.

On Windows, Qwen is now loaded **before the script even resolves or opens the ~1.17 GB lens artifact**. The loader also uses conservative default Accelerate placement caps of 8 GiB on CUDA:0 and 10 GiB on host RAM, with disk offload available for overflow. This is deliberately below a 12 GB GPU / 24 GB RAM machine's headline capacity so Windows, CUDA staging and file mappings retain headroom.

The safe first retry is:

~~~bash
python3.13 gate05_jspace_readout.py --layers 18,24,30
~~~

If that works and Task Manager shows plenty of committed-memory headroom, the caps can be raised explicitly:

~~~bash
python3.13 gate05_jspace_readout.py --layers 18,24,30 --gpu-memory 10GiB --cpu-memory 12GiB
~~~

Interpretation boundary:

- top J-lens delta tokens are a **readability diagnostic**;
- they are not yet the paper's sparse nonnegative \`k <= 25\` J-space coordinates;
- a nonlinear collision must still beat plain J-coordinate edits, matched linear/nonlinear attackers, provenance swaps, and a broadcastability control.

The stronger Gate-0 target is now:

> produce a nameable, task-relevant composition that is weak or absent in either branch alone, survives held-out branch pairings, and cannot be explained by a sparse coordinate edit.


### Gate 0.5d — does the readable route survive paraphrase?

The first successful J-lens run gives a useful but confounded result. At layer 30 the "nearby multiples" branch promotes tokens such as \`nearby\`, \`closest\`, \`nearest\`, \`proximity\`, while the repeated-addition branch promotes \`correction\` / \`incorrect\`. Those are readable branch-conditioned consequences, but the branch prompts themselves contain the corresponding words.

Before training any constructive composer, run the lexical-echo attacker:

~~~bash
python3.13 gate05_strategy_invariance.py --layers 30
~~~

This uses:

- an identical suffix and final token for every prompt;
- two paraphrases of each of the three strategies;
- three negated-keyword decoys.

It compares same-strategy versus different-strategy similarity in raw residual space, Jacobian-transported space, and full J-lens delta-logit space. The decoys ask whether the readout follows the **intended method** or merely the strategy word that appears in the prompt.

See [GATE05_STRATEGY_INVARIANCE.md](GATE05_STRATEGY_INVARIANCE.md) and [RESULTS_GATE05_JSPACE_ANALYSIS.md](RESULTS_GATE05_JSPACE_ANALYSIS.md).

### Gate 0.6 — does the route actually steer computation?

Gate 0.5d passed: at layer 30, same-strategy paraphrases remain substantially closer than different strategies in raw, transported, and J-lens space, and all three negated-keyword decoys follow the intended method rather than the misleading word.

The next gate therefore stops reading the route and **edits it**:

~~~bash
python3.13 gate06_causal_route_steering.py
~~~

It derives centered route directions from the 17x24 paraphrase pairs, writes one route vector once into the neutral prompt state entering block 31, and scores three valid strategy-specific continuations. The same frozen vectors are then tested on held-out multiplication problems without refitting.

See [GATE06_CAUSAL_ROUTE_STEERING.md](GATE06_CAUSAL_ROUTE_STEERING.md). If this passes, simple linear route steering becomes a mandatory attacker for the later AInstein collision.

### Gate 0.7 — numeric first action, no method-language

Gate 0.6 passed strongly: the centered route vectors derived only from 17x24 produced matching-route preference on all three rows for the source problem and both held-out multiplication problems. Mean held-out diagonal advantage was +0.08422, essentially unchanged from the source +0.08778.

The remaining attacker is that Gate 0.6 scored **verbal descriptions** of the methods. Gate 0.7 removes those words entirely:

~~~bash
python3.13 gate07_numeric_first_action.py
~~~

The same frozen D_A/D_B/D_C vectors now compete over bare arithmetic moves such as:

~~~text
A: 13 * 20 = 260
B: 13 + 13 = 26
C: 13 * 27 = 351
~~~

No strategy labels or explanatory prose appear in the candidates. The script aborts if method words leak in.

See [GATE07_NUMERIC_FIRST_ACTION.md](GATE07_NUMERIC_FIRST_ACTION.md) and [RESULTS_GATE06_ANALYSIS.md](RESULTS_GATE06_ANALYSIS.md). A strong pass earns only the narrow claim that a route vector biases the next concrete arithmetic operation; free rollout remains the next attacker.

### Gate 0.7b — route as a numeric trajectory

Gate 0.7 did not strongly pass. A and C retained positive target margins on all three problems, but B failed on 17x24 and 19x23 and only barely won on 13x26.

That failure suggests a sharper hypothesis: A and C have distinctive first transforms, while B is defined by a recurrence across several states. Gate 0.7b therefore scores **equation-only trajectories**, not one isolated arithmetic move:

~~~bash
python3.13 gate07b_numeric_trajectory.py
~~~

Example:

~~~text
A: 17 * 20 = 340; 17 * 4 = 68; 340 + 68 = 408
B: 17 + 17 = 34; 34 + 17 = 51; 51 + 17 = 68
C: 17 * 25 = 425; 425 - 17 = 408; 408 = 17 * 24
~~~

The script rejects any alphabetic character in the candidates. If B recovers here, that supports the idea that some route identity lives in a **transformation law across states**, not one next action.

Gate 0.7b now defaults to a deliberately conservative Windows memory profile (`6GiB` GPU + `6GiB` CPU placement budget, with overflow forced to `.offload_qwen_gate07b`). This is slower, but it avoids relying on transient RAM/VRAM headroom while Qwen shards are loaded. Override the caps only after a successful run.

See [GATE07B_NUMERIC_TRAJECTORY.md](GATE07B_NUMERIC_TRAJECTORY.md) and [RESULTS_GATE07_ANALYSIS.md](RESULTS_GATE07_ANALYSIS.md).

### Gate 0.7c — causal temporal footprint

Gate 0.7b produced a route-dependent horizon pattern rather than one uniform steering result: A weakened when extended beyond its first move, B improved when recurrence depth was included, and C stayed positive. The next gate therefore treats **support horizon** as part of the causal object:

```text
route direction + how long its influence remains useful
```

Run:

```bash
python3.13 gate07c_temporal_footprint.py
```

The script scores the same frozen D_A/D_B/D_C interventions at numeric horizons k=1,2,3 and reports both cumulative and newly-added-transition margins. It also attacks B with equation-only recurrence decoys that preserve the plus operator, recurrence depth, or even the exact first edge.

The interpretation is intentionally narrow: a pass would establish a route-specific **causal temporal footprint**, not yet a dynamically rotating vector field. See [GATE07C_TEMPORAL_FOOTPRINT.md](GATE07C_TEMPORAL_FOOTPRINT.md) and [RESULTS_GATE07B_ANALYSIS.md](RESULTS_GATE07B_ANALYSIS.md).

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

A real frozen-model depth microscope is also included:

~~~bash
python gate1_depth_trace.py --model Qwen/Qwen3-8B
~~~

It records how branch residues change across transformer depth. It deliberately uses full forward passes and therefore does **not** claim a compute saving; its role is to choose and audit coarse probe depths before Gate 1b implements genuine partial execution.

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
