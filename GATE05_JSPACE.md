# Gate 0.5 — workspace alignment before constructive collision

## Why this gate exists

The first Gate-0 scaffold treated a branch residue

R_i = h_i - h_anchor

as if every direction in the residual stream were equally meaningful downstream.

Gurnee et al. (2026), *Verbalizable Representations Form a Global Workspace in Language Models*, gives a strong reason not to assume that.

Their Jacobian lens identifies a sparse, verbalizable frame ("J-space") that:

- carries only a small fraction of residual-stream variance;
- appears coherently only after an early processing regime;
- is used for report and flexible internal reasoning;
- is preferentially amplified and broadcast by downstream model structure;
- gives way to a late motor/output regime.

Therefore a nonlinear collision that is geometrically novel but lands mostly outside a readable/broadcastable format may simply produce a large perturbation that the model does not know how to use.

Gate 0.5 is a **precondition gate**: before claiming constructive residue composition, ask whether the branch material and the synthesized intervention live in a format the frozen model can actually read and propagate.

## Important correction: J-space is a sparse subframe, not a low-dimensional subspace

The paper defines J-space as points approximable by sparse nonnegative combinations of J-lens vectors. The J-lens dictionary is overcomplete; the full set may span the whole residual stream.

So this gate must not say:

"project onto the J-space subspace."

The correct object is closer to:

h ~= sum_(j in K) alpha_j v_j

with:

alpha_j >= 0
|K| <= k

where k is typically at most about 25 in the paper.

The first repo implementation stops one step earlier: it applies a pre-fitted Jacobian lens and records human-readable **lens-logit deltas** for each branch. Those top tokens are useful diagnostics, but they are **not yet sparse J-space coefficients**.

Sparse nonnegative decomposition is a later subgate and must be labeled separately.

## Gate 0.5a — raw trace hygiene

Before any J-lens analysis:

1. capture raw decoder-block outputs with forward hooks;
2. do not use Hugging Face's final output_hidden_states slot as if it were a raw final block output;
3. report layer-scale-normalized residue magnitudes;
4. report the common branch mode separately from branch-specific departures.

Reason: Qwen/Llama-style Hugging Face models append a final normalized state after the decoder stack. Treating that state as raw "layer 35 output" creates an artificial norm collapse at the last layer.

Required metrics:

- raw residue norm;
- residue norm relative to anchor/state norm;
- anchor-to-branch state cosine;
- pairwise residue cosine;
- branch-specific fraction after subtracting the common branch residue;
- common-mode RMS / total-residue RMS.

## Gate 0.5b — Jacobian-lens readability

Use a pre-fitted Qwen3-8B Jacobian lens or fit one with the Anthropic reference implementation.

For each branch and candidate layer:

1. capture the raw residual state h_l;
2. transport it with the fitted average Jacobian J_l;
3. apply the model's final normalization and unembedding;
4. compare branch lens logits with the common-anchor lens logits.

Report:

- top positive branch-specific lens-token deltas;
- top negative branch-specific lens-token deltas;
- raw-residue pairwise cosine;
- transported-residue pairwise cosine;
- overlap/Jaccard of top branch-specific readable tokens;
- raw norm of J_l R_i relative to raw R_i.

The last quantity is only a **Jacobian transport norm diagnostic**. It is not the paper's MLP broadcast gain.

## Gate 0.5c — sparse J-space decomposition

Only after 0.5b gives stable, interpretable branch-specific readouts:

1. form the source-layer J-lens dictionary vectors;
2. approximate each branch residue with a sparse nonnegative combination;
3. use the same sparsity budget k across all methods;
4. retain provenance binding between coefficient sets and branch roles.

Then the constructive-collision question can be asked in readable coordinates:

(alpha_A, Z_A), (alpha_B, Z_B) -> H -> alpha_new / Delta O.

A useful result is not "the vector left span(R_A,R_B)." A more informative target is:

- neither branch alone carries the required concept/operation strongly;
- simple coordinate addition/swap does not solve the task;
- the nonlinear collision creates or strengthens a **nameable, task-relevant, broadcastable** representation;
- the result generalizes to held-out branch pairings.

## Required new attacker: plain J-coordinate edit

The paper demonstrates that changing a workspace coordinate can redirect downstream reasoning and planning without a nonlinear composer.

Therefore before crediting the bilinear composer, compare against:

1. copy one branch's strongest readable coordinates;
2. add readable coordinates from A and B;
3. swap role-bound coordinates;
4. optimize a sparse linear coordinate edit under the same norm/budget;
5. only then compare the nonlinear collision.

If a simple coordinate edit produces the same behavioral effect, the bilinear composer is decoration.

## Required broadcastability attacker

The paper reports that J-lens-aligned directions are preferentially amplified by MLP blocks in the workspace band, whereas random or ordinary MLP-output directions are not.

A synthesized intervention therefore needs a broadcastability audit.

Do not implement this approximately and then call it the paper's metric. The subgate should reproduce the paper's definition:

- take a unit source-layer direction v;
- apply the next MLP block to v under the paper's convention;
- measure output norm;
- normalize by the median gain of isotropic random directions.

Compare:

- branch J-aligned directions;
- bilinear collision direction;
- matched-norm random direction;
- rotated J-space control;
- linear coordinate-edit attacker.

A collision that is "novel" but has baseline broadcast gain is not yet a useful computational invention.

## Nameable-but-absent target

The Jacobian lens only directly names concepts associated with vocabulary tokens, so a totally alien invention may be invisible.

Gate 0 therefore uses a deliberately narrower target:

> a composition whose result is nameable/readable by the lens but absent or weak in either branch separately.

This avoids the false choice between:

- "readable, therefore not novel"; and
- "unreadable, therefore impossible to test."

The test is **novel relation/composition in context**, not creation of a brand-new vocabulary atom.

## Pre-registered semantic-distance prediction

The paper's workspace is capacity-limited and sparse. The current architecture also has a fixed branch budget.

Freeze this prediction before large behavioral runs:

- semantically/operationally related branch residues should be easier to co-represent and compose;
- unrelated branch residues should consume more workspace budget and should show weaker useful collision under the same compute;
- however, a successful invention task may require preserving a small number of low-probability or semantically distant alternatives long enough for a later collision.

This is descriptive until measured on held-out tasks.

## Available tooling

The Anthropic reference implementation is public:

https://github.com/anthropics/jacobian-lens

Neuronpedia hosts pre-fitted lenses, including Qwen3-8B:

https://huggingface.co/neuronpedia/jacobian-lens

The repo script:

    python gate05_jspace_readout.py

auto-discovers the Qwen3-8B .pt artifact under the Neuronpedia lens repository, downloads it, and reads the existing speculative branches through the fitted Jacobian transport.

This avoids fitting a 4096x4096 lens locally before the first audit.

## Pass / fail

### Pass

Branch strategies that were only geometrically distinct in raw residual space become stably distinct in readable J-lens content at a reproducible intermediate layer band, and later constructive interventions survive plain coordinate-edit and broadcastability attackers.

### Weak pass

Readable branch-specific content exists, but nonlinear collision adds nothing over sparse linear/J-coordinate edits.

Interpretation: the peripheral-future representation is useful, but the constructive-invention mechanism is unnecessary.

### Fail

Raw branch divergence is mostly common prompt-extension/bookkeeping structure; J-lens readouts do not separate strategies reliably; or strategy identity is not reproducible under paraphrase controls.

### Kill

If branch identity is not predictably recoverable in a readable/broadcastable format before answer information appears, stop treating raw residual departures as the right material for constructive collision.
