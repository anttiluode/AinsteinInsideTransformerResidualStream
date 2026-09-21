# Gate 0.5d — strategy invariance versus lexical echo

The first real Jacobian-lens result is readable at layer 30, but its strongest evidence is confounded.

The third branch was prompted with "nearby multiples" and its layer-30 readout strongly promoted tokens such as "closest", "nearby", "nearest", "closer", "neighboring" and "proximity". The repeated-addition branch promoted "incorrect" and "correction", while its prompt explicitly contained "correction".

That is useful — the lens is sensitive to branch-conditioned readable content — but it is not yet evidence that the residual encodes a **reasoning trajectory** rather than the lexical instruction that named that trajectory.

## Control

Every experimental prompt now has the same suffix:

    Do not state the answer yet. Continue from this method.
    Work:

The captured final token is therefore identical across all conditions.

Three strategy families are tested:

- A: split one factor and combine partial products;
- B: running/repeated addition;
- C: start from a convenient nearby product and compensate.

Each strategy gets two paraphrases that avoid relying on one fixed label.

Then three lexical decoys are added:

- intended A while explicitly saying to ignore "nearby" multiples;
- intended B while explicitly saying not to use "decomposition";
- intended C while explicitly saying not to use "correction".

## Pre-registered prediction

If the readable layer-30 residue is strategy-like rather than merely lexical echo:

1. same-strategy paraphrases should have more similar J-lens delta-logit vectors than different-strategy paraphrases;
2. the within-minus-between cosine gap should be positive;
3. a negated-keyword decoy should be closer to the centroid of the **intended strategy** than to the strategy named by the decoy word.

The same comparisons are reported in:

- raw residual space;
- Jacobian-transported residual space;
- full vocabulary J-lens delta-logit space.

The full vocabulary comparison matters because top-k token lists can be dominated by punctuation or one unstable token.

## Strong pass

At a reproducible intermediate layer, same-strategy paraphrases are more similar than different strategies in lens-logit space, and at least two of three lexical decoys follow the intended strategy rather than the negated keyword.

## Weak pass

Paraphrase invariance exists but decoys are ambiguous, or only raw/transported geometry is invariant while readable lens logits remain surface-sensitive.

## Fail

Within-strategy similarity is no better than between-strategy similarity, or decoys systematically follow the surface keyword.

Interpretation: the previous "nearby / correction" readout was mainly prompt echo and cannot yet be used as a substrate for AInstein collision.

## Why this gate comes before collision

If we collide residues before establishing strategy invariance, a bilinear composer can learn interactions between **phrases** rather than between computational routes.

The constructive-invention experiment only becomes interesting after branch residue survives a change in how the branch is described.
