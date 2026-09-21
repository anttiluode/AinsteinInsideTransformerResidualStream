# Gate 0.5 — analysis of the first J-space readout

Date: 2026-09-21

Files:

- results/gate1_depth_trace.json
- results/gate05_jspace_readout.json

## Raw geometry

The corrected raw-block depth trace shows a clear differentiate-then-reconverge shape among the three prompted branches.

| Layer | mean pairwise residue cosine | mean branch-specific fraction | common-mode RMS / residue RMS |
|---:|---:|---:|---:|
| 6 | 0.8869 | 0.2758 | 0.9606 |
| 12 | 0.8255 | 0.3410 | 0.9399 |
| 18 | 0.8286 | 0.3381 | 0.9408 |
| 24 | 0.7594 | 0.4014 | 0.9161 |
| 30 | **0.6904** | **0.4538** | **0.8907** |
| 35 | 0.8107 | 0.3568 | 0.9337 |

Among the sampled layers, layer 30 is the strongest raw separation point. Layer 35 partially reconverges.

This is descriptive only. The branch prompts differ lexically and in length.

## Jacobian transport

The fitted Qwen3-8B Jacobian lens changes the geometry substantially.

At layer 30:

- raw residue cosines: 0.7664, 0.6328, 0.6721;
- transported residue cosines: 0.8554, 0.8348, 0.8125.

So the average Jacobian maps the three increasingly different raw departures toward a **more common downstream-effective direction**.

At the same time, top-20 positive token overlap remains low:

- A/B Jaccard: 0.212;
- A/C: 0.081;
- B/C: 0.081.

That combination is interesting: a large common transported component can coexist with sparse, branch-specific readable consequences.

## Readability by depth

Layer 18 is not convincing as a strategy readout. Its top deltas are dominated by unrelated web/Wikitext-like tokens ("Reddit", "subreddit", etc.), although the repeated-addition branch does contain "iterative", "iterations" and "iterating".

Layer 24 is dominated by punctuation/formatting deltas. The repeated-addition branch also surfaces "Incorrect".

Layer 30 is the first sampled layer with an unmistakably legible branch-specific signal:

- branch B includes "incorrect" and "correction";
- branch C is dominated by "closest", "nearby", "nearest", "closer", "neighboring", "proximity", "near" and multilingual equivalents.

But this is also the central confound: those concepts are very close to words explicitly present in the branch instructions.

Therefore the current result is a **weak pass for readable branch conditioning**, not yet a pass for latent reasoning-route representation.

## Consequence

Do not train the bilinear AInstein composer on these residues yet.

The next gate is lexical invariance:

- same final suffix/token for all prompts;
- two paraphrases per strategy;
- negated-keyword decoys;
- compare same-strategy vs between-strategy similarity in full J-lens delta-logit space.

If strategy identity survives that attack, then layer 30 becomes a credible first collision layer.

If it fails, the current result is still useful: it tells us that the J-lens reads prompt-conditioned lexical state, but not yet the hidden route object we need.


## New control suggested by the data: separate carrier from route residue

The raw and transported geometry together suggest a nuisance decomposition worth testing explicitly:

    R_i = C + D_i

with:

- C = common task / answer-preparation carrier shared by all branches;
- D_i = branch-specific departure.

At layer 30 the raw routes are relatively separated, but Jacobian transport increases all three pairwise cosines substantially. This is consistent with a common downstream-effective component dominating the transported geometry while smaller branch-specific components determine the sparse readable extremes.

That makes the naive bilinear collision

    H(R_A, R_B)

dangerous: it can obtain an apparent win from C x C or C x D interactions without ever composing the branch-specific computational material.

Gate 0 now therefore requires a centered control:

    C = mean_i R_i
    D_i = R_i - C
    output = C + H(D_A, D_B)

and compares it against the raw-residue composer, carrier-only, departure-only and shuffled-departure attackers.

This is also a cleaner Temporal-Sihti interpretation: the shared carrier is what survives across the speculative futures; the centered D_i are the departures that distinguish one future from another.
