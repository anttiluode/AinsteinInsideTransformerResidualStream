# First real Qwen3-8B depth-microscope run

Date: 2026-09-21

Command:

    python3.13 gate1_depth_trace.py

Model:

    Qwen/Qwen3-8B

Anchor:

    Problem: 17 * 24 = ?
    Think:

Branches:

1. decomposition into 20 + 4
2. repeated addition and correction
3. checking nearby multiples first

## Original receipt

The first script version reported:

| Layer | Mean residue norm | Mean pairwise residue cosine |
|---:|---:|---:|
| 6 | 33.217 | 0.8869 |
| 12 | 54.333 | 0.8255 |
| 18 | 71.921 | 0.8287 |
| 24 | 151.067 | 0.7594 |
| 30 | 282.232 | 0.6904 |
| 35 | 46.113 | 0.6254 |

Pairwise detail at layer 30:

- decomposition vs repeated-addition: 0.7664
- decomposition vs nearby-multiples: 0.6327
- repeated-addition vs nearby-multiples: 0.6721

Descriptively, the three prompt-conditioned departures share a strong common direction early and become less aligned deeper in the stack. The third strategy ("nearby multiples") separates most strongly by layer 30.

## What this result does establish

It establishes that the frozen 8B model is loaded and the branch prompts produce measurable, depth-dependent differences in the residual stream.

It also gives a concrete reason to stop treating layer 18 as a uniquely privileged default. In this one prompt, the largest decrease in pairwise similarity happens later, between the layer-18 and layer-30 measurements.

That is a microscope result, not yet a reasoning result.

## What it does NOT establish

The run does not show that:

- the branch vectors encode distinct reasoning strategies rather than prompt wording/position;
- larger residue norm means "more thought";
- layer 30 is the workspace;
- any branch is useful;
- a collision of these branches invents a new operator.

The prompts have different lexical content and sequence lengths. A same-strategy paraphrase control and length/wording-matched controls are required.

## Important instrumentation bug discovered after the run

The first version used Hugging Face output_hidden_states and treated hidden_states[layer+1] as raw block output at every layer.

For Qwen/Llama-style decoder stacks, the final hidden-state slot is the model's **post-final-normalization** state rather than a raw final decoder-block output.

Therefore the apparent norm collapse:

    layer 30 mean norm ~282
    layer 35 mean norm ~46

must not be interpreted as late purification, Sihti-like collapse, or loss of branch residue.

The branch has now been changed to capture raw decoder-block outputs directly with forward hooks. Re-run gate1_depth_trace.py; the model files should already be cached locally.

The corrected script also reports scale-normalized residue magnitude, common-mode strength, and branch-specific fraction so that layer scale and the shared "prompt got longer" direction are not confused with strategy separation.

## Next receipt

Run:

    python3.13 gate1_depth_trace.py

then:

    python3.13 gate05_jspace_readout.py

The second command downloads a pre-fitted Qwen3-8B Jacobian lens and asks a much sharper question:

> what branch-specific, human-readable concepts become more or less available to future computation at each fitted layer?

Top J-lens deltas are still not the paper's sparse k<=25 J-space decomposition. They are the next microscope, not the verdict.
