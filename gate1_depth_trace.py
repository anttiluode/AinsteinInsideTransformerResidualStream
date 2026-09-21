from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from latent_fork import final_token_block_trace


def parse_args():
    p = argparse.ArgumentParser(
        description="Inspect speculative branch departures as a function of transformer depth."
    )
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--anchor", default="Problem: 17 * 24 = ?\nThink:")
    p.add_argument(
        "--branch",
        action="append",
        default=[],
        help="Repeat for multiple speculative branches.",
    )
    p.add_argument(
        "--layers",
        default="6,12,18,24,30,35",
        help="Comma-separated raw decoder-block output indices (0-based).",
    )
    p.add_argument("--out", default="results/gate1_depth_trace.json")
    return p.parse_args()


def _norm(x: torch.Tensor) -> float:
    return torch.linalg.vector_norm(x).item()


def main():
    args = parse_args()
    branches = args.branch or [
        "Problem: 17 * 24 = ?\nThink using decomposition into 20 + 4:",
        "Problem: 17 * 24 = ?\nThink using repeated addition and correction:",
        "Problem: 17 * 24 = ?\nThink by checking nearby multiples first:",
    ]
    layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]

    tok = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        device_map="auto",
    ).eval()

    # Capture raw decoder-block outputs with hooks. In Qwen/Llama-style HF
    # models output_hidden_states[-1] is post-final-norm, so using that slot
    # would create a fake last-layer norm collapse.
    anchor_trace = final_token_block_trace(model, tok, args.anchor)
    branch_traces = [
        final_token_block_trace(model, tok, branch)
        for branch in branches
    ]

    max_transformer_layer = len(anchor_trace) - 1
    for layer in layers:
        if layer < 0 or layer > max_transformer_layer:
            raise ValueError(
                f"layer {layer} outside 0..{max_transformer_layer}"
            )

    receipt = {
        "scope": (
            "real frozen-model depth microscope using raw decoder-block outputs; "
            "all branches are still fully evaluated, so this is not a compute-saving result"
        ),
        "model": args.model,
        "anchor": args.anchor,
        "branches": branches,
        "layer_semantics": "raw decoder block output before the model's final norm",
        "layers": {},
    }

    for layer in layers:
        anchor = anchor_trace[layer].float().cpu()
        states = [trace[layer].float().cpu() for trace in branch_traces]
        vecs = [state - anchor for state in states]
        norms = [_norm(v) for v in vecs]

        anchor_norm = _norm(anchor)
        state_norms = [_norm(state) for state in states]
        relative_norms = [
            _norm(v) / max(1e-12, 0.5 * (anchor_norm + state_norm))
            for v, state_norm in zip(vecs, state_norms)
        ]
        anchor_state_cosines = [
            torch.nn.functional.cosine_similarity(anchor, state, dim=-1).mean().item()
            for state in states
        ]

        pairwise = []
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                c = torch.nn.functional.cosine_similarity(
                    vecs[i], vecs[j], dim=-1
                ).mean().item()
                pairwise.append({"i": i, "j": j, "cosine": c})

        stack = torch.cat(vecs, dim=0)
        common = stack.mean(dim=0, keepdim=True)
        specific_fractions = [
            _norm(v - common) / max(1e-12, _norm(v))
            for v in vecs
        ]
        rms_residue = torch.sqrt(torch.mean(stack.square())).item()
        common_rms = torch.sqrt(torch.mean(common.square())).item()

        receipt["layers"][str(layer)] = {
            "anchor_norm": anchor_norm,
            "branch_state_norms": state_norms,
            "residue_norms": norms,
            "relative_residue_norms": relative_norms,
            "anchor_branch_state_cosines": anchor_state_cosines,
            "pairwise_residue_cosines": pairwise,
            "mean_pairwise_residue_cosine": (
                sum(x["cosine"] for x in pairwise) / len(pairwise)
                if pairwise else None
            ),
            "branch_specific_fraction": specific_fractions,
            "mean_branch_specific_fraction": (
                sum(specific_fractions) / len(specific_fractions)
            ),
            "common_mode_rms_over_residue_rms": (
                common_rms / max(1e-12, rms_residue)
            ),
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))
    print()
    print("Use this to choose and audit coarse probe depths for Gate 1a.")
    print("Raw norms are not compared alone: layer scale and common prompt-extension")
    print("directions are reported separately from branch-specific divergence.")
    print("All branches were fully evaluated here; no FLOP saving is claimed.")


if __name__ == "__main__":
    main()
