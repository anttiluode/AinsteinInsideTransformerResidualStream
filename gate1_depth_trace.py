from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from latent_fork import branch_residue_trace


def parse_args():
    p = argparse.ArgumentParser(
        description="Inspect branch residues as a function of transformer depth."
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
        help="Comma-separated transformer layer indices (0-based).",
    )
    p.add_argument("--out", default="results/gate1_depth_trace.json")
    return p.parse_args()


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
        torch_dtype=dtype,
        device_map="auto",
    ).eval()

    traces = [
        branch_residue_trace(model, tok, args.anchor, branch)
        for branch in branches
    ]

    max_transformer_layer = len(traces[0]) - 2
    for layer in layers:
        if layer < 0 or layer > max_transformer_layer:
            raise ValueError(
                f"layer {layer} outside 0..{max_transformer_layer}"
            )

    receipt = {
        "scope": (
            "real frozen-model depth microscope; hidden states are cached from full "
            "forward passes, so this is not a compute-saving result"
        ),
        "model": args.model,
        "anchor": args.anchor,
        "branches": branches,
        "layers": {},
    }

    for layer in layers:
        # hidden_states[0] is embeddings; layer i output is hidden_states[i+1]
        vecs = [trace[layer + 1].float().cpu() for trace in traces]
        norms = [torch.linalg.vector_norm(v).item() for v in vecs]
        cos = []
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                c = torch.nn.functional.cosine_similarity(
                    vecs[i], vecs[j], dim=-1
                ).mean().item()
                cos.append({"i": i, "j": j, "cosine": c})
        receipt["layers"][str(layer)] = {
            "residue_norms": norms,
            "pairwise_cosines": cos,
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))
    print()
    print("Use this to choose coarse probe depths for Gate 1a.")
    print("All branches were fully evaluated here; no FLOP saving is claimed.")


if __name__ == "__main__":
    main()
