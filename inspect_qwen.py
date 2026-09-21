from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from latent_fork import branch_residue, relative_outside_span


def parse_args():
    p = argparse.ArgumentParser(description="Inspect branch residues inside a frozen causal LM.")
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--layer", type=int, default=18)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--anchor", default="Problem: 17 * 24 = ?\\nThink:")
    p.add_argument(
        "--branch-a",
        default="Problem: 17 * 24 = ?\\nThink using decomposition into 20 + 4:",
    )
    p.add_argument(
        "--branch-b",
        default="Problem: 17 * 24 = ?\\nThink using repeated addition and correction:",
    )
    return p.parse_args()


def main():
    args = parse_args()
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto",
    ).eval()

    a = branch_residue(
        model, tok, args.anchor, args.branch_a, args.layer, branch="A", role="strategy-a"
    )
    b = branch_residue(
        model, tok, args.anchor, args.branch_b, args.layer, branch="B", role="strategy-b"
    )

    cos = torch.nn.functional.cosine_similarity(a.vector, b.vector).item()
    print(f"model={args.model}")
    print(f"layer={args.layer}")
    print(f"hidden={a.vector.shape[-1]}")
    print(f"||R_A||={torch.linalg.vector_norm(a.vector).item():.6f}")
    print(f"||R_B||={torch.linalg.vector_norm(b.vector).item():.6f}")
    print(f"cos(R_A,R_B)={cos:.6f}")

    # A deliberately nonlinear, parameter-free diagnostic only.
    crossed = a.vector * b.vector
    print(
        "elementwise-cross outside branch span="
        f"{relative_outside_span(crossed, a.vector, b.vector).item():.6f}"
    )
    print("No capability claim: Gate 0 must train/freeze a composer and beat matched attackers.")


if __name__ == "__main__":
    main()
