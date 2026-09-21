from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download, list_repo_files
from transformers import AutoModelForCausalLM, AutoTokenizer

from latent_fork import final_token_block_trace


DEFAULT_LENS_REPO = "neuronpedia/jacobian-lens"
DEFAULT_LENS_PREFIX = "qwen3-8b/"


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Read speculative branch departures through a pre-fitted Jacobian lens. "
            "This is a readability/broadcast-format audit, not yet sparse J-space decomposition."
        )
    )
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--anchor", default="Problem: 17 * 24 = ?\nThink:")
    p.add_argument(
        "--branch",
        action="append",
        default=[],
        help="Repeat for multiple speculative branches.",
    )
    p.add_argument("--lens-repo", default=DEFAULT_LENS_REPO)
    p.add_argument(
        "--lens-file",
        default="auto",
        help=(
            "Path inside the HF lens repo. 'auto' searches under --lens-prefix "
            "for a .pt Jacobian-lens checkpoint."
        ),
    )
    p.add_argument("--lens-prefix", default=DEFAULT_LENS_PREFIX)
    p.add_argument(
        "--layers",
        default="",
        help="Comma-separated layer subset. Default: all layers available in the lens.",
    )
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--out", default="results/gate05_jspace_readout.json")
    return p.parse_args()


def resolve_lens_file(repo_id: str, filename: str, prefix: str) -> str:
    if filename != "auto":
        return filename

    files = list_repo_files(repo_id, repo_type="model")
    candidates = [
        path
        for path in files
        if path.startswith(prefix)
        and path.lower().endswith(".pt")
        and ("jacobian" in path.lower() or "lens" in path.lower())
    ]
    if not candidates:
        raise RuntimeError(
            f"no Jacobian-lens .pt file found under {repo_id}:{prefix}; "
            "pass --lens-file explicitly"
        )

    preferred = [p for p in candidates if "n1000" in p.lower()]
    if len(preferred) == 1:
        return preferred[0]
    if len(candidates) == 1:
        return candidates[0]

    # Prefer an obvious jacobian_lens artifact over convergence/checkpoint files.
    obvious = [
        p for p in candidates
        if "jacobian_lens" in Path(p).name.lower()
        or Path(p).name.lower() in {"lens.pt", "jacobian-lens.pt"}
    ]
    if len(obvious) == 1:
        return obvious[0]

    raise RuntimeError(
        "multiple candidate lens files found; pass --lens-file explicitly:\n"
        + "\n".join(f"  {p}" for p in candidates)
    )


def load_jacobians(path: str) -> tuple[dict[int, torch.Tensor], dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if "J" in ckpt:
        raw = ckpt["J"]
    elif "jacobians" in ckpt:
        raw = ckpt["jacobians"]
    else:
        raise ValueError(
            f"unrecognized lens checkpoint keys: {sorted(ckpt.keys())}"
        )
    jacobians = {int(layer): tensor for layer, tensor in raw.items()}
    meta = {
        key: ckpt[key]
        for key in ("n_prompts", "d_model", "source_layers")
        if key in ckpt
    }
    return jacobians, meta


def final_norm_and_unembed(model, residual: torch.Tensor) -> torch.Tensor:
    if not (
        hasattr(model, "model")
        and hasattr(model.model, "norm")
        and hasattr(model, "lm_head")
    ):
        raise TypeError("expected Qwen/Llama-style model.model.norm + model.lm_head")

    norm = model.model.norm
    lm_head = model.lm_head
    norm_param = next(norm.parameters())
    x = residual.to(device=norm_param.device, dtype=norm_param.dtype)
    x = norm(x)
    x = x.to(device=lm_head.weight.device, dtype=lm_head.weight.dtype)
    return lm_head(x).float().cpu()


def lens_logits(
    model,
    residual: torch.Tensor,
    jacobian: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return (logits, transported_residual) for one [1,d] residual activation.

    This mirrors the paper/reference implementation:
        h_l -> J_l h_l -> final norm -> unembedding.
    """
    h = residual.float().cpu()
    J = jacobian.float().cpu()
    transported = h @ J.T
    logits = final_norm_and_unembed(model, transported)
    return logits, transported


def token_rows(tokenizer, values: torch.Tensor, top_k: int, largest: bool) -> list[dict]:
    values = values.reshape(-1)
    k = min(top_k, values.numel())
    scores, ids = torch.topk(values, k=k, largest=largest)
    rows = []
    for score, token_id in zip(scores.tolist(), ids.tolist()):
        rows.append(
            {
                "token_id": int(token_id),
                "token": tokenizer.decode([int(token_id)]),
                "score": float(score),
            }
        )
    return rows


def pairwise_cosines(vecs: list[torch.Tensor]) -> list[dict]:
    out = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            c = torch.nn.functional.cosine_similarity(
                vecs[i], vecs[j], dim=-1
            ).mean().item()
            out.append({"i": i, "j": j, "cosine": c})
    return out


def top_token_jaccards(
    deltas: list[torch.Tensor],
    top_k: int,
) -> list[dict]:
    sets = []
    for delta in deltas:
        ids = torch.topk(delta.reshape(-1), k=min(top_k, delta.numel())).indices
        sets.append(set(int(i) for i in ids.tolist()))
    out = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = sets[i] | sets[j]
            score = 0.0 if not union else len(sets[i] & sets[j]) / len(union)
            out.append({"i": i, "j": j, "jaccard": score})
    return out


def main():
    args = parse_args()
    branches = args.branch or [
        "Problem: 17 * 24 = ?\nThink using decomposition into 20 + 4:",
        "Problem: 17 * 24 = ?\nThink using repeated addition and correction:",
        "Problem: 17 * 24 = ?\nThink by checking nearby multiples first:",
    ]

    lens_file = resolve_lens_file(args.lens_repo, args.lens_file, args.lens_prefix)
    lens_path = hf_hub_download(args.lens_repo, lens_file)
    jacobians, lens_meta = load_jacobians(lens_path)
    available_layers = sorted(jacobians)

    if args.layers.strip():
        layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
        missing = sorted(set(layers) - set(available_layers))
        if missing:
            raise ValueError(
                f"layers {missing} absent from lens; available={available_layers}"
            )
    else:
        layers = available_layers

    tok = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        device_map="auto",
    ).eval()

    anchor_trace = final_token_block_trace(model, tok, args.anchor)
    branch_traces = [
        final_token_block_trace(model, tok, branch)
        for branch in branches
    ]

    receipt = {
        "scope": (
            "J-lens readout of real frozen-model branch departures. This reports "
            "human-readable lens-logit changes and transported geometry; it is not "
            "yet the paper's sparse nonnegative k<=25 J-space decomposition."
        ),
        "model": args.model,
        "lens_repo": args.lens_repo,
        "lens_file": lens_file,
        "lens_meta": {
            k: (v.tolist() if isinstance(v, torch.Tensor) else v)
            for k, v in lens_meta.items()
        },
        "anchor": args.anchor,
        "branches": branches,
        "top_k": args.top_k,
        "layers": {},
    }

    for layer in layers:
        if layer >= len(anchor_trace):
            raise ValueError(
                f"lens layer {layer} exceeds model decoder depth {len(anchor_trace)}"
            )

        anchor = anchor_trace[layer]
        states = [trace[layer] for trace in branch_traces]
        raw_residues = [state - anchor for state in states]

        anchor_logits, anchor_transport = lens_logits(
            model, anchor, jacobians[layer]
        )

        branch_logits = []
        branch_transport = []
        delta_logits = []
        transported_residues = []

        for state in states:
            logits, transported = lens_logits(model, state, jacobians[layer])
            branch_logits.append(logits)
            branch_transport.append(transported)
            delta_logits.append(logits - anchor_logits)
            transported_residues.append(transported - anchor_transport)

        branches_out = []
        for i, delta in enumerate(delta_logits):
            raw_norm = torch.linalg.vector_norm(raw_residues[i]).item()
            transport_norm = torch.linalg.vector_norm(transported_residues[i]).item()
            branches_out.append(
                {
                    "branch_index": i,
                    "raw_residue_norm": raw_norm,
                    "jacobian_transport_residue_norm": transport_norm,
                    "raw_transport_gain": (
                        transport_norm / max(1e-12, raw_norm)
                    ),
                    "top_positive_delta_tokens": token_rows(
                        tok, delta, args.top_k, largest=True
                    ),
                    "top_negative_delta_tokens": token_rows(
                        tok, delta, args.top_k, largest=False
                    ),
                }
            )

        receipt["layers"][str(layer)] = {
            "raw_residue_pairwise_cosines": pairwise_cosines(raw_residues),
            "transported_residue_pairwise_cosines": pairwise_cosines(
                transported_residues
            ),
            "top_positive_token_jaccards": top_token_jaccards(
                delta_logits, args.top_k
            ),
            "branches": branches_out,
        }

        # Free the largest per-layer matrix before the next iteration can move it.
        del jacobians[layer]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, default=str))
    print(json.dumps(receipt, indent=2, default=str))
    print()
    print("Interpretation boundary:")
    print("- readable branch-specific lens deltas are evidence about verbalizable format;")
    print("- top-k tokens are NOT sparse J-space coordinates;")
    print("- a useful collision still has to beat coordinate edits and behavioral attackers.")


if __name__ == "__main__":
    main()
