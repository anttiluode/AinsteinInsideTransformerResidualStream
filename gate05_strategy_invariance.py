from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from statistics import mean

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from gate05_jspace_readout import (
    DEFAULT_LENS_PREFIX,
    DEFAULT_LENS_REPO,
    lens_logits,
    load_jacobians,
    resolve_lens_file,
)
from latent_fork import final_token_block_trace


PREFIX = "Problem: 17 * 24 = ?\nInstruction: "
SUFFIX = "\nDo not state the answer yet. Continue from this method.\nWork:"

ANCHOR_INSTRUCTION = "solve the problem carefully."

VARIANTS = {
    "A": [
        "split 24 into 20 and 4, multiply 17 by each part, then combine the partial products.",
        "break one factor into an easy round chunk plus its remainder, compute both partial products, then add them.",
    ],
    "B": [
        "build the product as a running sum by adding 17 twenty-four times, checking the accumulation as you go.",
        "accumulate twenty-four copies of 17 one after another, repairing any slip in the running total before finishing.",
    ],
    "C": [
        "start from a convenient product close to 17 times 24, then compensate for the difference to reach the target.",
        "choose an easy multiplication around the target value and adjust for the offset instead of calculating it directly.",
    ],
}

DECOYS = [
    {
        "name": "A_with_nearby_decoy",
        "intended": "A",
        "lexical_decoy": "C",
        "instruction": (
            "ignore nearby multiples; split 24 into 20 and 4, multiply 17 by each "
            "part, then combine the partial products."
        ),
    },
    {
        "name": "B_with_decomposition_decoy",
        "intended": "B",
        "lexical_decoy": "A",
        "instruction": (
            "do not use decomposition; build the product as a running sum by "
            "adding 17 twenty-four times and check the accumulation."
        ),
    },
    {
        "name": "C_with_correction_decoy",
        "intended": "C",
        "lexical_decoy": "B",
        "instruction": (
            "do not use repeated addition or correction; start from a convenient "
            "product close to the target and compensate for the difference."
        ),
    },
]


def prompt(instruction: str) -> str:
    return PREFIX + instruction + SUFFIX


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Attack lexical echo in the Gate-0.5 J-lens result with common-suffix "
            "strategy paraphrases and negated-keyword decoys."
        )
    )
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--lens-repo", default=DEFAULT_LENS_REPO)
    p.add_argument("--lens-file", default="auto")
    p.add_argument("--lens-prefix", default=DEFAULT_LENS_PREFIX)
    p.add_argument(
        "--layers",
        default="30",
        help="Comma-separated J-lens layers. Layer 30 is the first cheap target.",
    )
    p.add_argument("--top-k", type=int, default=15)
    p.add_argument("--gpu-memory", default="8GiB")
    p.add_argument("--cpu-memory", default="10GiB")
    p.add_argument("--offload-dir", default=".offload_qwen")
    p.add_argument(
        "--out",
        default="results/gate05_strategy_invariance.json",
    )
    return p.parse_args()


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(1, -1).float()
    b = b.reshape(1, -1).float()
    return torch.nn.functional.cosine_similarity(a, b, dim=-1).item()


def group_pair_stats(
    vectors: dict[str, torch.Tensor],
    labels: dict[str, str],
    base_names: list[str],
) -> dict:
    within = []
    between = []
    pair_rows = []
    for i, name_i in enumerate(base_names):
        for name_j in base_names[i + 1 :]:
            c = cosine(vectors[name_i], vectors[name_j])
            same = labels[name_i] == labels[name_j]
            (within if same else between).append(c)
            pair_rows.append(
                {
                    "a": name_i,
                    "b": name_j,
                    "same_strategy": same,
                    "cosine": c,
                }
            )
    within_mean = mean(within) if within else None
    between_mean = mean(between) if between else None
    return {
        "within_strategy_mean": within_mean,
        "between_strategy_mean": between_mean,
        "invariance_gap": (
            None
            if within_mean is None or between_mean is None
            else within_mean - between_mean
        ),
        "pairs": pair_rows,
    }


def centroid(vectors: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack([v.float().reshape(-1) for v in vectors], dim=0).mean(dim=0)


def wordlike_top_tokens(
    tokenizer,
    values: torch.Tensor,
    top_k: int,
    *,
    candidate_k: int = 500,
) -> list[dict]:
    flat = values.reshape(-1)
    k = min(candidate_k, flat.numel())
    scores, ids = torch.topk(flat, k=k, largest=True)
    rows = []
    for score, token_id in zip(scores.tolist(), ids.tolist()):
        token = tokenizer.decode([int(token_id)])
        if not any(ch.isalpha() for ch in token):
            continue
        rows.append(
            {
                "token_id": int(token_id),
                "token": token,
                "score": float(score),
            }
        )
        if len(rows) >= top_k:
            break
    return rows


def common_suffix_token_count(sequences: list[list[int]]) -> int:
    if not sequences:
        return 0
    n = 0
    limit = min(len(s) for s in sequences)
    for offset in range(1, limit + 1):
        token = sequences[0][-offset]
        if any(s[-offset] != token for s in sequences[1:]):
            break
        n += 1
    return n


def load_model(args):
    tok = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    kwargs = {
        "dtype": dtype,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }
    if torch.cuda.is_available():
        kwargs["max_memory"] = {0: args.gpu_memory, "cpu": args.cpu_memory}
        kwargs["offload_folder"] = args.offload_dir
        kwargs["offload_state_dict"] = True
        print(
            "Loading Qwen first with placement caps: "
            f"cuda:0={args.gpu_memory}, cpu={args.cpu_memory}"
        )
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs).eval()
    return model, tok


def main():
    args = parse_args()
    layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]

    # Build a common-suffix family. Every analyzed prompt ends in the exact same
    # suffix, so final-token identity is no longer confounded with strategy name.
    prompts = {}
    labels = {}
    kinds = {}

    anchor_name = "anchor"
    prompts[anchor_name] = prompt(ANCHOR_INSTRUCTION)
    labels[anchor_name] = "anchor"
    kinds[anchor_name] = "anchor"

    base_names = []
    for label, instructions in VARIANTS.items():
        for index, instruction in enumerate(instructions, start=1):
            name = f"{label}{index}"
            prompts[name] = prompt(instruction)
            labels[name] = label
            kinds[name] = "paraphrase"
            base_names.append(name)

    decoy_meta = {}
    for row in DECOYS:
        name = row["name"]
        prompts[name] = prompt(row["instruction"])
        labels[name] = row["intended"]
        kinds[name] = "lexical_decoy"
        decoy_meta[name] = row

    model, tok = load_model(args)
    print("Qwen load complete. Capturing prompt traces.")

    token_ids = {
        name: tok(text, add_special_tokens=True)["input_ids"]
        for name, text in prompts.items()
    }
    analyzed_sequences = [token_ids[name] for name in prompts]
    common_suffix = common_suffix_token_count(analyzed_sequences)
    final_ids = {ids[-1] for ids in analyzed_sequences}
    if len(final_ids) != 1:
        raise RuntimeError(
            f"common-suffix control failed: final token ids differ: {sorted(final_ids)}"
        )

    traces = {}
    for index, (name, text) in enumerate(prompts.items(), start=1):
        print(f"[{index}/{len(prompts)}] forward: {name}")
        traces[name] = final_token_block_trace(model, tok, text)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("Prompt traces complete. Opening Jacobian lens.")
    lens_file = resolve_lens_file(args.lens_repo, args.lens_file, args.lens_prefix)
    lens_path = hf_hub_download(args.lens_repo, lens_file)
    jacobians, lens_meta = load_jacobians(lens_path)

    missing = sorted(set(layers) - set(jacobians))
    if missing:
        raise ValueError(
            f"layers {missing} absent from lens; available={sorted(jacobians)}"
        )

    receipt = {
        "scope": (
            "Gate-0.5 lexical-echo attacker. Same-strategy paraphrases and "
            "negated-keyword decoys share an identical suffix/final token."
        ),
        "model": args.model,
        "lens_repo": args.lens_repo,
        "lens_file": lens_file,
        "lens_meta": {
            k: (v.tolist() if isinstance(v, torch.Tensor) else v)
            for k, v in lens_meta.items()
        },
        "prompt_control": {
            "common_suffix_text": SUFFIX,
            "common_suffix_token_count_across_all_prompts": common_suffix,
            "shared_final_token_id": next(iter(final_ids)),
            "shared_final_token": tok.decode([next(iter(final_ids))]),
            "token_counts": {name: len(ids) for name, ids in token_ids.items()},
        },
        "prompts": {
            name: {
                "label": labels[name],
                "kind": kinds[name],
                "text": text,
                **(
                    {
                        "lexical_decoy_label": decoy_meta[name]["lexical_decoy"],
                    }
                    if name in decoy_meta
                    else {}
                ),
            }
            for name, text in prompts.items()
        },
        "layers": {},
    }

    for layer in layers:
        print(f"J-lens layer {layer}")
        J = jacobians[layer].float().cpu()

        anchor_state = traces[anchor_name][layer]
        anchor_logits, anchor_transport = lens_logits(model, anchor_state, J)

        raw_residues = {}
        transported_residues = {}
        delta_logits = {}
        prompt_rows = {}

        for name in prompts:
            if name == anchor_name:
                continue
            state = traces[name][layer]
            raw = state - anchor_state
            logits, transported = lens_logits(model, state, J)
            delta = logits - anchor_logits
            trans_residue = transported - anchor_transport

            raw_residues[name] = raw.cpu()
            transported_residues[name] = trans_residue.cpu()
            delta_logits[name] = delta.cpu()

            prompt_rows[name] = {
                "label": labels[name],
                "kind": kinds[name],
                "raw_residue_norm": torch.linalg.vector_norm(raw).item(),
                "transported_residue_norm": torch.linalg.vector_norm(trans_residue).item(),
                "top_wordlike_positive_delta_tokens": wordlike_top_tokens(
                    tok, delta, args.top_k
                ),
            }

        # Invariance test uses only the two paraphrases per strategy.
        raw_stats = group_pair_stats(raw_residues, labels, base_names)
        transport_stats = group_pair_stats(
            transported_residues, labels, base_names
        )
        readable_stats = group_pair_stats(delta_logits, labels, base_names)

        centroids = {
            label: centroid(
                [delta_logits[name] for name in base_names if labels[name] == label]
            )
            for label in sorted(VARIANTS)
        }

        decoy_rows = []
        decoy_correct = 0
        decoy_keyword_wins = 0
        for name, meta in decoy_meta.items():
            sims = {
                label: cosine(delta_logits[name], center)
                for label, center in centroids.items()
            }
            predicted = max(sims, key=sims.get)
            if predicted == meta["intended"]:
                decoy_correct += 1
            if predicted == meta["lexical_decoy"]:
                decoy_keyword_wins += 1
            decoy_rows.append(
                {
                    "name": name,
                    "intended": meta["intended"],
                    "lexical_decoy": meta["lexical_decoy"],
                    "centroid_cosines": sims,
                    "nearest_strategy_centroid": predicted,
                }
            )

        receipt["layers"][str(layer)] = {
            "raw_residue_invariance": raw_stats,
            "transported_residue_invariance": transport_stats,
            "lens_logit_invariance": readable_stats,
            "decoy_test": {
                "rows": decoy_rows,
                "intended_strategy_accuracy": decoy_correct / len(decoy_rows),
                "lexical_decoy_capture_rate": decoy_keyword_wins / len(decoy_rows),
            },
            "prompts": prompt_rows,
        }

        del J
        del jacobians[layer]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, default=str))
    print(json.dumps(receipt, indent=2, default=str))
    print()
    print("Interpretation:")
    print("- positive within-minus-between lens-logit cosine supports paraphrase invariance;")
    print("- decoys should follow intended strategy, not the negated surface keyword;")
    print("- failure means the previous readable L30 result was largely lexical echo.")


if __name__ == "__main__":
    main()
