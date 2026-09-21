from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from statistics import mean, median

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gate05_strategy_invariance import (
    ANCHOR_INSTRUCTION,
    PREFIX,
    SUFFIX,
    VARIANTS,
    prompt as source_prompt,
)
from latent_fork import final_token_block_trace, inject_token_position


SOURCE_A = 17
SOURCE_B = 24


def parse_problem(text: str) -> tuple[int, int]:
    a, b = text.lower().split('x', 1)
    return int(a), int(b)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            'Test whether the paraphrase-invariant layer-30 route residue is causal: '
            'a one-shot linear edit at the neutral workspace token should bias later '
            'computation toward the corresponding arithmetic method.'
        )
    )
    p.add_argument('--model', default='Qwen/Qwen3-8B')
    p.add_argument('--source-layer', type=int, default=30)
    p.add_argument('--scale', type=float, default=1.0)
    p.add_argument('--heldout', default='13x26,19x23')
    p.add_argument('--gpu-memory', default='8GiB')
    p.add_argument('--cpu-memory', default='10GiB')
    p.add_argument('--offload-dir', default='.offload_qwen')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='results/gate06_causal_route_steering.json')
    return p.parse_args()


def problem_prompt(a: int, b: int, instruction: str = ANCHOR_INSTRUCTION) -> str:
    return (
        f'Problem: {a} * {b} = ?\nInstruction: {instruction}'
        + SUFFIX
    )


def continuations(a: int, b: int) -> dict[str, str]:
    tens = (b // 10) * 10
    rem = b - tens
    near = b + 1
    return {
        'A': (
            f' {a} * {tens} = {a*tens}, and {a} * {rem} = {a*rem}; '
            'combine those partial products.'
        ),
        'B': (
            f' Start at 0 and add {a} repeatedly: {a}, {2*a}, {3*a}, {4*a}, '
            'continuing the running total.'
        ),
        'C': (
            f' Use {a} * {near} = {a*near}, then subtract {a} once to compensate.'
        ),
    }


def input_device(model) -> torch.device:
    emb = model.get_input_embeddings()
    hook = getattr(emb, '_hf_hook', None)
    execution_device = getattr(hook, 'execution_device', None)
    if execution_device is not None:
        return torch.device(execution_device)
    for param in emb.parameters():
        if param.device.type != 'meta':
            return param.device
    return torch.device('cpu')


def load_model(args):
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    kwargs = {
        'dtype': dtype,
        'device_map': 'auto',
        'low_cpu_mem_usage': True,
    }
    if torch.cuda.is_available():
        kwargs['max_memory'] = {0: args.gpu_memory, 'cpu': args.cpu_memory}
        kwargs['offload_folder'] = args.offload_dir
        kwargs['offload_state_dict'] = True
        print(
            'Loading Qwen with placement caps: '
            f'cuda:0={args.gpu_memory}, cpu={args.cpu_memory}'
        )
    return AutoModelForCausalLM.from_pretrained(args.model, **kwargs).eval(), tok


def centroid(rows: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack([x.reshape(-1).float() for x in rows], dim=0).mean(dim=0)


def orthogonal_random(
    basis_vectors: list[torch.Tensor],
    target_norm: float,
    seed: int,
) -> torch.Tensor:
    g = torch.Generator(device='cpu')
    g.manual_seed(seed)
    d = basis_vectors[0].numel()
    x = torch.randn(d, generator=g)
    basis = torch.stack([v.reshape(-1).float() for v in basis_vectors], dim=1)
    x = x - basis @ (torch.linalg.pinv(basis) @ x)
    norm = torch.linalg.vector_norm(x).item()
    if norm < 1e-8:
        raise RuntimeError('random control collapsed inside route span')
    return x * (target_norm / norm)


@torch.inference_mode()
def score_continuations_batch(
    model,
    tok,
    prompt_text: str,
    candidates: dict[str, str],
    *,
    injection_layer: int,
    delta: torch.Tensor | None,
    scale: float = 1.0,
) -> dict[str, float]:
    prompt_ids = tok(prompt_text, add_special_tokens=True)['input_ids']
    candidate_ids = {
        label: tok(text, add_special_tokens=False)['input_ids']
        for label, text in candidates.items()
    }
    labels = list(candidates)
    full = [prompt_ids + candidate_ids[label] for label in labels]
    max_len = max(len(ids) for ids in full)
    pad = tok.pad_token_id
    input_rows = []
    mask_rows = []
    for ids in full:
        n = len(ids)
        input_rows.append(ids + [pad] * (max_len - n))
        mask_rows.append([1] * n + [0] * (max_len - n))

    device = input_device(model)
    input_ids = torch.tensor(input_rows, dtype=torch.long, device=device)
    attention_mask = torch.tensor(mask_rows, dtype=torch.long, device=device)

    if delta is None:
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
    else:
        with inject_token_position(
            model,
            injection_layer,
            delta,
            position=len(prompt_ids) - 1,
            scale=scale,
        ):
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )

    scores = {}
    start = len(prompt_ids) - 1
    for row, label in enumerate(labels):
        ids = candidate_ids[label]
        stop = start + len(ids)
        token_logits = out.logits[row, start:stop, :].float()
        targets = torch.tensor(ids, dtype=torch.long, device=token_logits.device)
        logp = torch.log_softmax(token_logits, dim=-1)
        chosen = logp.gather(1, targets.unsqueeze(1)).squeeze(1)
        scores[label] = float(chosen.mean().cpu())

    del out
    return scores


def lift_row(scores: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    return {label: scores[label] - baseline[label] for label in baseline}


def selectivity(lifts: dict[str, float]) -> float:
    vals = list(lifts.values())
    return max(vals) - min(vals)


def main():
    args = parse_args()
    heldout = [parse_problem(x.strip()) for x in args.heldout.split(',') if x.strip()]
    model, tok = load_model(args)
    print('Qwen load complete. Deriving source-problem route vectors.')

    anchor_text = source_prompt(ANCHOR_INSTRUCTION)
    anchor = final_token_block_trace(model, tok, anchor_text)[args.source_layer]

    strategy_residues = {}
    for label, instructions in VARIANTS.items():
        rows = []
        for index, instruction in enumerate(instructions, start=1):
            print(f'source trace {label}{index}')
            state = final_token_block_trace(model, tok, source_prompt(instruction))[
                args.source_layer
            ]
            rows.append((state - anchor).float().cpu().reshape(-1))
        strategy_residues[label] = centroid(rows)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    labels = sorted(strategy_residues)
    carrier = centroid([strategy_residues[label] for label in labels])
    departures = {label: strategy_residues[label] - carrier for label in labels}
    departure_norms = {
        label: float(torch.linalg.vector_norm(departures[label]))
        for label in labels
    }
    random_control = orthogonal_random(
        [carrier] + [departures[label] for label in labels],
        target_norm=median(departure_norms.values()),
        seed=args.seed,
    )

    if args.source_layer + 1 >= len(model.model.layers):
        raise ValueError('source layer has no downstream block for writeback')
    injection_layer = args.source_layer + 1

    problems = [(SOURCE_A, SOURCE_B)] + heldout
    receipt = {
        'scope': (
            'Gate-0.6 causal route steering. Strategy vectors are derived only from '
            '17x24 paraphrases, centered to remove the common carrier, then written '
            'once into the neutral workspace token before the next decoder block.'
        ),
        'model': args.model,
        'source_problem': [SOURCE_A, SOURCE_B],
        'source_layer': args.source_layer,
        'injection_layer': injection_layer,
        'scale': args.scale,
        'route_norms': {
            'carrier': float(torch.linalg.vector_norm(carrier)),
            'departures': departure_norms,
            'raw_strategy_centroids': {
                label: float(torch.linalg.vector_norm(strategy_residues[label]))
                for label in labels
            },
            'random_control': float(torch.linalg.vector_norm(random_control)),
        },
        'problems': {},
    }

    for problem_index, (a, b) in enumerate(problems):
        key = f'{a}x{b}'
        print(f'scoring {key}')
        ptext = problem_prompt(a, b)
        cands = continuations(a, b)
        baseline = score_continuations_batch(
            model, tok, ptext, cands,
            injection_layer=injection_layer, delta=None, scale=args.scale,
        )

        centered_scores = {}
        centered_lifts = {}
        for label in labels:
            row = score_continuations_batch(
                model, tok, ptext, cands,
                injection_layer=injection_layer,
                delta=departures[label],
                scale=args.scale,
            )
            centered_scores[label] = row
            centered_lifts[label] = lift_row(row, baseline)

        target_wins = sum(
            max(centered_lifts[label], key=centered_lifts[label].get) == label
            for label in labels
        )
        diag = [centered_lifts[label][label] for label in labels]
        offdiag = [
            centered_lifts[source][target]
            for source in labels
            for target in labels
            if source != target
        ]

        row = {
            'baseline_avg_logprob': baseline,
            'centered_route_avg_logprob': centered_scores,
            'centered_route_lift': centered_lifts,
            'centered_target_wins': target_wins,
            'centered_target_win_rate': target_wins / len(labels),
            'centered_diagonal_mean_lift': mean(diag),
            'centered_offdiagonal_mean_lift': mean(offdiag),
            'centered_diagonal_advantage': mean(diag) - mean(offdiag),
        }

        # Expensive controls are needed only on the source problem. Held-out
        # problems test whether the centered route itself transfers.
        if problem_index == 0:
            common_scores = score_continuations_batch(
                model, tok, ptext, cands,
                injection_layer=injection_layer, delta=carrier, scale=args.scale,
            )
            random_scores = score_continuations_batch(
                model, tok, ptext, cands,
                injection_layer=injection_layer, delta=random_control, scale=args.scale,
            )
            raw_scores = {}
            raw_lifts = {}
            for label in labels:
                scores = score_continuations_batch(
                    model, tok, ptext, cands,
                    injection_layer=injection_layer,
                    delta=strategy_residues[label],
                    scale=args.scale,
                )
                raw_scores[label] = scores
                raw_lifts[label] = lift_row(scores, baseline)

            row['carrier_control'] = {
                'avg_logprob': common_scores,
                'lift': lift_row(common_scores, baseline),
                'selectivity': selectivity(lift_row(common_scores, baseline)),
            }
            row['random_control'] = {
                'avg_logprob': random_scores,
                'lift': lift_row(random_scores, baseline),
                'selectivity': selectivity(lift_row(random_scores, baseline)),
            }
            row['raw_route_control'] = {
                'avg_logprob': raw_scores,
                'lift': raw_lifts,
            }

        receipt['problems'][key] = row
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    heldout_rows = [receipt['problems'][f'{a}x{b}'] for a, b in heldout]
    receipt['summary'] = {
        'source_target_win_rate': receipt['problems'][f'{SOURCE_A}x{SOURCE_B}'][
            'centered_target_win_rate'
        ],
        'source_diagonal_advantage': receipt['problems'][f'{SOURCE_A}x{SOURCE_B}'][
            'centered_diagonal_advantage'
        ],
        'heldout_mean_target_win_rate': (
            mean(row['centered_target_win_rate'] for row in heldout_rows)
            if heldout_rows else None
        ),
        'heldout_mean_diagonal_advantage': (
            mean(row['centered_diagonal_advantage'] for row in heldout_rows)
            if heldout_rows else None
        ),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))
    print()
    print('Gate interpretation:')
    print('- target continuation should receive the largest lift in >=2/3 route rows;')
    print('- diagonal mean lift should exceed off-diagonal lift;')
    print('- transfer to held-out multiplication problems is the stronger causal test.')


if __name__ == '__main__':
    main()