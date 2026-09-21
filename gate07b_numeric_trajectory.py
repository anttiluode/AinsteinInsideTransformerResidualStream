from __future__ import annotations

import argparse
import gc
import json
import re
from pathlib import Path
from statistics import mean, median

import torch

from gate06_causal_route_steering import (
    SOURCE_A,
    SOURCE_B,
    lift_row,
    load_model,
    orthogonal_random,
    parse_problem,
    problem_prompt,
    score_continuations_batch,
)
from gate07_numeric_first_action import derive_centered_routes, row_stats


NUMERIC_ONLY = re.compile(r'^[0-9+*\-=;\s]+$')


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            'Gate 0.7b: test route steering on short arithmetic trajectories. '
            'Candidates contain equations only, no method-language.'
        )
    )
    p.add_argument('--model', default='Qwen/Qwen3-8B')
    p.add_argument('--source-layer', type=int, default=30)
    p.add_argument('--scale', type=float, default=1.0)
    p.add_argument('--heldout', default='13x26,19x23')
    p.add_argument('--gpu-memory', default='6GiB')
    p.add_argument('--cpu-memory', default='6GiB')
    p.add_argument('--offload-dir', default='.offload_qwen_gate07b')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='results/gate07b_numeric_trajectory.json')
    return p.parse_args()


def numeric_trajectories(a: int, b: int) -> dict[str, str]:
    """Three equation-only trajectories with distinct transformation laws."""
    tens = (b // 10) * 10
    if tens == 0:
        tens = 10
    rem = b - tens
    near = b + 1
    product = a * b

    a1 = a * tens
    a2 = a * rem

    b1 = 2 * a
    b2 = 3 * a
    b3 = 4 * a

    c1 = a * near

    return {
        'A': f' {a} * {tens} = {a1}; {a} * {rem} = {a2}; {a1} + {a2} = {product}',
        'B': f' {a} + {a} = {b1}; {b1} + {a} = {b2}; {b2} + {a} = {b3}',
        'C': f' {a} * {near} = {c1}; {c1} - {a} = {product}; {product} = {a} * {b}',
    }


def assert_numeric_only(cands: dict[str, str]) -> None:
    for label, text in cands.items():
        if not NUMERIC_ONLY.fullmatch(text):
            raise RuntimeError(
                f'non-numeric character leaked into trajectory {label}: {text!r}'
            )


def main():
    args = parse_args()
    heldout = [parse_problem(x.strip()) for x in args.heldout.split(',') if x.strip()]

    print(
        'Gate 0.7b safe-memory profile: '
        f'cuda:0={args.gpu_memory}, cpu={args.cpu_memory}, '
        f'disk overflow={args.offload_dir}'
    )
    print(
        'The deliberately low combined placement budget forces part of Qwen '
        'to disk instead of relying on transient Windows RAM/VRAM headroom.'
    )
    model, tok = load_model(args)
    print('Qwen load complete. Reconstructing centered route vectors.')

    labels, raw_routes, carrier, departures = derive_centered_routes(
        model, tok, args.source_layer
    )
    if args.source_layer + 1 >= len(model.model.layers):
        raise ValueError('source layer has no downstream block for writeback')
    injection_layer = args.source_layer + 1

    departure_norms = {
        label: float(torch.linalg.vector_norm(departures[label]))
        for label in labels
    }
    random_control = orthogonal_random(
        [carrier] + [departures[label] for label in labels],
        target_norm=median(departure_norms.values()),
        seed=args.seed,
    )

    problems = [(SOURCE_A, SOURCE_B)] + heldout
    receipt = {
        'scope': (
            'Gate-0.7b numeric trajectory steering. Same route vectors as Gates '
            '0.6/0.7, derived only from 17x24 method paraphrases. Readout contains '
            'equations and operators only; no alphabetic method-language.'
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
                label: float(torch.linalg.vector_norm(raw_routes[label]))
                for label in labels
            },
            'random_control': float(torch.linalg.vector_norm(random_control)),
        },
        'problems': {},
    }

    for problem_index, (a, b) in enumerate(problems):
        key = f'{a}x{b}'
        print(f'numeric trajectory scoring {key}')
        ptext = problem_prompt(a, b)
        cands = numeric_trajectories(a, b)
        assert_numeric_only(cands)

        baseline = score_continuations_batch(
            model, tok, ptext, cands,
            injection_layer=injection_layer, delta=None, scale=args.scale,
        )

        centered_scores = {}
        centered_lifts = {}
        for label in labels:
            scores = score_continuations_batch(
                model, tok, ptext, cands,
                injection_layer=injection_layer,
                delta=departures[label],
                scale=args.scale,
            )
            centered_scores[label] = scores
            centered_lifts[label] = lift_row(scores, baseline)

        row = {
            'numeric_trajectories': cands,
            'candidate_token_ids': {
                label: tok(text, add_special_tokens=False)['input_ids']
                for label, text in cands.items()
            },
            'baseline_avg_logprob': baseline,
            'centered_route_avg_logprob': centered_scores,
            'centered_route_lift': centered_lifts,
            **row_stats(centered_lifts, labels),
        }

        if problem_index == 0:
            carrier_scores = score_continuations_batch(
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
                    delta=raw_routes[label],
                    scale=args.scale,
                )
                raw_scores[label] = scores
                raw_lifts[label] = lift_row(scores, baseline)

            row['carrier_control'] = {
                'avg_logprob': carrier_scores,
                'lift': lift_row(carrier_scores, baseline),
            }
            row['random_control'] = {
                'avg_logprob': random_scores,
                'lift': lift_row(random_scores, baseline),
            }
            row['raw_route_control'] = {
                'avg_logprob': raw_scores,
                'lift': raw_lifts,
                'stats': row_stats(raw_lifts, labels),
            }

        receipt['problems'][key] = row
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    source_key = f'{SOURCE_A}x{SOURCE_B}'
    heldout_rows = [receipt['problems'][f'{a}x{b}'] for a, b in heldout]
    source = receipt['problems'][source_key]

    receipt['summary'] = {
        'source_target_win_rate': source['target_win_rate'],
        'source_diagonal_advantage': source['diagonal_advantage'],
        'source_mean_target_margin': source['mean_target_margin'],
        'source_route_B_margin': source['target_margins']['B'],
        'heldout_mean_target_win_rate': (
            mean(row['target_win_rate'] for row in heldout_rows) if heldout_rows else None
        ),
        'heldout_mean_diagonal_advantage': (
            mean(row['diagonal_advantage'] for row in heldout_rows) if heldout_rows else None
        ),
        'heldout_mean_target_margin': (
            mean(row['mean_target_margin'] for row in heldout_rows) if heldout_rows else None
        ),
        'heldout_route_B_margins': [
            row['target_margins']['B'] for row in heldout_rows
        ],
        'all_problem_min_target_margin': min(
            row['min_target_margin'] for row in receipt['problems'].values()
        ),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))
    print()
    print('Gate 0.7b interpretation:')
    print('- A/B/C candidates are short numeric trajectories, not prose descriptions;')
    print('- B now has several successive +a transitions, so its identity is temporal;')
    print('- recovery of B would support route-as-transformation-law;')
    print('- persistent B failure would narrow the causal claim to A/C-like routes.')


if __name__ == '__main__':
    main()