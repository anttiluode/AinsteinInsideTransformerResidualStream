from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from statistics import mean, median

import torch

from gate05_strategy_invariance import ANCHOR_INSTRUCTION, VARIANTS, prompt as source_prompt
from gate06_causal_route_steering import (
    SOURCE_A,
    SOURCE_B,
    centroid,
    lift_row,
    load_model,
    orthogonal_random,
    parse_problem,
    problem_prompt,
    score_continuations_batch,
)
from latent_fork import final_token_block_trace


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            'Gate 0.7: test whether causal route vectors steer the first concrete '
            'arithmetic move rather than merely method-language.'
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
    p.add_argument('--out', default='results/gate07_numeric_first_action.json')
    return p.parse_args()


def numeric_first_actions(a: int, b: int) -> dict[str, str]:
    """Three correct first moves with no strategy labels."""
    tens = (b // 10) * 10
    if tens == 0:
        tens = 10
    near = b + 1
    return {
        'A': f' {a} * {tens} = {a * tens}',
        'B': f' {a} + {a} = {2 * a}',
        'C': f' {a} * {near} = {a * near}',
    }


def derive_centered_routes(model, tok, source_layer: int):
    anchor = final_token_block_trace(
        model, tok, source_prompt(ANCHOR_INSTRUCTION)
    )[source_layer]
    strategy_residues = {}
    for label, instructions in VARIANTS.items():
        rows = []
        for index, instruction in enumerate(instructions, start=1):
            print(f'source route trace {label}{index}')
            state = final_token_block_trace(
                model, tok, source_prompt(instruction)
            )[source_layer]
            rows.append((state - anchor).float().cpu().reshape(-1))
        strategy_residues[label] = centroid(rows)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    labels = sorted(strategy_residues)
    carrier = centroid([strategy_residues[label] for label in labels])
    departures = {label: strategy_residues[label] - carrier for label in labels}
    return labels, strategy_residues, carrier, departures


def row_stats(lifts: dict[str, dict[str, float]], labels: list[str]) -> dict:
    target_wins = sum(
        max(lifts[source], key=lifts[source].get) == source
        for source in labels
    )
    diag = [lifts[label][label] for label in labels]
    offdiag = [
        lifts[source][target]
        for source in labels
        for target in labels
        if source != target
    ]
    margins = {
        label: lifts[label][label] - max(
            value for target, value in lifts[label].items() if target != label
        )
        for label in labels
    }
    return {
        'target_wins': target_wins,
        'target_win_rate': target_wins / len(labels),
        'diagonal_mean_lift': mean(diag),
        'offdiagonal_mean_lift': mean(offdiag),
        'diagonal_advantage': mean(diag) - mean(offdiag),
        'target_margins': margins,
        'mean_target_margin': mean(margins.values()),
        'min_target_margin': min(margins.values()),
    }


def main():
    args = parse_args()
    heldout = [parse_problem(x.strip()) for x in args.heldout.split(',') if x.strip()]
    model, tok = load_model(args)
    print('Qwen load complete. Reconstructing the Gate-0.6 route vectors.')

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
    forbidden_words = [
        'split', 'decompose', 'partial', 'repeat', 'running', 'nearby',
        'approximate', 'compensate', 'correction', 'strategy', 'method', 'route',
    ]
    receipt = {
        'scope': (
            'Gate-0.7 numeric first-action steering. Route vectors are unchanged '
            'from Gate 0.6 and derived only from 17x24 method paraphrases. The '
            'readout now contains only concrete arithmetic equations, with no '
            'strategy words.'
        ),
        'model': args.model,
        'source_problem': [SOURCE_A, SOURCE_B],
        'source_layer': args.source_layer,
        'injection_layer': injection_layer,
        'scale': args.scale,
        'forbidden_method_words': forbidden_words,
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
        print(f'numeric first-action scoring {key}')
        ptext = problem_prompt(a, b)
        cands = numeric_first_actions(a, b)
        lower = ' '.join(cands.values()).lower()
        leaked = [word for word in forbidden_words if word in lower]
        if leaked:
            raise RuntimeError(f'method-language leaked into numeric candidates: {leaked}')

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
            'numeric_candidates': cands,
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
    receipt['summary'] = {
        'source_target_win_rate': receipt['problems'][source_key]['target_win_rate'],
        'source_diagonal_advantage': receipt['problems'][source_key]['diagonal_advantage'],
        'source_mean_target_margin': receipt['problems'][source_key]['mean_target_margin'],
        'heldout_mean_target_win_rate': (
            mean(row['target_win_rate'] for row in heldout_rows) if heldout_rows else None
        ),
        'heldout_mean_diagonal_advantage': (
            mean(row['diagonal_advantage'] for row in heldout_rows) if heldout_rows else None
        ),
        'heldout_mean_target_margin': (
            mean(row['mean_target_margin'] for row in heldout_rows) if heldout_rows else None
        ),
        'all_problem_min_target_margin': min(
            row['min_target_margin'] for row in receipt['problems'].values()
        ),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))
    print()
    print('Gate 0.7 interpretation:')
    print('- no candidate continuation contains method-language;')
    print('- each centered route should preferentially lift its matching numeric move;')
    print('- held-out transfer is the critical test;')
    print('- if this passes, method-language alone no longer explains Gate 0.6.')


if __name__ == '__main__':
    main()