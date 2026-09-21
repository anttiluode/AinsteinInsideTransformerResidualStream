from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from statistics import mean, median

import torch

from gate06_causal_route_steering import (
    SOURCE_A,
    SOURCE_B,
    load_model,
    orthogonal_random,
    parse_problem,
    problem_prompt,
    score_continuations_batch,
)
from gate07_numeric_first_action import derive_centered_routes, row_stats
from temporal_footprint_candidates import HORIZONS, ROUTES, temporal_candidates


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Gate 0.7c: measure the causal temporal footprint of the frozen "
            "route directions and attack route B with operator-matched recurrences."
        )
    )
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--source-layer", type=int, default=30)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--heldout", default="13x26,19x23")
    p.add_argument("--gpu-memory", default="6GiB")
    p.add_argument("--cpu-memory", default="6GiB")
    p.add_argument("--offload-dir", default=".offload_qwen_gate07c")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/gate07c_temporal_footprint.json")
    return p.parse_args()



def token_counts_and_prefix_check(tok, candidates: dict[str, str]) -> dict[str, int]:
    ids = {
        label: tok(text, add_special_tokens=False)["input_ids"]
        for label, text in candidates.items()
    }
    for route in ROUTES:
        for horizon in (2, 3):
            prev = ids[f"{route}@{horizon - 1}"]
            cur = ids[f"{route}@{horizon}"]
            if cur[: len(prev)] != prev:
                raise RuntimeError(
                    f"tokenization changed inside {route} prefix at horizon {horizon}; "
                    "incremental footprint would be invalid"
                )
    return {label: len(row) for label, row in ids.items()}


def route_lift_matrix(
    route_scores: dict[str, dict[str, float]],
    baseline: dict[str, float],
    horizon: int,
) -> dict[str, dict[str, float]]:
    return {
        source: {
            target: route_scores[source][f"{target}@{horizon}"]
            - baseline[f"{target}@{horizon}"]
            for target in ROUTES
        }
        for source in ROUTES
    }


def incremental_score(
    scores: dict[str, float],
    counts: dict[str, int],
    route: str,
    horizon: int,
) -> float:
    current_label = f"{route}@{horizon}"
    current_total = scores[current_label] * counts[current_label]
    if horizon == 1:
        return current_total / counts[current_label]

    previous_label = f"{route}@{horizon - 1}"
    previous_total = scores[previous_label] * counts[previous_label]
    added_tokens = counts[current_label] - counts[previous_label]
    if added_tokens <= 0:
        raise RuntimeError(f"non-positive token increment for {current_label}")
    return (current_total - previous_total) / added_tokens


def incremental_lift_matrix(
    route_scores: dict[str, dict[str, float]],
    baseline: dict[str, float],
    counts: dict[str, int],
    horizon: int,
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for source in ROUTES:
        result[source] = {}
        for target in ROUTES:
            steered = incremental_score(route_scores[source], counts, target, horizon)
            clean = incremental_score(baseline, counts, target, horizon)
            result[source][target] = steered - clean
    return result


def b_law_attacker(
    b_scores: dict[str, float],
    baseline: dict[str, float],
) -> dict:
    target = "B@3"
    decoys = [
        "B_decoy_same_first_drift",
        "B_decoy_unit_recurrence",
        "B_decoy_other_constant",
    ]
    lifts = {target: b_scores[target] - baseline[target]}
    lifts.update({name: b_scores[name] - baseline[name] for name in decoys})
    best_decoy = max(decoys, key=lambda name: lifts[name])
    return {
        "target": target,
        "decoys": decoys,
        "lift": lifts,
        "best_decoy": best_decoy,
        "target_margin_over_best_decoy": lifts[target] - lifts[best_decoy],
        "target_wins": lifts[target] > lifts[best_decoy],
    }


def temporal_shape_summary(problem_rows: dict[str, dict]) -> dict:
    cumulative_means: dict[str, dict[str, float]] = {route: {} for route in ROUTES}
    incremental_means: dict[str, dict[str, float]] = {route: {} for route in ROUTES}

    for route in ROUTES:
        for horizon in HORIZONS:
            cumulative_means[route][str(horizon)] = mean(
                row["horizons"][str(horizon)]["cumulative_stats"]["target_margins"][route]
                for row in problem_rows.values()
            )
            incremental_means[route][str(horizon)] = mean(
                row["horizons"][str(horizon)]["incremental_stats"]["target_margins"][route]
                for row in problem_rows.values()
            )

    law_margins = {
        key: row["b_law_attacker"]["target_margin_over_best_decoy"]
        for key, row in problem_rows.items()
    }
    law_wins = sum(row["b_law_attacker"]["target_wins"] for row in problem_rows.values())

    shape_checks = {
        "A_short_horizon": (
            cumulative_means["A"]["1"] > 0
            and cumulative_means["A"]["1"] > cumulative_means["A"]["3"]
        ),
        "B_late_horizon": cumulative_means["B"]["3"] > cumulative_means["B"]["1"],
        "C_persistent_positive": all(
            cumulative_means["C"][str(h)] > 0 for h in HORIZONS
        ),
        "B_true_law_wins_majority": law_wins >= 2,
    }

    return {
        "mean_cumulative_target_margin_by_route_and_horizon": cumulative_means,
        "mean_incremental_target_margin_by_route_and_horizon": incremental_means,
        "b_law_margin_by_problem": law_margins,
        "b_law_target_wins": law_wins,
        "shape_checks": shape_checks,
        "all_shape_checks_pass": all(shape_checks.values()),
    }


def main():
    args = parse_args()
    heldout = [parse_problem(x.strip()) for x in args.heldout.split(",") if x.strip()]

    print(
        "Gate 0.7c safe-memory profile: "
        f"cuda:0={args.gpu_memory}, cpu={args.cpu_memory}, "
        f"disk overflow={args.offload_dir}"
    )
    model, tok = load_model(args)
    print("Qwen load complete. Reconstructing frozen Gate-0.6 route vectors.")

    labels, raw_routes, carrier, departures = derive_centered_routes(
        model, tok, args.source_layer
    )
    if tuple(labels) != ROUTES:
        raise RuntimeError(f"expected routes {ROUTES}, got {tuple(labels)}")
    if args.source_layer + 1 >= len(model.model.layers):
        raise ValueError("source layer has no downstream block for writeback")
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

    receipt = {
        "scope": (
            "Gate-0.7c causal temporal footprint. The same frozen centered route "
            "vectors from Gates 0.6-0.7b are injected once at the neutral prompt "
            "state. Numeric route prefixes are scored at horizons 1,2,3. Route B "
            "is also attacked with same-operator recurrence decoys."
        ),
        "pre_registered_hypothesis": {
            "A": "strongest early; causal target margin should decay with horizon",
            "B": "weak early; causal target margin should improve over recurrence depth",
            "C": "positive and comparatively persistent across short horizons",
            "B_law": (
                "the true +a recurrence should beat numeric decoys that preserve "
                "the + operator, recurrence depth, or the first edge"
            ),
        },
        "interpretation_boundary": (
            "A temporal footprint is not proof that the route vector itself rotates "
            "with hidden state. It shows only that one fixed intervention has a "
            "route-specific causal support profile across successive transitions."
        ),
        "model": args.model,
        "source_problem": [SOURCE_A, SOURCE_B],
        "source_layer": args.source_layer,
        "injection_layer": injection_layer,
        "scale": args.scale,
        "route_norms": {
            "carrier": float(torch.linalg.vector_norm(carrier)),
            "departures": departure_norms,
            "raw_strategy_centroids": {
                label: float(torch.linalg.vector_norm(raw_routes[label]))
                for label in labels
            },
            "random_control": float(torch.linalg.vector_norm(random_control)),
        },
        "problems": {},
    }

    problems = [(SOURCE_A, SOURCE_B)] + heldout
    for problem_index, (a, b) in enumerate(problems):
        key = f"{a}x{b}"
        print(f"temporal footprint scoring {key}")
        ptext = problem_prompt(a, b)
        candidates = temporal_candidates(a, b)
        counts = token_counts_and_prefix_check(tok, candidates)

        baseline = score_continuations_batch(
            model,
            tok,
            ptext,
            candidates,
            injection_layer=injection_layer,
            delta=None,
            scale=args.scale,
        )
        route_scores = {}
        for route in ROUTES:
            route_scores[route] = score_continuations_batch(
                model,
                tok,
                ptext,
                candidates,
                injection_layer=injection_layer,
                delta=departures[route],
                scale=args.scale,
            )

        horizons = {}
        for horizon in HORIZONS:
            cumulative = route_lift_matrix(route_scores, baseline, horizon)
            incremental = incremental_lift_matrix(
                route_scores, baseline, counts, horizon
            )
            horizons[str(horizon)] = {
                "cumulative_lift": cumulative,
                "cumulative_stats": row_stats(cumulative, list(ROUTES)),
                "incremental_lift": incremental,
                "incremental_stats": row_stats(incremental, list(ROUTES)),
            }

        row = {
            "candidates": candidates,
            "candidate_token_counts": counts,
            "baseline_avg_logprob": baseline,
            "route_avg_logprob": route_scores,
            "horizons": horizons,
            "b_law_attacker": b_law_attacker(route_scores["B"], baseline),
        }

        if problem_index == 0:
            carrier_scores = score_continuations_batch(
                model,
                tok,
                ptext,
                candidates,
                injection_layer=injection_layer,
                delta=carrier,
                scale=args.scale,
            )
            random_scores = score_continuations_batch(
                model,
                tok,
                ptext,
                candidates,
                injection_layer=injection_layer,
                delta=random_control,
                scale=args.scale,
            )
            row["source_controls"] = {
                "carrier_b_law": b_law_attacker(carrier_scores, baseline),
                "random_b_law": b_law_attacker(random_scores, baseline),
            }

        receipt["problems"][key] = row
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    receipt["summary"] = temporal_shape_summary(receipt["problems"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))
    print()
    print("Gate 0.7c interpretation:")
    print("- compare route target margins as a function of horizon, not one score;")
    print("- inspect incremental margins to see which transition receives the effect;")
    print("- B must beat matched recurrence decoys before transformation-law earns credit;")
    print("- this gate measures support horizon, not hidden-state vector-field rotation.")


if __name__ == "__main__":
    main()
