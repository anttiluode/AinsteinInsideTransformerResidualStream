from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from gate06_causal_route_steering import (
    SOURCE_A,
    SOURCE_B,
    load_model,
    problem_prompt,
    score_continuations_batch,
)
from gate07_numeric_first_action import (
    derive_centered_routes,
    numeric_first_actions,
    row_stats,
)
from gate07b_numeric_trajectory import numeric_trajectories
from temporal_footprint_candidates import ROUTES, temporal_candidates


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Gate 0.7d: verify that the tiny causal route effects are invariant "
            "to harmless candidate batching changes before interpreting temporal shape."
        )
    )
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--source-layer", type=int, default=30)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--gpu-memory", default="6GiB")
    p.add_argument("--cpu-memory", default="6GiB")
    p.add_argument("--offload-dir", default=".offload_qwen_gate07d")
    p.add_argument("--out", default="results/gate07d_scoring_invariance.json")
    return p.parse_args()


def score_condition(model, tok, prompt, candidates, injection_layer, departures, scale):
    baseline = score_continuations_batch(
        model,
        tok,
        prompt,
        candidates,
        injection_layer=injection_layer,
        delta=None,
        scale=scale,
    )
    route_scores = {}
    for route in ROUTES:
        route_scores[route] = score_continuations_batch(
            model,
            tok,
            prompt,
            candidates,
            injection_layer=injection_layer,
            delta=departures[route],
            scale=scale,
        )
    return baseline, route_scores


def select_labels(scores: dict[str, float], mapping: dict[str, str]) -> dict[str, float]:
    return {canonical: scores[source] for canonical, source in mapping.items()}


def select_route_scores(
    route_scores: dict[str, dict[str, float]],
    mapping: dict[str, str],
) -> dict[str, dict[str, float]]:
    return {
        route: select_labels(route_scores[route], mapping)
        for route in ROUTES
    }


def lifts(
    baseline: dict[str, float],
    route_scores: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    return {
        route: {
            target: route_scores[route][target] - baseline[target]
            for target in ROUTES
        }
        for route in ROUTES
    }


def winner_map(lift_matrix: dict[str, dict[str, float]]) -> dict[str, str]:
    return {
        route: max(lift_matrix[route], key=lift_matrix[route].get)
        for route in ROUTES
    }


def sign(value: float, eps: float = 0.0) -> int:
    if value > eps:
        return 1
    if value < -eps:
        return -1
    return 0


def compare_conditions(left: dict, right: dict) -> dict:
    raw_drifts = []
    lift_drifts = []
    for target in ROUTES:
        raw_drifts.append(abs(left["baseline"][target] - right["baseline"][target]))
    for route in ROUTES:
        for target in ROUTES:
            raw_drifts.append(
                abs(left["route_scores"][route][target] - right["route_scores"][route][target])
            )
            lift_drifts.append(
                abs(left["lift"][route][target] - right["lift"][route][target])
            )

    left_margins = left["stats"]["target_margins"]
    right_margins = right["stats"]["target_margins"]
    margin_sign_stable = {
        route: sign(left_margins[route]) == sign(right_margins[route])
        for route in ROUTES
    }
    left_winners = left["winners"]
    right_winners = right["winners"]
    winner_stable = {
        route: left_winners[route] == right_winners[route]
        for route in ROUTES
    }

    return {
        "max_abs_raw_score_drift": max(raw_drifts),
        "max_abs_lift_drift": max(lift_drifts),
        "target_margin_left": left_margins,
        "target_margin_right": right_margins,
        "target_margin_sign_stable": margin_sign_stable,
        "winner_left": left_winners,
        "winner_right": right_winners,
        "winner_stable": winner_stable,
        "all_margin_signs_stable": all(margin_sign_stable.values()),
        "all_winners_stable": all(winner_stable.values()),
    }


def package_condition(baseline, route_scores):
    lift_matrix = lifts(baseline, route_scores)
    return {
        "baseline": baseline,
        "route_scores": route_scores,
        "lift": lift_matrix,
        "stats": row_stats(lift_matrix, list(ROUTES)),
        "winners": winner_map(lift_matrix),
    }


def main():
    args = parse_args()
    print(
        "Gate 0.7d safe-memory profile: "
        f"cuda:0={args.gpu_memory}, cpu={args.cpu_memory}, "
        f"disk overflow={args.offload_dir}"
    )
    model, tok = load_model(args)
    print("Qwen load complete. Reconstructing frozen route vectors once.")

    labels, raw_routes, carrier, departures = derive_centered_routes(
        model, tok, args.source_layer
    )
    if tuple(labels) != ROUTES:
        raise RuntimeError(f"expected routes {ROUTES}, got {tuple(labels)}")
    injection_layer = args.source_layer + 1

    a, b = SOURCE_A, SOURCE_B
    prompt = problem_prompt(a, b)

    first = numeric_first_actions(a, b)
    first_reverse = {key: first[key] for key in reversed(ROUTES)}
    traj = numeric_trajectories(a, b)
    full = temporal_candidates(a, b)

    conditions = {}

    print("scoring legacy first-action batch")
    base, scores = score_condition(
        model, tok, prompt, first, injection_layer, departures, args.scale
    )
    conditions["legacy_first"] = package_condition(base, scores)

    print("scoring reversed first-action batch")
    base, scores = score_condition(
        model, tok, prompt, first_reverse, injection_layer, departures, args.scale
    )
    conditions["reversed_first"] = package_condition(
        select_labels(base, {route: route for route in ROUTES}),
        select_route_scores(scores, {route: route for route in ROUTES}),
    )

    print("scoring legacy trajectory batch")
    base, scores = score_condition(
        model, tok, prompt, traj, injection_layer, departures, args.scale
    )
    conditions["legacy_trajectory"] = package_condition(base, scores)

    print("scoring full Gate-0.7c candidate batch")
    full_base, full_scores = score_condition(
        model, tok, prompt, full, injection_layer, departures, args.scale
    )

    first_map = {route: f"{route}@1" for route in ROUTES}
    traj_map = {route: f"{route}@3" for route in ROUTES}
    conditions["full_batch_first"] = package_condition(
        select_labels(full_base, first_map),
        select_route_scores(full_scores, first_map),
    )
    conditions["full_batch_trajectory"] = package_condition(
        select_labels(full_base, traj_map),
        select_route_scores(full_scores, traj_map),
    )

    comparisons = {
        "first_legacy_vs_full": compare_conditions(
            conditions["legacy_first"], conditions["full_batch_first"]
        ),
        "first_legacy_vs_reversed": compare_conditions(
            conditions["legacy_first"], conditions["reversed_first"]
        ),
        "trajectory_legacy_vs_full": compare_conditions(
            conditions["legacy_trajectory"], conditions["full_batch_trajectory"]
        ),
    }

    primary = [
        comparisons["first_legacy_vs_full"],
        comparisons["trajectory_legacy_vs_full"],
    ]
    semantic_invariance_pass = all(
        row["all_margin_signs_stable"] and row["all_winners_stable"]
        for row in primary
    )

    commit_hash = getattr(getattr(model, "config", None), "_commit_hash", None)

    receipt = {
        "scope": (
            "Gate-0.7d numerical/scoring invariance. The same model, prompt, frozen "
            "route vectors and candidate strings are scored in different harmless "
            "batch contexts. A causal route claim this small must not reverse merely "
            "because unrelated candidate rows were added to the batch."
        ),
        "model": args.model,
        "model_commit_hash": commit_hash,
        "torch_version": torch.__version__,
        "source_problem": [a, b],
        "source_layer": args.source_layer,
        "injection_layer": injection_layer,
        "scale": args.scale,
        "route_norms": {
            "carrier": float(torch.linalg.vector_norm(carrier)),
            "departures": {
                route: float(torch.linalg.vector_norm(departures[route]))
                for route in ROUTES
            },
            "raw_strategy_centroids": {
                route: float(torch.linalg.vector_norm(raw_routes[route]))
                for route in ROUTES
            },
        },
        "conditions": conditions,
        "comparisons": comparisons,
        "summary": {
            "semantic_invariance_pass": semantic_invariance_pass,
            "kill_boundary": (
                "If target-margin signs or route winners change between the legacy "
                "three-candidate batch and the full Gate-0.7c batch, do not interpret "
                "the temporal-footprint curves. Stabilize the scoring path first."
            ),
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt["comparisons"], indent=2))
    print()
    print("Gate 0.7d:", "PASS" if semantic_invariance_pass else "FAIL")
    if not semantic_invariance_pass:
        print(
            "The current causal effects are not semantically invariant to harmless "
            "batching changes. Gate 0.7c temporal-shape interpretation is suspended."
        )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
