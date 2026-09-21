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
from gate07_numeric_first_action import derive_centered_routes, numeric_first_actions
from gate07d_scoring_invariance import (
    compare_conditions,
    package_condition,
    select_labels,
    select_route_scores,
)
from temporal_footprint_candidates import ROUTES, temporal_candidates


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Gate 0.7e: ask whether route steering rises above the BF16/batch-shape "
            "numerical floor when intervention strength is increased prospectively."
        )
    )
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--source-layer", type=int, default=30)
    p.add_argument("--scales", default="0.5,1.0,2.0")
    p.add_argument("--primary-scale", type=float, default=2.0)
    p.add_argument("--gpu-memory", default="6GiB")
    p.add_argument("--cpu-memory", default="6GiB")
    p.add_argument("--offload-dir", default=".offload_qwen_gate07e")
    p.add_argument("--out", default="results/gate07e_numerical_floor.json")
    return p.parse_args()


def flatten_lifts(condition: dict) -> torch.Tensor:
    vals = []
    for source in ROUTES:
        for target in ROUTES:
            vals.append(condition["lift"][source][target])
    return torch.tensor(vals, dtype=torch.float64)


def condition_similarity(left: dict, right: dict) -> dict:
    a = flatten_lifts(left)
    b = flatten_lifts(right)
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    cosine = float(torch.dot(a, b) / denom) if float(denom) > 0 else 0.0
    disagreement = float(torch.linalg.vector_norm(a - b))
    signal = float(torch.linalg.vector_norm((a + b) / 2.0))
    return {
        "lift_cosine": cosine,
        "lift_disagreement_norm": disagreement,
        "mean_lift_signal_norm": signal,
        "signal_to_disagreement": signal / disagreement if disagreement > 0 else None,
    }


def score_context(
    model,
    tok,
    prompt,
    candidates,
    injection_layer,
    departures,
    scale,
    baseline,
):
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
    return package_condition(baseline, route_scores)


def main():
    args = parse_args()
    scales = [float(x.strip()) for x in args.scales.split(",") if x.strip()]
    if args.primary_scale not in scales:
        raise ValueError("--primary-scale must be one of --scales")

    print(
        "Gate 0.7e safe-memory profile: "
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
    legacy = numeric_first_actions(a, b)
    full = temporal_candidates(a, b)

    print("scoring batch-context baselines once")
    legacy_baseline = score_continuations_batch(
        model,
        tok,
        prompt,
        legacy,
        injection_layer=injection_layer,
        delta=None,
        scale=1.0,
    )
    full_baseline_all = score_continuations_batch(
        model,
        tok,
        prompt,
        full,
        injection_layer=injection_layer,
        delta=None,
        scale=1.0,
    )
    first_map = {route: f"{route}@1" for route in ROUTES}
    full_baseline = select_labels(full_baseline_all, first_map)

    rows = {}
    for scale in scales:
        print(f"scale sweep {scale:g}: legacy three-row batch")
        legacy_condition = score_context(
            model,
            tok,
            prompt,
            legacy,
            injection_layer,
            departures,
            scale,
            legacy_baseline,
        )

        print(f"scale sweep {scale:g}: full Gate-0.7c batch")
        full_scores_all = {}
        for route in ROUTES:
            full_scores_all[route] = score_continuations_batch(
                model,
                tok,
                prompt,
                full,
                injection_layer=injection_layer,
                delta=departures[route],
                scale=scale,
            )
        full_condition = package_condition(
            full_baseline,
            select_route_scores(full_scores_all, first_map),
        )

        comparison = compare_conditions(legacy_condition, full_condition)
        comparison.update(condition_similarity(legacy_condition, full_condition))

        rows[f"{scale:g}"] = {
            "legacy": legacy_condition,
            "full_batch": full_condition,
            "comparison": comparison,
        }

    primary = rows[f"{args.primary_scale:g}"]["comparison"]
    primary_pass = (
        primary["all_margin_signs_stable"]
        and primary["all_winners_stable"]
    )

    receipt = {
        "scope": (
            "Gate-0.7e numerical-floor calibration. Gate 0.7d showed that scale-1 "
            "first-action route effects can flip when unrelated candidate rows change "
            "the BF16 batch shape. This gate prospectively tests whether doubling the "
            "same frozen intervention restores semantic agreement across those contexts."
        ),
        "model": args.model,
        "model_commit_hash": getattr(getattr(model, "config", None), "_commit_hash", None),
        "torch_version": torch.__version__,
        "source_problem": [a, b],
        "source_layer": args.source_layer,
        "injection_layer": injection_layer,
        "scales": scales,
        "primary_scale": args.primary_scale,
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
        "rows": rows,
        "summary": {
            "primary_scale_semantic_invariance_pass": primary_pass,
            "primary_comparison": primary,
            "interpretation": (
                "PASS: scale-2 steering is above the present semantic numerical floor; "
                "rerun the temporal gate at the pre-registered stable scale. "
                "FAIL: increasing effect size did not stabilize the causal readout; "
                "switch to a serial or higher-precision scorer before further route claims."
            ),
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2))

    print(json.dumps(receipt["summary"], indent=2))
    print()
    print("Gate 0.7e:", "PASS" if primary_pass else "FAIL")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
