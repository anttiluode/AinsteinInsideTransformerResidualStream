from __future__ import annotations

import argparse
import json
from math import exp
from pathlib import Path

from foveation import (
    adaptive_depth_allocation,
    allocation_entropy,
    total_utility,
    uniform_depth_allocation,
)


def payoff(branch_id: str, depth: int) -> float:
    """
    Deterministic scheduler receipt, not a transformer result.

    A: attractive immediately, then saturates.
    B: late bloomer; weak first probe, then high ceiling.
    C: weak route.
    D: strong route with ordinary diminishing returns.
    """
    if depth <= 0:
        return 0.0
    if branch_id == "A":
        return 1.8 * (1.0 - exp(-1.4 * depth))
    if branch_id == "B":
        return 4.0 * (1.0 - exp(-0.35 * max(0, depth - 1)))
    if branch_id == "C":
        return 0.8 * (1.0 - exp(-0.6 * depth))
    if branch_id == "D":
        return 3.0 * (1.0 - exp(-0.8 * depth))
    raise KeyError(branch_id)


def run(total_budget: int = 16) -> dict:
    branch_ids = ["A", "B", "C", "D"]

    uniform = uniform_depth_allocation(branch_ids, total_budget)
    greedy, greedy_trace = adaptive_depth_allocation(
        branch_ids,
        total_budget,
        payoff,
        min_probe_depth=1,
        exploration=0.05,
        diversity_reserve=0.0,
    )
    adaptive, adaptive_trace = adaptive_depth_allocation(
        branch_ids,
        total_budget,
        payoff,
        min_probe_depth=1,
        exploration=0.05,
        diversity_reserve=0.20,
    )

    return {
        "scope": "scheduler-only receipt; no transformer capability claim",
        "budget": total_budget,
        "profiles": {
            "A": "high early gain, low ceiling",
            "B": "late bloomer, high ceiling",
            "C": "weak",
            "D": "strong diminishing returns",
        },
        "uniform": {
            "depths": uniform,
            "utility": total_utility(uniform, payoff),
            "allocation_entropy": allocation_entropy(uniform),
        },
        "greedy_no_diversity": {
            "depths": greedy,
            "utility": total_utility(greedy, payoff),
            "allocation_entropy": allocation_entropy(greedy),
            "late_bloomer_depth": greedy["B"],
            "trace": greedy_trace,
        },
        "adaptive_with_diversity": {
            "depths": adaptive,
            "utility": total_utility(adaptive, payoff),
            "allocation_entropy": allocation_entropy(adaptive),
            "late_bloomer_depth": adaptive["B"],
            "trace": adaptive_trace,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=16)
    parser.add_argument("--out", default="results/gate1_allocator_receipt.json")
    args = parser.parse_args()

    receipt = run(args.budget)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))
    print()
    print("This validates scheduler bookkeeping and the late-bloomer control only.")
    print("Gate 1a must use frozen-transformer layer traces; Gate 1b must save real FLOPs.")


if __name__ == "__main__":
    main()
