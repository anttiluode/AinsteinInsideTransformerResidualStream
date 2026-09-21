from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Callable, Sequence


@dataclass
class BranchBudgetState:
    """Bookkeeping for one speculative branch under a shared compute budget."""

    branch_id: str
    depth: int = 0
    values: list[float] = field(default_factory=list)
    provenance: str | None = None

    @property
    def last_value(self) -> float:
        return self.values[-1] if self.values else 0.0

    @property
    def last_gain(self) -> float:
        if not self.values:
            return 0.0
        if len(self.values) == 1:
            return self.values[0]
        return self.values[-1] - self.values[-2]


def uniform_depth_allocation(
    branch_ids: Sequence[str],
    total_budget: int,
    *,
    max_depth: int | None = None,
) -> dict[str, int]:
    """Allocate integer depth as evenly as possible under an exact budget."""
    if total_budget < 0:
        raise ValueError("total_budget must be non-negative")
    if not branch_ids:
        if total_budget:
            raise ValueError("cannot spend budget without branches")
        return {}

    depths = {branch_id: 0 for branch_id in branch_ids}
    spent = 0
    while spent < total_budget:
        progressed = False
        for branch_id in branch_ids:
            if spent >= total_budget:
                break
            if max_depth is not None and depths[branch_id] >= max_depth:
                continue
            depths[branch_id] += 1
            spent += 1
            progressed = True
        if not progressed:
            break
    return depths


def _default_priority(state: BranchBudgetState) -> float:
    # Cheap local estimate: recent payoff per added unit of depth.
    return state.last_gain


def adaptive_depth_allocation(
    branch_ids: Sequence[str],
    total_budget: int,
    probe_fn: Callable[[str, int], float],
    *,
    min_probe_depth: int = 1,
    max_depth: int | None = None,
    exploration: float = 0.05,
    diversity_reserve: float = 0.15,
    priority_fn: Callable[[BranchBudgetState], float] = _default_priority,
    provenance: dict[str, str] | None = None,
) -> tuple[dict[str, int], list[dict[str, object]]]:
    """
    Spend a fixed branch-depth budget adaptively.

    probe_fn(branch_id, depth) returns the cumulative utility estimate available
    after computing that branch to depth. The foveator initially probes every
    branch coarsely, then purchases more depth where predicted marginal value is
    highest. A small diversity reserve forces periodic attention to the least
    explored routes so the scheduler does not immediately collapse onto one easy
    attractor.

    The scheduler is model-agnostic. Gate 1a plugs it into cached transformer
    layer traces; Gate 1b must replace layer-equivalent accounting with actual
    partial transformer execution before making a wall-clock claim.
    """
    if total_budget < 0:
        raise ValueError("total_budget must be non-negative")
    if min_probe_depth < 0:
        raise ValueError("min_probe_depth must be non-negative")
    if not 0.0 <= diversity_reserve <= 1.0:
        raise ValueError("diversity_reserve must lie in [0, 1]")
    if not branch_ids:
        if total_budget:
            raise ValueError("cannot spend budget without branches")
        return {}, []

    states = {
        branch_id: BranchBudgetState(
            branch_id=branch_id,
            provenance=None if provenance is None else provenance.get(branch_id),
        )
        for branch_id in branch_ids
    }
    trace: list[dict[str, object]] = []
    spent = 0

    # Coarse peripheral sweep: every route gets a minimal chance to leave a gist.
    for _ in range(min_probe_depth):
        for branch_id in branch_ids:
            if spent >= total_budget:
                break
            state = states[branch_id]
            if max_depth is not None and state.depth >= max_depth:
                continue
            state.depth += 1
            state.values.append(float(probe_fn(branch_id, state.depth)))
            spent += 1
            trace.append(
                {
                    "step": spent,
                    "branch": branch_id,
                    "depth": state.depth,
                    "reason": "coarse-probe",
                    "value": state.last_value,
                }
            )

    adaptive_steps = 0
    while spent < total_budget:
        eligible = [
            state
            for state in states.values()
            if max_depth is None or state.depth < max_depth
        ]
        if not eligible:
            break

        adaptive_steps += 1
        reserve_period = (
            None
            if diversity_reserve <= 0
            else max(1, round(1.0 / diversity_reserve))
        )

        if reserve_period is not None and adaptive_steps % reserve_period == 0:
            min_depth = min(state.depth for state in eligible)
            candidates = [state for state in eligible if state.depth == min_depth]
            chosen = max(candidates, key=lambda s: (priority_fn(s), s.branch_id))
            reason = "diversity-reserve"
        else:
            def score(state: BranchBudgetState) -> tuple[float, str]:
                ucb = priority_fn(state) + exploration / sqrt(state.depth + 1.0)
                return (ucb, state.branch_id)

            chosen = max(eligible, key=score)
            reason = "foveate"

        chosen.depth += 1
        chosen.values.append(float(probe_fn(chosen.branch_id, chosen.depth)))
        spent += 1
        trace.append(
            {
                "step": spent,
                "branch": chosen.branch_id,
                "depth": chosen.depth,
                "reason": reason,
                "value": chosen.last_value,
                "marginal_gain": chosen.last_gain,
            }
        )

    return {k: state.depth for k, state in states.items()}, trace


def allocation_entropy(depths: dict[str, int]) -> float:
    """Normalized allocation entropy in [0, 1] for at least two active branches."""
    from math import log

    total = sum(depths.values())
    active = [d for d in depths.values() if d > 0]
    if total <= 0 or len(active) <= 1:
        return 0.0
    probs = [d / total for d in active]
    h = -sum(p * log(p) for p in probs)
    return h / log(len(active))


def total_utility(
    depths: dict[str, int],
    payoff_fn: Callable[[str, int], float],
) -> float:
    """Sum terminal branch utilities; useful for scheduler controls and tests."""
    return sum(float(payoff_fn(branch_id, depth)) for branch_id, depth in depths.items())
