from __future__ import annotations

import re


ROUTES = ("A", "B", "C")
HORIZONS = (1, 2, 3)
NUMERIC_ONLY = re.compile(r"^[0-9+*\\-=;\\s]+$")


def trajectory_steps(a: int, b: int) -> dict[str, list[str]]:
    """Return the Gate-0.7b numeric routes split into transitions."""
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
        "A": [
            f"{a} * {tens} = {a1}",
            f"{a} * {rem} = {a2}",
            f"{a1} + {a2} = {product}",
        ],
        "B": [
            f"{a} + {a} = {b1}",
            f"{b1} + {a} = {b2}",
            f"{b2} + {a} = {b3}",
        ],
        "C": [
            f"{a} * {near} = {c1}",
            f"{c1} - {a} = {product}",
            f"{product} = {a} * {b}",
        ],
    }


def assert_numeric_only(candidates: dict[str, str]) -> None:
    for label, text in candidates.items():
        if not NUMERIC_ONLY.fullmatch(text):
            raise RuntimeError(
                f"non-numeric character leaked into candidate {label}: {text!r}"
            )


def temporal_candidates(a: int, b: int) -> dict[str, str]:
    """All route prefixes plus matched recurrence decoys for route B."""
    steps = trajectory_steps(a, b)
    out: dict[str, str] = {}
    for route in ROUTES:
        for horizon in HORIZONS:
            out[f"{route}@{horizon}"] = " " + "; ".join(steps[route][:horizon])

    b1 = 2 * a

    # Same first edge as true B, then abandon +a and drift by +1.
    drift2 = b1 + 1
    drift3 = drift2 + 1
    out["B_decoy_same_first_drift"] = (
        f" {a} + {a} = {b1}; {b1} + 1 = {drift2}; {drift2} + 1 = {drift3}"
    )

    # Same + operator and recurrence depth, but the increment is always 1.
    unit1 = a + 1
    unit2 = unit1 + 1
    unit3 = unit2 + 1
    out["B_decoy_unit_recurrence"] = (
        f" {a} + 1 = {unit1}; {unit1} + 1 = {unit2}; {unit2} + 1 = {unit3}"
    )

    # Same recurrent form, but uses the other problem operand as the increment.
    other1 = a + b
    other2 = other1 + b
    other3 = other2 + b
    out["B_decoy_other_constant"] = (
        f" {a} + {b} = {other1}; {other1} + {b} = {other2}; "
        f"{other2} + {b} = {other3}"
    )

    assert_numeric_only(out)
    return out
