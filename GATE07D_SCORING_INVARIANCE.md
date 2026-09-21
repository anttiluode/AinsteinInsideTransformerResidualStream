# Gate 0.7d — scoring invariance before temporal interpretation

Gate 0.7c did something scientifically more important than simply failing two pre-registered shape checks: its horizon-1 and horizon-3 values did not numerically reproduce the earlier Gate 0.7 and Gate 0.7b measurements even though those endpoints are supposed to use the same frozen route vectors and the same candidate strings.

That makes the immediate question **measurement stability**, not cognitive interpretation.

## Null expectation

Rows in a causal language-model batch do not interact with one another.

Therefore the score of

    " 17 * 20 = 340"

under the same prompt and same residual intervention should not change meaningfully just because unrelated candidate rows were added later in the batch.

Low-precision kernels can introduce small numerical drift. They must not reverse the qualitative route claim.

## Test

Load Qwen once and reconstruct D_A, D_B and D_C once.

On the source problem 17x24, score identical strings under:

1. the legacy Gate-0.7 three-candidate first-action batch;
2. the same three first actions in reversed row order;
3. the full Gate-0.7c candidate batch, then select A@1/B@1/C@1;
4. the legacy Gate-0.7b three-trajectory batch;
5. the same full Gate-0.7c batch, then select A@3/B@3/C@3.

For every condition retain raw average log-probabilities, intervention lifts, target margins and row winners.

## Kill boundary

The primary comparison is semantic, not an arbitrary decimal tolerance.

For both first-action and full-trajectory comparisons:

- the sign of each route's target margin must remain unchanged;
- the winning target under each route intervention must remain unchanged.

If either property changes solely because unrelated candidates were added to the batch, suspend the Gate-0.7c temporal-footprint interpretation.

The receipt also reports maximum absolute raw-score drift and lift drift so the size of the numerical instability is visible.

## Why this gate comes first

The current route effects are on the order of thousandths to hundredths of a nat per token. That is scientifically interesting only if the measurement is robust to harmless implementation choices.

A failure here does **not** mean route steering is false. It means the present scoring protocol cannot support claims as fine-grained as "A is early, B is recurrent, C is persistent."

The next move after a failure would be to stabilize the scoring path: fixed batch shape, paired baseline/intervention execution, stronger effect calibration, and possibly a higher-precision readout.

## Run

    python3.13 gate07d_scoring_invariance.py

The model is loaded once with the same conservative 6 GiB GPU + 6 GiB CPU + disk-offload profile used by Gates 0.7b and 0.7c.
