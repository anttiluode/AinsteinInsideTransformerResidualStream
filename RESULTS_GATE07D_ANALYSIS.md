# Gate 0.7d result — first-action steering is below the current semantic numerical floor

Date: 2026-09-21

Source receipt: `results/gate07d_scoring_invariance.json`

## Formal outcome

`semantic_invariance_pass = false`.

The failure is specific and informative.

### Reversing row order

The legacy three-candidate first-action batch was reversed.

Result:

- maximum raw-score drift: 0.0;
- maximum lift drift: 0.0;
- all target-margin signs stable;
- all route winners stable.

So this is not ordinary row-order nondeterminism.

### Adding unrelated longer rows

The exact same first-action strings were then scored inside the larger Gate-0.7c batch.

Maximum raw-score drift rose to approximately **0.01645**, and maximum intervention-lift drift to approximately **0.02843**.

Two of three route interpretations flipped:

| route | legacy margin | full-batch margin | legacy winner | full winner |
|---|---:|---:|---|---|
| A | +0.00931 | -0.01359 | A | B |
| B | -0.01657 | +0.01379 | A | B |
| C | +0.02110 | +0.02011 | C | C |

This is large relative to the effect being interpreted.

### Full numeric trajectory

The longer Gate-0.7b trajectory comparison was qualitatively more robust.

Its raw scores still moved (maximum drift about **0.01782**) and lifts moved (about **0.01169**), but all three target-margin signs and all three winners remained stable between legacy and full-batch contexts.

So the result is not "all route steering was noise."

It is narrower:

> the scale-1 first-action microscope is too close to the current BF16 / batch-shape numerical floor for fine causal interpretation.

## Likely mechanism

The scorer right-pads every candidate in a batch to that batch's longest sequence. Adding the longer Gate-0.7c candidates changes both batch dimensions and the low-precision matrix-kernel shapes, even though causal masking means the mathematical result for the earlier tokens should be unchanged.

The exact stability under row reversal, combined with the large change under batch enlargement, is consistent with that numerical explanation.

## Consequence

Gate 0.7c's temporal-shape story remains suspended.

Do not choose one of the conflicting scale-1 endpoint measurements as the "true" one.

The next experiment should first ask whether the causal signal can be made prospectively larger than this numerical disagreement. Gate 0.7e therefore tests the same frozen directions at scales 0.5, 1.0 and 2.0, with scale 2.0 fixed in advance as the primary test.

If scale 2.0 still changes semantic meaning across harmless batch contexts, switch scoring apparatus rather than increasing intervention strength further.
