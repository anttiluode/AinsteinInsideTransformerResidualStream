# Gate 0 contract — latent collision versus more tokens

## Question

Can two separately insufficient **latent branch residues** from a frozen transformer be composed into a useful intervention that generalizes to unseen branch pairings, and does that intervention provide an advantage over matched ordinary test-time reasoning?

A geometric novelty alone is insufficient.

## Backbone

Primary target:

- Qwen/Qwen3-8B
- frozen base-model weights
- initial capture layer: 18 (sweep later; do not tune on test)
- composer is the only trainable component in the first gate

The backbone can be changed if hardware forces it, but all compared methods must use the same backbone and precision.

## Task family

Start with procedurally generated **operator-composition tasks**.

Each example contains:

- a common query state S;
- branch role A exposing/inducing operation f_a;
- branch role B exposing/inducing operation g_b;
- a target requiring the ordered composition g_b(f_a(x)).

Train on individual operations and a subset of A/B pairings.

**Hold out entire A/B pair identities.** Test examples may contain familiar individual A and B operations, but their pairing must be unseen.

This asks whether the cross-residue law transfers rather than memorizing pair IDs.

## Latent residue

At capture layer l, measure the final-token state for the common anchor and each branch:

R_A = h_l(P_A) - h_l(P_0)

R_B = h_l(P_B) - h_l(P_0)

Later versions should replace this single difference with a compressed short branch trajectory:

R_A = C(Delta h^A_(l:l+m)).

Every residue carries a provenance stamp: role, branch address, operation identity during training, and actual/simulated source when applicable.

## Common carrier versus branch-specific residue

The first real Qwen/J-lens receipt motivates an additional decomposition.

For a set of branches forked from the same parent, define

    C = mean_i R_i
    D_i = R_i - C

where `C` is the common branch carrier and `D_i` is the branch-specific departure.

At layer 30 in the first receipt, raw branch pairwise cosines are substantially lower than Jacobian-transported cosines: downstream transport makes the different branches more similar even though their extreme readable token deltas remain distinct. A plausible nuisance explanation is that a strong shared task/answer-preparation direction dominates what the frozen model broadcasts.

Therefore Gate 0 must compare at least:

- composer on raw `R_A, R_B`;
- composer on centered `D_A, D_B`;
- common carrier `C` alone;
- branch-specific `D_A` and `D_B` alone;
- shuffled centered departures with the same carrier.

A raw bilinear win that disappears after carrier controls is not evidence that branch-specific computational residues composed.

The intended writeback, if centered composition works, is conceptually:

    common trajectory C
          +
    H(D_A, D_B)

rather than asking `H` to rediscover the common task carrier from every branch pair.

## Composer

First nonlinear family:

z = U_A R_A elementwise-multiplied-by U_B R_B

Delta r = W z.

This is an ordered low-rank bilinear interaction.

Inject Delta r into the common stream at a frozen injection layer and score the target answer.

Do not update the transformer's permanent weights.

## Required attackers

All are evaluated on the same held-out pairing split.

1. no intervention;
2. A residue only;
3. B residue only;
4. best learned scalar/convex combination of A and B;
5. unrestricted linear map of [R_A;R_B];
6. provenance erased;
7. provenance A/B swapped;
8. bilinear composer with shuffled pair bindings;
9. serialized chain-of-thought with a matched forward-pass/FLOP budget;
10. textual two-branch search with matched forward-pass/FLOP budget;
11. if practical, a stronger nonlinear post-concatenation MLP with parameter budget matched to the bilinear composer.

The nonlinear MLP is important: AInstein already established that transformer-like nonlinear post-mix machinery is allowed to succeed. A win over merely linear attackers is not enough.

## Measurements

Report at minimum:

- exact answer / task accuracy;
- target-token log probability;
- held-out pair generalization;
- relative distance of Delta r outside span(R_A, R_B);
- compute cost: forward token-equivalents and wall time;
- composer parameter count;
- provenance-shuffle damage;
- branch-only and convex-blend gaps.

For the temporary-operator version, also report the operator update's numerical rank and norm.

## Predeclared interpretation

### Strong pass

The latent collision improves held-out pair performance over branch-only, linear/convex, provenance attacks **and** matched textual rumination at comparable compute, with the advantage surviving a strong nonlinear matched-parameter attacker.

### Weak pass

Latent collision works and generalizes beyond branch-only/linear composition, but matched extra-token reasoning performs similarly or better.

Interpretation: useful representation/composition mechanism, **not** a superior reasoning architecture.

### Fail

Performance is explained by pair lookup, linear composition, extra compute, or provenance-insensitive mixing.

### Kill

If ordinary serialized reasoning repeatedly dominates at equal or lower compute and the latent composer contributes no complementary capability, stop selling the peripheral-field architecture. Keep only the instrumentation.

## Why this gate matters

The project is not trying to prove that transformers "cannot invent" without this mechanism. They can already create nonlinear cross-token interactions.

The narrower hypothesis is that **explicitly preserving alternative internal trajectories and compiling their interaction into the next temporary lens** may be a useful form of test-time computation distinct from selecting or verbalizing one trajectory at a time.
