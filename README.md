# BeliefLab

Exact rational finite-horizon planning for supplied two-hidden-state, two-observation POMDPs. Dependency-free Python 3.10+, strict JSON CLI, policy-tree witnesses, Bayesian observation branches and an exact all-belief alpha-vector envelope. No learned model or belief-grid approximation.

![Exact two-step value envelope and the value of a sensing action](assets/belief-audit.svg)

## Pay for information, then act on it

```sh
python -m belieflab demo
python -m unittest discover -s tests -v
PYTHONPATH=. python examples/render_audit.py --check
```

The directly specified model has two fixed hidden states and three actions. Exploit action 0 rewards +1 in state 0 and -1 in state 1; exploit action 1 reverses those rewards. Their observations are uninformative. Probe costs 1/10 now and reports the state correctly with probability 9/10. Transitions preserve the state. The horizon is two actions, discount one, zero terminal value and initial belief `(1/2,1/2)`.

| First action | Best total reward over the same two-action horizon |
|---|---:|
| Exploit state 0 | 0 |
| Exploit state 1 | 0 |
| Probe | 7/10 |

A myopic one-action decision picks an exploit action: immediate reward zero beats probe's -1/10. Two-action planning instead probes, then chooses the exploit action based on the observation. Each observation has probability 1/2, producing posterior `(9/10,1/10)` or `(1/10,9/10)`. The expected second reward is 4/5, so total value is `-1/10 + 4/5 = 7/10`. The root-action table above uses the same two-action horizon for every action, rather than comparing totals across horizons. This is a prescribed value-of-information example, not a sensor benchmark or a claim that probing always helps.

For any starting belief `b = P(state 0)`, the exact two-step value is `max(2*(1-2*b), 7/10, 2*(2*b-1))`. Probe is optimal on `[13/40,27/40]`, including a tie at either boundary. Outside that interval, exploit the more likely state. The SVG connects exact rational envelope breakpoints, not a sampled or fitted curve. Full exact model, layers, intervals, selected trees, root values and posterior branches are embedded in its metadata.

## Alpha vectors are policy-tree witnesses

A finite observation-contingent policy is a tree: take one action, then choose a child tree for each possible observation. Its alpha vector stores its exact total return from each known starting state. At a belief, the tree's value is the belief-weighted dot product with that vector. The optimal finite-horizon value is the maximum over those linear functions.

Each backup enumerates all root actions and all pairs of previous-layer continuation representatives. With transition `T[a][s][next]`, post-transition observation `O[a][next][obs]`, and child alpha selected per observation, it computes:

```text
alpha_new[s] = reward[a][s]
             + discount * sum_next T[a][s][next]
                        * sum_obs O[a][next][obs] * alpha_child[obs][next]
```

At horizon zero the only vector is `[0,0]`, with no action or tree. Every subsequent layer returns retained alpha vectors, root actions, child IDs referring to the previous layer, exact belief intervals, interval endpoint values and complete candidate/pruning counts. The selected policy is also expanded into a plain observation-contingent tree. This is not a receding-horizon controller or an infinite-horizon convergence certificate.

With two hidden states, each alpha vector is a line in the scalar belief b. BeliefLab sorts lines by slope and constructs their exact upper hull with rational intersections. It clips pieces to `[0,1]`. No numerical tolerance, belief grid or floating-point LP decides dominance.

**Values are preserved, not every tied policy tree.** Duplicate vectors retain the first generated tree; parallel dominated vectors, zero-width-only pieces and pieces outside the belief domain may be removed. At boundaries either adjacent retained tree attains the optimal value. `retained_maximizers` names only retained representatives and is not a complete policy-tree tie set. `root_action_values` and `optimal_root_actions` are computed separately from all final root candidates before pruning, so they preserve every optimal root action at the supplied belief, including duplicate-action ties. The selected tree uses the first retained maximizing representative; it need not be the lexicographically first tree among all original policies.

## Bayesian branch contract

`belief_update` first predicts the next state using the selected action's transition matrix, then multiplies by its new-state-conditioned observation likelihood. It returns the prediction, unnormalized masses, exact observation probability and normalized posterior. If the observation probability is zero, `posterior` is null. No belief is invented for an impossible event. Observation indices 0 and 1 label the corresponding two child branches in each policy tree.

```python
from belieflab import solve, belief_update
result = solve(model, [[1,2],[1,2]], horizon=2)
branch = belief_update(model, [[1,2],[1,2]], action=2, observation=0)
```

```sh
python -m belieflab solve examples/sensing.json --belief '[[1,2],[1,2]]' --horizon 2
python -m belieflab update examples/sensing.json --belief '[[1,2],[1,2]]' --action 2 --observation 0
python -m pip install .
belieflab demo
```

## Strict input and resource preflight

Model JSON is `{"version":1,"discount":rational,"actions":[...]}`. Every action has exactly `reward` (length two), `transition` (2x2) and `observation` (2x2). Every probability row and the length-two input belief sum exactly to one. Rationals are integers or `[numerator,positive denominator]`, never floats, booleans or decimal strings. Unknown/duplicate object keys, invalid shapes, negative probabilities and out-of-range indices fail explicitly. Rewards have absolute value <=100; raw rational components have magnitude <=1,000,000 with positive denominators. The LCM of normalized model and input-belief denominators must be <=1,000,000.

Scope is two hidden states, two observations, 1..3 actions, discount in `[0,1]`, horizon 0..4 and zero terminal value. JSON files are capped at 256 KiB. Each layer preflights `actions * previous_retained_vectors**2` before allocating candidate vectors. `candidate_limit` is an integer in 1..4,096; the aggregate generated-candidate cap is 8,192, excluding the initial zero vector. If a layer would exceed either cap, the call raises an error identifying that layer, and returns no partial solution. Pruning can reduce the next layer's branching, but a supplied model is not promised to fit every requested horizon/limit.

Counts describe candidate vectors and pruning categories. They are not the full unpruned policy-tree count, arithmetic operations, FLOPs, serialized bytes, wall time or RAM. Fraction integer sizes and full layer histories have costs not bounded by the candidate counter. No network, model download, paid API, GPU or dataset is needed. The four-module wheel's demo matches source output outside the source tree.

## Independent checks

29 tests include 81 prescribed stochastic/sensing/cost/discount models with all 27 depth-two policy trees evaluated by an independent trajectory recursion: 2,187 policy-tree references. For each model, exact oracle alpha intersections partition the whole belief interval; endpoints and every cell midpoint verify envelope equality, rather than relying on a fixed belief grid. Every layer's retained alpha backup, child ID, interval coverage, endpoint value and count reconciliation is reconstructed. All root-action values and root ties are compared against the full oracle before pruning.

A second check enumerates all 2,187 depth-three policy trees for the sensing fixture, compares their values at the solver's boundaries plus 21 stated belief points, and checks an independent Bayesian belief recursion. This depth-three check is not claimed as an exhaustive all-belief intersection audit. Tests also cover the analytic sensing thresholds/posteriors/tree, horizon zero/four, impossible observations, transition-before-observation semantics, mixture identities, duplicate roots, exact tiny intersections, zero discount, constant negative rewards, independent per-layer/aggregate budget boundaries, preflight before backup, strict schemas/rational/simplex/LCM/CLI errors and input/output/RNG isolation. Four SVG tests check every exact envelope vertex and breakpoint, complete metadata, regeneration and rejection of an undisplayed changed layer.

Finite-horizon policy-tree alpha-vector backups follow [MIT 6.0411/16.420, Fall 2023, L19: Solving POMDPs Offline](https://airr.mit.edu/_static/fall23/lectures/lec19/L19-POMDP-Offline.pdf). The two-state exact hull, schema, resource contract, code, fixture, independent oracles and figure are original. No external code or dataset is copied. [DESIGN.md](DESIGN.md) states the value-preservation and tie semantics. MIT licensed.

Only the supplied finite model is solved. No transition/reward/observation learning, statistical calibration, more-than-two-state solver, arbitrary observation count, infinite-horizon guarantee, complete enumeration of tied policies or real-world control-safety claim is made. Exact arithmetic cannot validate whether the model describes the world.
