# Evolutionary AFL-UAV V2 development report

Status: **development only; not frozen; not research-claim eligible**

V2 was developed and evaluated only on the fixed `uav2d-v1` Validation split.
It did not read or rerun Hidden Test-v2, and it made zero model/API calls.
Frozen V1 remains unchanged.

## Objective

The primary objective was reliability: retain a previously validated feasible
best-so-far path and leave enough time for result finalization, trusted
validation, and serialization. Multi-source population initialization and a
rooms/maze portal graph were implemented as secondary hypotheses.

## Implemented mechanisms

- Cooperative evolution soft deadline at 0.8536 of the one-second budget.
- Operation deadline at 0.88, with a 0.06 operation-start/finalization guard.
- Fast cost-ordered elite finalization near the deadline.
- Successful return of a trusted best-so-far path after a local cooperative
  deadline, without masking a true global timeout.
- Optional multi-source initialization from frozen AFL, A*, Theta*, PRM, and
  redistributed/topological path families.
- Optional rooms/maze wall-gap portal extraction, visibility graph routing,
  portal-aware insertion/movement, and topology-diverse archive behavior.
- Existing Train/Validation-only runner guard retained. Hidden Test-v2 remains
  forbidden to V2 development.

## Validation comparison

All rows use 60 Validation maps, five shared seeds, the same one-second and
2,000-evaluation limits, shared collision checker, objective, and trusted final
validator.

| Variant | Source | Trusted feasible | Timeouts | Median cost | Rooms/maze median | Min finalization reserve |
|---|---:|---:|---:|---:|---:|---:|
| V1 comparison | frozen | 300/300 | 0 | 117.363310 | 123.696151 | n/a |
| reliability-only, initial | `80e91189` | 300/300 | 0 | 117.438071 | 123.747485 | 0.079203 s |
| multi-source, initial | `80e91189` | 299/300 | 1 | 117.537604 | 123.647662 | 0.000000 s |
| full topology, initial | `80e91189` | 300/300 | 0 | 117.359212 | 124.300548 | 0.077366 s |
| multi-source, guarded | `081e9b2a` | 300/300 | 0 | 117.540097 | 123.804563 | 0.144031 s |
| reliability-only, guarded | `081e9b2a` | 300/300 | 0 | 117.460978 | 123.903570 | 0.134303 s |

The initial multi-source variant reproduced the reliability defect on one
corridor run: it returned a feasible path at 1.143 seconds and was correctly
classified as a timeout. On the same map and seed, guarded V2 returned
successfully inside the budget. The final guarded experiments had no timeouts.

## Decision

The selected V2 candidate configuration is **guarded reliability-only**. Its
Validation median cost is about 0.083% above V1 while its minimum planner-side
finalization reserve is 134 ms. This is the best supported tradeoff for V2's
primary reliability objective.

Multi-source initialization remains implemented as an ablation, but is not
selected: under the guarded protocol it cost more than reliability-only and did
not improve rooms/maze. The portal topology strategy is also not selected. It
successfully detected up to seven portals and generated topology paths, but its
rooms/maze median was worse and its per-map rooms comparison was 3 wins and 7
losses against V1 in the initial full run.

This Validation evidence does **not** prove that V2 has eliminated the two V1
timeouts observed on Hidden Test-v2. Hidden Test-v2 cannot be reused. A claim
about unseen-data reliability requires freezing V2 first and evaluating it on a
new preregistered Hidden Test-v3.

## Verification

- V2 and evolutionary experiment tests: 13 passed.
- Full suite: all failures were limited to two obsolete pre-opening assertions
  that require Hidden Test-v2 `SEALED.json` to exist. Hidden Test-v2 was already
  legitimately opened and completed in the prior phase, so those assertions no
  longer describe the repository state. Frozen final-evaluation code and old
  tests were not modified.
- Selected run: exactly 300 unique records, 300 success, 300 trusted feasible,
  zero timeout.

## Integrity hashes

- Frozen V1 source: `79f0a085a0f26b246d2f1e0d0bc1ac7e8a6288b34dd0fbc8f29bc9387e1a7d4f`
- V2 development source: `081e9b2a876332696d61df778fa25b6f1916a8684b2dc58a56863539bcea76d4`
- Candidate config: `3e395c06e7af785f07b4ba1830202e939d3b2ce4d62227cfd5cc7891207779fd`
- Selected benchmark CSV: `b089509fb6639bb1ec833d504ee6e9d13981c2910531423b4e83f607acfd7b29`
- Selected comparison JSON: `a751fd8d47d25e82268bb67b4ed14ebb34627d153aba90e5b37f58107fbb67df`

