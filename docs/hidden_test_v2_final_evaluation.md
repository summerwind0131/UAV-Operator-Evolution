# Hidden Test-v2 final-evaluation protocol

The final evaluation is implemented but remains unopened and unrun.  The base
preregistration is immutable; the disclosed addendum records the dedicated
authorization entry, exact execution configuration, runner change, executor
hash, analyzer hash, and Validation-only preflight evidence.

## Current sealed state

- Base preregistration ID: `8d8dcbb568f25a5fd9da74afbe865c5576ce585e25fc37da36662ad1787c3c99`
- Addendum ID: `ab1e83123806d7924b5439e02fea1ff6f837af03684d5249eddf34676380b3b6`
- Preflight receipt ID: `f5c1a62a0443198206556ae30ef8097d8767e377f49187365b84df585a6797b4`
- `data/benchmarks/uav2d-hidden-test-v2/SEALED.json` is present.
- No opening receipt or final-results directory exists.

The generic benchmark command remains unable to run AFL-UAV or Evolutionary
AFL-UAV on a Test split.  The dedicated runner also refuses immediately while
`SEALED.json` exists.

## Preflight evidence

The surrogate used one Validation map from each of the six map classes, never
Hidden Test-v2 map JSON.  All 14 arms produced an exact 150-record matrix using
a 0.10 second/50-evaluation short budget and two stochastic repetitions.  The
first run stopped after five complete arms; the second run verified their file
hashes, skipped them, and completed the remaining nine arms.  The report and
receipt generator then exercised the frozen statistics code.

The short preflight verifies loading, budget termination, checkpoint recovery,
record integrity, merge, and report generation.  Its path-quality numbers are
not research results.

## Future authorized sequence

Only after an explicit user authorization should the following opening command
be invoked.  It validates all hashes, validates every sealed map file, writes an
opening receipt, and archives (rather than deletes) the lock marker.

```powershell
python scripts/open_hidden_test_v2.py `
  --preregistration-id 8d8dcbb568f25a5fd9da74afbe865c5576ce585e25fc37da36662ad1787c3c99 `
  --authorization-phrase "AUTHORIZE UAV2D-HIDDEN-TEST-V2 FINAL"
```

The final runner has no flags for changing the split, arms, budget, seeds, or
statistics:

```powershell
python scripts/run_authorized_hidden_test_v2.py `
  --preregistration-id 8d8dcbb568f25a5fd9da74afbe865c5576ce585e25fc37da36662ad1787c3c99
```

It requires the matching opening receipt and exact frozen hashes.  Complete
arms are resumable; partial or modified arm outputs are rejected.  The merged
matrix must equal the 6,960-row seed schedule exactly before the frozen auditor
runs.

If an execution or analysis protocol change becomes unavoidable before any
result is seen, create a new append-only preregistration addendum.  Do not edit
the existing base receipt, addendum, or preflight receipt.
