# Hidden Test-v2 final-evaluation protocol and closure

The one-time final evaluation was explicitly authorized, executed, and audited on 2026-08-17. The base preregistration and disclosed addendum remain immutable. This document records the closed protocol state; it is not an instruction to reopen or rerun the test.

## Immutable protocol chain

- Base preregistration ID: `8d8dcbb568f25a5fd9da74afbe865c5576ce585e25fc37da36662ad1787c3c99`
- Addendum ID: `ab1e83123806d7924b5439e02fea1ff6f837af03684d5249eddf34676380b3b6`
- Preflight receipt ID: `f5c1a62a0443198206556ae30ef8097d8767e377f49187365b84df585a6797b4`
- Opening receipt ID: `e569d34d03272dd3debd23a0ea4b5331d1bb2a1631915c9dbaa54ed34611b76b`
- Execution receipt ID: `d5ddb5658846940f3b95b7b894dc304f5e03d0e536e4d0d8d9235661c48e4a46`
- Audit receipt ID: `6789dc4901faab1cccdc8af38668fdb09a2357c403f50eb1492810631e2ab838`
- Final status: `completed_audited`

The authorization phrase was `AUTHORIZE UAV2D-HIDDEN-TEST-V2 FINAL`. The opening step validated every frozen hash and map, wrote the opening receipt, and archived the lock as `SEALED.preopening.json`. No opening or planner command should be invoked again.

## Preflight boundary

The earlier surrogate preflight used one Validation map from each of six classes and never read Hidden Test-v2 map JSON. It exercised all 14 arms, checkpoint recovery, matrix integrity, merge, and frozen report generation with a short budget. Its path-quality numbers were never treated as research results.

## Final execution and audit

The dedicated runner used the exact frozen split, arms, budgets, seeds, executor and statistics implementation. All 14 arms completed. The merged matrix matched the preregistered 6,960-row seed schedule exactly, with 6,960 unique records and zero API calls.

The frozen auditor treated timeouts as failures, ranked feasibility first and trusted cost second, and produced status `passed`. Raw run and path matrices, normalized rows, summary, report, execution receipt and audit receipt are content-addressed. The human-readable outcome is maintained in [`hidden_test_v2_final_report.md`](hidden_test_v2_final_report.md).

If a future study needs another terminal evaluation, it must create a new dataset ID, preregistration, seal and population. It must not reuse Hidden Test-v2 as an adaptive validation set.
