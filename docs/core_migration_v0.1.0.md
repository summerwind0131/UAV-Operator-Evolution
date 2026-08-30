# Core v0.1.0 packaging and migration

Core, UAV and JSSP are maintained in one repository and versioned by one source
commit. Domain code imports shared APIs from `operator_evolution_core`; it must
not copy or fork those modules.

The standalone core wheel is a release artifact for consumers that need only
the domain-independent API. Install the wheel downloaded from the
`trajectory-core-v0.1.0` GitHub prerelease and verify it against the accompanying
SHA-256 receipt. The monorepo itself remains installable through its root
`pyproject.toml` and includes all three packages.

For a new domain package:

1. Implement the `DomainAdapter` components and a versioned `DomainKit`.
2. Keep domain models, features, IR, compiler and smoke fixtures outside the
   core package.
3. Import generic search, trajectory, diagnosis, memory, proposal and validation
   APIs from `operator_evolution_core`.
4. Run the adapter contract suite and domain-specific identity tests.
5. Fail closed on domain/version mismatch and compile only whitelisted typed IR.

The physical UAV SQLite `map_id` column remains a v1 compatibility alias for
the core's `instance_id`. Existing UAV proposal JSON and evidence hashes retain
their compatibility projection. Core v0.1.0 does not permit arbitrary generated
Python or define a universal operator DSL.
