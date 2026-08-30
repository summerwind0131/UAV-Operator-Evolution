# Core API overview

The standalone distribution exposes the Python package
`operator_evolution_core`. Its public surface is re-exported from that package's
top-level `__init__` module.

## Domain contracts

A domain supplies a `DomainAdapter` composed from an initializer, evaluator,
solution guard, solution codec, feature extractor and trace encoder. Stable
instance identity is represented by `InstanceRef`; evaluation returns an
`ObjectiveEvaluation` with a finite scalar cost plus named components and
violations.

## Search and trajectories

`GenericSearchKernel` runs a fixed `SearchBudget` using domain-neutral
`SearchOperator`, `OperatorScheduler` and `AcceptancePolicy` protocols. It emits
`SearchStep` records and a `SearchResult`. `TrajectoryRecorder` stores the
before, candidate and accepted states without knowing a domain's solution type.

## Diagnosis and memory

`FeatureCatalog`, `OperatorDiagnoser` and delayed rewards convert trajectories
into operator profiles and sequential synergies. `MechanismMemory` stores typed,
provenanced mechanism records while leaving target-domain realization to its
own compiler and validation gate.

`MechanismRecordV1` is the cross-domain boundary. It carries only abstract
ordinal context, direction-only effects, semantic mechanism tags and
train/validation provenance. `MechanismBankV1` and fixed top-4 retrieval exclude
domain IR, executable payloads, raw feature values and test evidence.

## Proposals and validation

`CandidateProposalEnvelope` is the versioned outer proposal protocol. A
domain-specific `DomainKit` parses and validates the typed payload, exposes a
capability catalog, compiles it, and runs bounded smoke checks. The generic
paired validator provides common-random-number schedules, ABBA timing,
bootstrap intervals, retention gates and population-slot replacement.

## Evolution lifecycle

`EvolutionManagerDependencies` and `EvolutionSplitCapabilities` keep train,
validation and test access explicit. A `PopulationFreezeReceipt` must exist
before a test capability is opened. The core does not import UAV or JSSP types.
