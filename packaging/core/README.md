# trajectory-operator-evolution

Domain-independent core for trajectory-informed search-operator evolution.
It contains the shared contracts, fixed-budget search kernel, three-state
trajectory recording, delayed credit and diagnosis, mechanism memory, typed
proposal envelope, `DomainKit`, paired validation and population-freeze
lifecycle used by both the UAV and JSSP packages in this repository.

The `0.1.0` interface was qualified by continuous two-dimensional UAV path
planning and discrete job-shop scheduling. The core imports neither domain.

The standalone wheel and sdist are published only as assets of the
`trajectory-core-v0.1.0` GitHub prerelease in `UAV-Operator-Evolution`; PyPI
publication is intentionally out of scope.
