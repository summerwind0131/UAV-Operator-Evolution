"""Domain-neutral dependency and split-capability contracts for evolution."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, Protocol, TypeVar

from ..contracts import DomainAdapter
from ..proposal import DomainKit, proposal_hash

InstanceT = TypeVar("InstanceT")
SolutionT = TypeVar("SolutionT")
OperatorT = TypeVar("OperatorT")
IRT = TypeVar("IRT")


@dataclass(frozen=True, slots=True)
class PopulationSeed(Generic[OperatorT, IRT]):
    operators: tuple[OperatorT, ...]
    ir_by_id: Mapping[str, IRT]

    def __post_init__(self) -> None:
        if not self.operators:
            raise ValueError("initial operator population cannot be empty")
        object.__setattr__(self, "ir_by_id", dict(self.ir_by_id))


@dataclass(frozen=True, slots=True)
class PopulationFreezeReceipt:
    population_ids: tuple[str, ...]
    population_fingerprint: str
    receipt_id: str


class EvolutionSplitCapabilities(Generic[InstanceT]):
    """Keep held-out test instances inaccessible until population freeze."""

    def __init__(
        self,
        *,
        train: Sequence[InstanceT],
        validation: Sequence[InstanceT],
        test: Sequence[InstanceT],
    ) -> None:
        if not train or not validation or not test:
            raise ValueError("train, validation, and test capabilities must be non-empty")
        self._train = tuple(train)
        self._validation = tuple(validation)
        self._test = tuple(test)
        self._active_receipt: PopulationFreezeReceipt | None = None

    @classmethod
    def from_mapping(
        cls,
        datasets: Mapping[str, Sequence[InstanceT]],
    ) -> "EvolutionSplitCapabilities[InstanceT]":
        required = {"train", "validation", "test"}
        if not required.issubset(datasets):
            raise ValueError(f"datasets must contain {sorted(required)}")
        return cls(
            train=datasets["train"],
            validation=datasets["validation"],
            test=datasets["test"],
        )

    def open_train(self) -> tuple[InstanceT, ...]:
        return self._train

    def open_validation(self) -> tuple[InstanceT, ...]:
        return self._validation

    def freeze_population(
        self,
        population_ids: Sequence[str],
        population_fingerprint: str,
    ) -> PopulationFreezeReceipt:
        identifiers = tuple(str(item) for item in population_ids)
        if not identifiers or len(identifiers) != len(set(identifiers)):
            raise ValueError("frozen population IDs must be unique and non-empty")
        if len(population_fingerprint) != 64:
            raise ValueError("population_fingerprint must be a SHA-256 hex digest")
        receipt_id = proposal_hash(
            {
                "schema": "population-freeze-v1",
                "population_ids": identifiers,
                "population_fingerprint": population_fingerprint,
            }
        )
        receipt = PopulationFreezeReceipt(
            population_ids=identifiers,
            population_fingerprint=population_fingerprint,
            receipt_id=receipt_id,
        )
        self._active_receipt = receipt
        return receipt

    def open_test(
        self,
        receipt: PopulationFreezeReceipt | None = None,
    ) -> tuple[InstanceT, ...]:
        if receipt is None or receipt is not self._active_receipt:
            raise PermissionError(
                "test capability requires this split set's active population freeze receipt"
            )
        return self._test


class EvolutionArtifactSink(Protocol):
    def emit(self, event: str, payload: Mapping[str, Any]) -> None: ...


@dataclass(slots=True)
class NullEvolutionArtifactSink:
    def emit(self, event: str, payload: Mapping[str, Any]) -> None:
        del event, payload


@dataclass(slots=True)
class EvolutionManagerDependencies(
    Generic[InstanceT, SolutionT, OperatorT, IRT]
):
    domain_adapter: DomainAdapter[InstanceT, SolutionT]
    domain_kit: DomainKit[IRT, OperatorT, Any]
    population_factory: Callable[[], PopulationSeed[OperatorT, IRT]]
    candidate_validator: Any
    designer: Any
    orchestrator_factory: Callable[..., Any]
    artifact_sink: EvolutionArtifactSink = field(
        default_factory=NullEvolutionArtifactSink
    )

    def __post_init__(self) -> None:
        if self.domain_adapter.domain_id != self.domain_kit.domain_id:
            raise ValueError(
                "DomainAdapter and DomainKit must declare the same domain_id"
            )


def population_fingerprint(
    population_ids: Sequence[str],
    ir_by_id: Mapping[str, Any],
    domain_kit: DomainKit[Any, Any, Any],
) -> str:
    payload = []
    for operator_id in population_ids:
        if operator_id not in ir_by_id:
            raise KeyError(f"population IR missing for operator: {operator_id}")
        ir = domain_kit.parse_ir(ir_by_id[operator_id])
        payload.append(
            {
                "operator_id": operator_id,
                "behavior_fingerprint": domain_kit.behavior_fingerprint(ir),
                "topology_fingerprint": domain_kit.topology_fingerprint(ir),
            }
        )
    return proposal_hash(
        {
            "schema": "population-fingerprint-v1",
            "domain_id": domain_kit.domain_id,
            "ir_version": domain_kit.ir_version,
            "operators": payload,
        }
    )


__all__ = [
    "EvolutionArtifactSink",
    "EvolutionManagerDependencies",
    "EvolutionSplitCapabilities",
    "NullEvolutionArtifactSink",
    "PopulationFreezeReceipt",
    "PopulationSeed",
    "population_fingerprint",
]
