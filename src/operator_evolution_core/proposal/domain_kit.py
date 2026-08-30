"""Domain-owned IR capability boundary used by proposal infrastructure."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, JsonValue

IRT = TypeVar("IRT")
OperatorT = TypeVar("OperatorT")
FixtureT = TypeVar("FixtureT")


class DomainCompatibilityError(ValueError):
    """Raised when an artifact is routed to the wrong domain or IR version."""


class DomainSmokeReport(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    domain_id: str
    ir_version: str
    smoke_passed: bool
    seeds_tested: int = Field(ge=0)
    successful_calls: int = Field(ge=0)
    failures: list[str] = Field(default_factory=list)


@runtime_checkable
class DomainKit(Protocol, Generic[IRT, OperatorT, FixtureT]):
    """The only capability surface proposal infrastructure may call."""

    domain_id: str
    ir_version: str
    accepts_legacy_unversioned: bool

    def parse_ir(self, payload: Any) -> IRT: ...

    def serialize_ir(self, ir: IRT) -> JsonValue: ...

    def capability_catalog(self) -> Mapping[str, Sequence[str]]: ...

    def compile(self, ir: IRT) -> OperatorT: ...

    def smoke(self, ir: IRT, fixture: FixtureT) -> DomainSmokeReport: ...

    def capability_usage(self, ir: IRT) -> tuple[str, ...]: ...

    def topology_fingerprint(self, ir: IRT) -> str: ...

    def behavior_fingerprint(self, ir: IRT) -> str: ...

    def static_safety_score(self, ir: IRT) -> float: ...

    def ir_name(self, ir: IRT) -> str: ...

    def ir_parent_ids(self, ir: IRT) -> tuple[str, ...]: ...

    def builtin_ir(self, operator_id: str) -> IRT | None: ...


def _value(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def ensure_domain_compatibility(
    kit: DomainKit[Any, Any, Any],
    artifact: Any,
    *,
    allow_legacy_unversioned: bool = False,
) -> tuple[str, str]:
    """Fail closed unless an artifact is explicitly bound to this kit.

    UAV v1 artifacts predate domain/version fields. Their compatibility facade
    opts into the legacy branch; new envelopes must always carry both fields.
    """

    domain_id = _value(artifact, "domain_id")
    ir_version = _value(artifact, "ir_version")
    if domain_id is None and ir_version is None:
        if allow_legacy_unversioned and kit.accepts_legacy_unversioned:
            return kit.domain_id, kit.ir_version
        raise DomainCompatibilityError("artifact is missing domain_id and ir_version")
    if domain_id is None or ir_version is None:
        raise DomainCompatibilityError(
            "artifact must declare domain_id and ir_version together"
        )
    if str(domain_id) != kit.domain_id:
        raise DomainCompatibilityError(
            f"domain mismatch: expected {kit.domain_id}, received {domain_id}"
        )
    if str(ir_version) != kit.ir_version:
        raise DomainCompatibilityError(
            f"IR version mismatch: expected {kit.ir_version}, received {ir_version}"
        )
    return kit.domain_id, kit.ir_version


def flattened_capabilities(
    kit: DomainKit[Any, Any, Any],
) -> frozenset[str]:
    return frozenset(
        str(name)
        for names in kit.capability_catalog().values()
        for name in names
    )


__all__ = [
    "DomainCompatibilityError",
    "DomainKit",
    "DomainSmokeReport",
    "ensure_domain_compatibility",
    "flattened_capabilities",
]
