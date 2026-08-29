"""Deterministic cross-validation and review for operator proposals.

The checks in this module never execute proposal content.  They compare a
validated DSL object with its declared evidence bundle and parent specs.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Literal

from operator_evolution_core.proposal import (
    DomainKit,
    ensure_domain_compatibility,
    flattened_capabilities,
)

from ..domain.uav_kit import UAVDomainKit
from .designer_base import OperatorProposal
from .design_models import DiagnosisClaim, OperatorReview


class ProposalValidationError(ValueError):
    """Raised when a proposal violates a non-negotiable static invariant."""

    def __init__(self, errors: Iterable[str]) -> None:
        self.errors = tuple(dict.fromkeys(str(error) for error in errors))
        super().__init__("; ".join(self.errors))


def _value(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _as_json(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


def _bundle_evidence_ids(bundle: Any) -> frozenset[str]:
    direct = getattr(bundle, "evidence_ids", None)
    if callable(direct):
        direct = direct()
    if direct is not None:
        return frozenset(str(item) for item in direct)

    evidence_fields = (
        "effective_contexts",
        "failure_contexts",
        "failure_modes",
        "synergy_evidence",
        "counterfactual_evidence",
        "representative_success_cases",
        "representative_failure_cases",
    )
    identifiers: set[str] = set()
    for field in evidence_fields:
        for item in _value(bundle, field, []) or []:
            evidence_id = _value(item, "evidence_id")
            if evidence_id:
                identifiers.add(str(evidence_id))
    return frozenset(identifiers)


def _bundle_parent_specs(
    bundle: Any,
    domain_kit: DomainKit[Any, Any, Any],
) -> dict[str, Any]:
    raw = _value(bundle, "parent_specs", {}) or {}
    if isinstance(raw, Mapping):
        items = raw.items()
    else:
        items = ((_value(spec, "name", ""), spec) for spec in raw)
    result: dict[str, Any] = {}
    for name, value in items:
        try:
            spec = domain_kit.parse_ir(value)
        except Exception:
            continue
        result[str(name or domain_kit.ir_name(spec))] = spec
    return result


def _bundle_allowed_primitives(
    bundle: Any,
    domain_kit: DomainKit[Any, Any, Any],
) -> frozenset[str]:
    expected = flattened_capabilities(domain_kit)
    raw = _value(bundle, "allowed_primitives", None)
    if isinstance(raw, Mapping):
        supplied = frozenset(
            str(name) for names in raw.values() for name in names
        )
        if supplied != expected:
            raise ProposalValidationError(
                ["evidence capability catalog does not match DomainKit"]
            )
    elif raw:
        supplied = frozenset(str(name) for name in raw)
        if supplied != expected:
            raise ProposalValidationError(
                ["evidence capability catalog does not match DomainKit"]
            )
    return expected


def used_primitive_names(
    spec: Any,
    domain_kit: DomainKit[Any, Any, Any] | None = None,
) -> tuple[str, ...]:
    """Return all primitive ``kind`` values referenced by a spec."""

    kit = domain_kit or UAVDomainKit()
    return kit.capability_usage(kit.parse_ir(spec))


def topology_payload(
    spec: Any,
    domain_kit: DomainKit[Any, Any, Any] | None = None,
) -> dict[str, Any]:
    """Return structural shape while deliberately excluding tuned values."""

    kit = domain_kit or UAVDomainKit()
    parsed = kit.parse_ir(spec)
    payload = getattr(kit, "topology_payload", None)
    if not callable(payload):
        raise TypeError("DomainKit does not expose a topology payload projection")
    return payload(parsed)


def topology_fingerprint(
    spec: Any,
    domain_kit: DomainKit[Any, Any, Any] | None = None,
) -> str:
    kit = domain_kit or UAVDomainKit()
    return kit.topology_fingerprint(kit.parse_ir(spec))


def _diagnosis_claims(proposal: OperatorProposal) -> tuple[DiagnosisClaim, ...]:
    diagnosis = proposal.diagnosis
    if diagnosis is None:
        return ()
    return tuple(
        [
            *diagnosis.effective_mechanisms,
            *diagnosis.failure_modes,
            *diagnosis.useful_synergies,
        ]
    )


class ProposalValidator:
    """Apply evidence, DSL, lineage and deterministic rule-review checks."""

    EVIDENCE_THRESHOLD = 0.60
    SAFETY_THRESHOLD = 0.80
    TESTABILITY_THRESHOLD = 0.60

    def __init__(
        self,
        domain_kit: DomainKit[Any, Any, Any] | None = None,
    ) -> None:
        self.domain_kit = domain_kit or UAVDomainKit()

    def validate(
        self,
        proposal: OperatorProposal,
        bundle: Any,
        *,
        review_mode: Literal["none", "rule_based"] = "rule_based",
    ) -> OperatorReview:
        """Validate hard invariants, then return a deterministic review.

        ``review_mode='none'`` skips score-based rejection only.  Evidence,
        parent, primitive and non-rename checks always run.
        """

        errors: list[str] = []
        ensure_domain_compatibility(
            self.domain_kit,
            bundle,
            allow_legacy_unversioned=True,
        )
        ensure_domain_compatibility(
            self.domain_kit,
            proposal,
            allow_legacy_unversioned=True,
        )
        spec = self.domain_kit.parse_ir(proposal.spec)
        parent_specs = _bundle_parent_specs(bundle, self.domain_kit)
        evidence_ids = _bundle_evidence_ids(bundle)
        allowed_primitives = _bundle_allowed_primitives(bundle, self.domain_kit)
        parent_ids = self.domain_kit.ir_parent_ids(spec)

        if proposal.diagnosis is None:
            errors.append("structured diagnosis is required")
        if proposal.hypothesis is None:
            errors.append("design hypothesis is required")
        if not proposal.used_evidence_ids:
            errors.append("used_evidence_ids must not be empty")

        referenced_ids: set[str] = set(proposal.used_evidence_ids)
        for claim in _diagnosis_claims(proposal):
            referenced_ids.update(claim.evidence_ids)
        if proposal.hypothesis is not None:
            referenced_ids.update(proposal.hypothesis.evidence_ids)
        missing_ids = sorted(referenced_ids - evidence_ids)
        if missing_ids:
            errors.append(f"unknown evidence IDs: {missing_ids}")

        if not parent_ids:
            errors.append("operator_spec must declare at least one parent operator")
        unknown_parents = sorted(set(parent_ids) - set(parent_specs))
        if unknown_parents:
            errors.append(f"unknown parent operators: {unknown_parents}")
        if proposal.diagnosis is not None and (
            proposal.diagnosis.parent_operator not in parent_ids
        ):
            errors.append("diagnosis parent_operator must be one of operator_spec.parent_operators")

        unknown_primitives = sorted(
            set(self.domain_kit.capability_usage(spec)) - allowed_primitives
        )
        if unknown_primitives:
            errors.append(f"non-whitelisted primitives: {unknown_primitives}")

        if proposal.hypothesis is not None and proposal.diagnosis is not None:
            failure_claims = {
                claim.claim: set(claim.evidence_ids) for claim in proposal.diagnosis.failure_modes
            }
            target = proposal.hypothesis.target_failure_mode
            if target not in failure_claims:
                errors.append("hypothesis target_failure_mode must exactly match a diagnosis failure claim")
            elif not (set(proposal.hypothesis.evidence_ids) & failure_claims[target]):
                errors.append("target failure hypothesis must cite evidence from its diagnosis claim")

        declared_parents = [
            parent_specs[name] for name in parent_ids if name in parent_specs
        ]
        candidate_behavior = self.domain_kit.behavior_fingerprint(spec)
        if any(
            candidate_behavior == self.domain_kit.behavior_fingerprint(parent)
            for parent in declared_parents
        ):
            errors.append("rename-only or metadata-only proposals are not accepted")

        if errors:
            raise ProposalValidationError(errors)

        fingerprint = self.domain_kit.topology_fingerprint(spec)
        parameter_variant = any(
            fingerprint == self.domain_kit.topology_fingerprint(parent)
            for parent in declared_parents
        )
        lineage_relation = "parameter_variant" if parameter_variant else "structural_variant"
        novelty_score = 0.35 if parameter_variant else 0.85

        reasoning_ids = {
            evidence_id
            for claim in _diagnosis_claims(proposal)
            for evidence_id in claim.evidence_ids
        }
        if proposal.hypothesis is not None:
            reasoning_ids.update(proposal.hypothesis.evidence_ids)
        if reasoning_ids:
            evidence_alignment = len(reasoning_ids & set(proposal.used_evidence_ids)) / len(
                reasoning_ids
            )
        else:
            evidence_alignment = 0.0

        safety_score = self.domain_kit.static_safety_score(spec)
        testability_score = 1.0 if (
            proposal.expected_advantages
            and proposal.hypothesis is not None
            and proposal.hypothesis.expected_effective_context.strip()
        ) else 0.6

        concerns: list[str] = []
        revisions: list[str] = []
        if evidence_alignment < self.EVIDENCE_THRESHOLD:
            concerns.append("proposal does not carry enough diagnosis evidence into used_evidence_ids")
            revisions.append("cite at least 60% of diagnosis and hypothesis evidence IDs")
        if safety_score < self.SAFETY_THRESHOLD:
            concerns.append("proposal lacks a safe bounded fallback")
            revisions.append("add rollback_on_failure fallback")
        if testability_score < self.TESTABILITY_THRESHOLD:
            concerns.append("expected effects are not experimentally testable")
            revisions.append("state measurable advantages and an effective context")

        passes_scores = not concerns
        decision: Literal["approve", "revise", "reject"]
        decision = "approve" if review_mode == "none" or passes_scores else "revise"
        return OperatorReview(
            decision=decision,
            evidence_alignment_score=evidence_alignment,
            novelty_score=novelty_score,
            safety_score=safety_score,
            testability_score=testability_score,
            concerns=concerns,
            required_revisions=revisions,
            lineage_relation=lineage_relation,
            topology_fingerprint=fingerprint,
        )

    def validate_and_review(
        self,
        proposal: OperatorProposal,
        bundle: Any,
        *,
        review_mode: Literal["none", "rule_based"] = "rule_based",
    ) -> OperatorReview:
        """Explicit alias useful to deterministic orchestrators."""

        return self.validate(proposal, bundle, review_mode=review_mode)


__all__ = [
    "ProposalValidationError",
    "ProposalValidator",
    "topology_fingerprint",
    "topology_payload",
    "used_primitive_names",
]
