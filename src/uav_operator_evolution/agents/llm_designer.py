"""Optional structured-output LLM designer with a compatibility fallback.

The historical :meth:`propose` entry point intentionally retains its offline
heuristic fallback.  Phase-8 callers use :meth:`propose_from_evidence`, which
is evidence-strict and never switches experimental arms after an error.
"""

from __future__ import annotations

import os
from typing import Any, Literal, Protocol

from pydantic import ValidationError

from ..operators.specs import OperatorSpec
from ..reproducibility import stable_hash
from .designer_base import OperatorProposal
from .design_models import DiagnosisReport, OperatorReview
from .heuristic_designer import HeuristicDesigner
from .prompts import DESIGNER_V1, DIAGNOSER_V1, REVIEWER_V1, PromptTemplate
from .proposal_validation import ProposalValidationError, ProposalValidator
from .providers import LLMCallConfig, LLMProvider


class JSONClient(Protocol):
    def complete_json(self, *, model: str, prompt: str, schema: dict[str, Any]) -> str:
        ...


class LLMDesignError(ValueError):
    """Raised when a structured design result violates evidence contracts."""


class LLMDesignerUnavailableError(RuntimeError):
    """Raised when the explicit Phase-8 LLM arm has no configured provider."""


class LLMDesignerAdapter:
    """Accept only validated JSON; never execute text returned by a model."""

    def __init__(
        self,
        client: JSONClient | None = None,
        fallback: HeuristicDesigner | None = None,
        *,
        provider: LLMProvider | None = None,
        proposal_validator: ProposalValidator | None = None,
    ) -> None:
        self.client = client
        self.fallback = fallback or HeuristicDesigner()
        self.provider = provider
        self.proposal_validator = proposal_validator or ProposalValidator()
        self.model = os.getenv("UOE_LLM_MODEL", "")
        self.api_key_present = bool(os.getenv("OPENAI_API_KEY") or os.getenv("UOE_LLM_API_KEY"))
        self.last_error: str | None = None
        self.last_diagnosis: DiagnosisReport | None = None
        self.last_review: OperatorReview | None = None
        self.last_prompt_template: PromptTemplate | None = None

    @property
    def available(self) -> bool:
        return self.client is not None and bool(self.model) and self.api_key_present

    def propose(
        self,
        problem_description: str,
        parent_specs: list[OperatorSpec],
        parent_profiles: list[Any],
        memory_context: list[Any],
        success_cases: list[dict[str, Any]],
        failure_cases: list[dict[str, Any]],
    ) -> OperatorProposal:
        if not self.available:
            self.last_error = "LLM adapter unavailable; using deterministic heuristic designer"
            return self.fallback.propose(
                problem_description,
                parent_specs,
                parent_profiles,
                memory_context,
                success_cases,
                failure_cases,
            )

        prompt = self._build_prompt(
            problem_description,
            parent_specs,
            parent_profiles,
            memory_context,
            success_cases,
            failure_cases,
        )
        try:
            assert self.client is not None
            response = self.client.complete_json(
                model=self.model,
                prompt=prompt,
                schema=OperatorProposal.model_json_schema(),
            )
            return OperatorProposal.model_validate_json(response)
        except (ValidationError, ValueError, TypeError, RuntimeError) as exc:
            self.last_error = f"LLM proposal rejected: {exc}"
            return self.fallback.propose(
                problem_description,
                parent_specs,
                parent_profiles,
                memory_context,
                success_cases,
                failure_cases,
            )

    def propose_from_evidence(
        self,
        bundle: Any,
        mode: Literal["single_call", "staged"] = "single_call",
        call_config: LLMCallConfig | None = None,
    ) -> OperatorProposal:
        """Generate one evidence-grounded proposal through a structured provider.

        Unlike the legacy method, this is an explicit experimental arm: provider,
        schema, evidence, or prompt failures propagate to the orchestrator and are
        never converted into heuristic proposals.
        """

        if self.provider is None:
            raise LLMDesignerUnavailableError(
                "propose_from_evidence requires an explicit structured LLM provider"
            )
        if mode not in {"single_call", "staged"}:
            raise ValueError(f"unsupported LLM designer mode: {mode}")

        config = call_config or LLMCallConfig()
        bundle_payload = self._bundle_payload(bundle)
        self.last_error = None
        self.last_diagnosis = None
        self.last_review = None

        if mode == "single_call":
            proposal = self._generate(
                template=DESIGNER_V1,
                user_payload={
                    "task": "diagnose evidence and return one complete OperatorProposal",
                    "bundle": bundle_payload,
                },
                output_model=OperatorProposal,
                config=config,
            )
            proposal = OperatorProposal.model_validate(proposal)
        else:
            diagnosis = self._generate(
                template=DIAGNOSER_V1,
                user_payload={
                    "task": "return one evidence-grounded DiagnosisReport",
                    "bundle": bundle_payload,
                },
                output_model=DiagnosisReport,
                config=config,
            )
            diagnosis = DiagnosisReport.model_validate(diagnosis)
            self._validate_diagnosis(diagnosis, bundle)
            self.last_diagnosis = diagnosis
            diagnosis_hash = stable_hash(diagnosis.model_dump(mode="json"))

            proposal = self._generate(
                template=DESIGNER_V1,
                user_payload={
                    "task": "return one complete OperatorProposal using the exact diagnosis",
                    "bundle": bundle_payload,
                    "diagnosis": diagnosis.model_dump(mode="json"),
                    "diagnosis_hash": diagnosis_hash,
                },
                output_model=OperatorProposal,
                config=config,
            )
            proposal = OperatorProposal.model_validate(proposal)
            if proposal.diagnosis is None:
                raise LLMDesignError("staged proposal omitted the validated diagnosis")
            returned_hash = stable_hash(proposal.diagnosis.model_dump(mode="json"))
            if returned_hash != diagnosis_hash:
                raise LLMDesignError(
                    "staged proposal diagnosis does not match the validated diagnosis hash"
                )

        # This performs all hard checks even if a later orchestrator chooses a
        # different review policy.  Scores are retained for audit/CLI display.
        self.last_review = self.proposal_validator.validate_and_review(
            proposal,
            bundle,
            review_mode="rule_based",
        )
        self.last_diagnosis = proposal.diagnosis
        return proposal

    def review_from_evidence(
        self,
        bundle: Any,
        proposal: OperatorProposal,
        call_config: LLMCallConfig | None = None,
    ) -> OperatorReview:
        """Run optional structured review without allowing it to waive hard checks."""

        if self.provider is None:
            raise LLMDesignerUnavailableError(
                "review_from_evidence requires an explicit structured LLM provider"
            )
        static_review = self.proposal_validator.validate_and_review(
            proposal,
            bundle,
            review_mode="none",
        )
        returned = self._generate(
            template=REVIEWER_V1,
            user_payload={
                "task": "review one hard-valid proposal; do not claim formal validation",
                "bundle": self._bundle_payload(bundle),
                "proposal": proposal.model_dump(mode="json", by_alias=True),
                "static_review": static_review.model_dump(mode="json"),
            },
            output_model=OperatorReview,
            config=call_config or LLMCallConfig(),
        )
        model_review = OperatorReview.model_validate(returned)
        thresholds_pass = (
            model_review.evidence_alignment_score
            >= self.proposal_validator.EVIDENCE_THRESHOLD
            and model_review.safety_score >= self.proposal_validator.SAFETY_THRESHOLD
            and model_review.testability_score
            >= self.proposal_validator.TESTABILITY_THRESHOLD
        )
        decision = model_review.decision
        if not thresholds_pass and decision == "approve":
            decision = "revise"
        review = model_review.model_copy(
            update={
                "decision": decision,
                "novelty_score": static_review.novelty_score,
                "lineage_relation": static_review.lineage_relation,
                "topology_fingerprint": static_review.topology_fingerprint,
            }
        )
        self.last_review = review
        return review

    def _generate(
        self,
        *,
        template: PromptTemplate,
        user_payload: Any,
        output_model: type[Any],
        config: LLMCallConfig,
    ) -> Any:
        assert self.provider is not None
        self.last_prompt_template = template
        return self.provider.generate_structured(
            system_prompt=template.system_text,
            user_payload=user_payload,
            output_model=output_model,
            config=config,
            prompt_version=template.version,
            prompt_hash=template.prompt_hash,
        )

    @staticmethod
    def _bundle_payload(bundle: Any) -> Any:
        if hasattr(bundle, "model_dump"):
            return bundle.model_dump(mode="json")
        if isinstance(bundle, dict):
            return bundle
        raise TypeError("evidence bundle must be a Pydantic model or JSON mapping")

    @staticmethod
    def _validate_diagnosis(diagnosis: DiagnosisReport, bundle: Any) -> None:
        """Cross-check diagnosis evidence before permitting the design stage."""

        evidence_source = getattr(bundle, "evidence_ids", None)
        if callable(evidence_source):
            evidence_source = evidence_source()
        if evidence_source is None and isinstance(bundle, dict):
            evidence_source = {
                item["evidence_id"]
                for field in (
                    "effective_contexts",
                    "failure_contexts",
                    "failure_modes",
                    "synergy_evidence",
                    "counterfactual_evidence",
                    "representative_success_cases",
                    "representative_failure_cases",
                )
                for item in bundle.get(field, [])
            }
        known_evidence = {str(value) for value in (evidence_source or [])}

        parent_specs = getattr(bundle, "parent_specs", None)
        if parent_specs is None and isinstance(bundle, dict):
            parent_specs = bundle.get("parent_specs", [])
        parent_names = {
            str(spec.get("name") if isinstance(spec, dict) else getattr(spec, "name", ""))
            for spec in (parent_specs or [])
        }

        errors: list[str] = []
        if diagnosis.parent_operator not in parent_names:
            errors.append(
                f"diagnosis references unknown parent operator: {diagnosis.parent_operator}"
            )
        claims = (
            *diagnosis.effective_mechanisms,
            *diagnosis.failure_modes,
            *diagnosis.useful_synergies,
        )
        for claim in claims:
            unknown = sorted(set(claim.evidence_ids) - known_evidence)
            if unknown:
                errors.append(f"diagnosis claim references unknown evidence IDs: {unknown}")
        if errors:
            raise ProposalValidationError(errors)

    @staticmethod
    def _build_prompt(
        problem_description: str,
        parent_specs: list[OperatorSpec],
        parent_profiles: list[Any],
        memory_context: list[Any],
        success_cases: list[dict[str, Any]],
        failure_cases: list[dict[str, Any]],
    ) -> str:
        def dump(value: Any) -> Any:
            if hasattr(value, "model_dump"):
                return value.model_dump(mode="json")
            return value

        payload = {
            "instruction": "Return only JSON matching the supplied OperatorProposal schema.",
            "problem": problem_description,
            "parents": [dump(value) for value in parent_specs],
            "profiles": [dump(value) for value in parent_profiles],
            "memory": [dump(value) for value in memory_context],
            "success_cases": success_cases[:5],
            "failure_cases": failure_cases[:5],
            "safety": "Use only the DSL primitives in the schema; never return Python code.",
        }
        import json

        return json.dumps(payload, ensure_ascii=False, sort_keys=True)


__all__ = [
    "JSONClient",
    "LLMDesignError",
    "LLMDesignerAdapter",
    "LLMDesignerUnavailableError",
]
