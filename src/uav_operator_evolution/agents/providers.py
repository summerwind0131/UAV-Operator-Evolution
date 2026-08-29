"""Structured-output provider boundary for optional LLM-backed designers.

The core package deliberately does not import an LLM SDK at module import
time.  Providers return validated Pydantic objects and retain only compact
request metadata; prompts, API keys, and arbitrary model text are not stored.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from time import perf_counter
from typing import Any, Literal, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..reproducibility import canonical_json, stable_hash


StructuredModel = TypeVar("StructuredModel", bound=BaseModel)
ProviderName = Literal["mock", "openai", "deepseek", "gemini"]
CallStatus = Literal[
    "success",
    "schema_error",
    "refusal",
    "timeout",
    "rate_limit",
    "server_error",
    "configuration_error",
    "provider_error",
    "budget_exceeded",
]
MockMode = Literal[
    "success",
    "schema_error",
    "timeout",
    "refusal",
    "rate_limit",
    "server_error",
]


class ProviderModel(BaseModel):
    """Strict JSON-native base for provider configuration and audit records."""

    model_config = ConfigDict(extra="forbid", strict=True)


class LLMCallConfig(ProviderModel):
    """Hard limits for one structured call and its enclosing design run."""

    model: str | None = Field(default=None, min_length=1, max_length=200)
    timeout_seconds: float = Field(default=60.0, gt=0.0, le=600.0)
    max_retries: int = Field(default=2, ge=0, le=10)
    max_output_tokens: int = Field(default=4_096, ge=1, le=100_000)
    max_total_tokens: int = Field(default=20_000, ge=1, le=1_000_000)
    max_logical_calls: int = Field(default=64, ge=1, le=1_000)


class LLMUsage(ProviderModel):
    """Provider-neutral token usage."""

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)


class LLMCallRecord(ProviderModel):
    """Compact local record for a logical call, including all retry attempts."""

    call_id: str
    provider: str
    output_model: str
    status: CallStatus
    model: str | None = None
    response_id: str | None = None
    prompt_version: str | None = None
    prompt_hash: str
    request_hash: str
    usage: LLMUsage = Field(default_factory=LLMUsage)
    cumulative_total_tokens: int = Field(default=0, ge=0)
    attempts: int = Field(default=1, ge=1)
    retry_count: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0.0)
    error: str | None = None


class LLMProviderError(RuntimeError):
    """Base exception for a rejected or failed structured generation."""

    status: CallStatus = "provider_error"

    def __init__(self, message: str, *, record: LLMCallRecord | None = None) -> None:
        super().__init__(message)
        self.record = record


class LLMConfigurationError(LLMProviderError):
    status: CallStatus = "configuration_error"


class LLMStructuredOutputError(LLMProviderError):
    status: CallStatus = "schema_error"


class LLMRefusalError(LLMProviderError):
    status: CallStatus = "refusal"


class LLMTimeoutError(LLMProviderError):
    status: CallStatus = "timeout"


class LLMRateLimitError(LLMProviderError):
    status: CallStatus = "rate_limit"


class LLMServerError(LLMProviderError):
    status: CallStatus = "server_error"


class LLMTokenBudgetError(LLMProviderError):
    status: CallStatus = "budget_exceeded"


@runtime_checkable
class LLMProvider(Protocol):
    """SDK-independent structured generation interface."""

    call_records: list[LLMCallRecord]

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_payload: Any,
        output_model: type[StructuredModel],
        config: LLMCallConfig,
        prompt_version: str | None = None,
        prompt_hash: str | None = None,
    ) -> StructuredModel:
        ...


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", by_alias=True)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _request_metadata(
    system_prompt: str,
    user_payload: Any,
    supplied_prompt_hash: str | None,
) -> tuple[str, str, str]:
    payload = _jsonable(user_payload)
    user_text = user_payload if isinstance(user_payload, str) else canonical_json(payload)
    computed_prompt_hash = stable_hash({"system_prompt": system_prompt})
    prompt_hash = supplied_prompt_hash or computed_prompt_hash
    request_hash = stable_hash(
        {"prompt_hash": prompt_hash, "system_prompt": system_prompt, "user_payload": payload}
    )
    return prompt_hash, request_hash, str(user_text)


_API_KEY_PATTERN = re.compile(r"(?i)\b(?:sk|sess)-[A-Za-z0-9_-]{8,}\b")


def _safe_error(value: object, secrets: Sequence[str] = ()) -> str:
    message = str(value)
    for secret in secrets:
        if secret:
            message = message.replace(secret, "[REDACTED]")
    message = _API_KEY_PATTERN.sub("[REDACTED]", message)
    return message[:2_000]


def _usage_from(value: Any) -> LLMUsage:
    if value is None:
        return LLMUsage()

    def read(*names: str) -> int:
        for name in names:
            item = value.get(name) if isinstance(value, Mapping) else getattr(value, name, None)
            if item is not None:
                try:
                    return max(0, int(item or 0))
                except (TypeError, ValueError):
                    return 0
        return 0

    input_tokens = read("input_tokens", "prompt_tokens")
    output_tokens = read("output_tokens", "completion_tokens")
    total_tokens = read("total_tokens") or input_tokens + output_tokens
    return LLMUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


class _ProviderState:
    provider_name = "provider"

    def __init__(self) -> None:
        self.call_records: list[LLMCallRecord] = []
        self.cumulative_total_tokens = 0
        self._call_counter = 0
        self._logical_calls_since_reset = 0

    @property
    def last_record(self) -> LLMCallRecord | None:
        return self.call_records[-1] if self.call_records else None

    def reset_usage(self) -> None:
        """Start a new per-design cumulative token budget."""

        self.cumulative_total_tokens = 0
        self._logical_calls_since_reset = 0

    def reset_token_budget(self) -> None:
        """Compatibility spelling for callers that name the enforced budget."""

        self.reset_usage()

    def continue_call_ids_after(self, sequence: int) -> None:
        """Continue an audit's call-id sequence before this provider makes new calls."""

        if self.call_records or self._logical_calls_since_reset:
            raise RuntimeError("call-id sequence can only be set before provider use")
        if sequence < 0:
            raise ValueError("call-id sequence must be non-negative")
        self._call_counter = int(sequence)

    def _next_call_id(self) -> str:
        self._call_counter += 1
        self._logical_calls_since_reset += 1
        return f"llmcall_{self.provider_name}_{self._call_counter:06d}"

    def _append_record(
        self,
        *,
        call_id: str,
        output_model: type[BaseModel],
        status: CallStatus,
        model: str | None,
        response_id: str | None,
        prompt_version: str | None,
        prompt_hash: str,
        request_hash: str,
        usage: LLMUsage,
        attempts: int,
        started_at: float,
        error: str | None = None,
    ) -> LLMCallRecord:
        record = LLMCallRecord(
            call_id=call_id,
            provider=self.provider_name,
            output_model=output_model.__name__,
            status=status,
            model=model,
            response_id=response_id,
            prompt_version=prompt_version,
            prompt_hash=prompt_hash,
            request_hash=request_hash,
            usage=usage,
            cumulative_total_tokens=self.cumulative_total_tokens,
            attempts=attempts,
            retry_count=max(0, attempts - 1),
            latency_ms=max(0.0, float((perf_counter() - started_at) * 1_000.0)),
            error=error,
        )
        self.call_records.append(record)
        return record

    def _enforce_logical_call_budget(
        self,
        *,
        config: LLMCallConfig,
        call_id: str,
        output_model: type[BaseModel],
        model: str | None,
        prompt_version: str | None,
        prompt_hash: str,
        request_hash: str,
        started_at: float,
    ) -> None:
        if self._logical_calls_since_reset <= config.max_logical_calls:
            return
        message = "logical LLM call budget is exhausted"
        record = self._append_record(
            call_id=call_id,
            output_model=output_model,
            status="budget_exceeded",
            model=model,
            response_id=None,
            prompt_version=prompt_version,
            prompt_hash=prompt_hash,
            request_hash=request_hash,
            usage=LLMUsage(),
            attempts=1,
            started_at=started_at,
            error=message,
        )
        raise LLMTokenBudgetError(message, record=record)


class MockLLMProvider(_ProviderState):
    """Deterministic provider for offline tests, demos, and ablations.

    Fixtures can be keyed by output model class, class name, or fully-qualified
    class name.  A factory may instead accept ``(output_model, user_payload)``
    and return a Pydantic object or data accepted by ``model_validate``.
    """

    provider_name = "mock"

    def __init__(
        self,
        *,
        fixtures: Mapping[object, Any] | None = None,
        factory: Callable[..., Any] | None = None,
        mode: MockMode = "success",
        failure_sequence: Sequence[MockMode] | None = None,
        usage: LLMUsage | Mapping[str, int] | None = None,
    ) -> None:
        super().__init__()
        self.fixtures = dict(fixtures or {})
        self.factory = factory
        self.mode: MockMode = mode
        self.failure_sequence = list(failure_sequence or ())
        self._outcome_index = 0
        self.fixture_usage = _usage_from(
            usage.model_dump(mode="json") if isinstance(usage, LLMUsage) else usage
        )

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_payload: Any,
        output_model: type[StructuredModel],
        config: LLMCallConfig,
        prompt_version: str | None = None,
        prompt_hash: str | None = None,
    ) -> StructuredModel:
        started_at = perf_counter()
        call_id = self._next_call_id()
        resolved_prompt_hash, request_hash, _ = _request_metadata(
            system_prompt, user_payload, prompt_hash
        )
        model_name = config.model or "mock-structured-v1"
        self._enforce_logical_call_budget(
            config=config,
            call_id=call_id,
            output_model=output_model,
            model=model_name,
            prompt_version=prompt_version,
            prompt_hash=resolved_prompt_hash,
            request_hash=request_hash,
            started_at=started_at,
        )
        if self.cumulative_total_tokens >= config.max_total_tokens:
            return self._raise(
                LLMTokenBudgetError,
                "cumulative LLM token budget is already exhausted",
                call_id=call_id,
                output_model=output_model,
                status="budget_exceeded",
                model=model_name,
                prompt_version=prompt_version,
                prompt_hash=resolved_prompt_hash,
                request_hash=request_hash,
                usage=LLMUsage(),
                attempts=1,
                started_at=started_at,
            )

        for attempt in range(1, config.max_retries + 2):
            outcome = self._next_outcome()
            if outcome in {"timeout", "rate_limit", "server_error"}:
                if attempt <= config.max_retries:
                    continue
                error_type: type[LLMProviderError]
                if outcome == "timeout":
                    error_type = LLMTimeoutError
                elif outcome == "rate_limit":
                    error_type = LLMRateLimitError
                else:
                    error_type = LLMServerError
                return self._raise(
                    error_type,
                    f"mock provider simulated {outcome}",
                    call_id=call_id,
                    output_model=output_model,
                    status=outcome,
                    model=model_name,
                    prompt_version=prompt_version,
                    prompt_hash=resolved_prompt_hash,
                    request_hash=request_hash,
                    usage=LLMUsage(),
                    attempts=attempt,
                    started_at=started_at,
                )
            if outcome == "refusal":
                return self._raise(
                    LLMRefusalError,
                    "mock provider simulated a model refusal",
                    call_id=call_id,
                    output_model=output_model,
                    status="refusal",
                    model=model_name,
                    prompt_version=prompt_version,
                    prompt_hash=resolved_prompt_hash,
                    request_hash=request_hash,
                    usage=LLMUsage(),
                    attempts=attempt,
                    started_at=started_at,
                )
            if outcome == "schema_error":
                return self._raise(
                    LLMStructuredOutputError,
                    "mock provider simulated invalid structured output",
                    call_id=call_id,
                    output_model=output_model,
                    status="schema_error",
                    model=model_name,
                    prompt_version=prompt_version,
                    prompt_hash=resolved_prompt_hash,
                    request_hash=request_hash,
                    usage=LLMUsage(),
                    attempts=attempt,
                    started_at=started_at,
                )

            try:
                raw = self._fixture(output_model, user_payload)
                parsed = raw if isinstance(raw, output_model) else output_model.model_validate(raw)
            except (ValidationError, ValueError, TypeError, KeyError) as exc:
                return self._raise(
                    LLMStructuredOutputError,
                    f"mock fixture failed {output_model.__name__} validation: {exc}",
                    call_id=call_id,
                    output_model=output_model,
                    status="schema_error",
                    model=model_name,
                    prompt_version=prompt_version,
                    prompt_hash=resolved_prompt_hash,
                    request_hash=request_hash,
                    usage=LLMUsage(),
                    attempts=attempt,
                    started_at=started_at,
                )

            usage = self.fixture_usage
            if usage.total_tokens == 0:
                output_text = canonical_json(parsed.model_dump(mode="json", by_alias=True))
                input_text = canonical_json(_jsonable(user_payload))
                input_tokens = max(1, (len(system_prompt) + len(input_text) + 3) // 4)
                output_tokens = max(1, (len(output_text) + 3) // 4)
                usage = LLMUsage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=input_tokens + output_tokens,
                )
            self.cumulative_total_tokens += usage.total_tokens
            status: CallStatus = "success"
            error: str | None = None
            if self.cumulative_total_tokens > config.max_total_tokens:
                status = "budget_exceeded"
                error = "structured output exceeded cumulative LLM token budget"
            record = self._append_record(
                call_id=call_id,
                output_model=output_model,
                status=status,
                model=model_name,
                response_id=f"mock_response_{self._call_counter:06d}",
                prompt_version=prompt_version,
                prompt_hash=resolved_prompt_hash,
                request_hash=request_hash,
                usage=usage,
                attempts=attempt,
                started_at=started_at,
                error=error,
            )
            if status == "budget_exceeded":
                raise LLMTokenBudgetError(error or "LLM token budget exceeded", record=record)
            return parsed

        raise AssertionError("unreachable mock provider state")

    def _next_outcome(self) -> MockMode:
        if self._outcome_index < len(self.failure_sequence):
            outcome = self.failure_sequence[self._outcome_index]
            self._outcome_index += 1
            return outcome
        return self.mode

    def _fixture(self, output_model: type[StructuredModel], user_payload: Any) -> Any:
        if self.factory is not None:
            try:
                return self.factory(output_model=output_model, user_payload=user_payload)
            except TypeError as keyword_error:
                try:
                    return self.factory(output_model, user_payload)
                except TypeError:
                    raise keyword_error
        qualified = f"{output_model.__module__}.{output_model.__qualname__}"
        for key in (output_model, output_model.__name__, qualified):
            if key in self.fixtures:
                fixture = self.fixtures[key]
                return fixture(output_model, user_payload) if callable(fixture) else fixture
        return _default_mock_fixture(output_model, user_payload)

    def _raise(
        self,
        error_type: type[LLMProviderError],
        message: str,
        **record_fields: Any,
    ) -> Any:
        record = self._append_record(error=message, response_id=None, **record_fields)
        raise error_type(message, record=record)


def _payload_bundle(user_payload: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    root = _jsonable(user_payload)
    if not isinstance(root, dict):
        return {}, {}
    bundle = root
    for key in ("bundle", "evidence_bundle", "operator_evidence_bundle"):
        candidate = root.get(key)
        if isinstance(candidate, dict):
            bundle = candidate
            break
    return root, bundle


def _default_mock_fixture(output_model: type[StructuredModel], user_payload: Any) -> Any:
    """Create deterministic Phase-8 fixtures while remaining model-generic."""

    if output_model.__name__ == "DiagnosisReport":
        return _default_diagnosis(user_payload)
    if output_model.__name__ == "OperatorProposal":
        return _default_proposal(user_payload)
    if output_model.__name__ == "OperatorReview":
        root, _ = _payload_bundle(user_payload)
        static_review = root.get("static_review")
        if isinstance(static_review, dict):
            return static_review
        return {
            "decision": "approve",
            "evidence_alignment_score": 1.0,
            "novelty_score": 0.85,
            "safety_score": 1.0,
            "testability_score": 1.0,
            "concerns": [],
            "required_revisions": [],
            "lineage_relation": "structural_variant",
            "topology_fingerprint": None,
        }
    if output_model.__name__ == "PortfolioCritique":
        return _default_portfolio_critique(user_payload)
    # Models with only defaults are still useful in provider unit tests.
    return output_model.model_validate({})


def _first_evidence(bundle: Mapping[str, Any], fields: Sequence[str]) -> dict[str, Any] | None:
    for field in fields:
        values = bundle.get(field)
        if isinstance(values, list) and values and isinstance(values[0], dict):
            return values[0]
    return None


def _default_diagnosis(user_payload: Any) -> dict[str, Any]:
    root, bundle = _payload_bundle(user_payload)
    parents = bundle.get("parent_specs", [])
    parent_name = "unknown_parent"
    if isinstance(parents, list) and parents and isinstance(parents[0], dict):
        parent_name = str(parents[0].get("name") or parent_name)

    failure = _first_evidence(bundle, ("failure_modes", "failure_contexts"))
    effective = _first_evidence(bundle, ("effective_contexts",))
    synergy = _first_evidence(bundle, ("synergy_evidence",))

    def claim(item: dict[str, Any] | None, text: str) -> list[dict[str, Any]]:
        if item is None or not item.get("evidence_id"):
            return []
        confidence = float(item.get("confidence", 0.5))
        return [
            {
                "claim": text,
                "evidence_ids": [str(item["evidence_id"])],
                "confidence": min(1.0, max(0.0, confidence)),
                "alternative_explanation": "Observed association may reflect map or search-stage mix.",
            }
        ]

    failure_text = "underperformance in the cited context"
    if failure is not None:
        failure_text = str(failure.get("failure_mode") or failure_text)
    effective_text = "the parent is effective in the cited context"
    if effective is not None and effective.get("context"):
        effective_text = "positive reward in the cited operator context"
    synergy_text = "the cited ordered operator pair is positively associated"
    limitations = bundle.get("limitations", [])
    unresolved = [str(item) for item in limitations[:8]] if isinstance(limitations, list) else []
    diagnosis = {
        "parent_operator": parent_name,
        "effective_mechanisms": claim(effective, effective_text),
        "failure_modes": claim(failure, failure_text),
        "useful_synergies": claim(synergy, synergy_text),
        "unresolved_questions": unresolved,
    }
    # A staged caller may explicitly provide the authoritative diagnosis.
    supplied = root.get("diagnosis") or root.get("diagnosis_report")
    return supplied if isinstance(supplied, dict) else diagnosis


def _default_proposal(user_payload: Any) -> dict[str, Any]:
    from ..domain.uav_kit import UAVDomainKit
    from .design_models import DiagnosisReport

    domain_kit = UAVDomainKit()
    root, bundle = _payload_bundle(user_payload)
    design_role = str(root.get("design_role", ""))
    multi_role = design_role in {"exploitation_designer", "exploration_designer"}
    exploration = design_role == "exploration_designer"
    parents = bundle.get("parent_specs", [])
    if not isinstance(parents, list) or not parents:
        raise ValueError("mock OperatorProposal requires at least one parent spec")
    parent = domain_kit.parse_ir(parents[0])
    diagnosis = DiagnosisReport.model_validate(_default_diagnosis(user_payload))
    failure_claim = diagnosis.failure_modes[0] if diagnosis.failure_modes else None
    # Carry the diagnosis evidence through to the proposal, not merely the
    # targeted failure claim.  This makes the deterministic fixture exercise
    # the same cross-evidence alignment gate as a real structured provider.
    evidence_ids = list(
        dict.fromkeys(
            evidence_id
            for claim_group in (
                diagnosis.effective_mechanisms,
                diagnosis.failure_modes,
                diagnosis.useful_synergies,
            )
            for claim in claim_group
            for evidence_id in claim.evidence_ids
        )
    )
    failure_evidence_ids = (
        list(failure_claim.evidence_ids) if failure_claim is not None else []
    )
    target_failure = failure_claim.claim if failure_claim is not None else ""

    specification = parent.model_dump(mode="json")
    role_prefix = (
        "MockExplore"
        if exploration
        else "MockExploit"
        if multi_role
        else "MockEvolved"
    )
    specification["name"] = f"{role_prefix}_{parent.name}"[:100]
    specification["description"] = (
        "Deterministic exploratory evidence-grounded mock composite operator."
        if exploration
        else "Deterministic exploitative evidence-grounded mock composite operator."
        if multi_role
        else "Deterministic evidence-grounded mock composite operator."
    )
    specification["parent_operators"] = [item["name"] for item in parents[:4] if isinstance(item, dict)]
    transformations = list(specification["transformations"])
    if exploration:
        specification["selection_strategy"] = {"kind": "select_low_clearance_segment"}
        transformations = [
            {
                "kind": "try_alternative_side",
                "clearance_factor": 1.5,
                "repeat": 1,
                "when": None,
            },
            {
                "kind": "reconstruct_segment",
                "max_points": 8,
                "repeat": 1,
                "when": None,
            },
            {
                "kind": "smooth_segment",
                "strength": 0.25,
                "repeat": 1,
                "when": None,
            },
        ]
        specification["repair_strategy"] = {
            "kind": "repeat_until_feasible",
            "transformations": [
                {
                    "kind": "generate_obstacle_detour",
                    "clearance_factor": 1.5,
                    "repeat": 1,
                    "when": None,
                }
            ],
            "max_attempts": 2,
        }
    else:
        added_step = {
            "kind": "smooth_segment",
            "strength": 0.35,
            "repeat": 1,
            "when": None,
        }
        if len(transformations) < 8:
            transformations.append(added_step)
        else:
            transformations[-1] = added_step
        if multi_role:
            specification["repair_strategy"] = {
                "kind": "repeat_until_feasible",
                "transformations": [
                    {
                        "kind": "try_alternative_side",
                        "clearance_factor": 1.5,
                        "repeat": 1,
                        "when": None,
                    },
                    {
                        "kind": "smooth_segment",
                        "strength": 0.25,
                        "repeat": 1,
                        "when": None,
                    },
                ],
                "max_attempts": 2,
            }
    specification["transformations"] = transformations
    specification["fallback_strategy"] = {"kind": "rollback_on_failure"}
    specification["expected_mechanism"] = (
        "Explore bounded obstacle-side detours while retaining rollback protection."
        if exploration
        else "Preserve the parent move while adding bounded smoothing and rollback protection."
    )
    specification["target_failure_modes"] = [target_failure] if target_failure else []
    validated_spec = domain_kit.parse_ir(specification)

    hypothesis: dict[str, Any] | None = None
    if failure_claim is not None:
        hypothesis = {
            "hypothesis": (
                "A bounded detour should explore a distinct route around the diagnosed failure."
                if exploration
                else "A bounded smoothing pass should reduce the diagnosed failure without bypassing safety."
            ),
            "target_failure_mode": failure_claim.claim,
            "expected_mechanism": (
                "Obstacle-aware side detour followed by rollback on structural failure."
                if exploration
                else "Post-move smoothing followed by rollback on structural failure."
            ),
            "expected_effective_context": (
                "Blocked or low-clearance path segments."
                if exploration
                else "Paths with excess curvature or locally noisy waypoints."
            ),
            "possible_side_effects": (
                ["Detours can increase path length."]
                if exploration
                else ["Smoothing can reduce obstacle clearance."]
            ),
            "evidence_ids": failure_evidence_ids,
        }
    return {
        "operator_spec": domain_kit.serialize_ir(validated_spec),
        "design_rationale": (
            "The exploration role tries a distinct bounded detour grounded in the shared diagnosis."
            if exploration
            else "The exploitation role refines the parent using the highest-priority cited evidence."
            if multi_role
            else "The deterministic mock follows the highest-priority cited failure evidence."
        ),
        "evidence_used": evidence_ids,
        "target_failure_modes": [target_failure] if target_failure else [],
        "changes_from_parents": (
            [
                "switched to low-clearance selection",
                "added alternative-side reconstruction",
                "added bounded repair and rollback",
            ]
            if exploration
            else ["added bounded smoothing", "added bounded repair and rollback"]
        ),
        "expected_contexts": (
            ["blocked or low-clearance path segments"]
            if exploration
            else ["high-curvature or locally irregular paths"]
        ),
        "expected_risks": (
            ["may increase path length"]
            if exploration
            else ["may trade clearance for smoothness"]
        ),
        "evidence_level": "exploratory",
        "diagnosis": diagnosis.model_dump(mode="json"),
        "hypothesis": hypothesis,
        "expected_advantages": (
            ["higher route diversity around obstacles"]
            if exploration
            else ["lower smoothness penalty"]
        ),
        "used_evidence_ids": evidence_ids,
    }


def _default_portfolio_critique(user_payload: Any) -> dict[str, Any]:
    """Deterministically assess two siblings without making the selection."""

    root, _ = _payload_bundle(user_payload)
    candidates = root.get("candidates", [])
    if not isinstance(candidates, list) or len(candidates) != 2:
        raise ValueError("mock PortfolioCritique requires exactly two candidates")
    assessments: list[dict[str, Any]] = []
    for item in candidates:
        if not isinstance(item, Mapping):
            raise ValueError("portfolio candidate summaries must be mappings")
        candidate_id = str(item.get("candidate_id", ""))
        role = str(item.get("role", ""))
        review = (
            item.get("static_review")
            if isinstance(item.get("static_review"), Mapping)
            else {}
        )
        approved = str(review.get("decision", "reject")) == "approve"
        assessments.append(
            {
                "candidate_id": candidate_id,
                "decision": "approve" if approved else "revise",
                "evidence_alignment_score": float(
                    review.get("evidence_alignment_score", 0.0)
                ),
                "safety_score": float(review.get("safety_score", 0.0)),
                "testability_score": float(review.get("testability_score", 0.0)),
                "mechanism_fit_score": (
                    0.95 if role == "exploitation_designer" else 0.90
                ),
                "causal_overclaim": False,
                "evidence_ids": sorted(
                    {str(value) for value in root.get("used_evidence_ids", [])}
                ),
                "strengths": ["proposal is testable under bounded local checks"]
                if approved
                else [],
                "concerns": []
                if approved
                else ["static review did not approve the candidate"],
                "required_revisions": []
                if approved
                else ["address deterministic static review concerns"],
            }
        )
    evidence_ids = root.get("used_evidence_ids", [])
    return {
        "assessments": assessments,
        "comparative_rationale": (
            "The critic assessed evidence and mechanism fit only; deterministic Python "
            "owns portfolio scoring, tie-breaking, compilation, and smoke checks."
        ),
        "used_evidence_ids": (
            sorted({str(value) for value in evidence_ids})
            if isinstance(evidence_ids, list)
            else []
        ),
    }


class OpenAIProvider(_ProviderState):
    """Optional OpenAI Responses API provider using Structured Outputs.

    The SDK import and client construction are delayed until the first call so
    the default offline installation has no optional dependency requirement.
    """

    provider_name = "openai"

    def __init__(
        self,
        *,
        client: Any | None = None,
        api_key: str | None = None,
        model: str | None = None,
        allow_legacy_environment: bool = True,
    ) -> None:
        super().__init__()
        self._client = client
        self._api_key = api_key
        self.model = model
        self.allow_legacy_environment = allow_legacy_environment

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_payload: Any,
        output_model: type[StructuredModel],
        config: LLMCallConfig,
        prompt_version: str | None = None,
        prompt_hash: str | None = None,
    ) -> StructuredModel:
        started_at = perf_counter()
        call_id = self._next_call_id()
        resolved_prompt_hash, request_hash, user_text = _request_metadata(
            system_prompt, user_payload, prompt_hash
        )
        model_name = config.model or self.model
        if model_name is None and self.allow_legacy_environment:
            model_name = os.getenv("UOE_LLM_MODEL")
        api_key = self._api_key or os.getenv("OPENAI_API_KEY")
        if api_key is None and self.allow_legacy_environment:
            api_key = os.getenv("UOE_LLM_API_KEY")
        secrets = (api_key or "",)
        self._enforce_logical_call_budget(
            config=config,
            call_id=call_id,
            output_model=output_model,
            model=model_name,
            prompt_version=prompt_version,
            prompt_hash=resolved_prompt_hash,
            request_hash=request_hash,
            started_at=started_at,
        )
        if not model_name:
            return self._configuration_failure(
                (
                    "OpenAI provider requires an explicit model"
                    if not self.allow_legacy_environment
                    else "OpenAI provider requires LLMCallConfig.model or UOE_LLM_MODEL"
                ),
                call_id,
                output_model,
                model_name,
                prompt_version,
                resolved_prompt_hash,
                request_hash,
                started_at,
            )
        if self._client is None and not api_key:
            return self._configuration_failure(
                (
                    "OpenAI provider requires OPENAI_API_KEY"
                    if not self.allow_legacy_environment
                    else "OpenAI provider requires OPENAI_API_KEY or UOE_LLM_API_KEY"
                ),
                call_id,
                output_model,
                model_name,
                prompt_version,
                resolved_prompt_hash,
                request_hash,
                started_at,
            )
        if self.cumulative_total_tokens >= config.max_total_tokens:
            record = self._append_record(
                call_id=call_id,
                output_model=output_model,
                status="budget_exceeded",
                model=model_name,
                response_id=None,
                prompt_version=prompt_version,
                prompt_hash=resolved_prompt_hash,
                request_hash=request_hash,
                usage=LLMUsage(),
                attempts=1,
                started_at=started_at,
                error="cumulative LLM token budget is already exhausted",
            )
            raise LLMTokenBudgetError(record.error or "token budget exhausted", record=record)

        try:
            client = self._get_client(api_key, config)
        except (ImportError, ModuleNotFoundError) as exc:
            message = (
                "OpenAI SDK is not installed; install the project's 'llm' optional dependency"
            )
            record = self._append_record(
                call_id=call_id,
                output_model=output_model,
                status="configuration_error",
                model=model_name,
                response_id=None,
                prompt_version=prompt_version,
                prompt_hash=resolved_prompt_hash,
                request_hash=request_hash,
                usage=LLMUsage(),
                attempts=1,
                started_at=started_at,
                error=f"{message}: {_safe_error(exc, secrets)}",
            )
            raise LLMConfigurationError(message, record=record) from exc
        except Exception as exc:
            message = _safe_error(exc, secrets)
            record = self._append_record(
                call_id=call_id,
                output_model=output_model,
                status="configuration_error",
                model=model_name,
                response_id=None,
                prompt_version=prompt_version,
                prompt_hash=resolved_prompt_hash,
                request_hash=request_hash,
                usage=LLMUsage(),
                attempts=1,
                started_at=started_at,
                error=message,
            )
            raise LLMConfigurationError(message, record=record) from exc
        for attempt in range(1, config.max_retries + 2):
            try:
                response = client.responses.parse(
                    model=model_name,
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_text},
                    ],
                    text_format=output_model,
                    max_output_tokens=config.max_output_tokens,
                    timeout=config.timeout_seconds,
                )
            except Exception as exc:
                status = _classify_openai_error(exc)
                retriable = status in {"timeout", "rate_limit", "server_error"}
                if retriable and attempt <= config.max_retries:
                    continue
                error_type: type[LLMProviderError] = {
                    "timeout": LLMTimeoutError,
                    "rate_limit": LLMRateLimitError,
                    "server_error": LLMServerError,
                }.get(status, LLMProviderError)
                message = _safe_error(exc, secrets)
                record = self._append_record(
                    call_id=call_id,
                    output_model=output_model,
                    status=status,
                    model=model_name,
                    response_id=None,
                    prompt_version=prompt_version,
                    prompt_hash=resolved_prompt_hash,
                    request_hash=request_hash,
                    usage=LLMUsage(),
                    attempts=attempt,
                    started_at=started_at,
                    error=message,
                )
                raise error_type(message, record=record) from exc

            response_id = str(getattr(response, "id", "")) or None
            response_model = str(getattr(response, "model", "")) or model_name
            usage = _usage_from(getattr(response, "usage", None))
            self.cumulative_total_tokens += usage.total_tokens
            refusal = _response_refusal(response)
            if refusal is not None:
                message = "OpenAI model refused the structured-output request"
                record = self._append_record(
                    call_id=call_id,
                    output_model=output_model,
                    status="refusal",
                    model=response_model,
                    response_id=response_id,
                    prompt_version=prompt_version,
                    prompt_hash=resolved_prompt_hash,
                    request_hash=request_hash,
                    usage=usage,
                    attempts=attempt,
                    started_at=started_at,
                    error=message,
                )
                raise LLMRefusalError(message, record=record)

            parsed = getattr(response, "output_parsed", None)
            if parsed is None:
                message = "OpenAI response contained no parsed structured output"
                record = self._append_record(
                    call_id=call_id,
                    output_model=output_model,
                    status="schema_error",
                    model=response_model,
                    response_id=response_id,
                    prompt_version=prompt_version,
                    prompt_hash=resolved_prompt_hash,
                    request_hash=request_hash,
                    usage=usage,
                    attempts=attempt,
                    started_at=started_at,
                    error=message,
                )
                raise LLMStructuredOutputError(message, record=record)
            try:
                result = parsed if isinstance(parsed, output_model) else output_model.model_validate(parsed)
            except (ValidationError, ValueError, TypeError) as exc:
                message = f"OpenAI parsed output failed local validation: {_safe_error(exc, secrets)}"
                record = self._append_record(
                    call_id=call_id,
                    output_model=output_model,
                    status="schema_error",
                    model=response_model,
                    response_id=response_id,
                    prompt_version=prompt_version,
                    prompt_hash=resolved_prompt_hash,
                    request_hash=request_hash,
                    usage=usage,
                    attempts=attempt,
                    started_at=started_at,
                    error=message,
                )
                raise LLMStructuredOutputError(message, record=record) from exc

            status: CallStatus = "success"
            error: str | None = None
            if self.cumulative_total_tokens > config.max_total_tokens:
                status = "budget_exceeded"
                error = "structured output exceeded cumulative LLM token budget"
            record = self._append_record(
                call_id=call_id,
                output_model=output_model,
                status=status,
                model=response_model,
                response_id=response_id,
                prompt_version=prompt_version,
                prompt_hash=resolved_prompt_hash,
                request_hash=request_hash,
                usage=usage,
                attempts=attempt,
                started_at=started_at,
                error=error,
            )
            if status == "budget_exceeded":
                raise LLMTokenBudgetError(error or "LLM token budget exceeded", record=record)
            return result

        raise AssertionError("unreachable OpenAI provider state")

    def _get_client(self, api_key: str | None, config: LLMCallConfig) -> Any:
        if self._client is not None:
            return self._client
        from openai import OpenAI  # type: ignore[import-not-found]

        self._client = OpenAI(api_key=api_key, max_retries=0, timeout=config.timeout_seconds)
        return self._client

    def _configuration_failure(
        self,
        message: str,
        call_id: str,
        output_model: type[BaseModel],
        model: str | None,
        prompt_version: str | None,
        prompt_hash: str,
        request_hash: str,
        started_at: float,
    ) -> Any:
        record = self._append_record(
            call_id=call_id,
            output_model=output_model,
            status="configuration_error",
            model=model,
            response_id=None,
            prompt_version=prompt_version,
            prompt_hash=prompt_hash,
            request_hash=request_hash,
            usage=LLMUsage(),
            attempts=1,
            started_at=started_at,
            error=message,
        )
        raise LLMConfigurationError(message, record=record)


class DeepSeekProvider(_ProviderState):
    """DeepSeek Chat Completions adapter using its OpenAI-compatible JSON mode."""

    provider_name = "deepseek"

    def __init__(
        self,
        *,
        client: Any | None = None,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str = "https://api.deepseek.com",
    ) -> None:
        super().__init__()
        self._client = client
        self._api_key = api_key
        self.model = model
        self.base_url = base_url

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_payload: Any,
        output_model: type[StructuredModel],
        config: LLMCallConfig,
        prompt_version: str | None = None,
        prompt_hash: str | None = None,
    ) -> StructuredModel:
        started_at = perf_counter()
        call_id = self._next_call_id()
        resolved_prompt_hash, request_hash, user_text = _request_metadata(
            system_prompt, user_payload, prompt_hash
        )
        model_name = config.model or self.model
        api_key = self._api_key or os.getenv("DEEPSEEK_API_KEY")
        secrets = (api_key or "",)
        self._enforce_logical_call_budget(
            config=config,
            call_id=call_id,
            output_model=output_model,
            model=model_name,
            prompt_version=prompt_version,
            prompt_hash=resolved_prompt_hash,
            request_hash=request_hash,
            started_at=started_at,
        )
        if not model_name:
            return self._configuration_failure(
                "DeepSeek provider requires an explicit model",
                call_id,
                output_model,
                model_name,
                prompt_version,
                resolved_prompt_hash,
                request_hash,
                started_at,
            )
        if self._client is None and not api_key:
            return self._configuration_failure(
                "DeepSeek provider requires DEEPSEEK_API_KEY",
                call_id,
                output_model,
                model_name,
                prompt_version,
                resolved_prompt_hash,
                request_hash,
                started_at,
            )
        if self.cumulative_total_tokens >= config.max_total_tokens:
            return self._budget_failure(
                call_id,
                output_model,
                model_name,
                prompt_version,
                resolved_prompt_hash,
                request_hash,
                started_at,
            )
        try:
            client = self._get_client(api_key, config)
        except (ImportError, ModuleNotFoundError) as exc:
            message = "OpenAI SDK is required for DeepSeek; install the project's 'llm' extra"
            record = self._append_record(
                call_id=call_id,
                output_model=output_model,
                status="configuration_error",
                model=model_name,
                response_id=None,
                prompt_version=prompt_version,
                prompt_hash=resolved_prompt_hash,
                request_hash=request_hash,
                usage=LLMUsage(),
                attempts=1,
                started_at=started_at,
                error=f"{message}: {_safe_error(exc, secrets)}",
            )
            raise LLMConfigurationError(message, record=record) from exc
        except Exception as exc:
            message = _safe_error(exc, secrets)
            record = self._append_record(
                call_id=call_id,
                output_model=output_model,
                status="configuration_error",
                model=model_name,
                response_id=None,
                prompt_version=prompt_version,
                prompt_hash=resolved_prompt_hash,
                request_hash=request_hash,
                usage=LLMUsage(),
                attempts=1,
                started_at=started_at,
                error=message,
            )
            raise LLMConfigurationError(message, record=record) from exc

        schema_text = canonical_json(output_model.model_json_schema())
        json_prompt = (
            system_prompt
            + "\nReturn JSON only. The JSON must validate against this schema: "
            + schema_text
        )
        schema_retry_feedback: str | None = None
        logical_usage = LLMUsage()
        for attempt in range(1, config.max_retries + 2):
            attempt_started_at = perf_counter()
            try:
                attempt_system_prompt = json_prompt
                if schema_retry_feedback:
                    attempt_system_prompt += schema_retry_feedback
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": attempt_system_prompt},
                        {"role": "user", "content": user_text},
                    ],
                    response_format={"type": "json_object"},
                    # DeepSeek V4 defaults to thinking mode.  Its reasoning
                    # tokens share max_tokens with the final JSON and can
                    # exhaust a bounded structured-output call before any
                    # content is emitted.  This adapter needs only the typed
                    # artifact, so make the non-thinking contract explicit.
                    extra_body={"thinking": {"type": "disabled"}},
                    max_tokens=config.max_output_tokens,
                    timeout=config.timeout_seconds,
                )
            except Exception as exc:
                status = _classify_provider_error(exc)
                if status in {"timeout", "rate_limit", "server_error"} and attempt <= config.max_retries:
                    continue
                return self._raise_provider_error(
                    exc,
                    status=status,
                    call_id=call_id,
                    output_model=output_model,
                    model=model_name,
                    prompt_version=prompt_version,
                    prompt_hash=resolved_prompt_hash,
                    request_hash=request_hash,
                    attempt=attempt,
                    started_at=started_at,
                    secrets=secrets,
                )

            attempt_usage = _usage_from(_read_value(response, "usage"))
            self.cumulative_total_tokens += attempt_usage.total_tokens
            logical_usage = LLMUsage(
                input_tokens=logical_usage.input_tokens + attempt_usage.input_tokens,
                output_tokens=logical_usage.output_tokens + attempt_usage.output_tokens,
                total_tokens=logical_usage.total_tokens + attempt_usage.total_tokens,
            )
            usage = logical_usage
            response_id = _string_or_none(_read_value(response, "id"))
            response_model = _string_or_none(_read_value(response, "model")) or model_name
            attempt_latency = perf_counter() - attempt_started_at
            if attempt_latency > config.timeout_seconds:
                message = (
                    "DeepSeek response exceeded the end-to-end attempt deadline "
                    f"of {config.timeout_seconds:.3f}s"
                )
                if self.cumulative_total_tokens > config.max_total_tokens:
                    return self._terminal_failure(
                        LLMTokenBudgetError,
                        "structured output exceeded cumulative LLM token budget",
                        "budget_exceeded",
                        call_id,
                        output_model,
                        response_model,
                        response_id,
                        prompt_version,
                        resolved_prompt_hash,
                        request_hash,
                        usage,
                        attempt,
                        started_at,
                    )
                if attempt <= config.max_retries:
                    continue
                return self._terminal_failure(
                    LLMTimeoutError,
                    message,
                    "timeout",
                    call_id,
                    output_model,
                    response_model,
                    response_id,
                    prompt_version,
                    resolved_prompt_hash,
                    request_hash,
                    usage,
                    attempt,
                    started_at,
                )
            choices = _read_value(response, "choices", [])
            choice = choices[0] if isinstance(choices, (list, tuple)) and choices else None
            message = _read_value(choice, "message")
            refusal = _read_value(message, "refusal")
            if refusal:
                return self._terminal_failure(
                    LLMRefusalError,
                    "DeepSeek model refused the structured-output request",
                    "refusal",
                    call_id,
                    output_model,
                    response_model,
                    response_id,
                    prompt_version,
                    resolved_prompt_hash,
                    request_hash,
                    usage,
                    attempt,
                    started_at,
                )
            finish_reason = str(_read_value(choice, "finish_reason", "") or "").lower()
            content = _read_value(message, "content")
            if finish_reason == "length" or not isinstance(content, str) or not content.strip():
                return self._terminal_failure(
                    LLMStructuredOutputError,
                    "DeepSeek returned empty or truncated JSON output",
                    "schema_error",
                    call_id,
                    output_model,
                    response_model,
                    response_id,
                    prompt_version,
                    resolved_prompt_hash,
                    request_hash,
                    usage,
                    attempt,
                    started_at,
                )
            try:
                result = output_model.model_validate_json(content)
            except (ValidationError, ValueError, TypeError, json.JSONDecodeError) as exc:
                validation_error = _safe_error(exc, secrets)
                if attempt <= config.max_retries:
                    schema_retry_feedback = (
                        "\nThe previous JSON response failed local schema validation: "
                        + validation_error
                        + "\nReturn a corrected JSON object only. Respect every string "
                        "length, numeric bound, enum, required field, and extra-field rule."
                    )
                    continue
                return self._terminal_failure(
                    LLMStructuredOutputError,
                    f"DeepSeek JSON failed local validation: {validation_error}",
                    "schema_error",
                    call_id,
                    output_model,
                    response_model,
                    response_id,
                    prompt_version,
                    resolved_prompt_hash,
                    request_hash,
                    usage,
                    attempt,
                    started_at,
                )
            self._record_success_or_budget(
                call_id,
                output_model,
                response_model,
                response_id,
                prompt_version,
                resolved_prompt_hash,
                request_hash,
                usage,
                attempt,
                started_at,
                config,
            )
            return result
        raise AssertionError("unreachable DeepSeek provider state")

    def _get_client(self, api_key: str | None, config: LLMCallConfig) -> Any:
        if self._client is not None:
            return self._client
        from openai import OpenAI  # type: ignore[import-not-found]

        self._client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            max_retries=0,
            timeout=config.timeout_seconds,
        )
        return self._client

    def _configuration_failure(self, message: str, *fields: Any) -> Any:
        call_id, output_model, model, prompt_version, prompt_hash, request_hash, started_at = fields
        record = self._append_record(
            call_id=call_id,
            output_model=output_model,
            status="configuration_error",
            model=model,
            response_id=None,
            prompt_version=prompt_version,
            prompt_hash=prompt_hash,
            request_hash=request_hash,
            usage=LLMUsage(),
            attempts=1,
            started_at=started_at,
            error=message,
        )
        raise LLMConfigurationError(message, record=record)

    def _budget_failure(self, *fields: Any) -> Any:
        call_id, output_model, model, prompt_version, prompt_hash, request_hash, started_at = fields
        message = "cumulative LLM token budget is already exhausted"
        record = self._append_record(
            call_id=call_id,
            output_model=output_model,
            status="budget_exceeded",
            model=model,
            response_id=None,
            prompt_version=prompt_version,
            prompt_hash=prompt_hash,
            request_hash=request_hash,
            usage=LLMUsage(),
            attempts=1,
            started_at=started_at,
            error=message,
        )
        raise LLMTokenBudgetError(message, record=record)

    def _terminal_failure(
        self,
        error_type: type[LLMProviderError],
        message: str,
        status: CallStatus,
        call_id: str,
        output_model: type[BaseModel],
        model: str,
        response_id: str | None,
        prompt_version: str | None,
        prompt_hash: str,
        request_hash: str,
        usage: LLMUsage,
        attempts: int,
        started_at: float,
    ) -> Any:
        record = self._append_record(
            call_id=call_id,
            output_model=output_model,
            status=status,
            model=model,
            response_id=response_id,
            prompt_version=prompt_version,
            prompt_hash=prompt_hash,
            request_hash=request_hash,
            usage=usage,
            attempts=attempts,
            started_at=started_at,
            error=message,
        )
        raise error_type(message, record=record)

    def _raise_provider_error(self, exc: Exception, *, status: CallStatus, secrets: Sequence[str], **fields: Any) -> Any:
        error_type: type[LLMProviderError] = {
            "timeout": LLMTimeoutError,
            "rate_limit": LLMRateLimitError,
            "server_error": LLMServerError,
        }.get(status, LLMProviderError)
        message = _safe_error(exc, secrets)
        return self._terminal_failure(
            error_type,
            message,
            status,
            fields["call_id"],
            fields["output_model"],
            fields["model"],
            None,
            fields["prompt_version"],
            fields["prompt_hash"],
            fields["request_hash"],
            LLMUsage(),
            fields["attempt"],
            fields["started_at"],
        )

    def _record_success_or_budget(
        self,
        call_id: str,
        output_model: type[BaseModel],
        model: str,
        response_id: str | None,
        prompt_version: str | None,
        prompt_hash: str,
        request_hash: str,
        usage: LLMUsage,
        attempts: int,
        started_at: float,
        config: LLMCallConfig,
    ) -> None:
        exceeded = self.cumulative_total_tokens > config.max_total_tokens
        message = "structured output exceeded cumulative LLM token budget" if exceeded else None
        record = self._append_record(
            call_id=call_id,
            output_model=output_model,
            status="budget_exceeded" if exceeded else "success",
            model=model,
            response_id=response_id,
            prompt_version=prompt_version,
            prompt_hash=prompt_hash,
            request_hash=request_hash,
            usage=usage,
            attempts=attempts,
            started_at=started_at,
            error=message,
        )
        if exceeded:
            raise LLMTokenBudgetError(message or "token budget exceeded", record=record)


class GeminiProvider(DeepSeekProvider):
    """Google Gen AI adapter with JSON Schema output and local Pydantic validation."""

    provider_name = "gemini"

    def __init__(
        self,
        *,
        client: Any | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        _ProviderState.__init__(self)
        self._client = client
        self._api_key = api_key
        self.model = model

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_payload: Any,
        output_model: type[StructuredModel],
        config: LLMCallConfig,
        prompt_version: str | None = None,
        prompt_hash: str | None = None,
    ) -> StructuredModel:
        started_at = perf_counter()
        call_id = self._next_call_id()
        resolved_prompt_hash, request_hash, user_text = _request_metadata(
            system_prompt, user_payload, prompt_hash
        )
        model_name = config.model or self.model
        api_key = self._api_key or os.getenv("GEMINI_API_KEY")
        secrets = (api_key or "",)
        self._enforce_logical_call_budget(
            config=config,
            call_id=call_id,
            output_model=output_model,
            model=model_name,
            prompt_version=prompt_version,
            prompt_hash=resolved_prompt_hash,
            request_hash=request_hash,
            started_at=started_at,
        )
        if not model_name:
            return self._configuration_failure(
                "Gemini provider requires an explicit model",
                call_id,
                output_model,
                model_name,
                prompt_version,
                resolved_prompt_hash,
                request_hash,
                started_at,
            )
        if self._client is None and not api_key:
            return self._configuration_failure(
                "Gemini provider requires GEMINI_API_KEY",
                call_id,
                output_model,
                model_name,
                prompt_version,
                resolved_prompt_hash,
                request_hash,
                started_at,
            )
        if self.cumulative_total_tokens >= config.max_total_tokens:
            return self._budget_failure(
                call_id,
                output_model,
                model_name,
                prompt_version,
                resolved_prompt_hash,
                request_hash,
                started_at,
            )
        try:
            client = self._get_client(api_key, config)
        except (ImportError, ModuleNotFoundError) as exc:
            message = "Google Gen AI SDK is not installed; install the project's 'llm' extra"
            record = self._append_record(
                call_id=call_id,
                output_model=output_model,
                status="configuration_error",
                model=model_name,
                response_id=None,
                prompt_version=prompt_version,
                prompt_hash=resolved_prompt_hash,
                request_hash=request_hash,
                usage=LLMUsage(),
                attempts=1,
                started_at=started_at,
                error=f"{message}: {_safe_error(exc, secrets)}",
            )
            raise LLMConfigurationError(message, record=record) from exc
        except Exception as exc:
            message = _safe_error(exc, secrets)
            record = self._append_record(
                call_id=call_id,
                output_model=output_model,
                status="configuration_error",
                model=model_name,
                response_id=None,
                prompt_version=prompt_version,
                prompt_hash=resolved_prompt_hash,
                request_hash=request_hash,
                usage=LLMUsage(),
                attempts=1,
                started_at=started_at,
                error=message,
            )
            raise LLMConfigurationError(message, record=record) from exc

        for attempt in range(1, config.max_retries + 2):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=user_text,
                    config={
                        "system_instruction": system_prompt,
                        "response_mime_type": "application/json",
                        "response_json_schema": output_model.model_json_schema(),
                        "max_output_tokens": config.max_output_tokens,
                    },
                )
            except Exception as exc:
                status = _classify_provider_error(exc)
                if status in {"timeout", "rate_limit", "server_error"} and attempt <= config.max_retries:
                    continue
                return self._raise_provider_error(
                    exc,
                    status=status,
                    call_id=call_id,
                    output_model=output_model,
                    model=model_name,
                    prompt_version=prompt_version,
                    prompt_hash=resolved_prompt_hash,
                    request_hash=request_hash,
                    attempt=attempt,
                    started_at=started_at,
                    secrets=secrets,
                )

            usage = _gemini_usage(_read_value(response, "usage_metadata"))
            self.cumulative_total_tokens += usage.total_tokens
            response_id = _string_or_none(_read_value(response, "response_id"))
            response_model = _string_or_none(_read_value(response, "model_version")) or model_name
            candidates = _read_value(response, "candidates", [])
            candidate = candidates[0] if isinstance(candidates, (list, tuple)) and candidates else None
            finish_reason = str(_read_value(candidate, "finish_reason", "") or "").lower()
            if "safety" in finish_reason or "blocked" in finish_reason or "recitation" in finish_reason:
                return self._terminal_failure(
                    LLMRefusalError,
                    "Gemini blocked or refused the structured-output request",
                    "refusal",
                    call_id,
                    output_model,
                    response_model,
                    response_id,
                    prompt_version,
                    resolved_prompt_hash,
                    request_hash,
                    usage,
                    attempt,
                    started_at,
                )
            try:
                content = _read_value(response, "text")
            except Exception:
                content = None
            if "max_tokens" in finish_reason or not isinstance(content, str) or not content.strip():
                return self._terminal_failure(
                    LLMStructuredOutputError,
                    "Gemini returned empty or truncated JSON output",
                    "schema_error",
                    call_id,
                    output_model,
                    response_model,
                    response_id,
                    prompt_version,
                    resolved_prompt_hash,
                    request_hash,
                    usage,
                    attempt,
                    started_at,
                )
            try:
                result = output_model.model_validate_json(content)
            except (ValidationError, ValueError, TypeError, json.JSONDecodeError) as exc:
                return self._terminal_failure(
                    LLMStructuredOutputError,
                    f"Gemini JSON failed local validation: {_safe_error(exc, secrets)}",
                    "schema_error",
                    call_id,
                    output_model,
                    response_model,
                    response_id,
                    prompt_version,
                    resolved_prompt_hash,
                    request_hash,
                    usage,
                    attempt,
                    started_at,
                )
            self._record_success_or_budget(
                call_id,
                output_model,
                response_model,
                response_id,
                prompt_version,
                resolved_prompt_hash,
                request_hash,
                usage,
                attempt,
                started_at,
                config,
            )
            return result
        raise AssertionError("unreachable Gemini provider state")

    def _get_client(self, api_key: str | None, config: LLMCallConfig) -> Any:
        if self._client is not None:
            return self._client
        from google import genai  # type: ignore[import-not-found]

        self._client = genai.Client(
            api_key=api_key,
            http_options={"timeout": int(config.timeout_seconds * 1_000)},
        )
        return self._client


def _read_value(value: Any, name: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _string_or_none(value: Any) -> str | None:
    text = "" if value is None else str(value)
    return text or None


def _gemini_usage(value: Any) -> LLMUsage:
    if value is None:
        return LLMUsage()

    def read(*names: str) -> int:
        for name in names:
            item = _read_value(value, name)
            if item is not None:
                try:
                    return max(0, int(item or 0))
                except (TypeError, ValueError):
                    return 0
        return 0

    input_tokens = read("prompt_token_count", "prompt_tokens")
    output_tokens = read("candidates_token_count", "output_tokens") + read(
        "thoughts_token_count", "reasoning_tokens"
    )
    total_tokens = read("total_token_count", "total_tokens") or input_tokens + output_tokens
    return LLMUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


def _classify_provider_error(exc: Exception) -> CallStatus:
    name = exc.__class__.__name__.lower()
    status_code = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    if status_code is None and response is not None:
        status_code = getattr(response, "status_code", None)
    if "timeout" in name or isinstance(exc, TimeoutError):
        return "timeout"
    if "ratelimit" in name or "rate_limit" in name or status_code == 429:
        return "rate_limit"
    if "connection" in name or "connecterror" in name:
        return "server_error"
    if "internalserver" in name or (isinstance(status_code, int) and status_code >= 500):
        return "server_error"
    return "provider_error"


def _classify_openai_error(exc: Exception) -> CallStatus:
    return _classify_provider_error(exc)


def _response_refusal(response: Any) -> str | None:
    direct = getattr(response, "refusal", None)
    if direct:
        return str(direct)
    outputs = getattr(response, "output", None)
    if not isinstance(outputs, (list, tuple)):
        return None
    for output in outputs:
        contents = output.get("content", []) if isinstance(output, Mapping) else getattr(output, "content", [])
        if not isinstance(contents, (list, tuple)):
            continue
        for part in contents:
            part_type = part.get("type") if isinstance(part, Mapping) else getattr(part, "type", None)
            refusal = part.get("refusal") if isinstance(part, Mapping) else getattr(part, "refusal", None)
            if part_type == "refusal" or refusal:
                return str(refusal or "refused")
    return None


__all__ = [
    "DeepSeekProvider",
    "GeminiProvider",
    "LLMCallConfig",
    "LLMCallRecord",
    "LLMConfigurationError",
    "LLMProvider",
    "LLMProviderError",
    "LLMRateLimitError",
    "LLMRefusalError",
    "LLMServerError",
    "LLMStructuredOutputError",
    "LLMTimeoutError",
    "LLMTokenBudgetError",
    "LLMUsage",
    "MockLLMProvider",
    "OpenAIProvider",
    "ProviderName",
]
