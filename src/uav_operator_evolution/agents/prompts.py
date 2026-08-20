"""Versioned, immutable prompts used by optional LLM and agent designers.

Prompt text is treated like experiment configuration: callers can persist the
human-readable version together with a deterministic content hash and later
reconstruct exactly which policy was sent to a provider.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from collections.abc import Mapping

from ..reproducibility import stable_hash


# Kept as a public compatibility constant for the original schema-only
# adapter and downstream integrations.
SYSTEM_POLICY = (
    "Design bounded OperatorSpec JSON from computed evidence. "
    "Do not emit source code, tools, file access, network access, or shell commands."
)


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    """An immutable prompt artifact with a stable experiment identifier."""

    name: str
    version: str
    system_text: str
    prompt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("prompt name must not be empty")
        if not self.version.strip():
            raise ValueError("prompt version must not be empty")
        if not self.system_text.strip():
            raise ValueError("prompt system_text must not be empty")
        object.__setattr__(
            self,
            "prompt_hash",
            stable_hash(self.system_text),
        )

    @property
    def content_hash(self) -> str:
        """Readable alias used by audit and reporting code."""

        return self.prompt_hash

    @property
    def content(self) -> str:
        """Provider-neutral alias for the immutable system prompt text."""

        return self.system_text

    @property
    def system_prompt(self) -> str:
        """Alias matching :class:`LLMProvider` call terminology."""

        return self.system_text


DIAGNOSER_V1 = PromptTemplate(
    name="diagnoser",
    version="diagnoser_v1",
    system_text=(
        f"{SYSTEM_POLICY} Analyze only the supplied OperatorEvidenceBundle. "
        "Return a DiagnosisReport whose every claim cites evidence_id values "
        "present in that bundle. Separate observations, alternative explanations, "
        "and uncertainty; never invent traces, paths, maps, scores, or causal claims."
    ),
)

DESIGNER_V1 = PromptTemplate(
    name="designer",
    version="designer_v1",
    system_text=(
        f"{SYSTEM_POLICY} Return exactly one OperatorProposal. Every diagnosis, "
        "hypothesis, advantage, parent, and DSL choice must be grounded in the "
        "supplied evidence bundle. Use only catalogued primitives, preserve bounded "
        "fallback behavior, and propose an executable behavioral change rather than "
        "a rename or metadata-only variant. The deterministic validator, not this "
        "model, decides whether the candidate advances."
    ),
)

REVIEWER_V1 = PromptTemplate(
    name="reviewer",
    version="reviewer_v1",
    system_text=(
        f"{SYSTEM_POLICY} Review one already validated OperatorProposal against its "
        "evidence bundle. Score evidence alignment, novelty, DSL safety, and "
        "testability without executing the operator or claiming validation results. "
        "Hard schema, evidence, primitive, and lineage checks cannot be waived."
    ),
)

RESEARCH_AGENT_V1 = PromptTemplate(
    name="research_agent",
    version="research_agent_v1",
    system_text=(
        f"{SYSTEM_POLICY} Work only through the registered read-only evidence and "
        "bounded compiler/smoke tools. Observe turn, tool, candidate, revision, and "
        "token budgets. Do not request formal validation, test data, arbitrary code "
        "execution, shell, filesystem, network, hosted tools, or handoffs."
    ),
)

EXPLOITATION_DESIGNER_V1 = PromptTemplate(
    name="exploitation_designer",
    version="designer_exploitation_v1",
    system_text=(
        f"{SYSTEM_POLICY} Use the shared validated DiagnosisReport and return exactly "
        "one conservative OperatorProposal. Prefer evidence-backed refinements of the "
        "parent mechanism, bounded parameter choices, rollback safety, and contexts "
        "with observed support. Do not repeat diagnosis, call tools, or claim formal "
        "validation results."
    ),
)

EXPLORATION_DESIGNER_V1 = PromptTemplate(
    name="exploration_designer",
    version="designer_exploration_v1",
    system_text=(
        f"{SYSTEM_POLICY} Use the shared validated DiagnosisReport and return exactly "
        "one structurally distinct but bounded OperatorProposal. Explore a different "
        "catalogued mechanism from the exploitation candidate while preserving rollback "
        "safety and citing bundle evidence. Do not call tools or claim formal validation "
        "results."
    ),
)

PORTFOLIO_CRITIC_V1 = PromptTemplate(
    name="portfolio_critic",
    version="critic_v1",
    system_text=(
        f"{SYSTEM_POLICY} Compare exactly two sibling candidate summaries that share one "
        "DiagnosisReport. Return one approve, revise, or reject assessment per candidate. "
        "Do not select, rank, compile, smoke-test, or revise candidates: deterministic "
        "Python owns portfolio scoring and local gates. Do not claim formal validation "
        "results."
    ),
)

# Specification-facing names mirror the requested immutable prompt versions;
# uppercase aliases remain conventional for Python constants.
diagnoser_v1 = DIAGNOSER_V1
designer_v1 = DESIGNER_V1
reviewer_v1 = REVIEWER_V1
research_agent_v1 = RESEARCH_AGENT_V1
exploitation_designer_v1 = EXPLOITATION_DESIGNER_V1
exploration_designer_v1 = EXPLORATION_DESIGNER_V1
portfolio_critic_v1 = PORTFOLIO_CRITIC_V1
designer_exploitation_v1 = EXPLOITATION_DESIGNER_V1
designer_exploration_v1 = EXPLORATION_DESIGNER_V1
critic_v1 = PORTFOLIO_CRITIC_V1


PROMPT_TEMPLATES: Mapping[str, PromptTemplate] = MappingProxyType(
    {
        template.version: template
        for template in (
            DIAGNOSER_V1,
            DESIGNER_V1,
            REVIEWER_V1,
            RESEARCH_AGENT_V1,
            EXPLOITATION_DESIGNER_V1,
            EXPLORATION_DESIGNER_V1,
            PORTFOLIO_CRITIC_V1,
        )
    }
)


def get_prompt_template(version: str) -> PromptTemplate:
    """Return a prompt by immutable version name."""

    try:
        return PROMPT_TEMPLATES[version]
    except KeyError as exc:
        raise KeyError(f"unknown prompt template: {version}") from exc


__all__ = [
    "DESIGNER_V1",
    "DIAGNOSER_V1",
    "EXPLOITATION_DESIGNER_V1",
    "EXPLORATION_DESIGNER_V1",
    "PROMPT_TEMPLATES",
    "PromptTemplate",
    "RESEARCH_AGENT_V1",
    "REVIEWER_V1",
    "PORTFOLIO_CRITIC_V1",
    "SYSTEM_POLICY",
    "designer_v1",
    "designer_exploitation_v1",
    "designer_exploration_v1",
    "diagnoser_v1",
    "critic_v1",
    "exploitation_designer_v1",
    "exploration_designer_v1",
    "get_prompt_template",
    "research_agent_v1",
    "reviewer_v1",
    "portfolio_critic_v1",
]
