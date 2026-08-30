"""Strict preregistration schema for bidirectional mechanism transfer."""

from __future__ import annotations

import hashlib
import json
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TransferBudgetV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    train_search_calls: int = Field(gt=0)
    validation_search_calls: int = Field(gt=0)
    test_search_calls: int = Field(gt=0)
    generations: int = Field(gt=0)
    candidates_per_generation: int = Field(gt=0)
    population_slots: int = Field(gt=0)


class TransferStatisticsV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    bootstrap_resamples: int = Field(ge=1000)
    confidence_level: float = Field(gt=0.0, lt=1.0)
    multiple_comparison: Literal["holm"] = "holm"
    outcome_order: tuple[Literal["feasibility", "feasible_cost"], ...] = (
        "feasibility",
        "feasible_cost",
    )


class MechanismTransferPreregistrationV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["mechanism-transfer-preregistration-v1"]
    experiment_id: Literal["mechanism-transfer-v1"]
    directions: tuple[
        Literal["uav-to-jssp", "jssp-to-uav"],
        ...,
    ]
    arms: tuple[
        Literal["scratch", "same-domain", "cross-domain"],
        ...,
    ]
    final_groups: tuple[
        Literal["p0", "scratch", "same-domain", "cross-domain"],
        ...,
    ]
    master_seeds: tuple[int, ...] = Field(min_length=10, max_length=10)
    uav_bank_seeds: tuple[int, ...] = Field(min_length=1)
    jssp_bank_seeds: tuple[int, ...] = Field(min_length=1)
    retrieval_limit: Literal[4]
    designer: Literal["deterministic-heuristic"]
    remote_provider_allowed: Literal[False]
    uav_test_dataset: Literal["uav2d-transfer-v1"]
    jssp_test_dataset: Literal["orlib-jobshop1-sealed-41"]
    budget: TransferBudgetV1
    statistics: TransferStatisticsV1

    @model_validator(mode="after")
    def validate_registration(self) -> Self:
        if self.directions != ("uav-to-jssp", "jssp-to-uav"):
            raise ValueError("both transfer directions must appear in fixed order")
        if self.arms != ("scratch", "same-domain", "cross-domain"):
            raise ValueError("the three design arms must appear in fixed order")
        if self.final_groups != ("p0", *self.arms):
            raise ValueError("final groups must compare p0 and all registered arms")
        seed_groups = (
            self.master_seeds,
            self.uav_bank_seeds,
            self.jssp_bank_seeds,
        )
        if any(len(set(group)) != len(group) for group in seed_groups):
            raise ValueError("seed schedules must not contain duplicates")
        flattened = [seed for group in seed_groups for seed in group]
        if len(set(flattened)) != len(flattened):
            raise ValueError("experiment and bank seed schedules must be disjoint")
        if self.statistics.outcome_order != ("feasibility", "feasible_cost"):
            raise ValueError("feasibility must be tested before feasible cost")
        return self

    @property
    def content_hash(self) -> str:
        canonical = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()
