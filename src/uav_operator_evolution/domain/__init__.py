"""UAV implementations of the experimental domain/core contracts."""

from .adapters import (
    UAV_DOMAIN_ID,
    UAV_OBJECTIVE_ADAPTER_VERSION,
    environment_matches_instance_ref,
    environment_to_instance_ref,
    evaluation_result_to_objective,
    objective_to_evaluation_result,
)
from .uav_kit import UAVDomainKit, UAVSmokeFixture, UAV_IR_VERSION
from .uav_adapter import (
    UAVDomainAdapter,
    UAVObjectiveEvaluator,
    UAVPathCodec,
    UAVPathFeatureExtractor,
    UAVPathGuard,
    UAVPathInitializer,
    UAVTraceEncoder,
)

__all__ = [
    "UAV_DOMAIN_ID",
    "UAVDomainAdapter",
    "UAVDomainKit",
    "UAVSmokeFixture",
    "UAV_IR_VERSION",
    "UAV_OBJECTIVE_ADAPTER_VERSION",
    "UAVObjectiveEvaluator",
    "UAVPathCodec",
    "UAVPathFeatureExtractor",
    "UAVPathGuard",
    "UAVPathInitializer",
    "UAVTraceEncoder",
    "environment_matches_instance_ref",
    "environment_to_instance_ref",
    "evaluation_result_to_objective",
    "objective_to_evaluation_result",
]
