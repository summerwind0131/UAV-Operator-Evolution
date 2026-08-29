"""UAV-specific names for context grouping in generic diagnostics."""

from operator_evolution_core.diagnosis import FeatureCatalog

UAV_FEATURE_CATALOG = FeatureCatalog(
    # Kept literal here so importing diagnostics does not initialize the UAV
    # operator/search dependency cycle merely to identify the domain.
    domain_id="uav-path-planning-2d",
    version="uav-trace-features-v1",
    groups={
        "map_type": "context.analysis.map_type",
        "obstacle_density": "context.analysis.obstacle_density",
        "search_phase": "context.analysis.search_phase",
        "stagnation": "context.analysis.stagnation",
        "feasible_before": "context.analysis.feasible_before",
        "collision_count": "context.analysis.collision_count",
        "smoothness": "context.analysis.smoothness",
        "map_difficulty": "map_difficulty",
        "environment_difficulty": "context.environment_features.difficulty",
    },
)


__all__ = ["UAV_FEATURE_CATALOG"]
