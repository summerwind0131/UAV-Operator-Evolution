"""JSSP feature aliases for the shared domain-neutral diagnoser."""

from operator_evolution_core.diagnosis import FeatureCatalog

JSSP_FEATURE_CATALOG = FeatureCatalog(
    domain_id="jssp",
    version="jssp-trace-features-v1",
    groups={
        "instance_shape": "context.instance_shape",
        "source_family": "context.source_family",
        "critical_path_ratio": "context.analysis.critical_path_ratio",
        "bottleneck_utilization": (
            "context.analysis.bottleneck_machine_utilization"
        ),
        "load_imbalance": "context.analysis.machine_load_imbalance",
        "critical_block_count": "context.analysis.critical_block_count",
        "operation_displacement": "context.analysis.operation_displacement",
        "relative_initial_improvement": (
            "context.analysis.relative_initial_improvement"
        ),
    },
)

__all__ = ["JSSP_FEATURE_CATALOG"]
