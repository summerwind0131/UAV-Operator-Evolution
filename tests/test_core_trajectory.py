from __future__ import annotations

from operator_evolution_core.diagnosis import FeatureCatalog, OperatorDiagnoser
from operator_evolution_core.trajectory import (
    OperatorTrace as CoreOperatorTrace,
    TrajectoryRecorder as CoreTrajectoryRecorder,
)
from uav_operator_evolution.trajectory import (
    OperatorTrace as UAVOperatorTrace,
    TrajectoryRecorder as UAVTrajectoryRecorder,
)


def test_instance_id_uses_map_id_v1_storage_without_schema_migration() -> None:
    trace = CoreOperatorTrace(
        run_id="core",
        instance_id="instance-7",
        operator_id="move",
        before_objective=5.0,
        accepted_objective=4.0,
        accepted=True,
    )

    assert trace.instance_id == trace.map_id == "instance-7"
    payload = trace.model_dump(mode="json")
    assert payload["map_id"] == "instance-7"
    assert "instance_id" not in payload

    with CoreTrajectoryRecorder(":memory:") as recorder:
        recorder.record(trace)
        columns = {
            row["name"]
            for row in recorder.connection.execute(
                "PRAGMA table_info(operator_traces)"
            ).fetchall()
        }
        restored = recorder.list_traces("core")[0]

    assert "map_id" in columns
    assert "instance_id" not in columns
    assert restored.instance_id == "instance-7"


def test_uav_imports_are_identity_preserving_compatibility_facades() -> None:
    assert UAVOperatorTrace is CoreOperatorTrace
    assert UAVTrajectoryRecorder is CoreTrajectoryRecorder


def test_feature_catalog_resolves_domain_group_without_changing_trace_shape() -> None:
    catalog = FeatureCatalog(
        domain_id="example-domain",
        version="example-v1",
        groups={"phase": "context.analysis.phase"},
    )
    traces = [
        CoreOperatorTrace(
            run_id="run",
            instance_id="a",
            operator_id="move",
            context={"analysis": {"phase": "late"}},
            immediate_reward=2.0,
        ),
        CoreOperatorTrace(
            run_id="run",
            instance_id="b",
            operator_id="move",
            context={"analysis": {"phase": "late"}},
            immediate_reward=0.0,
        ),
    ]

    profiles = OperatorDiagnoser(
        minimum_context_samples=2,
        feature_catalog=catalog,
    ).diagnose(traces, group_by="phase")

    assert len(profiles) == 1
    assert profiles[0].context == {"context.analysis.phase": "late"}
    assert profiles[0].mean_immediate_reward == 1.0

