from __future__ import annotations

from uav_operator_evolution.memory import MechanismMemory
from uav_operator_evolution.trajectory import OperatorTrace


def test_mechanism_crud_and_evidence_queries(tmp_path) -> None:
    with MechanismMemory(tmp_path / "memory.sqlite") as memory:
        assert memory.add_mechanism(
            "parent",
            {"transform": "repair"},
            score=0.7,
            evidence_count=10,
            metadata={"context": {"difficulty": "dense"}},
        ) == "parent"
        memory.add_mechanism("child", {"transform": "smooth"}, score=0.9)
        memory.add_mechanism("other", {}, score=1.0)

        updated = memory.update_mechanism("child", description="specialist")
        assert updated.description == "specialist"
        assert memory.get_mechanism("child")["definition"] == {"transform": "smooth"}

        trace = OperatorTrace(
            trace_id=4,
            run_id="r",
            map_id="m",
            operator_id="repair",
            immediate_reward=2,
            delayed_rewards={5: 4},
            accepted=True,
            context={"difficulty": "dense"},
        )
        memory.record_operator_history(trace, mechanism_id="parent")
        memory.add_failure_mode(
            "collision", mechanism_id="parent", operator_id="repair", count=3
        )
        memory.add_synergy("repair", "smooth", 0.4, sample_count=5)
        profile_id = memory.add_operator_profile(
            {"operator_name": "repair", "total_calls": 10},
            operator_id="repair",
            run_id="r",
            generation=1,
        )
        insight_id = memory.add_insight(
            operator_id="repair",
            insight_type="effective_mechanism",
            evidence={"trace_ids": [4]},
            confidence=0.8,
            applicable_context={"difficulty": "dense"},
            source_profile_id="profile-1",
        )
        case_id = memory.add_case(
            mechanism_id="parent",
            operator_id="repair",
            outcome="success",
            score=2,
            context={"difficulty": "dense"},
            state={"objective": 10},
        )
        memory.add_lineage("parent", "child")

        assert memory.get_operator_history("repair")[0].trace_id == 4
        assert memory.get_failure_modes("repair")[0].count == 3
        assert memory.get_synergies("repair")[0].second_operator == "smooth"
        profile = memory.get_operator_profiles("repair")[0]
        assert profile.profile_id == profile_id
        assert profile.profile["total_calls"] == 10
        insight = memory.get_insights("repair")[0]
        assert insight.insight_id == insight_id
        assert insight.insight_type == "effective_mechanism"
        assert memory.get_relevant_cases({"difficulty": "dense"})[0].case_id == case_id
        assert memory.get_lineage("child", direction="ancestors")[0].parent_id == "parent"

        best = memory.get_best_mechanisms({"difficulty": "dense"})
        # Contextual relevance takes precedence over an unrelated raw score.
        assert best[0].mechanism_id == "parent"
        assert memory.delete_mechanism("other") is True
        assert memory.get_mechanism("other") is None
