"""Delayed operator-credit calculations."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Literal

from .models import OperatorTrace


def compute_delayed_rewards(
    traces: Sequence[OperatorTrace] | Iterable[OperatorTrace],
    horizons: Iterable[int] = (5, 10, 20),
    *,
    baseline: Literal["before", "accepted"] = "before",
    in_place: bool = False,
) -> list[OperatorTrace]:
    """Attach objective improvement observed after each requested horizon.

    Credit is computed independently inside each ``(run, episode, map)``
    trajectory.  With the default ``baseline='before'`` it is the cumulative
    improvement from immediately before the credited operator through the state
    accepted *horizon* decisions later.  ``baseline='accepted'`` instead reports
    only subsequent improvement.  Missing future observations are represented by
    ``None`` so right-censored samples are not mistaken for zero reward.

    The returned list retains the caller's ordering; ordering for the calculation
    itself is by iteration and timestamp inside each trajectory.
    """

    requested = sorted({int(horizon) for horizon in horizons})
    if any(horizon <= 0 for horizon in requested):
        raise ValueError("delayed reward horizons must be positive")

    original = [
        trace if isinstance(trace, OperatorTrace) else OperatorTrace.model_validate(trace)
        for trace in traces
    ]
    output = original if in_place else [trace.model_copy(deep=True) for trace in original]

    groups: dict[tuple[str, str | None, str], list[int]] = defaultdict(list)
    for index, trace in enumerate(output):
        groups[(trace.run_id, trace.episode_id, trace.instance_id)].append(index)

    for indexes in groups.values():
        indexes.sort(
            key=lambda index: (
                output[index].iteration,
                output[index].timestamp,
                index,
            )
        )
        for position, index in enumerate(indexes):
            trace = output[index]
            rewards = dict(trace.delayed_rewards)
            start = (
                trace.before_objective
                if baseline == "before"
                else trace.accepted_objective
            )
            for horizon in requested:
                future_position = position + horizon - 1
                if future_position >= len(indexes) or start is None:
                    rewards[horizon] = None
                    continue
                future = output[indexes[future_position]].accepted_objective
                rewards[horizon] = None if future is None else start - future
            trace.delayed_rewards = rewards
    return output
