"""Fixed simulated-annealing acceptance used by every search run."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp

import numpy as np


@dataclass(frozen=True, slots=True)
class SimulatedAnnealingAcceptance:
    """Exponential cooling with immutable, experiment-level parameters."""

    start_temperature_ratio: float = 0.05
    end_temperature_ratio: float = 0.001
    minimum_temperature: float = 1e-12

    def __post_init__(self) -> None:
        if self.start_temperature_ratio <= 0 or self.end_temperature_ratio <= 0:
            raise ValueError("temperature ratios must be positive")
        if self.minimum_temperature <= 0:
            raise ValueError("minimum_temperature must be positive")

    def temperature(self, iteration: int, max_iterations: int, cost_scale: float) -> float:
        """Return the fixed exponentially cooled temperature for one step."""

        progress = min(max(float(iteration) / max(1, int(max_iterations) - 1), 0.0), 1.0)
        start = max(abs(float(cost_scale)) * self.start_temperature_ratio, self.minimum_temperature)
        end = max(abs(float(cost_scale)) * self.end_temperature_ratio, self.minimum_temperature)
        return float(start * ((end / start) ** progress))

    def accept(
        self,
        current_cost: float,
        candidate_cost: float,
        temperature: float,
        rng: np.random.Generator,
    ) -> bool:
        """Accept every non-worsening candidate and probabilistically accept worse ones."""

        delta = float(candidate_cost) - float(current_cost)
        if delta <= 0.0:
            return True
        if not np.isfinite(delta) or temperature <= 0.0:
            return False
        probability = exp(-delta / max(float(temperature), self.minimum_temperature))
        return bool(rng.random() < probability)


def simulated_annealing_accept(
    current_cost: float,
    candidate_cost: float,
    temperature: float,
    rng: np.random.Generator,
) -> bool:
    """Functional convenience wrapper around the fixed acceptance rule."""

    return SimulatedAnnealingAcceptance().accept(current_cost, candidate_cost, temperature, rng)
