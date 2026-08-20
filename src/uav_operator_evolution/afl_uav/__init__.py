"""AFL-style end-to-end solver generation adapted to static 2D UAV planning."""

from .coordinator import AFLUAVCoordinator
from .buffer import SolverBuffer
from .artifact import (
    AFLSolverArtifact,
    AFLSolverCandidate,
    AFL_PROVIDER_MODELS,
    QUALIFICATION_MAP_IDS,
    build_solver_artifact,
    freeze_solver_candidate,
    generate_solver_candidate,
    load_solver_artifact,
    load_solver_candidate,
)
from .models import AFLUAVLimits, AFLUAVRunResult, UAVProblemDescription, UAVSolverInstance

__all__ = [
    "AFLUAVCoordinator",
    "AFLUAVLimits",
    "AFLUAVRunResult",
    "AFLSolverArtifact",
    "AFLSolverCandidate",
    "AFL_PROVIDER_MODELS",
    "QUALIFICATION_MAP_IDS",
    "SolverBuffer",
    "UAVProblemDescription",
    "UAVSolverInstance",
    "build_solver_artifact",
    "freeze_solver_candidate",
    "generate_solver_candidate",
    "load_solver_artifact",
    "load_solver_candidate",
]
