"""Trajectory-informed operator evolution for UAV path planning."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("uav-operator-evolution")
except PackageNotFoundError:  # pragma: no cover - source checkout
    __version__ = "0.1.0"

__all__ = ["__version__"]

