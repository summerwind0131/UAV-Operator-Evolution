"""Experimental domain-independent contracts for operator evolution.

The package remains internal to the UAV repository until a second domain has
validated the interfaces.  Its modules must never import UAV implementation
types.
"""

from .contracts import DatasetSplit, InstanceRef, ObjectiveEvaluation

__all__ = ["DatasetSplit", "InstanceRef", "ObjectiveEvaluation"]
