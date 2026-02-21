"""Observation module public API.

Expose `ObservationBuilder` while keeping module-level docs minimal to avoid
lint warnings about unused variables or ambiguous exports.
"""

from .observation_builder import ObservationBuilder

__all__ = ["ObservationBuilder"]