"""Problem and solver specification models for define/solve workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional


_PROBLEM_ALIASES = {
    "linear_program": "lp",
    "linear_programming": "lp",
    "linprog": "lp",
}

_SOLVER_ALIASES = {
    "simulated_bifurcation": "sb",
    "primal_dual_lp": "pdlp",
}


@dataclass(frozen=True)
class ProblemSpec:
    """User-facing problem definition independent from solver selection."""

    problem_type: str
    data_format: str = "generic"
    name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def normalized_type(self) -> str:
        raw = str(self.problem_type).strip().lower()
        return _PROBLEM_ALIASES.get(raw, raw)

    def signature_dict(self) -> Dict[str, Any]:
        return {
            "problem_type": self.normalized_type(),
            "data_format": str(self.data_format).strip().lower(),
            "name": str(self.name) if self.name else "",
            "metadata": dict(sorted((self.metadata or {}).items())),
        }


@dataclass(frozen=True)
class SolverSpec:
    """Solver-side configuration independent from problem data schema."""

    solver_type: str
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def normalized_type(self) -> str:
        raw = str(self.solver_type).strip().lower()
        return _SOLVER_ALIASES.get(raw, raw)

    def signature_dict(self) -> Dict[str, Any]:
        return {
            "solver_type": self.normalized_type(),
            "config": dict(sorted((self.config or {}).items())),
            "metadata": dict(sorted((self.metadata or {}).items())),
        }


def build_problem_spec(
    problem_type: str,
    *,
    data_format: str = "generic",
    name: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> ProblemSpec:
    return ProblemSpec(
        problem_type=problem_type,
        data_format=data_format,
        name=name,
        metadata=dict(metadata or {}),
    )


def build_solver_spec(
    solver_type: str,
    *,
    config: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> SolverSpec:
    return SolverSpec(
        solver_type=solver_type,
        config=dict(config or {}),
        metadata=dict(metadata or {}),
    )
