"""Pydantic data models (request / response contracts)."""
from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

from .graph import topo_order


class RelationType(str, Enum):
    hierarchical = "hierarchical"   # one task directly impacts the next
    synergistic = "synergistic"     # several tasks jointly impact one task


class Task(BaseModel):
    id: str
    name: str = ""
    # Risk factors in [0, 1] (0 = benign, 1 = severe)
    threat: float = Field(0.3, ge=0, le=1)
    resource_gap: float = Field(0.3, ge=0, le=1)
    uncertainty: float = Field(0.3, ge=0, le=1)
    complexity: float = Field(0.3, ge=0, le=1)
    # How important the task is to mission success, in (0, 1]
    criticality: float = Field(0.5, gt=0, le=1)
    # Optional expert override of the standalone failure probability
    base_risk: Optional[float] = Field(None, ge=0, le=1)


class Relation(BaseModel):
    type: RelationType
    sources: List[str] = Field(min_length=1)
    target: str
    strength: float = Field(0.5, ge=0, le=1)

    @model_validator(mode="after")
    def _check(self):
        if len(set(self.sources)) != len(self.sources):
            raise ValueError("Duplicate source tasks in relation.")
        if self.type == RelationType.hierarchical and len(self.sources) != 1:
            raise ValueError("A hierarchical relation needs exactly one source task.")
        if self.type == RelationType.synergistic and len(self.sources) < 2:
            raise ValueError("A synergistic relation needs two or more source tasks.")
        return self


class Mission(BaseModel):
    id: Optional[str] = None
    name: str
    description: str = ""
    tasks: List[Task] = Field(min_length=1)
    relations: List[Relation] = []

    @model_validator(mode="after")
    def _check(self):
        ids = [t.id for t in self.tasks]
        if len(set(ids)) != len(ids):
            raise ValueError("Task ids must be unique.")
        known = set(ids)
        for r in self.relations:
            for s in [*r.sources, r.target]:
                if s not in known:
                    raise ValueError(f"Relation references unknown task '{s}'.")
            if r.target in r.sources:
                raise ValueError("A task cannot depend on itself.")
        topo_order(ids, self.relations)  # raises on cycles
        return self


class WhatIfRequest(BaseModel):
    """Planner proposes mitigations: task_id -> new standalone risk in [0, 1]."""
    mitigations: Dict[str, float]


class BenchmarkRequest(BaseModel):
    n_scenarios: int = Field(30, ge=1, le=200)
    n_tasks: int = Field(15, ge=3, le=60)
    n_sim: int = Field(10000, ge=1000, le=50000)
    seed: int = 42
