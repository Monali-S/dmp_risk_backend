"""Correlation-based mission risk model (hierarchical + synergistic + independent).

Idea (after Du & Chen, 2024): task risks are NOT independent.
  * independent  : task has no incoming relation -> adjusted risk = own risk
  * hierarchical : failure of ONE source task directly raises the target's risk
  * synergistic  : SEVERAL source tasks jointly raise the target's risk
                   (effects add up, which is stronger than independent causes)

Adjusted risk of task j (processed in topological order):

    survive_j = (1 - r_j)
                * prod_{hierarchical h: i -> j} (1 - s_h * r'_i)
                * prod_{synergistic  g: S -> j} (1 - min(1, s_g * sum_{k in S} r'_k))
    r'_j      = 1 - survive_j

Mission-level outputs
  * mission_risk          : criticality-weighted expected fraction of failed tasks
  * critical_failure_prob : P(at least one critical task fails)
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Optional

import numpy as np

from .graph import topo_order
from .schemas import Mission, RelationType, Task

FACTOR_WEIGHTS: Dict[str, float] = {
    "threat": 0.35,
    "resource_gap": 0.25,
    "uncertainty": 0.20,
    "complexity": 0.20,
}
CRITICAL_THRESHOLD = 0.9  # tasks with criticality >= this are "critical"


# ------------------------------------------------------------------ helpers
def base_risk(task: Task, weights: Dict[str, float] = FACTOR_WEIGHTS) -> float:
    """Standalone failure probability of a task (expert override wins)."""
    if task.base_risk is not None:
        return float(task.base_risk)
    return float(sum(w * getattr(task, k) for k, w in weights.items()))


def base_risks(mission: Mission, overrides: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    out = {t.id: base_risk(t) for t in mission.tasks}
    if overrides:
        for k, v in overrides.items():
            if k not in out:
                raise KeyError(f"Unknown task '{k}'")
            out[k] = float(min(max(v, 0.0), 1.0))
    return out


def incoming_relations(mission: Mission):
    inc = defaultdict(list)
    for r in mission.relations:
        inc[r.target].append(r)
    return inc


def risk_level(x: float) -> str:
    if x < 0.25:
        return "LOW"
    if x < 0.50:
        return "MODERATE"
    if x < 0.75:
        return "HIGH"
    return "CRITICAL"


# --------------------------------------------------------------- core model
def propagate(mission: Mission, base: Dict[str, float]) -> Dict[str, float]:
    """Adjusted (correlation-aware) risk for every task."""
    order = topo_order([t.id for t in mission.tasks], mission.relations)
    inc = incoming_relations(mission)
    adj: Dict[str, float] = {}
    for tid in order:
        survive = 1.0 - base[tid]
        for rel in inc[tid]:
            if rel.type == RelationType.hierarchical:
                survive *= 1.0 - rel.strength * adj[rel.sources[0]]
            else:
                joint = min(1.0, rel.strength * sum(adj[s] for s in rel.sources))
                survive *= 1.0 - joint
        adj[tid] = 1.0 - survive
    return adj


def mission_risk(mission: Mission, task_risk: Dict[str, float]) -> float:
    num = sum(t.criticality * task_risk[t.id] for t in mission.tasks)
    den = sum(t.criticality for t in mission.tasks)
    return float(num / den)


def critical_failure_prob(mission: Mission, task_risk: Dict[str, float]) -> float:
    crit = [t for t in mission.tasks if t.criticality >= CRITICAL_THRESHOLD] or mission.tasks
    surv = 1.0
    for t in crit:
        surv *= 1.0 - task_risk[t.id]
    return float(1.0 - surv)


def sensitivity(mission: Mission, base: Dict[str, float]) -> list:
    """Mission-risk reduction if each task were fully mitigated (risk -> 0)."""
    ref = mission_risk(mission, propagate(mission, base))
    rows = []
    for t in mission.tasks:
        trial = dict(base)
        trial[t.id] = 0.0
        new = mission_risk(mission, propagate(mission, trial))
        rows.append({"task_id": t.id, "name": t.name, "mission_risk_reduction": float(ref - new)})
    return sorted(rows, key=lambda r: r["mission_risk_reduction"], reverse=True)


def predict(mission: Mission, overrides: Optional[Dict[str, float]] = None) -> dict:
    """Full decision-support output for one mission."""
    base = base_risks(mission, overrides)
    adj = propagate(mission, base)
    mr = mission_risk(mission, adj)
    cf = critical_failure_prob(mission, adj)
    tasks = [
        {
            "task_id": t.id,
            "name": t.name,
            "criticality": t.criticality,
            "base_risk": round(base[t.id], 4),
            "adjusted_risk": round(adj[t.id], 4),
            "propagated_increase": round(adj[t.id] - base[t.id], 4),
            "level": risk_level(adj[t.id]),
        }
        for t in mission.tasks
    ]
    return {
        "mission": mission.name,
        "mission_risk": round(mr, 4),
        "mission_risk_level": risk_level(mr),
        "critical_failure_prob": round(cf, 4),
        "tasks": tasks,
        "top_risk_drivers": sensitivity(mission, base)[:5],
        "note": "Decision-support estimate only; a human commander makes the final decision.",
    }


def predict_with_uncertainty(
    mission: Mission, n: int = 300, concentration: float = 20.0, seed: int = 0
) -> dict:
    """Resample standalone risks from Beta(mean=r, concentration) -> 90% interval."""
    rng = np.random.default_rng(seed)
    base = base_risks(mission)
    draws = []
    for _ in range(n):
        b = {}
        for k, m in base.items():
            m = min(max(m, 1e-3), 1 - 1e-3)
            b[k] = float(rng.beta(m * concentration, (1 - m) * concentration))
        draws.append(mission_risk(mission, propagate(mission, b)))
    p5, p50, p95 = np.percentile(draws, [5, 50, 95])
    return {"p5": float(p5), "median": float(p50), "p95": float(p95), "n": n}


def what_if(mission: Mission, mitigations: Dict[str, float]) -> dict:
    before = predict(mission)
    after = predict(mission, overrides=mitigations)
    return {
        "before": {"mission_risk": before["mission_risk"], "critical_failure_prob": before["critical_failure_prob"]},
        "after": {"mission_risk": after["mission_risk"], "critical_failure_prob": after["critical_failure_prob"]},
        "mission_risk_change": round(after["mission_risk"] - before["mission_risk"], 4),
        "tasks_after": after["tasks"],
    }
