"""Monte Carlo simulation of mission execution with dependent task failures.

This acts as the "ground truth" generator used to validate the analytic model.
Unlike the analytic model (which propagates marginal probabilities), the
simulator propagates ACTUAL failures: a task is more likely to fail when its
upstream tasks really failed in that simulated run.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from .graph import topo_order
from .risk_model import CRITICAL_THRESHOLD, base_risks, incoming_relations
from .schemas import Mission, RelationType


def monte_carlo(mission: Mission, n_sim: int = 20000, seed: int = 0) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    base = base_risks(mission)
    order = topo_order([t.id for t in mission.tasks], mission.relations)
    inc = incoming_relations(mission)

    fail: Dict[str, np.ndarray] = {}
    for tid in order:
        survive = np.full(n_sim, 1.0 - base[tid])
        for rel in inc[tid]:
            if rel.type == RelationType.hierarchical:
                survive = survive * (1.0 - rel.strength * fail[rel.sources[0]])
            else:
                k = np.sum([fail[s] for s in rel.sources], axis=0)  # failed sources per run
                survive = survive * (1.0 - np.minimum(1.0, rel.strength * k))
        fail[tid] = (rng.random(n_sim) < (1.0 - survive)).astype(float)

    crit_w = np.array([t.criticality for t in mission.tasks])
    stack = np.vstack([fail[t.id] for t in mission.tasks])          # tasks x sims
    weighted_fraction = (crit_w[:, None] * stack).sum(axis=0) / crit_w.sum()

    crit_idx = [i for i, t in enumerate(mission.tasks) if t.criticality >= CRITICAL_THRESHOLD]
    if not crit_idx:
        crit_idx = list(range(len(mission.tasks)))
    any_crit_fail = stack[crit_idx].max(axis=0)

    return {
        "task_risk": {t.id: float(fail[t.id].mean()) for t in mission.tasks},
        "mission_risk": float(weighted_fraction.mean()),
        "critical_failure_prob": float(any_crit_fail.mean()),
        "n_sim": n_sim,
    }
