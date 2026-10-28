"""Synthetic data generation, a sample 15-task mission, and the benchmark.

All data here is SIMULATED / illustrative - no operational information is used.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from .baselines import entropy_baseline, independence_baseline
from .risk_model import critical_failure_prob, mission_risk, propagate, base_risks
from .schemas import Mission, Relation, RelationType, Task
from .simulation import monte_carlo


# ------------------------------------------------------------ sample mission
def sample_air_defence_mission() -> Mission:
    """Illustrative 15-task air-defence planning problem (synthetic numbers)."""
    # id, name, threat, resource_gap, uncertainty, complexity, criticality
    rows = [
        ("T01", "Surveillance coverage setup", 0.40, 0.30, 0.50, 0.30, 0.60),
        ("T02", "Radar deployment", 0.50, 0.40, 0.30, 0.40, 0.70),
        ("T03", "Comms network establishment", 0.45, 0.35, 0.40, 0.50, 0.80),
        ("T04", "Target detection", 0.55, 0.30, 0.55, 0.40, 0.80),
        ("T05", "Track correlation", 0.40, 0.25, 0.45, 0.55, 0.70),
        ("T06", "Threat identification", 0.60, 0.30, 0.60, 0.50, 0.90),
        ("T07", "Command & control decision", 0.50, 0.35, 0.50, 0.60, 0.95),
        ("T08", "Asset allocation", 0.40, 0.55, 0.40, 0.50, 0.75),
        ("T09", "Logistics & ammunition supply", 0.35, 0.60, 0.35, 0.45, 0.65),
        ("T10", "Interceptor readiness", 0.45, 0.50, 0.30, 0.50, 0.85),
        ("T11", "Engagement coordination", 0.55, 0.40, 0.50, 0.65, 0.90),
        ("T12", "Electronic protection", 0.65, 0.45, 0.55, 0.60, 0.70),
        ("T13", "Civil-asset deconfliction", 0.30, 0.25, 0.45, 0.50, 0.60),
        ("T14", "Battle damage assessment", 0.35, 0.30, 0.50, 0.40, 0.50),
        ("T15", "Re-tasking & recovery", 0.40, 0.45, 0.45, 0.45, 0.55),
    ]
    tasks = [
        Task(id=i, name=n, threat=a, resource_gap=b, uncertainty=c, complexity=d, criticality=e)
        for i, n, a, b, c, d, e in rows
    ]
    H, S = RelationType.hierarchical, RelationType.synergistic
    rel = [
        Relation(type=H, sources=["T01"], target="T04", strength=0.55),
        Relation(type=H, sources=["T02"], target="T04", strength=0.50),
        Relation(type=H, sources=["T04"], target="T05", strength=0.60),
        Relation(type=H, sources=["T05"], target="T06", strength=0.60),
        Relation(type=S, sources=["T03", "T06"], target="T07", strength=0.35),
        Relation(type=H, sources=["T07"], target="T08", strength=0.50),
        Relation(type=H, sources=["T09"], target="T10", strength=0.45),
        Relation(type=S, sources=["T08", "T10", "T03"], target="T11", strength=0.25),
        Relation(type=H, sources=["T12"], target="T11", strength=0.35),
        Relation(type=H, sources=["T06"], target="T13", strength=0.30),
        Relation(type=H, sources=["T11"], target="T14", strength=0.55),
        Relation(type=S, sources=["T14", "T13"], target="T15", strength=0.30),
    ]
    return Mission(id="sample-air-defence", name="Air-defence planning (sample, 15 tasks)",
                   description="Synthetic example for prototype testing.", tasks=tasks, relations=rel)


# ------------------------------------------------------- synthetic generator
def generate_mission(n_tasks: int, rng: np.random.Generator, name: str = "synthetic") -> Mission:
    tasks: List[Task] = []
    for i in range(n_tasks):
        tasks.append(Task(
            id=f"T{i+1:02d}", name=f"Task {i+1}",
            threat=float(rng.beta(2, 3)), resource_gap=float(rng.beta(2, 3)),
            uncertainty=float(rng.beta(2, 3)), complexity=float(rng.beta(2, 3)),
            criticality=float(rng.uniform(0.2, 1.0)),
        ))
    relations: List[Relation] = []
    for j in range(1, n_tasks):               # edges only go low index -> high index => DAG
        target = tasks[j].id
        r = rng.random()
        if r < 0.45:                           # hierarchical parent
            src = tasks[int(rng.integers(0, j))].id
            relations.append(Relation(type=RelationType.hierarchical, sources=[src],
                                      target=target, strength=float(rng.uniform(0.25, 0.8))))
        elif r < 0.65 and j >= 2:              # synergistic group
            k = int(rng.integers(2, min(3, j) + 1))
            idx = rng.choice(j, size=k, replace=False)
            relations.append(Relation(type=RelationType.synergistic,
                                      sources=[tasks[int(x)].id for x in idx], target=target,
                                      strength=float(rng.uniform(0.1, 0.45))))
        # else: independent task
    return Mission(name=name, tasks=tasks, relations=relations)


# ------------------------------------------------------------------ benchmark
def compare_methods(mission: Mission, n_sim: int = 20000, seed: int = 0) -> Dict[str, Dict[str, float]]:
    truth = monte_carlo(mission, n_sim=n_sim, seed=seed)
    adj = propagate(mission, base_risks(mission))
    model = {"mission_risk": mission_risk(mission, adj),
             "critical_failure_prob": critical_failure_prob(mission, adj)}
    return {
        "monte_carlo_truth": {k: truth[k] for k in ("mission_risk", "critical_failure_prob")},
        "correlation_model": model,
        "independence_baseline": independence_baseline(mission),
        "entropy_baseline": entropy_baseline(mission),
    }


def run_benchmark(n_scenarios: int = 30, n_tasks: int = 15, n_sim: int = 10000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_scenarios):
        m = generate_mission(n_tasks, rng, name=f"scenario-{s}")
        res = compare_methods(m, n_sim=n_sim, seed=seed + s)
        truth = res["monte_carlo_truth"]
        row = {"scenario": s, "n_relations": len(m.relations),
               "truth_mission_risk": truth["mission_risk"],
               "truth_critical_prob": truth["critical_failure_prob"]}
        for method in ("correlation_model", "independence_baseline", "entropy_baseline"):
            row[f"{method}__mission_err"] = abs(res[method]["mission_risk"] - truth["mission_risk"])
            row[f"{method}__critical_err"] = abs(res[method]["critical_failure_prob"] - truth["critical_failure_prob"])
        rows.append(row)

    def mean(key: str) -> float:
        return float(np.mean([r[key] for r in rows]))

    summary = {}
    for metric in ("mission", "critical"):
        mae = {m: mean(f"{m}__{metric}_err") for m in
               ("correlation_model", "independence_baseline", "entropy_baseline")}
        summary[f"{metric}_MAE"] = mae
        summary[f"{metric}_deviation_reduction_vs_independence_%"] = round(
            100 * (1 - mae["correlation_model"] / mae["independence_baseline"]), 1)
        summary[f"{metric}_deviation_reduction_vs_entropy_%"] = round(
            100 * (1 - mae["correlation_model"] / mae["entropy_baseline"]), 1)
    return {"config": {"n_scenarios": n_scenarios, "n_tasks": n_tasks, "n_sim": n_sim, "seed": seed},
            "summary": summary, "rows": rows}
