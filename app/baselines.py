"""Baseline models that ASSUME task risks are independent.

1. independence_baseline : same task risks as our model, but relations ignored.
2. entropy_baseline      : entropy-weight method (factor weights learned from data),
                           also ignoring relations. Mirrors a baseline family
                           used in the risk-prediction literature.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from .risk_model import (
    CRITICAL_THRESHOLD,
    base_risks,
    critical_failure_prob,
    mission_risk,
)
from .schemas import Mission

FACTORS = ["threat", "resource_gap", "uncertainty", "complexity"]


def independence_baseline(mission: Mission) -> Dict[str, float]:
    base = base_risks(mission)
    return {
        "mission_risk": mission_risk(mission, base),
        "critical_failure_prob": critical_failure_prob(mission, base),
    }


def entropy_weights(matrix: np.ndarray) -> np.ndarray:
    """Entropy weight method over a (tasks x factors) matrix."""
    n = matrix.shape[0]
    if n < 2:
        return np.full(matrix.shape[1], 1.0 / matrix.shape[1])
    x = matrix + 1e-9
    p = x / x.sum(axis=0, keepdims=True)
    entropy = -(p * np.log(p)).sum(axis=0) / np.log(n)
    d = 1.0 - entropy
    if d.sum() <= 1e-12:
        return np.full(matrix.shape[1], 1.0 / matrix.shape[1])
    return d / d.sum()


def entropy_baseline(mission: Mission) -> Dict[str, float]:
    mat = np.array([[getattr(t, f) for f in FACTORS] for t in mission.tasks], dtype=float)
    w = entropy_weights(mat)
    scores = {t.id: float(mat[i] @ w) for i, t in enumerate(mission.tasks)}
    return {
        "mission_risk": mission_risk(mission, scores),
        "critical_failure_prob": critical_failure_prob(mission, scores),
    }
