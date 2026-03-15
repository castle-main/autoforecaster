"""Platt scaling calibration using logistic regression."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from .types import PlattParams, PROJECT_ROOT

CLAMP_MIN = 0.001
CLAMP_MAX = 0.999
MIN_SAMPLES = 10


def _clamp(p: float) -> float:
    return max(CLAMP_MIN, min(CLAMP_MAX, p))


def _logit(p: float) -> float:
    p = _clamp(p)
    return np.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def _brier(probs: list[float], outcomes: list[int]) -> float:
    return float(np.mean([(p - o) ** 2 for p, o in zip(probs, outcomes)]))


def fit_platt(raw_probs: list[float], outcomes: list[int]) -> PlattParams:
    """Fit Platt scaling parameters: sigmoid(a * logit(p) + b)."""
    if len(raw_probs) < MIN_SAMPLES:
        raise ValueError(f"Need at least {MIN_SAMPLES} samples, got {len(raw_probs)}")
    if len(raw_probs) != len(outcomes):
        raise ValueError("raw_probs and outcomes must have same length")

    X = np.array([_logit(p) for p in raw_probs]).reshape(-1, 1)
    y = np.array(outcomes)

    # C=1e10 for essentially no regularization
    lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
    lr.fit(X, y)

    a = float(lr.coef_[0][0])
    b = float(lr.intercept_[0])

    calibrated = [apply_platt_raw(p, a, b) for p in raw_probs]
    brier_before = _brier(raw_probs, outcomes)
    brier_after = _brier(calibrated, outcomes)

    return PlattParams(
        a=a, b=b,
        n_samples=len(raw_probs),
        brier_before=brier_before,
        brier_after=brier_after,
    )


def apply_platt_raw(p: float, a: float, b: float) -> float:
    """Apply Platt scaling given raw parameters."""
    return float(_sigmoid(a * _logit(p) + b))


def apply_platt(raw_prob: float, params: PlattParams) -> float:
    """Apply Platt scaling using PlattParams."""
    return apply_platt_raw(raw_prob, params.a, params.b)


def save_params(params: PlattParams, path: Path | None = None) -> None:
    path = path or PROJECT_ROOT / "data" / "platt_params.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(params.model_dump(), f, indent=2)


def load_params(path: Path | None = None) -> PlattParams | None:
    """Load Platt params. Returns None if file doesn't exist."""
    path = path or PROJECT_ROOT / "data" / "platt_params.json"
    if not path.exists():
        return None
    with open(path) as f:
        return PlattParams.model_validate(json.load(f))
