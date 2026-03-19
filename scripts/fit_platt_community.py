#!/usr/bin/env python3
"""Fit Platt scaling params against community predictions from test traces."""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent.parent))

from autoforecast.types import PROJECT_ROOT

CLAMP_MIN = 0.001
CLAMP_MAX = 0.999


def clamp(p):
    return max(CLAMP_MIN, min(CLAMP_MAX, p))


def logit(p):
    p = clamp(p)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    trace_dir = PROJECT_ROOT / "logs" / "test_traces"

    # Load all non-cluster traces
    raw_probs = []
    community_probs = []
    for path in sorted(trace_dir.glob("*.json")):
        if path.name.startswith("cluster_"):
            continue
        with open(path) as f:
            data = json.load(f)
        raw_p = data["raw_probability"]
        community_p = data["question"]["community_prediction_final"]
        raw_probs.append(raw_p)
        community_probs.append(community_p)

    raw_probs = np.array(raw_probs)
    community_probs = np.array(community_probs)
    logits = np.array([logit(p) for p in raw_probs])

    print(f"Loaded {len(raw_probs)} traces.")
    print(f"Raw probs: mean={raw_probs.mean():.4f}, std={raw_probs.std():.4f}")
    print(f"Community: mean={community_probs.mean():.4f}, std={community_probs.std():.4f}")

    # Current params
    with open(PROJECT_ROOT / "data" / "platt_params.json") as f:
        old_params = json.load(f)
    print(f"\nOld params: a={old_params['a']:.4f}, b={old_params['b']:.4f}")

    old_calibrated = np.array([sigmoid(old_params["a"] * l + old_params["b"]) for l in logits])
    old_mse = np.mean((old_calibrated - community_probs) ** 2)
    print(f"Old MSE vs community: {old_mse:.6f}")

    # Fit new params: minimize MSE(sigmoid(a*logit(raw) + b), community)
    def objective(params):
        a, b = params
        calibrated = sigmoid(a * logits + b)
        return np.mean((calibrated - community_probs) ** 2)

    result = minimize(objective, x0=[1.0, 0.0], method="Nelder-Mead")
    a_new, b_new = result.x

    new_calibrated = np.array([sigmoid(a_new * l + b_new) for l in logits])
    new_mse = np.mean((new_calibrated - community_probs) ** 2)

    print(f"\nNew params: a={a_new:.4f}, b={b_new:.4f}")
    print(f"New MSE vs community: {new_mse:.6f}")
    print(f"MSE improvement: {old_mse - new_mse:.6f} ({(1 - new_mse/old_mse)*100:.1f}%)")

    # Show before/after divergence stats
    old_div = old_calibrated - community_probs
    new_div = new_calibrated - community_probs
    print(f"\nOld divergence: mean={old_div.mean():+.4f}, abs_mean={np.abs(old_div).mean():.4f}")
    print(f"New divergence: mean={new_div.mean():+.4f}, abs_mean={np.abs(new_div).mean():.4f}")

    # Save
    new_params = {
        "a": float(a_new),
        "b": float(b_new),
        "n_samples": len(raw_probs),
        "brier_before": float(old_mse),
        "brier_after": float(new_mse),
    }
    params_path = PROJECT_ROOT / "data" / "platt_params.json"
    with open(params_path, "w") as f:
        json.dump(new_params, f, indent=2)
    print(f"\nSaved new params to {params_path}")


if __name__ == "__main__":
    main()
