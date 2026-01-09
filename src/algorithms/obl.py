from __future__ import annotations
import numpy as np


def opposition_point(x: np.ndarray,
                     lower_bound: np.ndarray,
                     upper_bound: np.ndarray) -> np.ndarray:
    """Opposition point: x' = lower + upper - x"""
    return lower_bound + upper_bound - x


def opposition_based_learning(
    population: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    n_routes: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Opposition-based learning for mixed variables.
    """
    pop_size = population.shape[0]

    idx_discrete = slice(0, n_routes)
    idx_continuous = slice(n_routes, 3 * n_routes)

    # Discrete block
    lb_d = np.broadcast_to(lower_bound[idx_discrete], (pop_size, n_routes))
    ub_d = np.broadcast_to(upper_bound[idx_discrete], (pop_size, n_routes))

    discrete = np.round(rng.random((pop_size, n_routes)) * (lb_d + ub_d)) - population[:, idx_discrete]

    lo = int(round(lower_bound[idx_discrete][0]))
    hi = int(round(upper_bound[idx_discrete][0]))

    invalid = (discrete < lo) | (discrete > hi)
    if np.any(invalid):
        discrete[invalid] = rng.integers(lo, hi + 1, size=invalid.sum())

    discrete = discrete.astype(float)

    # Continuous block
    lb_c = np.broadcast_to(lower_bound[idx_continuous], (pop_size, 2 * n_routes))
    ub_c = np.broadcast_to(upper_bound[idx_continuous], (pop_size, 2 * n_routes))

    continuous = rng.random((pop_size, 2 * n_routes)) * (lb_c + ub_c) - population[:, idx_continuous]

    invalid = (continuous < lb_c) | (continuous > ub_c)
    if np.any(invalid):
        continuous[invalid] = lb_c[invalid] + (ub_c[invalid] - lb_c[invalid]) * rng.random(invalid.sum())

    return np.hstack([discrete, continuous])
