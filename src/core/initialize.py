from __future__ import annotations
import numpy as np


def initialize_population(
    pop_size: int,
    n_edges: int,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Initialize mixed-integer population.

    Chromosome structure:
    - Block 1: discrete vehicle assignment (length = n_edges)
    - Block 2: continuous charging ratio  (length = n_edges)
    - Block 3: continuous charging time   (length = n_edges)

    Strategy:
    - First half: purely random
    - Second half: opposition-style initialization
    """
    rng = rng or np.random.default_rng()
    half = pop_size // 2

    # Index slices
    idx_assign = slice(0, n_edges)
    idx_ratio  = slice(n_edges, 2 * n_edges)
    idx_time   = slice(2 * n_edges, 3 * n_edges)

    # =====================================================
    # First half: random initialization
    # =====================================================

    assign_lo = int(np.round(lower_bound[idx_assign][0]))
    assign_hi = int(np.round(upper_bound[idx_assign][0]))

    assign_rand = rng.integers(
        low=assign_lo,
        high=assign_hi + 1,
        size=(half, n_edges),
    )

    ratio_lo = lower_bound[idx_ratio]
    ratio_hi = upper_bound[idx_ratio]
    ratio_rand = ratio_lo + (ratio_hi - ratio_lo) * rng.random((half, n_edges))

    time_lo = float(lower_bound[idx_time][-1])
    time_hi = float(upper_bound[idx_time][-1])
    time_rand = time_lo + (time_hi - time_lo) * rng.random((half, n_edges))

    pop_random = np.hstack([
        assign_rand.astype(float),
        ratio_rand,
        time_rand,
    ])

    # =====================================================
    # Second half: opposition-style initialization
    # =====================================================

    lb_assign = np.broadcast_to(lower_bound[idx_assign], (half, n_edges))
    ub_assign = np.broadcast_to(upper_bound[idx_assign], (half, n_edges))

    assign_opposite = np.round(
        rng.random((half, n_edges)) * (lb_assign + ub_assign)
    ) - assign_rand

    invalid = (assign_opposite < assign_lo) | (assign_opposite > assign_hi)
    if np.any(invalid):
        assign_opposite[invalid] = rng.integers(
            low=assign_lo,
            high=assign_hi + 1,
            size=invalid.sum(),
        )

    assign_opposite = assign_opposite.astype(float)

    lb_cont = np.broadcast_to(
        lower_bound[idx_ratio.start:idx_time.stop],
        (half, 2 * n_edges),
    )
    ub_cont = np.broadcast_to(
        upper_bound[idx_ratio.start:idx_time.stop],
        (half, 2 * n_edges),
    )

    base_cont = np.hstack([ratio_rand, time_rand])
    cont_opposite = rng.random((half, 2 * n_edges)) * (lb_cont + ub_cont) - base_cont

    invalid = (cont_opposite < lb_cont) | (cont_opposite > ub_cont)
    if np.any(invalid):
        repaired = lb_cont + (ub_cont - lb_cont) * rng.random((half, 2 * n_edges))
        cont_opposite[invalid] = repaired[invalid]

    pop_opposite = np.hstack([assign_opposite, cont_opposite])

    # =====================================================
    # Merge population
    # =====================================================

    population = np.vstack([pop_random, pop_opposite])

    # Handle odd pop_size (rare but consistent)
    if population.shape[0] < pop_size:
        extra = np.zeros((1, 3 * n_edges), dtype=float)
        extra[0, idx_assign] = rng.integers(assign_lo, assign_hi + 1, size=n_edges)
        extra[0, idx_ratio] = ratio_lo + (ratio_hi - ratio_lo) * rng.random(n_edges)
        extra[0, idx_time] = time_lo + (time_hi - time_lo) * rng.random(n_edges)
        population = np.vstack([population, extra])

    return population


def initialize_population_random(
    pop_size: int,
    n_edges: int,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Pure random initialization without opposition.
    """
    rng = rng or np.random.default_rng()

    idx_assign = slice(0, n_edges)
    idx_ratio  = slice(n_edges, 2 * n_edges)
    idx_time   = slice(2 * n_edges, 3 * n_edges)

    assign_lo = int(np.round(lower_bound[idx_assign][0]))
    assign_hi = int(np.round(upper_bound[idx_assign][0]))

    assign = rng.integers(assign_lo, assign_hi + 1, size=(pop_size, n_edges)).astype(float)

    ratio_lo = lower_bound[idx_ratio]
    ratio_hi = upper_bound[idx_ratio]
    ratio = ratio_lo + (ratio_hi - ratio_lo) * rng.random((pop_size, n_edges))

    time_lo = float(lower_bound[idx_time][-1])
    time_hi = float(upper_bound[idx_time][-1])
    time = time_lo + (time_hi - time_lo) * rng.random((pop_size, n_edges))

    return np.hstack([assign, ratio, time])
