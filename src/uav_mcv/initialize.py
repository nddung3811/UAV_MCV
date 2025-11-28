from __future__ import annotations
import numpy as np

def initialize(pop_size: int, n_route: int, lower: np.ndarray, upper: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:

    rng = rng or np.random.default_rng()
    half = pop_size // 2

    # Slices
    b1 = slice(0, n_route)
    b2 = slice(n_route, 2 * n_route)
    b3 = slice(2 * n_route, 3 * n_route)

    # ----- Half #1: pure random -----
    # pop1: int in [lower(1), upper(1)] for block1 (assumes uniform bound across edges for block1)
    lo1 = int(np.round(lower[b1][0]))
    hi1 = int(np.round(upper[b1][0]))
    # numpy integers: high is exclusive -> +1
    pop1 = rng.integers(low=lo1, high=hi1 + 1, size=(half, n_route))

    # pop2: block2 random in [lower_i, upper_i]
    lo2 = lower[b2]
    hi2 = upper[b2]
    pop2 = lo2 + (hi2 - lo2) * rng.random((half, n_route))

    # pop3: block3 random in [lower[2*n_route], upper[2*n_route]] (MATLAB used scalar at 3*n_route)
    lo3 = float(lower[b3][-1])
    hi3 = float(upper[b3][-1])
    pop3 = lo3 + (hi3 - lo3) * rng.random((half, n_route))

    pop4 = np.hstack([pop1.astype(float), pop2, pop3])

    # ----- Half #2: opposition-like (OBL-style) -----
    # For block1 (integers)
    a1 = np.broadcast_to(lower[b1], (half, n_route))
    b1u = np.broadcast_to(upper[b1], (half, n_route))
    # MATLAB: pop5 = round(rand*(a+b)) - pop1;
    pop5 = np.round(rng.random((half, n_route)) * (a1 + b1u)) - pop1
    # repair out-of-bound for block1:
    idx = (pop5 < lo1) | (pop5 > hi1)
    if np.any(idx):
        pop5[idx] = rng.integers(low=lo1, high=hi1 + 1, size=idx.sum())
    pop5 = pop5.astype(float)

    # For block2+block3 (floats)
    a23 = np.broadcast_to(lower[b2.start:b3.stop], (half, 2 * n_route))
    b23 = np.broadcast_to(upper[b2.start:b3.stop], (half, 2 * n_route))
    base23 = np.hstack([pop2, pop3])  # same shapes as in MATLAB: [pop2, pop3]
    pop6 = rng.random((half, 2 * n_route)) * (a23 + b23) - base23
    # repair
    idx23 = (pop6 < a23) | (pop6 > b23)
    if np.any(idx23):
        r2 = a23 + (b23 - a23) * rng.random((half, 2 * n_route))
        pop6[idx23] = r2[idx23]

    pop7 = np.hstack([pop5, pop6])

    # Final population
    pop = np.vstack([pop4, pop7])

    # If pop_size is odd, add one more random row to match MATLAB behavior (rare).
    if pop.shape[0] < pop_size:
        extra = np.zeros((1, 3 * n_route), dtype=float)
        extra[0, b1] = rng.integers(low=lo1, high=hi1 + 1, size=n_route)
        extra[0, b2] = lo2 + (hi2 - lo2) * rng.random(n_route)
        extra[0, b3] = lo3 + (hi3 - lo3) * rng.random(n_route)
        pop = np.vstack([pop, extra])

    return pop


def initialize_obl(pop_size: int, n_route: int, lower: np.ndarray, upper: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Python port of MATLAB initialize_obl.m
    """
    rng = rng or np.random.default_rng()

    b1 = slice(0, n_route)
    b2 = slice(n_route, 2 * n_route)
    b3 = slice(2 * n_route, 3 * n_route)

    lo1 = int(np.round(lower[b1][0]))
    hi1 = int(np.round(upper[b1][0]))

    pop1 = rng.integers(low=lo1, high=hi1 + 1, size=(pop_size, n_route)).astype(float)

    lo2 = lower[b2]; hi2 = upper[b2]
    pop2 = lo2 + (hi2 - lo2) * rng.random((pop_size, n_route))

    lo3 = float(lower[b3][-1]); hi3 = float(upper[b3][-1])
    pop3 = lo3 + (hi3 - lo3) * rng.random((pop_size, n_route))

    return np.hstack([pop1, pop2, pop3])
