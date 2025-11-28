from __future__ import annotations
import numpy as np

def opposition_based_learning(X: np.ndarray,
                              lower: np.ndarray,
                              upper: np.ndarray) -> np.ndarray:
    """Classic OBL: X' = lower + upper - X"""
    return lower + upper - X


def obl(pop: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        n_route: int,
        rng: np.random.Generator) -> np.ndarray:
    """
    Port của obl.m trong MATLAB (refactor dùng rng):
      - pop : (n, 3*n_route)
      - lower, upper: (3*n_route,)
      - n_route: số cung
      - rng: np.random.Generator được seed từ simulate.py
    """
    n = pop.shape[0]
    b1  = slice(0, n_route)
    b23 = slice(n_route, 3 * n_route)

    # ===== Block 1 (integer-like) =====
    a1  = np.broadcast_to(lower[b1], (n, n_route))
    b1u = np.broadcast_to(upper[b1], (n, n_route))

    # MATLAB: pop1 = round(rand(n, n_route) .* (a1 + b1u)) - pop(:, b1);
    pop1 = np.round(rng.random((n, n_route)) * (a1 + b1u)) - pop[:, b1]

    lo1 = int(round(lower[b1][0]))
    hi1 = int(round(upper[b1][0]))
    idx = (pop1 < lo1) | (pop1 > hi1)
    if np.any(idx):
        # random int trong [lo1, hi1]
        r1 = rng.integers(lo1, hi1 + 1, size=idx.sum())
        pop1[idx] = r1

    pop1 = pop1.astype(float)

    # ===== Block 2+3 (continuous) =====
    a = np.broadcast_to(lower[b23], (n, 2 * n_route))
    b = np.broadcast_to(upper[b23], (n, 2 * n_route))

    # MATLAB: pop2 = rand(n,2*n_route).*(a+b) - pop(:,b23);
    pop2 = rng.random((n, 2 * n_route)) * (a + b) - pop[:, b23]

    idx2 = (pop2 < a) | (pop2 > b)
    if np.any(idx2):
        # repair bằng uniform trong [a,b]
        r2 = a + (b - a) * rng.random((n, 2 * n_route))
        pop2[idx2] = r2[idx2]

    return np.hstack([pop1, pop2])
