from __future__ import annotations
import numpy as np

def opposition_based_learning(X: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    # classic OBL
    return lower + upper - X

def obl(pop: np.ndarray, lower: np.ndarray, upper: np.ndarray, n_route: int) -> np.ndarray:
    """
    Port của obl.m trong MATLAB
    """
    n = pop.shape[0]
    b1 = slice(0, n_route)
    b23 = slice(n_route, 3*n_route)

    # block1 integer-style
    a1 = np.broadcast_to(lower[b1], (n, n_route))
    b1u = np.broadcast_to(upper[b1], (n, n_route))
    pop1 = np.round(np.random.rand(n, n_route) * (a1 + b1u)) - pop[:, b1]
    lo1 = int(round(lower[b1][0]))
    hi1 = int(round(upper[b1][0]))
    idx = (pop1 < lo1) | (pop1 > hi1)
    if np.any(idx):
        r1 = np.random.randint(lo1, hi1 + 1, size=idx.sum())
        pop1[idx] = r1

    # block2+3
    a = np.broadcast_to(lower[b23], (n, 2 * n_route))
    b = np.broadcast_to(upper[b23], (n, 2 * n_route))
    pop2 = np.random.rand(n, 2 * n_route) * (a + b) - pop[:, b23]
    idx2 = (pop2 < a) | (pop2 > b)
    if np.any(idx2):
        r2 = a + (b - a) * np.random.rand(n, 2 * n_route)
        pop2[idx2] = r2[idx2]
    return np.hstack([pop1.astype(float), pop2])
