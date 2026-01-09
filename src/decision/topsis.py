from __future__ import annotations
import numpy as np


def topsis_generic(
    matrix: np.ndarray,
    weights: np.ndarray,
    benefit_mask: np.ndarray,
) -> np.ndarray:
    """
    General TOPSIS implementation.

    Parameters
    ----------
    matrix : (N, M) ndarray
        Decision matrix.
    weights : (M,) ndarray
        Criterion weights.
    benefit_mask : (M,) ndarray of bool
        True for benefit criteria, False for cost criteria.

    Returns
    -------
    scores : (N,) ndarray
        TOPSIS scores (higher is better).
    """
    X = np.asarray(matrix, dtype=float)
    n, m = X.shape

    w = np.asarray(weights, dtype=float).reshape(-1)
    w = w / (w.sum() + 1e-12)

    benefit = np.asarray(benefit_mask, dtype=bool).reshape(-1)

    norm = np.linalg.norm(X, axis=0, keepdims=True) + 1e-12
    V = (X / norm) * w

    ideal_best = np.where(benefit, V.max(axis=0), V.min(axis=0))
    ideal_worst = np.where(benefit, V.min(axis=0), V.max(axis=0))

    d_best = np.linalg.norm(V - ideal_best, axis=1)
    d_worst = np.linalg.norm(V - ideal_worst, axis=1)

    return d_worst / (d_best + d_worst + 1e-12)


def topsis_matlab(obj: np.ndarray) -> tuple[np.ndarray, int, np.ndarray]:
    """
    TOPSIS variant consistent with the original project implementation.

    Parameters
    ----------
    obj : (N, M) ndarray
        Objective matrix (all objectives are minimized).

    Returns
    -------
    obj_best : (M,) ndarray
        Selected objective vector.
    index_best : int
        Index of the selected solution.
    scores : (N,) ndarray
        TOPSIS scores.
    """
    X = np.asarray(obj, dtype=float)
    if X.ndim != 2:
        raise ValueError("obj must be a 2D array")

    n, m = X.shape

    max_col = X.max(axis=0, keepdims=True)
    norm_obj = max_col - X

    denom = np.sqrt((norm_obj ** 2).sum(axis=0, keepdims=True)) + 1e-12
    norm_obj = norm_obj / denom

    ideal_best = norm_obj.max(axis=0, keepdims=True)
    ideal_worst = norm_obj.min(axis=0, keepdims=True)

    d_best = np.sqrt(((ideal_best - norm_obj) ** 2).sum(axis=1))
    d_worst = np.sqrt(((ideal_worst - norm_obj) ** 2).sum(axis=1))

    scores = d_worst / (d_best + d_worst + 1e-12)

    index_best = int(np.argmax(scores))
    obj_best = X[index_best].copy()

    return obj_best, index_best, scores
